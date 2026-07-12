"""Adversarial, frozen-corpus equivalence tests for incremental episode ledgers.

Every path is deliberately redirected to ``tmp_path``.  These tests are a
semantic proof: the only tolerated difference is ``last_rebuild_ts``.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

import execution.episode_ledger as ledger_module


def _fill(
    ts: str,
    side: str,
    reduce_only: bool,
    *,
    symbol: str = "BTCUSDT",
    position_side: str = "LONG",
    price: float = 100.0,
    qty: float = 1.0,
    fee: float = 0.1,
    order_id: str = "",
    entry_price: float | None = None,
) -> dict:
    metadata: dict = {"strategy": "fixture", "exit": {"reason": "tp"}}
    if entry_price is not None:
        metadata["entry_price"] = entry_price
    return {
        "event_type": "order_fill", "ts": ts, "ts_fill_first": ts,
        "symbol": symbol, "positionSide": position_side, "side": side,
        "reduceOnly": reduce_only, "executedQty": qty, "avgPrice": price,
        "fee_total": fee, "orderId": order_id or f"{symbol}-{ts}-{side}",
        "metadata": metadata,
    }


def _append(path: Path, *events: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def _record(path: Path) -> dict:
    stat = path.stat()
    raw = path.read_bytes()
    return {
        "path": str(path), "size": stat.st_size,
        "lines": raw.count(b"\n"), "sha256": hashlib.sha256(raw).hexdigest(),
        "source_identity": {"device": stat.st_dev, "inode": stat.st_ino},
    }


class FrozenCorpus:
    """Fixture-only provenance manifest with strict semantic input checking."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.execution = root / "execution"
        self.state = root / "state"
        self.active = self.execution / "orders_executed.jsonl"
        self.dle = self.execution / "dle_shadow_events.jsonl"
        self.nav = self.state / "nav_state.json"
        self.manifest_path = root / "frozen_corpus_manifest.json"
        self.execution.mkdir(parents=True)
        self.state.mkdir()
        self.dle.write_text("", encoding="utf-8")
        self.nav.write_text(json.dumps({"total_equity": 10_000.0}), encoding="utf-8")
        self.snapshots: dict[str, dict] = {}

    def freeze(self, label: str, *, require_resume_state: bool = False) -> dict:
        required = [self.active, self.dle, self.nav]
        if require_resume_state:
            required += [self.state / "episode_ledger.json", self.state / "episode_ledger_checkpoint.json"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise AssertionError(f"frozen corpus missing required semantic input: {missing}")
        sources = sorted(self.execution.glob("orders_executed.*.jsonl"), reverse=True)
        if self.active.exists():
            sources.append(self.active)
        snapshot = {
            "label": label,
            "execution_log_read_order": [path.name for path in sources],
            "execution_logs": [_record(path) for path in sources],
            "dle_authority_log": _record(self.dle),
            "nav_state": _record(self.nav),
            "starting_ledger": _record(self.state / "episode_ledger.json") if (self.state / "episode_ledger.json").exists() else None,
            "starting_checkpoint": _record(self.state / "episode_ledger_checkpoint.json") if (self.state / "episode_ledger_checkpoint.json").exists() else None,
            "checkpoint_offsets_and_anchors": self._checkpoint_sources(),
            "canonicalization": {"excluded": ["last_rebuild_ts"], "json": "sorted keys, compact separators, UTF-8"},
            "builder": {"checkpoint_version": ledger_module.EPISODE_LEDGER_CHECKPOINT_VERSION, "git_sha": ledger_module._git_commit()},
        }
        self.snapshots[label] = snapshot
        self.manifest_path.write_text(json.dumps({"snapshots": self.snapshots}, indent=2, sort_keys=True), encoding="utf-8")
        return snapshot

    def assert_unchanged(self, label: str) -> None:
        expected = self.snapshots[label]
        now = self.freeze(f"{label}-verification", require_resume_state=bool(expected["starting_ledger"]))
        for key in ("execution_log_read_order", "execution_logs", "dle_authority_log", "nav_state", "starting_ledger", "starting_checkpoint"):
            assert now[key] == expected[key], f"frozen semantic input silently changed: {key}"

    def _checkpoint_sources(self) -> list[dict]:
        path = self.state / "episode_ledger_checkpoint.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8")).get("sources", [])


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FrozenCorpus:
    result = FrozenCorpus(tmp_path / "episode-ledger-corpus")
    monkeypatch.setattr(ledger_module, "EXECUTION_LOG_DIR", result.execution)
    monkeypatch.setattr(ledger_module, "EXECUTION_LOG_PATH", result.active)
    monkeypatch.setattr(ledger_module, "DLE_SHADOW_LOG_PATH", result.dle)
    monkeypatch.setattr(ledger_module, "NAV_STATE_PATH", result.nav)
    monkeypatch.setattr(ledger_module, "EPISODE_LEDGER_PATH", result.state / "episode_ledger.json")
    monkeypatch.setattr(ledger_module, "EPISODE_LEDGER_CHECKPOINT_PATH", result.state / "episode_ledger_checkpoint.json")
    monkeypatch.setattr(ledger_module, "EPISODE_LEDGER_REBUILD_LOG_PATH", result.execution / "rebuild.jsonl")
    return result


def _semantic(ledger: ledger_module.EpisodeLedger) -> dict:
    payload = deepcopy(ledger.to_dict())
    payload.pop("last_rebuild_ts", None)
    return payload


def _assert_equivalent(full: ledger_module.EpisodeLedger, incremental: ledger_module.EpisodeLedger) -> None:
    left, right = _semantic(full), _semantic(incremental)
    assert left.keys() == right.keys()
    assert left["episode_count"] == right["episode_count"]
    assert [x.get("episode_uid") for x in left.get("episodes_v2", [])] == [x.get("episode_uid") for x in right.get("episodes_v2", [])]
    assert {x.get("episode_uid") for x in left.get("episodes_v2", [])} == {x.get("episode_uid") for x in right.get("episodes_v2", [])}
    assert left["episodes"] == right["episodes"]
    assert left.get("episodes_v2", []) == right.get("episodes_v2", [])
    assert left["stats"]["reconciliation"] == right["stats"]["reconciliation"]
    assert left["stats"]["exit_reasons"] == right["stats"]["exit_reasons"]
    assert left["stats"] == right["stats"]
    assert ledger_module._canonical_ledger_hash(full) == ledger_module._canonical_ledger_hash(incremental)


def _initial(corpus: FrozenCorpus, *events: dict) -> ledger_module.EpisodeLedger:
    _append(corpus.active, *events)
    corpus.freeze("initial")
    result = ledger_module.rebuild_and_save()
    corpus.freeze("initial-resume", require_resume_state=True)
    return result


def _check(corpus: FrozenCorpus, label: str) -> ledger_module.EpisodeLedger:
    corpus.freeze(label, require_resume_state=True)
    incremental = ledger_module.rebuild_and_save()
    full = ledger_module.build_episode_ledger()
    _assert_equivalent(full, incremental)
    return incremental


def _last_telemetry(corpus: FrozenCorpus) -> dict:
    return json.loads((corpus.execution / "rebuild.jsonl").read_text(encoding="utf-8").splitlines()[-1])


@pytest.mark.parametrize("case", [
    "clean", "noop", "single_append", "unsorted_append", "equal_timestamp",
    "interleaved", "rounding", "active_growth", "rotation", "restart",
    "ledger_checkpoint_stale", "checkpoint_mismatch", "inode_replacement",
    "partial_line", "duplicate", "nav_only", "repeated_noop", "repeated_recovery",
])
def test_frozen_corpus_full_incremental_equivalence(corpus: FrozenCorpus, case: str) -> None:
    entry = _fill("2026-01-01T00:00:00+00:00", "BUY", False, order_id="entry")
    exit_ = _fill("2026-01-01T01:00:00+00:00", "SELL", True, price=101, order_id="exit", entry_price=100)
    if case == "clean":
        initial = _initial(corpus, entry, exit_)
        _assert_equivalent(ledger_module.build_episode_ledger(), initial)
        return
    _initial(corpus, *(() if case == "unsorted_append" else (entry,)))
    if case in {"noop", "restart", "repeated_noop"}:
        result = _check(corpus, case)
        if case == "repeated_noop":
            again = _check(corpus, "repeated-noop")
            assert ledger_module._canonical_ledger_hash(result) == ledger_module._canonical_ledger_hash(again)
    elif case in {"single_append", "active_growth"}:
        _append(corpus.active, exit_)
        _check(corpus, case)
    elif case == "unsorted_append":
        _append(corpus.active, exit_, _fill("2026-01-01T00:30:00+00:00", "BUY", False, order_id="late-entry"))
        _check(corpus, case)
    elif case == "equal_timestamp":
        _append(corpus.active, _fill("2026-01-01T01:00:00+00:00", "BUY", False, order_id="same-entry"), exit_)
        _check(corpus, case)
    elif case == "interleaved":
        _append(corpus.active,
            _fill("2026-01-01T02:00:00+00:00", "SELL", True, price=101, order_id="btc-exit", entry_price=100),
            _fill("2026-01-01T01:00:00+00:00", "BUY", False, symbol="ETHUSDT", order_id="eth-entry"),
            _fill("2026-01-01T01:30:00+00:00", "SELL", True, symbol="ETHUSDT", price=102, order_id="eth-exit", entry_price=100),
            _fill("2026-01-01T00:30:00+00:00", "BUY", False, order_id="btc-second-entry"),
        )
        _check(corpus, case)
    elif case == "rounding":
        _append(corpus.active,
            _fill("2026-01-01T00:10:00+00:00", "BUY", False, qty=1, order_id="entry-two"),
            _fill("2026-01-01T01:00:00+00:00", "SELL", True, price=100.004, order_id="round-one", entry_price=100, fee=0),
            _fill("2026-01-01T01:01:00+00:00", "SELL", True, price=100.004, order_id="round-two", entry_price=100, fee=0),
        )
        result = _check(corpus, case)
        assert result.stats["metadata_pnl"]["gross_pnl"] == 0.01
    elif case == "rotation":
        corpus.active.rename(corpus.execution / "orders_executed.1.jsonl")
        _append(corpus.active, exit_)
        _check(corpus, case)
    elif case == "ledger_checkpoint_stale":
        stale = (corpus.state / "episode_ledger_checkpoint.json").read_text(encoding="utf-8")
        _append(corpus.active, exit_)
        ledger_module.rebuild_and_save()
        (corpus.state / "episode_ledger_checkpoint.json").write_text(stale, encoding="utf-8")
        _check(corpus, case)
    elif case == "checkpoint_mismatch":
        checkpoint = json.loads((corpus.state / "episode_ledger_checkpoint.json").read_text(encoding="utf-8"))
        checkpoint["ledger_hash"] = "mismatch"
        (corpus.state / "episode_ledger_checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
        _check(corpus, case)
    elif case == "inode_replacement":
        replacement = corpus.execution / "replacement.jsonl"
        _append(replacement, exit_)
        replacement.replace(corpus.active)
        _check(corpus, case)
        assert _last_telemetry(corpus)["mode"] == "full_fallback"
    elif case == "partial_line":
        with corpus.active.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(exit_))
        corpus.freeze("partial-pending", require_resume_state=True)
        pending = ledger_module.rebuild_and_save()
        assert len(pending.episodes) == 0
        with corpus.active.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        _check(corpus, case)
    elif case == "duplicate":
        _append(corpus.active, entry, exit_)
        _check(corpus, case)
    elif case == "nav_only":
        _append(corpus.active, _fill("2026-01-01T01:00:00+00:00", "SELL", True, price=99, order_id="loss", entry_price=100))
        before = _check(corpus, "nav-before")
        corpus.nav.write_text(json.dumps({"total_equity": 100.0}), encoding="utf-8")
        after = _check(corpus, case)
        assert before.stats["max_drawdown_pct"] != after.stats["max_drawdown_pct"]
        assert ledger_module._canonical_ledger_hash(before) != ledger_module._canonical_ledger_hash(after)
    elif case == "repeated_recovery":
        checkpoint = json.loads((corpus.state / "episode_ledger_checkpoint.json").read_text(encoding="utf-8"))
        checkpoint["ledger_hash"] = "mismatch"
        (corpus.state / "episode_ledger_checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
        first = _check(corpus, "recovery-one")
        second = _check(corpus, "recovery-two")
        assert ledger_module._canonical_ledger_hash(first) == ledger_module._canonical_ledger_hash(second)


def test_authority_payloads_and_nav_are_semantic_inputs(corpus: FrozenCorpus) -> None:
    entry = _fill("2026-01-01T00:00:00+00:00", "BUY", False, order_id="entry")
    exit_ = _fill("2026-01-01T01:00:00+00:00", "SELL", True, price=99, order_id="exit", entry_price=100)
    for action, ts, decision in (("ENTRY", entry["ts"], "decision-entry"), ("EXIT", exit_["ts"], "decision-exit")):
        _append(corpus.dle, {"event_type": "LINK", "payload": {"ts": ts, "request_id": f"request-{action}", "decision_id": decision, "permit_id": f"permit-{action}", "symbol": "BTCUSDT", "requested_action": action, "strategy": "fixture"}})
        _append(corpus.dle, {"event_type": "DECISION", "payload": {"decision_id": decision, "context_snapshot": {"regime": "trend"}}})
    initial = _initial(corpus, entry, exit_)
    _assert_equivalent(ledger_module.build_episode_ledger(), initial)
    assert initial.episodes_v2 and initial.episodes_v2[0].authority_entry.decision_id == "decision-entry"
    original_hash = ledger_module._canonical_ledger_hash(initial)
    corpus.nav.write_text(json.dumps({"total_equity": 5000.0}), encoding="utf-8")
    changed = _check(corpus, "nav-semantic-change")
    assert changed.stats["max_drawdown_pct"] > 0
    assert ledger_module._canonical_ledger_hash(changed) != original_hash


def test_frozen_manifest_rejects_missing_and_mutated_semantic_inputs(corpus: FrozenCorpus) -> None:
    with pytest.raises(AssertionError, match="missing required semantic input"):
        corpus.freeze("missing-active-log")
    _append(corpus.active, _fill("2026-01-01T00:00:00+00:00", "BUY", False))
    corpus.freeze("mutable-input")
    corpus.nav.write_text(json.dumps({"total_equity": 123.0}), encoding="utf-8")
    with pytest.raises(AssertionError, match="silently changed: nav_state"):
        corpus.assert_unchanged("mutable-input")
