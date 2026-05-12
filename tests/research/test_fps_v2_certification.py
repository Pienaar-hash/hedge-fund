from __future__ import annotations

import csv
import json
from pathlib import Path

from research import fps_v2_certification as cert


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_fake_run(
    base: Path,
    *,
    run_id: str,
    output_hash: str,
    sample_size: int,
    gross_per_trade: float,
    fee_per_trade: float,
    nav_series: list[float],
    veto_count: int = 0,
) -> dict[str, object]:
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trades = []
    for i in range(sample_size):
        gross = gross_per_trade
        fees = fee_per_trade
        net = gross - fees
        trades.append(
            {
                "symbol": "BTCUSDT",
                "entry_ts": 1710000000 + i * 60,
                "exit_ts": 1710000030 + i * 60,
                "entry_px": 100.0,
                "exit_px": 100.0,
                "qty": 1.0,
                "gross_pnl": gross,
                "fees": fees,
                "net_pnl": net,
                "exit_reason": "signal",
            }
        )

    equity = [{"ts": 1710000000 + i * 60, "nav": v, "cash_nav": v, "unrealized": 0.0} for i, v in enumerate(nav_series)]
    veto = [
        {
            "ts": 1710000000 + i,
            "symbol": "BTCUSDT",
            "reason": "min_notional",
            "min_notional_action": "ABSTAIN_MIN_NOTIONAL",
            "intended_notional": 10.0,
            "adjusted_notional": 25.0,
        }
        for i in range(veto_count)
    ]

    _write_csv(
        run_dir / "trades.csv",
        ["symbol", "entry_ts", "exit_ts", "entry_px", "exit_px", "qty", "gross_pnl", "fees", "net_pnl", "exit_reason"],
        trades,
    )
    _write_csv(run_dir / "equity_curve.csv", ["ts", "nav", "cash_nav", "unrealized"], equity)
    _write_csv(
        run_dir / "veto_trace.csv",
        ["ts", "symbol", "reason", "min_notional_action", "intended_notional", "adjusted_notional"],
        veto,
    )

    (run_dir / "permit_trace.csv").write_text("ts,symbol,signal,permit,reason\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "strategy": "TREND_PULLBACK_V2_REPLAY_CANDIDATE",
                "trade_count": sample_size,
                "gross_pnl": sample_size * gross_per_trade,
                "fees": sample_size * fee_per_trade,
                "net_pnl": sample_size * (gross_per_trade - fee_per_trade),
            }
        ),
        encoding="utf-8",
    )

    manifest = {"output_hash": output_hash}
    (run_dir / "replay_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "manifest": manifest,
        "summary": {},
    }


def test_certification_report_contains_required_fields_and_output_dir(tmp_path: Path, monkeypatch) -> None:
    fake_base = tmp_path / "fake_runs"

    def _stub_run_replay(**kwargs):
        rid = str(kwargs["run_id"])
        return _make_fake_run(
            fake_base,
            run_id=rid,
            output_hash="stablehash",
            sample_size=35,
            gross_per_trade=5.0,
            fee_per_trade=1.0,
            nav_series=[10000.0, 10100.0, 10200.0, 10150.0, 10300.0],
            veto_count=3,
        )

    monkeypatch.setattr(cert, "run_replay", _stub_run_replay)

    result = cert.certify_replay(
        data_dir="ignored",
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframe="15m",
        config_path="ignored.json",
        starting_nav=10000,
        fee_bps=5,
        slippage_bps=3,
        run_id="cert_a",
        output_base_dir=str(tmp_path / "certs"),
    )

    report = result["report"]
    assert Path(str(result["certification_dir"])) == tmp_path / "certs" / "cert_a"
    required = {
        "setup_class",
        "symbols",
        "sample_size",
        "gross_pnl",
        "fees",
        "net_pnl",
        "win_rate",
        "max_drawdown",
        "veto_count",
        "output_hash",
        "verdict",
    }
    assert required.issubset(set(report.keys()))
    assert report["setup_class"] == "TREND_PULLBACK_V2_REPLAY_CANDIDATE"


def test_certification_verdict_insufficient_sample(tmp_path: Path, monkeypatch) -> None:
    fake_base = tmp_path / "fake_runs"

    def _stub_run_replay(**kwargs):
        return _make_fake_run(
            fake_base,
            run_id=str(kwargs["run_id"]),
            output_hash="stablehash",
            sample_size=10,
            gross_per_trade=5.0,
            fee_per_trade=1.0,
            nav_series=[10000.0, 10100.0],
        )

    monkeypatch.setattr(cert, "run_replay", _stub_run_replay)

    result = cert.certify_replay(
        data_dir="ignored",
        symbols=["BTCUSDT"],
        timeframe="15m",
        config_path="ignored.json",
        starting_nav=10000,
        fee_bps=5,
        slippage_bps=3,
        run_id="cert_insufficient",
        output_base_dir=str(tmp_path / "certs"),
    )
    assert result["report"]["verdict"] == "INSUFFICIENT_SAMPLE"


def test_certification_verdict_fail_when_output_hash_unstable(tmp_path: Path, monkeypatch) -> None:
    fake_base = tmp_path / "fake_runs"

    def _stub_run_replay(**kwargs):
        rid = str(kwargs["run_id"])
        out_hash = "hash_a" if rid.endswith("_a") else "hash_b"
        return _make_fake_run(
            fake_base,
            run_id=rid,
            output_hash=out_hash,
            sample_size=40,
            gross_per_trade=8.0,
            fee_per_trade=1.0,
            nav_series=[10000.0, 10200.0, 10150.0, 10300.0],
        )

    monkeypatch.setattr(cert, "run_replay", _stub_run_replay)

    result = cert.certify_replay(
        data_dir="ignored",
        symbols=["BTCUSDT"],
        timeframe="15m",
        config_path="ignored.json",
        starting_nav=10000,
        fee_bps=5,
        slippage_bps=3,
        run_id="cert_unstable",
        output_base_dir=str(tmp_path / "certs"),
    )
    assert result["report"]["output_hash_stable"] is False
    assert result["report"]["verdict"] == "FAIL"


def test_certification_verdict_pass_when_all_rules_hold(tmp_path: Path, monkeypatch) -> None:
    fake_base = tmp_path / "fake_runs"

    def _stub_run_replay(**kwargs):
        rid = str(kwargs["run_id"])
        return _make_fake_run(
            fake_base,
            run_id=rid,
            output_hash="stable_ok",
            sample_size=36,
            gross_per_trade=10.0,
            fee_per_trade=1.0,
            nav_series=[10000.0, 10100.0, 10050.0, 10200.0, 10180.0, 10300.0],
        )

    monkeypatch.setattr(cert, "run_replay", _stub_run_replay)

    result = cert.certify_replay(
        data_dir="ignored",
        symbols=["BTCUSDT"],
        timeframe="15m",
        config_path="ignored.json",
        starting_nav=10000,
        fee_bps=5,
        slippage_bps=3,
        run_id="cert_pass",
        output_base_dir=str(tmp_path / "certs"),
    )
    assert result["report"]["verdict"] == "PASS"
