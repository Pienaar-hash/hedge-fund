import json
from typing import Any, Dict


def test_unlisted_symbol_not_listed(monkeypatch, tmp_path):
    from execution import universe_resolver as ur

    # Monkeypatch exchange listing check to raise for NONFUT
    def fake_get_symbol_filters(symbol: str) -> Dict[str, Any]:
        if symbol.upper() == "NONFUTUSDT":
            raise RuntimeError("not listed")
        return {"LOT_SIZE": {"stepSize": "0.001", "minQty": "0.001"}}

    monkeypatch.setattr(ur, "get_symbol_filters", fake_get_symbol_filters, raising=True)

    assert ur.is_listed_on_futures("BTCUSDT") is True
    assert ur.is_listed_on_futures("NONFUTUSDT") is False


def test_discovery_merge_gate(monkeypatch, tmp_path):
    from execution import universe_resolver as ur

    # Point to temp settings + discovery
    settings = tmp_path / "settings.json"
    discovery = tmp_path / "discovery.yml"
    monkeypatch.setenv("SETTINGS_PATH", str(settings))
    monkeypatch.setenv("DISCOVERY_PATH", str(discovery))

    settings.write_text(json.dumps({"automerge_discovery": True}))
    discovery.write_text(
        "- symbol: FOOUSDT\n  rationale: stub\n  liquidity_ok: true\n  trend_ok: true\n"
    )

    # Listed gate stub
    def fake_get_symbol_filters(symbol: str) -> Dict[str, Any]:
        return {"LOT_SIZE": {"stepSize": "0.001", "minQty": "0.001"}}

    monkeypatch.setattr(ur, "get_symbol_filters", fake_get_symbol_filters, raising=True)

    allowed, _ = ur.resolve_allowed_symbols()
    assert "FOOUSDT" in allowed
