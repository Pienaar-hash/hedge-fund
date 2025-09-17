from execution.orderbook_features import evaluate_entry_gate


def test_orderbook_adverse_veto(monkeypatch):
    # Force imbalance adverse to BUY
    import execution.orderbook_features as ob

    monkeypatch.setattr(ob, "topn_imbalance", lambda _s, limit=10: -0.5, raising=True)
    veto, info = evaluate_entry_gate("BTCUSDT", "BUY", enabled=True)
    assert veto is True
    assert isinstance(info, dict)


def test_orderbook_feature_disabled_pass_through():
    veto, info = evaluate_entry_gate("BTCUSDT", "BUY", enabled=False)
    assert veto is False
    assert isinstance(info, dict)
