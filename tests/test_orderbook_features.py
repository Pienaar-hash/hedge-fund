from execution.orderbook_features import evaluate_entry_gate


def test_orderbook_feature_disabled_pass_through():
    veto, info = evaluate_entry_gate("BTCUSDT", "BUY", enabled=False)
    assert veto is False
    assert isinstance(info, dict)

