from execution.rules_sl_tp import compute_sl_tp, should_exit


def test_compute_sl_tp_long_fixed_vs_atr():
    sl, tp = compute_sl_tp(
        100.0, "LONG", atr=0.002, atr_mult=2.0, fixed_sl_pct=0.6, fixed_tp_pct=1.0
    )
    # atr*mult = 0.4%, fixed SL=0.6% wins; fixed TP=1.0% wins
    assert abs(sl - 99.4) < 1e-6
    assert abs(tp - 101.0) < 1e-6


def test_compute_sl_tp_short_only_fixed():
    sl, tp = compute_sl_tp(
        200.0, "SHORT", atr=0.0, atr_mult=0.0, fixed_sl_pct=0.5, fixed_tp_pct=0.8
    )
    assert abs(sl - 201.0) < 1e-6
    assert abs(tp - 198.4) < 1e-6


def test_should_exit_tp_sl_and_trail():
    entry = 100.0
    sl, tp = compute_sl_tp(
        entry, "LONG", atr=0.0025, atr_mult=2.0, fixed_sl_pct=0.6, fixed_tp_pct=1.0
    )
    # price hits TP
    assert (
        should_exit([99.8, 100.4, 101.05], entry, "LONG", sl, tp, max_bars=60) is True
    )
    # price hits SL
    assert should_exit([100.0, 99.9, 99.4], entry, "LONG", sl, tp, max_bars=60) is True
    # trail (0.6%) from peak 101.0 => trail ~100.394; last 100.30 triggers trail
    assert (
        should_exit(
            [100.0, 101.0, 100.30],
            entry,
            "LONG",
            sl,
            tp,
            max_bars=60,
            trail={"width_pct": 0.006},
        )
        is True
    )


def test_time_stop():
    sl, tp = compute_sl_tp(
        100.0, "SHORT", atr=0.0, atr_mult=0.0, fixed_sl_pct=0.6, fixed_tp_pct=1.0
    )
    prices = [100.0] * 60
    assert should_exit(prices, 100.0, "SHORT", sl, tp, max_bars=60) is True
