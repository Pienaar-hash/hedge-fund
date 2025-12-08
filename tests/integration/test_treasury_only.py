from execution import nav as navmod


def test_treasury_only_returns_zero():
    val, detail = navmod.compute_treasury_only()
    assert val == 0.0
    assert detail == {"treasury": {}}
