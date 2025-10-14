import pandas as pd

from dashboard import nav_helpers


def test_treasury_table_formats_sorted():
    summary = {
        "details": {
            "treasury": {
                "treasury": {
                    "XAUT": {"qty": 0.59, "val_usdt": 2295.0},
                    "BTC": {"qty": 0.025, "val_usdt": 650.0},
                    "USDC": {"qty": 100.0, "val_usdt": 100.0},
                }
            }
        }
    }
    df = nav_helpers.treasury_table_from_summary(summary)
    assert list(df["Asset"]) == ["XAUT", "BTC", "USDC"]
    assert pd.isna(df["Units"]).sum() == 0
    assert df.loc[df["Asset"] == "BTC", "Units"].iloc[0] == 0.025
    html = nav_helpers.format_treasury_table(df).to_html()
    assert "0.025000" in html
    assert "2,295.00" in html


def test_signal_attempts_summary_parses_latest():
    lines = [
        "[screener] attempted=5 emitted=1",
        "misc line",
        "[screener] attempted=7 emitted=3",
    ]
    msg = nav_helpers.signal_attempts_summary(lines)
    assert msg == "Signals: 7 attempted, 3 emitted (43%)"


def test_signal_attempts_summary_missing():
    lines: list[str] = ["no metrics here"]
    assert nav_helpers.signal_attempts_summary(lines) == "Signals: N/A"
