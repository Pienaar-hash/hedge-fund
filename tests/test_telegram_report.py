from execution import telegram_report as tr


def test_telegram_report_dry_run():
    ok, payload = tr.run(env="prod", dry_run=True)
    assert ok is True
    assert "note" in payload and isinstance(payload["note"], str)
    assert "png" in payload and isinstance(payload["png"], str)

