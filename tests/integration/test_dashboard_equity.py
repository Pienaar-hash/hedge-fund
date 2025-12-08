import importlib


def test_cached_zar_rate_reports_age_and_source():
    utils = importlib.import_module("execution.utils")

    original_get = utils.get_usd_to_zar

    def _mock_get_usd_to_zar(force: bool = False, with_meta: bool = False):  # type: ignore[override]
        meta = {"source": "cache", "age": 7 * 3600.0}
        return (18.25, meta) if with_meta else 18.25

    utils.get_usd_to_zar = _mock_get_usd_to_zar  # type: ignore[assignment]
    try:
        rate, meta = utils.get_usd_to_zar(with_meta=True)  # type: ignore[attr-defined]
        assert rate == 18.25
        assert isinstance(meta, dict)
        assert meta.get("source") == "cache"
        assert meta.get("age") == 7 * 3600.0
    finally:
        utils.get_usd_to_zar = original_get  # type: ignore[assignment]
