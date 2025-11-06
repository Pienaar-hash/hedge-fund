import importlib
import time


def test_cached_zar_rate_reports_age_and_source():
    utils = importlib.import_module("execution.utils")

    # Seed cache with a stale rate (7h old) to ensure metadata tracks freshness.
    utils._USD_ZAR_CACHE = {"rate": 18.25, "source": "cache"}  # type: ignore[attr-defined]
    utils._USD_ZAR_TS = time.time() - (7 * 3600)  # type: ignore[attr-defined]

    rate, meta = utils.get_usd_to_zar(with_meta=True)  # type: ignore[attr-defined]

    assert rate == 18.25
    assert isinstance(meta, dict)
    assert meta.get("source") == "cache"
    assert meta.get("age") and meta["age"] > 6 * 3600
