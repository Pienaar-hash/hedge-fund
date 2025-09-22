#!/usr/bin/env python3
import json
import os

from execution.utils import load_json
from execution.nav import PortfolioSnapshot


def main() -> None:
    cfg = load_json("config/strategy_config.json") or {}
    snapshot = PortfolioSnapshot(cfg)
    nav_usd = float(snapshot.current_nav_usd())
    gross_usd = float(snapshot.current_gross_usd())

    sizing = (cfg.get("sizing") or {})
    cap_pct = float(sizing.get("max_gross_exposure_pct", 0.0) or 0.0)
    if os.environ.get("EVENT_GUARD", "0") == "1":
        cap_pct *= 0.8
    cap_usd = nav_usd * (cap_pct / 100.0) if nav_usd > 0 and cap_pct > 0 else 0.0
    free_usd = max(cap_usd - gross_usd, 0.0) if cap_usd > 0 else 0.0
    gross_pct = (gross_usd / nav_usd * 100.0) if nav_usd > 0 else 0.0

    payload = {
        "nav_usd": nav_usd,
        "gross_usd": gross_usd,
        "gross_pct_of_nav": gross_pct,
        "cap_pct_effective": cap_pct,
        "cap_usd": cap_usd,
        "free_to_deploy_usd": free_usd,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
