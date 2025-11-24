#!/usr/bin/env python3
import json
from execution.utils import load_json
from execution.ml.predict import score_all


def main() -> None:
    cfg = load_json("config/strategy_config.json")
    results = score_all(cfg)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
