from __future__ import annotations

import sys

from execution import pipeline_v6_compare as _canonical

sys.modules[__name__] = _canonical
