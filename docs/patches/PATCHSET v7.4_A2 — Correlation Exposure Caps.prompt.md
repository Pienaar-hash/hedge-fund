# PATCHSET v7.4_A2 — Correlation Exposure Caps

## Objective

Introduce **correlation-aware position limits** so the engine cannot over-stack risk across highly correlated symbols.

Specifically:

- Define **correlation groups** (e.g., BTC/ETH/SOL as “L1_bluechips”).
- Compute **group-level NAV exposure** (before and after a proposed order).
- Veto new orders when a group’s NAV % would exceed its configured cap.
- Publish group exposure stats for observability and dashboard rendering.
- Add a clear veto reason: `"correlation_cap"`.

This must **not** break existing v7 risk, state, telemetry, or executor contracts, and must coexist with:

- Per-symbol caps  
- Portfolio DD circuit (A1)  
- Other risk vetoes  

---

## Files to Touch

- `config/correlation_groups.json` (new)
- `execution/risk_loader.py` (load/validate correlation groups)
- `execution/correlation_groups.py` (new) **or** `execution/universe_resolver.py` (extend)
- `execution/risk_limits.py`
- `execution/state_publish.py`
- `dashboard/risk_panel.py` (or equivalent risk panel)
- `tests/test_correlation_groups_resolver.py` (new)
- `tests/test_risk_limits_correlation_cap.py` (new)
- `tests/test_state_publish_correlation_groups.py` (new or extend existing)

Do **not** change executor entrypoints or screener logic.

---

## 1. Config: Correlation Groups

**File:** `config/correlation_groups.json` (new)

### Format

Create a new JSON config file:

```json
{
  "groups": {
    "L1_bluechips": {
      "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
      "max_group_nav_pct": 0.35
    },
    "L2_layer1s": {
      "symbols": ["AVAXUSDT", "NEARUSDT", "LINKUSDT"],
      "max_group_nav_pct": 0.25
    }
  }
}
````

Rules:

* `groups` is a mapping `group_name -> { symbols, max_group_nav_pct }`.
* `symbols` must be upper-case, matching the symbols used in positions, screener, and router.
* `max_group_nav_pct` is a **fraction** (0.35 = 35% of NAV) just like other risk percentages.
* Empty file (no groups) should effectively **disable** correlation caps.

---

## 2. Risk Loader: Load and Normalize Correlation Groups

**File:** `execution/risk_loader.py`

### Change

1. Add a function to load `config/correlation_groups.json`:

   * Read JSON from disk.
   * Validate structure (basic checks only: `groups` is dict, each entry has `symbols` list and `max_group_nav_pct`).
   * Normalize `max_group_nav_pct` via existing percentage helper, e.g. `normalize_percentage`.

2. Introduce a small config dataclass:

```python
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class CorrelationGroupConfig:
    max_group_nav_pct: float
    symbols: List[str]

@dataclass
class CorrelationGroupsConfig:
    groups: Dict[str, CorrelationGroupConfig]
```

3. Expose a loader function:

```python
def load_correlation_groups_config(config_path: Path = DEFAULT_CORR_GROUPS_PATH) -> CorrelationGroupsConfig:
    ...
```

4. Make `risk_loader` export this config for `risk_limits` and `correlation_groups` to consume.

5. If the file is missing, malformed, or empty, return `CorrelationGroupsConfig(groups={})`.

---

## 3. Resolver: Map Symbols → Correlation Groups and Aggregate NAV

You can either:

* Create a new module `execution/correlation_groups.py`, **preferred**, or
* Extend `execution/universe_resolver.py` if that’s where similar logic lives.

Below assumes a new `execution/correlation_groups.py`:

**File:** `execution/correlation_groups.py` (new)

### Responsibilities

1. **Index building**

   * From `CorrelationGroupsConfig`, construct:

     ```python
     symbol_to_groups: Dict[str, Set[str]]
     ```

     so we can quickly answer: “which groups does this symbol belong to?”

   * Allow for overlapping groups (a symbol can be in multiple groups).

2. **Group NAV exposure calculation**

   Provide a function:

```python
from typing import Dict, Iterable
from .correlation_groups_config import CorrelationGroupsConfig  # Or from risk_loader
from .types import Position  # Use whatever position type you already have

def compute_group_exposure_nav_pct(
    positions: Iterable[Position],
    nav_total_usd: float,
    corr_cfg: CorrelationGroupsConfig,
) -> Dict[str, float]:
    """
    Returns a mapping: group_name -> gross_nav_pct (0..1) based on current positions.

    Gross exposure is the sum of absolute notional of all positions whose symbol is in that group,
    divided by nav_total_usd.

    If nav_total_usd <= 0, return all zeros.
    """
```

* Use **gross** exposure: sum of `abs(notional)` for each group; do not net longs and shorts.
* If `nav_total_usd <= 0`, safely return `0.0` for all.

3. **Hypothetical group exposure with a proposed order**

Add a helper:

```python
def compute_hypothetical_group_exposure_nav_pct(
    positions: Iterable[Position],
    nav_total_usd: float,
    corr_cfg: CorrelationGroupsConfig,
    order_symbol: str,
    order_notional_usd: float,
) -> Dict[str, float]:
    """
    Returns group_name -> gross_nav_pct, assuming we add the proposed order's notional
    (abs(order_notional_usd)) to all groups containing order_symbol.
    """
```

* Implementation:

  * Start from existing group exposures (from `compute_group_exposure_nav_pct`).
  * Add `abs(order_notional_usd)` to all relevant groups (where `order_symbol` belongs).
  * Divide by `nav_total_usd` to yield final percentages.

4. **Edge cases**

* If `order_symbol` not in any group → return current exposures (no added risk).
* If `nav_total_usd <= 0` → treat as no exposure (and correlation veto should be effectively inactive).

---

## 4. Risk Limits: Integrate Correlation Caps

**File:** `execution/risk_limits.py`

### Change

1. Ensure `risk_limits` has access to:

* `CorrelationGroupsConfig` (from `risk_loader`).
* Current `positions` and `nav_total_usd`.
* The proposed order’s symbol and USD notional.

2. Inside `check_order(...)`, **after**:

* Portfolio DD circuit check (A1), and
* Before final per-symbol caps are applied (or just after, but still before returning success),

add a correlation exposure check:

```python
if correlation_groups_config.groups and nav_total_usd > 0:
    # 1) compute current group exposures
    current_exposure = compute_group_exposure_nav_pct(...)

    # 2) compute hypothetical exposures including this order
    hypothetical_exposure = compute_hypothetical_group_exposure_nav_pct(
        positions=positions,
        nav_total_usd=nav_total_usd,
        corr_cfg=correlation_groups_config,
        order_symbol=symbol,
        order_notional_usd=order_notional_usd,
    )

    # 3) check against caps
    for group_name, group_cfg in correlation_groups_config.groups.items():
        before = current_exposure.get(group_name, 0.0)
        after = hypothetical_exposure.get(group_name, 0.0)
        cap = group_cfg.max_group_nav_pct
        if after > cap:
            # emit veto and return
```

3. Emit a veto with:

* `veto_reason = "correlation_cap"`
* Observed:

```python
observed = {
    "group_name": group_name,
    "group_nav_pct_before": before,
    "group_nav_pct_after": after,
}
limits = {
    "max_group_nav_pct": cap
}
```

4. Ordering of veto precedence:

* If portfolio DD circuit triggers, it should still be **first**.
* If circuit does not trigger, but this correlation check does, `correlation_cap` should appear as the **primary** reason.

5. `risk_vetoes.jsonl` must now contain structured `correlation_cap` entries.

6. If:

* No correlation groups configured, or
* `nav_total_usd <= 0`,

then **skip** correlation checks to avoid false blocking.

---

## 5. State Publisher: Expose Group Exposures

**File:** `execution/state_publish.py`

### Change

When building the risk snapshot (`write_risk_snapshot_state`), extend the state with correlation exposure info:

```python
"correlation_groups": {
    group_name: {
        "gross_nav_pct": current_exposure[group_name],
        "max_group_nav_pct": group_cfg.max_group_nav_pct
    }
    for group_name, group_cfg in correlation_groups_config.groups.items()
}
```

Notes:

* This uses **current exposure** (not hypothetical).
* If no groups configured → `correlation_groups` can be `{}` or omitted, but prefer empty object for schema stability.
* Do not remove or rename existing fields; this is purely additive.

---

## 6. Dashboard: Show Correlation Group Risk

**File:** `dashboard/risk_panel.py` (or equivalent)

### Change

1. In the risk panel, read:

* `risk["correlation_groups"]`

Expected shape:

```json
"correlation_groups": {
  "L1_bluechips": {
    "gross_nav_pct": 0.31,
    "max_group_nav_pct": 0.35
  }
}
```

2. Render a small table, e.g.:

| Group        | Exposure | Cap | Status |
| ------------ | -------- | --- | ------ |
| L1_bluechips | 31%      | 35% | OK     |
| L2_layer1s   | 24%      | 25% | NEAR   |

Status suggestions (optional):

* **OK** when `gross_nav_pct < 0.8 * cap`
* **NEAR** when `0.8 * cap <= gross_nav_pct < cap`
* **BREACH** when `gross_nav_pct >= cap` (should only occur if config changed after positions existed)

3. Keep layout simple and non-intrusive; integrate in the same section that shows per-symbol exposure and portfolio DD.

---

## 7. Tests

### 7.1 Resolver Tests

**File:** `tests/test_correlation_groups_resolver.py` (new)

Test cases:

1. **Symbol in one group**

   * Config: `L1_bluechips` with `["BTCUSDT", "ETHUSDT"]`.
   * Positions in BTC & ETH with NAV = 10000.
   * Validate `compute_group_exposure_nav_pct` correctly sums absolute notionals.

2. **Symbol in multiple groups**

   * BTC appears in `L1_bluechips` and `MacroCrypto`.
   * Ensure BTC notional contributes to **both** groups.

3. **No groups configured**

   * Empty config → exposures map should be empty.

4. **NAV <= 0**

   * nav_total_usd = 0 → exposures all zero.

5. **Hypothetical exposure**

   * With a proposed BTC order:

     * Validate `group_nav_pct_after = (current_abs_notional + abs(order_notional)) / nav_total`.

### 7.2 Risk Limits Correlation Tests

**File:** `tests/test_risk_limits_correlation_cap.py` (new)

Scenarios:

1. **No correlation groups**

   * Config: `CorrelationGroupsConfig(groups={})`.
   * Ensure correlation check is skipped; no `correlation_cap` veto.

2. **Within cap**

   * Group `L1_bluechips`: cap 0.35.
   * Positions lead to current exposure 0.20.
   * Proposed order increases group exposure to 0.30.
   * Ensure order is **not** vetoed.

3. **Exceeds cap**

   * Same group cap 0.35.
   * Current exposure 0.32.
   * Proposed order pushes `after` to 0.37.
   * Ensure:

     * `check_order` vetoes.
     * `veto_reason == "correlation_cap"`.
     * Observed payload contains `group_name`, `group_nav_pct_before`, `group_nav_pct_after`, `max_group_nav_pct`.

4. **NAV <= 0**

   * nav_total_usd = 0.
   * Ensure correlation caps do **not** block orders due purely to bad NAV denominator (this is handled by NAV freshness logic elsewhere).

5. **Interaction with other vetoes**

   * When both per-symbol cap and correlation cap could trigger:

     * Confirm ordering: if correlation is checked after symbol cap, then ensure tests are written accordingly; or explicitly define correlation to be checked first.
   * The important contract: behaviour is deterministic and tests assert the chosen order.

### 7.3 State Publish Tests

**File:** `tests/test_state_publish_correlation_groups.py` (new or extend existing risk state tests)

* Ensure `risk["correlation_groups"]` key exists when groups are configured.
* Validate that each group entry contains:

```python
{
  "gross_nav_pct": float,
  "max_group_nav_pct": float
}
```

* Edge case: no groups configured → `correlation_groups` is `{}`.

All existing tests (including A1) must remain green.

---

## 8. Acceptance Criteria

The patch is complete when:

1. `config/correlation_groups.json` exists and can define named correlation groups with `max_group_nav_pct` caps.
2. `risk_loader` can load and normalize these caps into `CorrelationGroupsConfig`.
3. `correlation_groups` resolver correctly:

   * Maps symbols to groups.
   * Computes current and hypothetical NAV % exposure per group.
4. `risk_limits.check_order`:

   * Applies correlation caps when groups exist and `nav_total_usd > 0`.
   * Emits `correlation_cap` veto with correct payload when a group would exceed its cap.
   * Does not block when no groups are configured or NAV is invalid.
5. `logs/state/risk.json` includes a stable `correlation_groups` section with each group’s `gross_nav_pct` and `max_group_nav_pct`.
6. The dashboard risk panel renders correlation group exposure in a simple table.
7. All tests (existing + new) pass with the standard test command (respecting `PYTHONPATH=.` per copilot-instructions).

Do not change any other behaviour or contracts beyond what is described here.

```
