## `v7_nav_freshness_fix.prompt.md`

````markdown
# v7 NAV Freshness Fix Patch
# Mode: Codex IDE / CLI
# Branch: v7-risk-tuning

Goal:  
Make the dashboard’s NAV age + “stale vs live” status accurately reflect the same NAV freshness used by the executor/risk engine.

Right now:
- Executor/risk logs show `nav_age=0.0s`, `fresh=True`.
- Dashboard header occasionally shows `NAV Age ≈ 30s` and/or implies “stale”.
- `logs/state/nav_state.json` is written by `execution/sync_state.py` and is the canonical NAV surface for the dashboard.
- `dashboard/nav_helpers.py` and `dashboard/live_helpers.py` still implement legacy `nav_state_age_seconds()` semantics that don't fully align with how v7 nav_state is populated.

We want to:
- Ensure `nav_state.json` carries explicit, accurate freshness fields.
- Make the dashboard read and display those fields directly.
- Keep behaviour backward-compatible where reasonable.

---

## Files in scope

Only touch these unless absolutely necessary:

- `execution/sync_state.py`
- `dashboard/nav_helpers.py`
- `dashboard/live_helpers.py`
- (Optional) update docs if you see them in the repo, e.g. `docs/v7_Telemetry.md`, `docs/v7_Dashboard_Spec.md`.

Do not change risk logic in `execution/risk_limits.py` or `execution/nav.py` in this patch.

---

## 1) Harden nav_state payload in sync_state

In `execution/sync_state.py`, locate the function where we:

- compute `nav_payload` (from nav_log + tail metrics + Firestore)
- write it to `logs/state/nav_state.json` via `_write_json(...)`.

Currently we log series, peak, etc and then:

```python
_write_json(
    os.path.join(LOGS_DIR, "state", "nav_state.json"),
    nav_payload,
    label="nav_state",
)
````

### Add / enforce these fields on every nav_state write:

1. `nav_payload["updated_at"]`

   * Set to `time.time()` immediately before writing.
   * Always float seconds since epoch.

2. `nav_payload["age_s"]`

   * Age in seconds of the NAV used in this payload.
   * If we have a nav-tail metric like `tail_kpis.get("nav_age_s")` then use it.
   * Else, compute from:

     * `source_ts = nav_payload.get("updated_at")` or
     * last series point: `max(p["t"] for p in nav_payload.get("series", []) if "t" in p)`
   * Then: `age_s = max(0.0, time.time() - source_ts)`.
   * Ensure it’s always a float.

3. Make sure we don’t break any existing keys:

   * If `nav_payload` already has `updated_at` or `age_s`, we can overwrite with fresh values (that’s fine).
   * Keep all other fields intact.

**Reasoning:**
The dashboard should not try to reconstruct age from implicit series timestamps; we’ll hand it explicit `age_s` and `updated_at` as part of v7.

---

## 2) Fix nav_state_age_seconds on the dashboard

In `dashboard/nav_helpers.py`:

* We have:

```python
NAV_STATE_PATH = Path(os.getenv("NAV_STATE_PATH") or (STATE_DIR / "nav_state.json"))
...
def nav_state_age_seconds(nav_state: Dict[str, Any]) -> Optional[float]:
    ...
```

Update `nav_state_age_seconds` to follow this logic:

1. If `nav_state` is not a dict or is empty → return `None`.

2. First, if `"age_s"` is present and is numeric:

   * Return `float(nav_state["age_s"])`.

3. Else, if `"updated_at"` is present and numeric:

   * `age = max(0.0, time.time() - float(nav_state["updated_at"]))`
   * Return that.

4. Else, fall back to any time-like field we already used today:

   * Look for keys like `"updated_at"`, `"ts"`, etc if you see them in the current nav_state structure.
   * If none of that is available, try the series:

     * `series = nav_state.get("series") or []`
     * If non-empty, take the max `"t"` and compute age from it.
   * If still nothing, return `None`.

Make sure the function never raises; it should handle bad/missing data gracefully and either return a float or `None`.

This ensures that once `age_s` is present (from sync_state), the dashboard shows the exact age that the execution side computed.

---

## 3) Make the dashboard header prefer nav_state["age_s"]

In `dashboard/app.py`, we currently do something like:

```python
from dashboard.nav_helpers import load_nav_state, nav_state_age_seconds
...
nav_doc, nav_source = load_nav_state()
nav_age = nav_state_age_seconds(nav_doc)
```

and use `nav_age` in the header.

No big structural changes are needed here, but:

* Confirm we’re always using `nav_state_age_seconds(nav_doc)` for the NAV header.
* If there is any legacy fallback path that recomputes age independently (e.g. by file mtime or series tail), remove that; we want a single source of truth: `nav_state["age_s"]` (or the helper’s fallback logic).

If needed, add a tiny safety clamp where we render:

* For negative or `None` ages, display a placeholder like `"—"` rather than misrepresenting the age.

---

## 4) Optional: keep v7 docs consistent

If `docs/v7_Telemetry.md` and `docs/v7_Dashboard_Spec.md` exist:

* Add a short note under NAV state schema summarising the new fields:

  * `updated_at: float (epoch seconds, when nav_state was last written)`
  * `age_s: float (seconds, NAV age used by dashboard + risk; non-negative)`

Do not mention historical nav files (nav.json/nav_confirmed.json) beyond what they already say; just document the `nav_state` enrichment.

---

## 5) Validation

After patching:

1. Run basic syntax check:

```bash
python -m py_compile execution/sync_state.py dashboard/nav_helpers.py dashboard/app.py
```

2. Restart sync + dashboard (however you normally do it, e.g.):

```bash
sudo supervisorctl restart hedge:sync
sudo supervisorctl restart hedge:dashboard
```

3. Tail logs for NAV:

```bash
tail -f logs/cache/nav_confirmed.json
tail -f logs/state/nav_state.json
```

Verify that `nav_state.json` now contains `updated_at` and `age_s` and that `age_s` is small and changing.

4. Check executor log remains consistent:

* `nav_age=... fresh=True` lines still look sane.

5. Open the dashboard:

* The NAV header should show an age that matches `age_s` (roughly 0–a few seconds, depending on timing).
* No “NAV is stale” unless `age_s` genuinely exceeds the configured freshness threshold.

---

## 6) Return summary

When you’re done, summarise:

* What was changed (files + functions).
* New fields added to `nav_state.json`.
* How NAV age is now computed and displayed.
* Any follow-up cleanups you recommend.

# ---------------------------------------------------------------------------
# END OF PATCH PROMPT