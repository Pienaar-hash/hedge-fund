# Code-change prompt: Telegram Alerts v7 — State-driven, Low-noise

Purpose

Stop the current noisy, useless Telegram messages and replace them with a single deterministic v7 state summary sent only on 4-hour candle close. Persist canonical state to `logs/state/telegram_state.json` so alerts are state-change driven and auditable.

Target files to modify

- `execution/telegram_utils.py` — add state persistence helpers, strict filter for 4h-only mode, clear logging markers, and atomic writes.
- `execution/telegram_alerts_v7.py` — ensure `run_alerts()` only triggers the 4h-close routine, build the canonical JSON payload and persist it after successful send.
- Optional (non-invasive): leave direct callers (e.g., `execution/executor_live.py`, `execution/telegram_report.py`) unchanged — rely on `telegram_utils` filtering for immediate mitigation.

Operator flags (env)

- `EXEC_TELEGRAM_4H_ONLY=1` — strict mode; only 4h JSON payloads allowed.
- `EXEC_TELEGRAM_MAX_PER_MIN` — integer rate cap. For immediate mitigation the operator can set `0` to block sends.
- `TELEGRAM_ENABLED=0` — top-level disable.

Requirements (must)

- Only send one JSON object per 4-hour close. The message must contain exactly these keys (types shown):

  - `atr_regime`: string
  - `drawdown_state`: string or null
  - `router_quality`: string or null
  - `aum_total`: number (float)
  - `last_4h_close_ts`: integer (epoch seconds)

- Persist canonical state to `logs/state/telegram_state.json` atomically (tmp file + `os.replace`). The file structure should match the JSON above and include a `last_sent` map for internal bookkeeping.
- Suppress all other Telegram messages unless explicitly enabled by operator flags.
- When suppressed, log a short searchable line, e.g. "⏳ Telegram suppressed (4h-only mode)" or "❌ Telegram suppressed (rate limit reached)".
- Do not alter strategy/risk logic: alerts are telemetry-only.

Implementation hints

- `execution/telegram_utils.py` additions:
  - `load_telegram_state(path='logs/state/telegram_state.json') -> dict`
  - `save_telegram_state(state, path=...)` (atomic)
  - Top-level `send_telegram(message: str, ...)` should check `EXEC_TELEGRAM_4H_ONLY` and allow only JSON messages having `last_4h_close_ts` (or `atr_regime`) keys. If `EXEC_TELEGRAM_MAX_PER_MIN` is `0`, send should be blocked by rate cap immediately.
  - Keep existing identical-message suppression logic as a secondary defense.
  - Distinguish allowed 4h sends in logs with "✅ Telegram 4h-state sent".

- `execution/telegram_alerts_v7.py` changes:
  - `run_alerts()` should only call `_maybe_send_4h_close_alert(...)`.
  - `_maybe_send_4h_close_alert()` should:
    - Load current `telegram_state`.
    - If `last_kline_close_ts > state['last_4h_close_ts']`, build payload using the canonical keys.
    - Call `send_telegram(json.dumps(payload))`.
    - On send success, update the `state` canonical keys and `last_4h_close_ts` and `save_telegram_state(state)`; also call `write_alert_jsonl(payload, path='logs/alerts/alerts_v7.jsonl')`.

Testing / Acceptance

- Unit tests (`tests/test_telegram_v7.py`):
  - Mock `requests.post` and set env `EXEC_TELEGRAM_4H_ONLY=1`.
  - Verify non-JSON or JSON without required keys is suppressed (no POST).
  - Verify proper 4h JSON payload is POSTed and file `logs/state/telegram_state.json` is updated.
  - Verify a duplicate attempt with same `last_4h_close_ts` is suppressed.

- Manual integration steps (for ops):

  1. Update Supervisor env or `.env` with `EXEC_TELEGRAM_4H_ONLY=1` and `EXEC_TELEGRAM_MAX_PER_MIN=0` during mitigation.
  2. `sudo supervisorctl restart hedge:hedge-executor`
  3. `sudo tail -n 200 /var/log/hedge-executor.out.log | grep -E "Telegram|suppressed"`
  4. After verifying quietness, set `EXEC_TELEGRAM_MAX_PER_MIN` to 1 and enable the new code changes via a PR.

Rollout and safety

- Stage to test environment first if available.
- Keep tokens in Supervisor environment (not committed to repo).
- Fallback operator actions:
  - `TELEGRAM_ENABLED=0` + restart to immediately cut all sends.
  - `EXEC_TELEGRAM_MAX_PER_MIN=0` for immediate block (applied as mitigation now).

Example JSON payload (single-line) — this is the only message that should be sent per 4h close:

{"atr_regime":"low","drawdown_state":"none","router_quality":"good","aum_total":11173.87,"last_4h_close_ts":1764177600}

Deliverables for the PR

- Code changes for `execution/telegram_utils.py` and `execution/telegram_alerts_v7.py` implementing the logic above.
- Unit tests `tests/test_telegram_v7.py` covering suppression and send behavior.
- A short README note in `docs/` describing the operator env flags and the mitigation commands.

If you want, I can implement the PR now (patch the two Python files and add tests) or just leave this prompt for developers to implement; tell me which you'd prefer.