# Binary Lab P2 Shadow Campaign Log

Start: 2026-02-19 (UTC)  
Mode: Path A (P2 shadow campaign, Binary Lab intentionally fail-closed)

## Daily Checks (10 minutes)

1. Process/status: `supervisorctl status hedge:hedge-executor`
2. Env invariants:
   - `PREDICTION_PHASE=P2_PRODUCTION`
   - `PREDICTION_DLE_ENABLED=1`
   - `BINARY_LAB_LIMITS_HASH` absent
3. Binary Lab state: `jq -r '.status + " " + (.last_checkpoint_utc_date // "null")' logs/state/binary_lab_state.json`
4. Atomic writes: `find logs/state -maxdepth 1 -name 'binary_lab_state.json.tmp' -print` (expect no output)
5. Containment signal: firewall denial activity (last 24h count)

## Done Criteria (3 days minimum)

- No restart loops
- Binary Lab status never leaves `DISABLED`
- No `.tmp` artifacts observed
- Deny logs observed for non-binary consumers
- No evidence of cross-sleeve influence
- Manifest audit sampled as `MANIFEST_OK`

## Entry Template

- Date (UTC):
- Supervisor status:
- Env invariants:
- Binary Lab status:
- Temp artifacts:
- Firewall deny count (last 24h):
- Notes:

## Entries

### 2026-02-19 (Day 0)

- Date (UTC): 2026-02-19T09:57:56Z
- Supervisor status: `hedge:hedge-executor RUNNING pid 3110049 uptime 0:10:57`
- Env invariants: `PREDICTION_PHASE=P2_PRODUCTION`, `PREDICTION_DLE_ENABLED=1`, `BINARY_LAB_LIMITS_HASH` absent
- Binary Lab status: `DISABLED null`
- Temp artifacts: none
- Firewall deny count (last 24h): `354` (`logs/prediction/firewall_denials.jsonl`)
- Notes: P2 boundary remains active with Binary Lab fail-closed as intended.
