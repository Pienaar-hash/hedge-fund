# Firestore Publishing

## Default
Live runs set `FIRESTORE_ENABLED=0` in `scripts/go_live_now.sh` to avoid noisy warnings when ADC is missing.

## Enable
1. Provision a GCP service account with Firestore access.  
2. Place credentials on the host and export:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   export FIRESTORE_ENABLED=1
   ```
3. Restart the executor.

If libraries or ADC are missing, the client raises a friendly `firestore disabled`/`libs unavailable` error without crashing the loop.
