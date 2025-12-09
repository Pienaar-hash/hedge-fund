# PATCHSET V7.6-P9 — Schema Reconciliation & Test Contract Finalization
# Context:
# - All v7.6 functional patches (P1–P8) are complete.
# - Documentation + activation runbooks + tag pack are complete.
# - One pre-existing failing test remains:
#       tests/integration/test_state_files_schema.py::test_state_files_have_minimal_schema
#   due to router_health.json schema mismatch.
# - No engine semantics or trading logic may change.
# - Goal: align the schema test expectations with the actual canonical state surfaces in v7.6.

## Objectives

1. **Reconcile the schema expectations in tests with the actual v7.6 surfaces**
   - Update tests/integration/test_state_files_schema.py so that:
       - It expects the **correct v7.6 router_health structure**:
           {
             "updated_ts": "...",
             "router_health": {
                 "global": {...},
                 "per_symbol": {...}
             }
           }
       - NOT the legacy “{router_health.json must have router_health key at root-level w/o updated_ts}”
   - This is the current canonical structure established in P1 & P3 via state_publish.  
     (router_health.json always has a top-level updated_ts and a router_health object.)

2. **Update test_manifest_state_contract.py if needed**
   - Ensure that manifest state entries for router_health match:
       - path: logs/state/router_health.json
       - owner: executor
       - update_frequency: per_loop
       - description includes microstructure fields
   - Ensure tests do not expect legacy keys removed in v7.6.

3. **Update schema fixtures to match v7.6 contract**
   - In tests/integration/test_state_files_schema.py:
       - Relax or update minimal schema for:
         - factor_diagnostics.json
         - symbol_scores_v6.json
       - Ensure only required fields declared in v7.6_State_Contract.md are enforced:
           * updated_ts required
           * canonical nested objects present
           * values are dict/list/float/str as applicable

4. **Add a small schema-normalization utility for tests (optional but recommended)**
   - Create tests/helpers/schema_utils.py:
       - get_minimal_schema() 
       - assert_required_keys_present()
       - Used for router_health, diagnostics, nav_state, risk_snapshot
   - These helpers must be **test-only** (never imported by runtime).

5. **ABSOLUTELY NO RUNTIME CHANGES**
   - No edits to state_publish.py or execution routines.
   - Only tests + optional test helpers updated.
   - All test contracts must reflect the existing stable runtime, not the reverse.

6. **Re-run tests to confirm:**
   - make test-fast
   - pytest tests/integration/test_state_files_schema.py
   - Full suite (optional): make test-all
   - CI workflow must pass.

---

## Files to Modify

- tests/integration/test_state_files_schema.py
- tests/integration/test_manifest_state_contract.py (if expectations updated)
- tests/helpers/schema_utils.py (new optional helper)
- No runtime files touched.

---

## Expected Test Requirements

The updated test for router_health must **assert all of**:

- file exists  
- top-level `updated_ts` exists and is a string  
- top-level `router_health` is an object  
- router_health.global exists  
- router_health.per_symbol exists (object, possibly empty)  
- router_health.global.quality_score is float (or numeric) if present  
- No assumptions about which per_symbol keys must exist.

---

## Expected Output

A patch that:

- Modifies only tests + test helpers.
- Reconciles minimal schema expectations to match v7.6 canonical outputs.
- Removes false assumptions from schema tests.
- Brings test suite to ✨100% green✨ for v7.6 tagging.
- Leaves runtime behaviour unchanged.

