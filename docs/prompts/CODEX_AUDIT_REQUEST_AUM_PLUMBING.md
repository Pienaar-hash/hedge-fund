# CODEX AUDIT REQUEST
# v7 Off-Exchange AUM Path Audit (config/offexchange_holdings.json → nav.json.aum.offexchange)

# OBJECTIVE:
#   Understand exactly why logs/state/nav.json.aum.offexchange == {} even though:
#     - v7 AUM wiring is in place
#     - aum.futures and aum.total are present
#
#   We want a precise map of:
#     1) How off-exchange holdings are loaded from config/offexchange_holdings.json.
#     2) How they are transformed into USD values.
#     3) How they are attached into nav_snapshot["aum"]["offexchange"].
#     4) Any conditions that might cause offexchange to be empty {}.
#     5) Any testnet/ENV flags that gate offexchange behavior.
#
#   NO CODE CHANGES. AUDIT ONLY.

# SCOPE:
#   Search the repo for:
#     - "offexchange_holdings"
#     - "off-exchange"
#     - "off_exchange"
#     - "config/offexchange_holdings.json"
#     - ".aum[\"offexchange\"]"
#     - "attach_offexchange"
#
#   Focus on:
#     - execution/nav.py
#     - execution/state_publish.py
#     - execution/executor_live.py
#     - any utils/ or config loaders that touch off-exchange/AUM
#
# QUESTIONS TO ANSWER:

# 1) CONFIG LOADING
#   - Where exactly is config/offexchange_holdings.json loaded?
#   - What is the function name and call stack?
#   - What is the expected schema? (e.g. symbol → {qty, avg_cost}).
#   - What happens if:
#       a) The file does not exist?
#       b) The file is empty?
#       c) The file is malformed (wrong keys or types)?
#   - Is there any ENV/flag (e.g. OFFEXCHANGE_ENABLED, ENV="prod") that controls whether
#     off-exchange holdings are loaded?

# 2) VALUE COMPUTATION
#   - Where are off-exchange quantities converted to USD value?
#   - Which price source(s) are used? (e.g. get_mark_price_for_symbol, coingecko, etc.)
#   - Are there any try/except blocks that silently drop entries (e.g. on symbol mismatch)?
#   - Are any symbols restricted by exchange (e.g. only Binance spot/testnet)?

# 3) AUM ATTACHMENT
#   - In execution/nav.py, identify the function(s) that attach off-exchange holdings to
#     nav_snapshot["aum"]["offexchange"].
#   - Confirm whether an empty dict {} is the default value, or whether it is the result
#     of filtered-out holdings.
#   - Document the logic path from config holdings → computed offexchange dict → final
#     nav_snapshot["aum"]["offexchange"].

# 4) CALL SITES / RUNTIME FLOW
#   - List each call site where the off-exchange loader / AUM attachment is invoked.
#   - Identify whether those calls are active in the main executor path:
#       execution/executor_live.py → nav helpers → state_publish.write_nav_state
#   - Check for any alternate nav builders that skip offexchange (e.g. "minimal" or
#     "report-only" snapshots) and see whether those are used in production.

# 5) TESTNET VS PROD BEHAVIOR
#   - Is offexchange logic conditional on ENV ("prod" vs testnet) or any other flag?
#   - Could running on BINANCE_TESTNET be causing the code to skip or zero out the
#     offexchange block?

# 6) RISK / STRATEGY SAFETY
#   - Confirm again that no risk/sizer/router/pipeline code reads nav.aum or
#     nav.aum.offexchange (telemetry-only).
#   - If any such references exist, list them and mark them clearly as "DANGEROUS"
#     in the audit report.

# 7) SUMMARY & ROOT CAUSE HYPOTHESES
#   - Provide a short summary:
#       * Why offexchange == {} under current runtime?
#       * What would need to be true for it to be populated?
#       * Any config/schema issues that would cause silent dropping of holdings?
#   - NO CODE CHANGES. Just a clear explanation and references to the functions
#     and line numbers involved.

# FORMAT:
#   Please structure the report with the following headings:
#
#     1. Off-Exchange Config Loader
#     2. Off-Exchange Value Computation
#     3. AUM Attachment Path (nav_snapshot["aum"]["offexchange"])
#     4. Executor → Nav → State Publish Flow
#     5. ENV/Testnet Conditions
#     6. Risk/Strategy Safety Check (.aum usage)
#     7. Most Likely Reason offexchange == {}
#     8. Suggested Fix Options (NO code, just design)
#
# END AUDIT REQUEST.
