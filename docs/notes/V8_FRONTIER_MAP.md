# v8 Frontier Map (High-Signal)

## 1) Core v8 Frontier

**Frontier definition:** v8 introduces an **activation/certification control-plane** that wraps the execution data-plane with time-bounded structural governance.

- **Data-plane (existing):** order routing, sizing, risk vetoing, position lifecycle (`execution/executor_live.py`).
- **Control-plane (v8):** `execution/activation_window.py` enforces 14-day certification, structural integrity checks, and post-window scale authorization.
- **Verification-plane (v8):** `scripts/activation_verify.py` evaluates 7 gates and records the authoritative GO/EXTEND/NO-GO verdict consumed by the scale gate.

---

## 2) Architectural Boundaries (What v8 draws hard lines around)

### A. Activation boundary (dual-key)

**Boundary:** runtime config alone cannot activate v8 behavior.

- Requires both `activation_window.enabled: true` **and** `ACTIVATION_WINDOW_ACK=1`.
- Fail-closed behavior for malformed/missing activation config (window remains inactive).

**Why it matters:** separates declarative config drift from intentional operational activation.

### B. Structural-integrity boundary

**Boundary:** v8 treats config/manifest immutability as first-class runtime invariants.

- Boot snapshots of manifest + runtime config hashes.
- Per-loop drift checks against snapshots.
- Drift is treated as a structural halt condition (KILL_SWITCH), not advisory telemetry.

**Why it matters:** moves integrity from documentation/process into executable policy.

### C. Certification boundary vs strategy boundary

**Boundary:** v8 certification is explicitly **not** expectancy tuning.

- Time-bounded window (14 calendar days) replaces episode-only calibration as primary certification semantics.
- Window completion triggers halt lock (completion state), distinct from failure semantics.

**Why it matters:** decouples machine trustworthiness from strategy alpha outcomes.

### D. Scale authorization boundary

**Boundary:** post-window production sizing is gated by persisted verification evidence.

- `get_scale_gate_cap()` applies cap until recorded 7/7 GO + manifest hash match.
- Gate persists even when activation window is disabled (if `require_go_for_scale` enabled).

**Why it matters:** introduces durable, auditable promotion criteria across lifecycle transitions.

---

## 3) Architectural Seams (Where v8 stitches into existing system)

### Seam 1: Executor loop hook

- `_sync_dry_run()` calls `check_activation_window()` every loop.
- Halt state is projected into existing kill-switch path via environment mutation (`KILL_SWITCH=1`).

**Interpretation:** low-intrusion seam; v8 composes with existing stop mechanisms rather than replacing executor core.

### Seam 2: Sizing arbitration seam

- Executor applies sizing caps from: calibration window, activation window, and production scale gate.
- Effective cap is the **tightest** active constraint.

**Interpretation:** v8 overlays constraints additively; no single-source sizing authority.

### Seam 3: Boot-time observability seam

- Startup calls `log_activation_boot_status()` to snapshot hashes and emit initial lifecycle state.

**Interpretation:** v8 makes boot an auditable state transition rather than a silent init phase.

### Seam 4: DLE governance seam

- Activation lifecycle emits `STRUCTURAL_GUARD` events (`STARTED`, `HALTED`, `COMPLETED`, `VERIFIED`) into DLE shadow stream.

**Interpretation:** v8 extends DLE from trade-decision governance into system-governance telemetry.

### Seam 5: Ops channel seam

- `logs/state/activation_window_state.json` consumed by ops scripts (`aw_status`, Telegram heartbeat block).
- `activation_verification_verdict.json` becomes downstream source of truth for certification outcome.

**Interpretation:** state surfaces are now contract artifacts, not implementation detail.

---

## 4) Emerging Interfaces (Contracts becoming stable)

1. **Activation state contract** (`activation_window_state.json`)
   - High-frequency operational contract: integrity, drawdown, veto/mismatch counts, lifecycle fields.
2. **Verification verdict contract** (`activation_verification_verdict.json`)
   - Low-frequency governance contract: gate-by-gate evidence and promotion decision.
3. **Structural-guard event contract** (DLE log payload)
   - Cross-subsystem lifecycle signaling for constitutional/audit workflows.
4. **Config handshake contract** (`runtime.yaml` + env ACK)
   - Explicit operator intent interface for enabling certification logic.

---

## 5) Frontier Expansion / Contraction / New Operational Surface

### Expands

- **Scope expansion:** from trading-calibration concerns into full-stack integrity (manifest/config drift, Binary Lab freeze, DLE mismatch telemetry).
- **Temporal expansion:** from episode windows to calendar-time certification.
- **Governance expansion:** from local module checks to persisted verification + promotion gate.
- **Observability expansion:** new state and verdict surfaces integrated into ops tooling and heartbeat channels.

### Contracts (tightens)

- Dual-key activation, immutable-while-running assumptions, and manifest-hash coupling to scale authorization.
- Explicit distinction between governance vetoes and plumbing vetoes in metrics.

### Exposes new operational surfaces

- **Runtime state surface:** `logs/state/activation_window_state.json` (per-loop).
- **Governance verdict surface:** `logs/state/activation_verification_verdict.json` (on verification).
- **DLE lifecycle stream:** structural guard events as auditable boundary transitions.
- **Operator CLI/reporting:** `scripts/activation_verify.py` and `scripts/aw_status.py` for control-plane decisions.

---

## 6) Net Frontier Read (One-screen)

v8’s frontier is a **certification envelope** around the executor: it introduces a dual-key activation boundary, elevates structural integrity to kill-level policy, binds promotion to persisted 7-gate evidence, and projects this governance state across DLE and ops surfaces. The system’s architectural center of gravity shifts from "can this strategy trade" to "is the whole machine safe, unchanged, and certifiably promotable." 
