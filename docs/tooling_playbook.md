# Tooling Playbook — GPT, Codex, Copilot, Reviews

Goal: Use each tool for what it’s best at. No redundancy, no chaos.

---

## 1. GPT (ChatGPT, Quant Infra Lead)

Use GPT when you need:

- Sprint planning (like this v7 sprint plan).
- High-level architecture design.
- Writing or revising docs (contracts, specs, runbooks).
- Designing prompts for Codex / Copilot.
- Thinking through risk trade-offs, workflows, pipelines.

You **do not** use GPT to modify repo files directly — that’s Codex’s job.

---

## 2. Codex CLI / IDE (repo-attached agent)

Use Codex when you need:

- Actual **code changes** in the repo:
  - new modules, refactors, bug fixes.
- Schema / contract implementation:
  - state files, telemetry, risk snapshots.
- Multi-file refactors.
- Tests or smoke scripts around the runtime.

Workflow:

1. Open Codex IDE on the repo.
2. Paste a focused prompt (e.g. `v7_risk_tuning_patch.prompt.md`).
3. Let Codex:
   - read relevant files,
   - propose diffs,
   - apply patches.
4. Run tests / import checks.
5. Commit with normal Git flows.

---

## 3. GitHub Copilot & Copilot Chat

Use Copilot for:

- Inline suggestions while typing code.
- Small refactors, e.g.:
  - extracting helper functions,
  - simplifying loops,
  - adding type hints.
- Pull request summarisation and micro-reviews:
  - “What changed and what should I double-check?”

Copilot is best for **short-horizon changes** tied to what’s open in your editor.

---

## 4. Code Reviews (Human + AI)

- **Small PRs:** use Copilot / GPT to review and then do a quick human pass.
- **Large PRs / Risk-critical:** rely on:
  - human review first,
  - GPT to help summarise and verify contracts and telemetry invariants.

Suggested pattern:
- Ask GPT: “Given this PR description and diff, what invariants should I validate manually in a live system?”

---

## 5. Putting It All Together (Example)

**You want v7 risk tuning:**

1. In GPT:
   - Ask for sprint plan and patch prompt (done).
2. In Codex:
   - Run `v7_agent_bootstrap.prompt.md` to generate v7 docs.
   - Then run `v7_risk_tuning_patch.prompt.md` to implement tuning.
3. In Copilot:
   - Improve small bits of code surfaced in the patch (e.g. logging structure or docstrings).
   - Help write good commit messages and PR descriptions.
4. In Git:
   - Commit and open a PR.
5. In GPT again:
   - Paste the PR diff/summary for an additional audit if needed.

That’s the coordination loop.

---
