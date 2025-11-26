# Using GitHub Copilot in Git Workflow (GPT-Hedge v7)

This describes how we integrate GitHub Copilot / Copilot Chat with our normal Git flows.

---

## 1. Branching

- Use clear, small branches:
  - `v7-ws1-risk-diagnostics`
  - `v7-ws2-telemetry-kpis`
  - `v7-ws3-aum-donut`
- Copilot can help:
  - suggest branch names in your IDE
  - generate short branch descriptions you can paste into PRs.

---

## 2. Commit Messages

Use Copilot Chat with prompts like:

- “Generate a concise commit message summarising the staged changes, max 72 characters in the title.”
- “Group these changes into 2–3 logical commits and suggest messages.”

Pattern:
- Prefix with area:
  - `risk: add thresholds/observations to veto logs`
  - `telemetry: add kpis_v7 state file`
  - `dash: wire AUM donut to new state`

---

## 3. Pull Requests

In PR descriptions, use Copilot to:

- Summarise the diff:
  - “Explain this PR in 5 bullet points, focusing on behaviour and risk.”
- Generate checklist:
  - “List risk items to double-check before merging this PR.”

In GitHub UI, Copilot can also **review PRs**:
- Ask: “Highlight potential issues with this PR in terms of performance, safety, and risk controls.”

---

## 4. Code Suggestions

In VS Code / IDE:
- Use Copilot for **local refactors** (e.g. extracting helpers, aligning logging patterns).
- Prefer Copilot for micro-level suggestions (loops, dict building, type hints).
- Use GPT/Codex for macro-level refactors and architectural changes.

---

## 5. Guardrails

- Never accept large Copilot suggestions blindly; always inspect:
  - logging behaviour
  - error handling
  - interactions with risk / order routing.
- Prefer manual control over anything touching:
  - `executor_live`
  - `risk_limits`
  - order placement paths.

Copilot is a power tool for small edits, not a replacement for careful review in critical paths.
