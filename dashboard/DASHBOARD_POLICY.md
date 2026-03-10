# Dashboard Policy — GPT Hedge

> Canonical reference for dashboard architecture, layering, and styling.
> Violations of this policy produce architectural drift that is expensive to repair.

## Entry Point

One entry point: `dashboard/app.py`.

```
dashboard/
├── app.py              # Entry point (40–80 LOC max)
├── layout.py           # Section ordering, column splits, expanders
├── state_v7.py         # All state loaders (read-only from logs/state/)
├── components/         # Individual widgets (pure render functions)
├── static/
│   └── quant_theme.css # Single styling surface
└── DASHBOARD_POLICY.md
```

`app.py` does exactly three things:

1. `st.set_page_config(...)` — page title, layout, favicon
2. Load state via `_load_dashboard_state()`
3. Call render functions from `layout.py` and `components/`

No plotting, no widgets, no file reads, no inline CSS.

## Layering

| Layer | File(s) | Responsibility |
|-------|---------|----------------|
| **Entry** | `app.py` | Page config → state load → layout call |
| **Layout** | `layout.py` | Section ordering, column splits, expanders |
| **Components** | `components/*.py` | Individual widgets — pure render functions |
| **State** | `state_v7.py` | All state loading from `logs/state/*.json` |
| **Style** | `static/quant_theme.css` | All CSS — the only styling surface |

Each layer may only import from the layer below it.
Components never import from `app.py` or `layout.py`.

## Component Contract

Every widget follows this pattern:

```python
def render_widget(data: dict) -> None:
    """Pure render function. No file I/O, no global state."""
    st.metric("Label", data["value"])
```

Rules:

- **No file I/O** inside components — all data arrives as function arguments
- **No global state** access — no `st.session_state` writes, no singletons
- **Semantic HTML classes** only — components emit `<div class="status-ok">`, CSS does the rest
- **Fail-safe rendering** — wrap external calls in try/except, show "unavailable" on error

## State Loading

All dashboard data comes from `logs/state/*.json` via `state_v7.py`.

Rules:

- State files are **read-only** — dashboard never writes to `logs/state/`
- New state fields must be registered in `v7_manifest.json`
- New state files require schema tests in `tests/integration/test_state_*.py`
- All state loading happens **once** in `_load_dashboard_state()` — never inside render functions

## CSS Policy

One styling surface: `static/quant_theme.css`.

Rules:

- **No inline `st.markdown("<style>...")`** in any component or layout file
- CSS injection happens **once** in `app.py` via `_inject_css()`
- Color palette is defined in the CSS file (status colors, tier colors, severity)
- Components emit semantic classes: `status-ok`, `status-warning`, `status-critical`

## Layout Policy

Flat structure with expanders for detail sections:

```
Header + KPI Strip
Regime Visibility
NAV Composition
Equity Curve
Multi-Engine Soak
Positions
Episode Ledger
Strategy Performance
[Expander] System Internals
[Expander] Diagnostics & Raw State
```

Rules:

- **No tabs** — they break Streamlit refresh semantics and fragment the mental model
- Use `st.expander()` for detail/internals sections (collapsed by default)
- Primary surfaces are always visible; operational detail is collapsed

## Adding a New Panel

1. Create `dashboard/components/my_panel.py` with a pure `render_my_panel(data)` function
2. Add state loader to `state_v7.py` if new state file is needed
3. Register any new state files in `v7_manifest.json`
4. Import and call from `app.py` or `layout.py` in the appropriate section
5. Add schema test in `tests/integration/`
6. All styling via CSS classes in `quant_theme.css` — no inline styles

## What Not to Do

- Do not create additional entry points (`app_v2.py`, `main.py`, etc.)
- Do not add tabs — use expanders for collapsible sections
- Do not load files inside render functions
- Do not write inline CSS in components
- Do not import from `execution/` — dashboard is read-only from `logs/state/`
