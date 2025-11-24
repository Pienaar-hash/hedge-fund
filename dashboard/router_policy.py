from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:0.25em 0.7em;'
        f'border-radius:0.6em;font-weight:700;color:#fff;background:{color};">{text}</span>'
    )


def _diff_style(value: Any, other: Any) -> str:
    return "background-color: #fef08a" if value != other else ""


def render_router_policy_panel(
    current_policy: Dict[str, Any] | None,
    suggestions: Dict[str, Any] | None,
    apply_enabled: bool = False,
) -> None:
    st.markdown("### Router Auto-Tune (v6)")

    apply_badge = _badge("Apply: OFF (configured)", "#db2828" if not apply_enabled else "#21ba45")
    st.markdown(apply_badge, unsafe_allow_html=True)

    current_symbols = current_policy.get("symbols") if isinstance(current_policy, dict) else []
    sugg_symbols = suggestions.get("symbols") if isinstance(suggestions, dict) else []
    if not isinstance(current_symbols, list):
        current_symbols = []
    if not isinstance(sugg_symbols, list):
        sugg_symbols = []

    current_map = {str(item.get("symbol")).upper(): item for item in current_symbols if isinstance(item, dict)}
    rows = []
    for entry in sugg_symbols:
        if not isinstance(entry, dict):
            continue
        sym = str(entry.get("symbol") or "").upper()
        if not sym:
            continue
        current = current_map.get(sym, {})
        proposed = entry.get("proposed_policy") or {}
        current_policy_block = current.get("policy") or {}
        rows.append(
            {
                "symbol": sym,
                "curr_maker_first": current_policy_block.get("maker_first"),
                "curr_offset_bps": current_policy_block.get("offset_bps"),
                "curr_quality": current_policy_block.get("quality"),
                "prop_maker_first": proposed.get("maker_first"),
                "prop_offset_bps": proposed.get("offset_bps"),
                "prop_quality": proposed.get("quality"),
            }
        )

    if not rows:
        st.info("No router suggestions available.")
        return

    df = pd.DataFrame(rows)
    styler = df.style
    styler = styler.apply(
        lambda s: [
            _diff_style(s["curr_maker_first"], s["prop_maker_first"]),
            _diff_style(s["curr_offset_bps"], s["prop_offset_bps"]),
            _diff_style(s["curr_quality"], s["prop_quality"]),
            "",
            "",
            "",
            "",
        ],
        axis=1,
    )
    st.dataframe(styler, use_container_width=True, height=360)


__all__ = ["render_router_policy_panel"]
