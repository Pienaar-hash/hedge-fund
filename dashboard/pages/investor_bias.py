import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import time
import math
import pandas as pd

# Initialize Firebase if not already done
if not firebase_admin._apps:
    cred = credentials.Certificate("config/firebase_creds.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

st.set_page_config(page_title="Investor Bias", layout="centered")
st.title("📈 Investor Bias Monitor")

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "LTCUSDT"]
selected_asset = st.selectbox("Select Asset", ASSETS)

# --- Source Weights (case-insensitive keys) ---
SOURCE_WEIGHTS = {
    "strategy": 1.2,
    "llm": 0.8,
    "tradervote": 1.0,
    "votes": 1.0,
}

# Per-source weights
SOURCE_WEIGHTS = {
    "Strategy": 1.2,
    "LLM": 0.8,
    "Votes": 1.0
}

# --- Helpers ---
def parse_ts(ts_val):
    try:
        if hasattr(ts_val, "seconds"):
            sec = int(ts_val.seconds)
        else:
            sec = int(float(ts_val))
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(sec))
    except Exception:
        return "—"

def majority_vote(components, fallback_signal):
    if not components:
        return fallback_signal
    counts = {}
    for c in components:
        s = str(c.get("signal", "")).lower()
        if not s:
            continue
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return fallback_signal
    return max(counts, key=counts.get)

def compute_confidence(components, vote):
    """Agreement-weighted mean of component confidences with source weights and optional recency decay.
    - Each component contributes: (base_conf × source_weight × recency_decay) if it aligns with the majority vote.
    - Normalized by total source weights across ALL components (so disagreement reduces score).
    """
    if not components:
        return 0.0

    total_weight = 0.0
    aligned_sum = 0.0

    for c in components:
        # base confidence
        try:
            base_conf = float(c.get("confidence", 0))
        except Exception:
            base_conf = 0.0

        # source weight (case-insensitive)
        src_key = str(c.get("source", "")).strip().lower()
        w = float(SOURCE_WEIGHTS.get(src_key, 1.0))

        # optional recency decay: half-life ~ 3 days if timestamp provided
        ts = c.get("timestamp")
        decay = 1.0
        if ts is not None:
            try:
                secs = ts.seconds if hasattr(ts, "seconds") else float(ts)
                age_days = max(0.0, (time.time() - secs) / 86400.0)
                decay = 0.5 ** (age_days / 3.0)
            except Exception:
                pass

        # accumulate total weight regardless of alignment (penalize disagreement)
        total_weight += w

        if str(c.get("signal", "")).lower() == vote:
            aligned_sum += (base_conf * w * decay)

    score = (aligned_sum / total_weight) if total_weight > 0 else 0.0
    return round(score, 3)

with st.spinner(f"Fetching latest signal for {selected_asset}..."):
    doc_ref = db.collection("signals").document(selected_asset)
    doc = doc_ref.get()

    if doc.exists:
        signal = doc.to_dict()
        ts_str = parse_ts(signal.get("timestamp", ""))
        components = signal.get("confidence_components", [])
        vote = majority_vote(components, str(signal.get("signal", "")).lower())
        conf = compute_confidence(components, vote)

        st.markdown(f"""
        ### {selected_asset}
        **🕒 Timestamp**: `{ts_str}`  
        **📊 Majority Vote**: `{vote.upper() or '—'}`  
        **🧠 Source**: `{signal.get('source','—')}`
        """)
        st.metric("🔥 Blended Confidence", f"{conf:.3f}")

        if components:
            rows = []
            for c in components:
                src = c.get("source", "—")
                s = str(c.get("signal", "")).upper() or "—"
                base = c.get("confidence", 0)
                ts_c = parse_ts(c.get("timestamp")) if c.get("timestamp") is not None else "—"
                aligned = (str(c.get("signal", "")).lower() == vote)
                                # compute effective weight & weighted conf (post-decay) for display
                src_key = str(c.get("source", "")).strip().lower()
                w = float(SOURCE_WEIGHTS.get(src_key, 1.0))
                # recency decay for display parity
                ts_raw = c.get("timestamp")
                decay = 1.0
                if ts_raw is not None:
                    try:
                        secs = ts_raw.seconds if hasattr(ts_raw, "seconds") else float(ts_raw)
                        age_days = max(0.0, (time.time() - secs) / 86400.0)
                        decay = 0.5 ** (age_days / 3.0)
                    except Exception:
                        pass
                weighted_conf = round(float(base) * w * decay, 3)

                rows.append({
                    "Source": src,
                    "Signal": s,
                    "Base Conf": round(float(base), 3),
                    "Weight": w,
                    "Weighted Conf": weighted_conf,
                    "Aligned": "✅" if aligned else "❌",
                    "Timestamp": ts_c,
                })
            df = pd.DataFrame(rows)
            st.subheader("🔬 Confidence Components (weighted)")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No component-level confidence found. Add `confidence_components` to the Firestore doc to enable blended confidence.")
    else:
        st.warning(f"No signal data found for {selected_asset}.")

st.button("🔄 Refresh")
