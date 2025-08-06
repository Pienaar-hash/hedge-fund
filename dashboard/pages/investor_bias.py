import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import time

# Initialize Firebase if not already done
if not firebase_admin._apps:
    cred = credentials.Certificate("/root/hedge-fund/firebase_creds.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

st.set_page_config(page_title="Investor Bias", layout="centered")
st.title("📈 Investor Bias")

with st.spinner("Fetching latest signal..."):
    doc_ref = db.collection("signals").document("BTCUSDT")
    doc = doc_ref.get()

    if doc.exists:
        signal = doc.to_dict()
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(signal["timestamp"]))
        st.markdown(f"""
        **🕒 Timestamp**: `{ts}`  
        **📊 Signal**: `{signal['signal'].upper()}`  
        **🧠 Source**: `{signal['source']}`  
        **🔥 Confidence**: `{signal['confidence']:.2f}`
        """)
    else:
        st.warning("No signal data found.")

st.button("🔄 Refresh")
