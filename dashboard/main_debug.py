import os, streamlit as st, time
st.set_page_config(page_title="Debug", layout="wide")
st.title("✅ Streamlit is rendering")
st.write({"ENV": os.getenv("ENV"), "PYTHONPATH": os.getenv("PYTHONPATH")})
if st.button("Ping"):
    st.success("Pong")
