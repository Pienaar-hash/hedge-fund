import streamlit as st
import os

st.header("📚 Research & Insights")

st.markdown("""
This section surfaces AI-generated insights, notes, and research from LLM workflows, grid searches,
and any modeling investigations relevant to strategy development.

You can drop markdown, CSVs, or visual research summaries into a `/research_outputs/` folder.
""")

folder = "research_outputs"

if not os.path.exists(folder):
    st.warning("No research outputs folder found.")
else:
    files = [f for f in os.listdir(folder) if f.endswith(".md") or f.endswith(".txt") or f.endswith(".csv")]
    if not files:
        st.info("No markdown or CSV research files found in /research_outputs.")
    else:
        selected = st.selectbox("Select a file to view", files)
        ext = selected.split(".")[-1]
        path = os.path.join(folder, selected)
        if ext == "csv":
            df = pd.read_csv(path)
            st.dataframe(df, use_container_width=True)
        else:
            with open(path, "r") as f:
                content = f.read()
            st.markdown(f"```{ext}\n{content}\n```")

def render():
    import streamlit as st
    import os
    import pandas as pd

    st.header("📚 Research & Insights")

    st.markdown("""
    This section surfaces AI-generated insights, notes, and research from LLM workflows, grid searches,
    and any modeling investigations relevant to strategy development.

    You can drop markdown, CSVs, or visual research summaries into a `/research_outputs/` folder.
    """)

    folder = "research_outputs"

    if not os.path.exists(folder):
        st.warning("No research outputs folder found.")
    else:
        files = [f for f in os.listdir(folder) if f.endswith(".md") or f.endswith(".txt") or f.endswith(".csv")]
        if not files:
            st.info("No markdown or CSV research files found in /research_outputs.")
        else:
            selected = st.selectbox("Select a file to view", files)
            ext = selected.split(".")[-1]
            path = os.path.join(folder, selected)
            if ext == "csv":
                df = pd.read_csv(path)
                st.dataframe(df, use_container_width=True)
            else:
                with open(path, "r") as f:
                    content = f.read()
                st.markdown(f"```{ext}\n{content}\n```")
