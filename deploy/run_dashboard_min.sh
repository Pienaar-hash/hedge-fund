#!/usr/bin/env bash
cd /root/hedge-fund
exec /root/hedge-fund/venv/bin/python -m streamlit run dashboard/app.py \
  --server.port=8501 --server.address=0.0.0.0 --server.headless true --logger.level=info
