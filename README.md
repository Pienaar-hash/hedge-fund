# 🧠 Crypto Hedge Fund Project

A modular, data-driven crypto hedge fund platform built in Python.  
We combine quantitative strategies with transparent dashboards and rigorous backtesting to deliver sustainable trading performance.

---

## 🏗️ Project Structure


---

## 📈 Available Strategies

| Strategy | Description |
|----------|-------------|
| **ETH/BTC Spread (Relative Value)** | Mean-reversion logic between ETH/USDT and BTC/USDT, using z-score thresholds |
| **Volatility Targeting** | Dynamic position sizing based on rolling volatility |
| **Factor-Based Momentum** | Momentum filters derived from rolling returns and volume dynamics |

---

## 🚀 How to Run

### 1. Clone & Setup
```bash
git clone https://github.com/your-repo/hedge-fund.git
cd hedge-fund
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
