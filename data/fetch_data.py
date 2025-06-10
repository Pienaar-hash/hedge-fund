import requests
import pandas as pd

def fetch_prices(symbol, id):
    url = f"https://api.coingecko.com/api/v3/coins/{id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': '365',
        'interval': 'daily'
    }
    r = requests.get(url, params=params)
    
    if r.status_code != 200:
        print(f"❌ Error fetching {symbol}: HTTP {r.status_code}")
        print(r.text)
        return pd.DataFrame()

    data = r.json()

    if 'prices' not in data:
        print(f"❌ 'prices' not in response for {symbol}:")
        print(data)
        return pd.DataFrame()

    df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

btc = fetch_prices("BTC", "bitcoin")
eth = fetch_prices("ETH", "ethereum")

if not btc.empty:
    btc.to_csv("data/processed/btc.csv", index=False)
    print("✅ BTC data saved.")

if not eth.empty:
    eth.to_csv("data/processed/eth.csv", index=False)
