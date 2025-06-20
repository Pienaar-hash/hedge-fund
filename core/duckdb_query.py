import duckdb
import pandas as pd

def get_top_momentum_symbols(n=3):
    con = duckdb.connect("data/factors.duckdb")
    query = """
    SELECT symbol, sharpe
    FROM momentum_stats
    ORDER BY sharpe DESC
    LIMIT ?
    """
    df = con.execute(query, [n]).fetchdf()
    con.close()
    return df["symbol"].tolist()
