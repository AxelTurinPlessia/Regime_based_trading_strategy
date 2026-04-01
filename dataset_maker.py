import pandas as pd
import yfinance as yf
import requests
import time
from io import StringIO

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# -----------------------------------
# 1. Helpers to fetch HTML safely
# -----------------------------------

def get_html(url):
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return response.text

# -----------------------------------
# 2. Get ETF constituent proxies
# -----------------------------------

def get_sp500_tickers():
    """VOO proxy = S&P 500 constituents"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = get_html(url)
    tables = pd.read_html(StringIO(html))
    table = tables[0]
    tickers = table["Symbol"].astype(str).tolist()
    return [t.replace(".", "-") for t in tickers]

def get_russell2000_tickers():
    """VB proxy = Russell 2000 constituents approximation"""
    url = "https://en.wikipedia.org/wiki/Russell_2000_Index"
    html = get_html(url)
    tables = pd.read_html(StringIO(html))

    for table in tables:
        cols = [str(c) for c in table.columns]
        if "Ticker" in cols:
            tickers = table["Ticker"].astype(str).tolist()
            return [t.replace(".", "-") for t in tickers]
        if "Symbol" in cols:
            tickers = table["Symbol"].astype(str).tolist()
            return [t.replace(".", "-") for t in tickers]

    raise ValueError("Could not find ticker column on Russell 2000 page.")

# -----------------------------------
# 3. Download data
# -----------------------------------

def download_prices_and_volumes(tickers, start="2016-01-01", end=None):
    all_rows = []

    for i, ticker in enumerate(tickers, start=1):
        try:
            print(f"[{i}/{len(tickers)}] Downloading {ticker} ...")
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False
            )

            if df.empty:
                print(f"  -> No data for {ticker}")
                continue

            # Keep only Close and Volume
            cols_to_keep = []
            if "Close" in df.columns:
                cols_to_keep.append("Close")
            if "Volume" in df.columns:
                cols_to_keep.append("Volume")

            if len(cols_to_keep) < 2:
                print(f"  -> Missing Close/Volume for {ticker}")
                continue

            tmp = df[cols_to_keep].copy()
            tmp = tmp.reset_index()
            tmp["Ticker"] = ticker

            # Reorder columns
            tmp = tmp[["Date", "Ticker", "Close", "Volume"]]
            all_rows.append(tmp)

            time.sleep(0.1)

        except Exception as e:
            print(f"  -> Error for {ticker}: {e}")

    if not all_rows:
        raise ValueError("No data downloaded.")

    return pd.concat(all_rows, ignore_index=True)

# -----------------------------------
# 4. Main
# -----------------------------------

def main():
    print("Fetching tickers...")
    voo_tickers = get_sp500_tickers()
    vb_tickers = get_russell2000_tickers()

    print(f"VOO tickers: {len(voo_tickers)}")
    print(f"VB tickers: {len(vb_tickers)}")

    # -----------------------------
    # Download VOO data
    # -----------------------------
    print("\nDownloading VOO data...")
    voo_data = download_prices_and_volumes(
        voo_tickers,
        start="2016-01-01"
    )

    voo_file = "voo_prices_volumes.csv"
    voo_data.to_csv(voo_file, index=False)
    print(f"Saved VOO data to {voo_file}")

    # -----------------------------
    # Download VB data
    # -----------------------------
    print("\nDownloading VB data...")
    vb_data = download_prices_and_volumes(
        vb_tickers,
        start="2016-01-01"
    )

    vb_file = "vb_prices_volumes.csv"
    vb_data.to_csv(vb_file, index=False)
    print(f"Saved VB data to {vb_file}")

if __name__ == "__main__":
    main()