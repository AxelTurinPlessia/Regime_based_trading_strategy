import pandas as pd
import yfinance as yf
import requests
import time
from io import StringIO
from pathlib import Path

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

DATA_DIR = Path("data")
LARGE_CAPS_DIR = DATA_DIR / "large_caps"
SMALL_CAPS_DIR = DATA_DIR / "small_caps"

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

def extract_close_and_volume(df):
    if isinstance(df.columns, pd.MultiIndex):
        flattened = {}
        for column in df.columns:
            for level in column:
                if level in {"Close", "Volume"}:
                    flattened[level] = df[column]
                    break
        df = pd.DataFrame(flattened, index=df.index)

    missing_columns = [column for column in ("Close", "Volume") if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    output = df[["Close", "Volume"]].copy()
    output = output.reset_index()
    output = output[["Date", "Close", "Volume"]]
    return output


def safe_ticker_filename(ticker):
    return ticker.replace("/", "-")


def download_ticker_prices_and_volumes(ticker, start="2010-01-01", end=None):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if df.empty:
        raise ValueError("No data returned.")

    return extract_close_and_volume(df)


def download_group_to_folder(tickers, output_dir, start="2010-01-01", end=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    failed_tickers = []

    for i, ticker in enumerate(tickers, start=1):
        try:
            print(f"[{i}/{len(tickers)}] Downloading {ticker} ...")
            ticker_data = download_ticker_prices_and_volumes(
                ticker,
                start=start,
                end=end,
            )
            file_path = output_path / f"{safe_ticker_filename(ticker)}.csv"
            ticker_data.to_csv(file_path, index=False)
            print(f"  -> Saved {file_path}")
            saved_count += 1
            time.sleep(0.1)
        except Exception as e:
            print(f"  -> Error for {ticker}: {e}")
            failed_tickers.append(ticker)

    if saved_count == 0:
        raise ValueError(f"No data downloaded for {output_path}.")

    if failed_tickers:
        print(f"\nFailed tickers for {output_path}: {len(failed_tickers)}")
        print(", ".join(failed_tickers))

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
    print(f"\nDownloading large-cap data to {LARGE_CAPS_DIR} ...")
    download_group_to_folder(
        voo_tickers,
        output_dir=LARGE_CAPS_DIR,
        start="2010-01-01"
    )

    # -----------------------------
    # Download VB data
    # -----------------------------
    print(f"\nDownloading small-cap data to {SMALL_CAPS_DIR} ...")
    download_group_to_folder(
        vb_tickers,
        output_dir=SMALL_CAPS_DIR,
        start="2010-01-01"
    )

if __name__ == "__main__":
    main()
