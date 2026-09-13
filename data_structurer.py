import pandas as pd
from pathlib import Path

def extract_tickers_from_wide_panel(wide_file_path: Path, output_dir: Path):
    """
    Reads a wide-panel CSV (Date, AAPL, MSFT...) and splits it into 
    individual ticker CSVs inside the target directory.
    """
    if not wide_file_path.exists():
        print(f"⚠️ Could not find {wide_file_path}. Skipping.")
        return
        
    print(f"Reading {wide_file_path.name}...")
    df = pd.read_csv(wide_file_path)
    
    # Assume the first column is the Date column
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.set_index(date_col)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for ticker in df.columns:
        ticker_series = df[ticker].dropna()
        if ticker_series.empty:
            continue
            
        # Create the exact format the orchestrator needs
        out_df = pd.DataFrame({
            "Date": ticker_series.index,
            "Close": ticker_series.values,
            "Volume": 0.0  # Dummy volume required by the orchestrator
        })
        
        # Save the individual ticker
        out_path = output_dir / f"{ticker}.csv"
        out_df.to_csv(out_path, index=False)
        count += 1
        
    print(f"  -> Successfully extracted {count} tickers into {output_dir.name}/")

def main():
    print("Starting Local Data Restructuring...")
    data_dir = Path("data")
    
    # Target directories
    large_caps_dir = data_dir / "large_caps"
    small_caps_dir = data_dir / "small_caps"
    
    # Locate the wide panel files (checking both inside the subfolders and in the root data folder)
    large_wide_file = large_caps_dir / "large_caps_prices_no_dividends.csv"
    if not large_wide_file.exists():
        large_wide_file = data_dir / "large_caps_prices_no_dividends.csv"
        
    small_wide_file = small_caps_dir / "small_caps_prices_no_dividends.csv"
    if not small_wide_file.exists():
        small_wide_file = data_dir / "small_caps_prices_no_dividends.csv"

    # Execute the split
    extract_tickers_from_wide_panel(large_wide_file, large_caps_dir)
    extract_tickers_from_wide_panel(small_wide_file, small_caps_dir)
    
    print("\n✅ Data is perfectly staged! You can now run your strategy orchestrator.")

if __name__ == "__main__":
    main()