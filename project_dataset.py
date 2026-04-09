from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from dataset_contracts import DatasetBundle


PriceUniverse = Literal["large_caps", "small_caps"]
DividendPolicy = Literal["with_dividends", "no_dividends"]
TradedPriceSource = Literal["portfolio_returns", "asset_prices"]
MarketReturnMode = Literal["average", "large_caps", "small_caps"]
UniversePriceSource = Literal["folders", "wide_csv"]


PRICE_PANEL_FILES: dict[tuple[PriceUniverse, DividendPolicy], str] = {
    ("large_caps", "with_dividends"): "large_caps_prices_with_dividends.csv",
    ("large_caps", "no_dividends"): "large_caps_prices_no_dividends.csv",
    ("small_caps", "with_dividends"): "small_caps_prices_with_dividends.csv",
    ("small_caps", "no_dividends"): "small_caps_prices_no_dividends.csv",
}


DEFAULT_PORTFOLIO_COLUMNS: dict[tuple[PriceUniverse, DividendPolicy], str] = {
    ("large_caps", "with_dividends"): "EW_Large_WithDiv",
    ("large_caps", "no_dividends"): "EW_Large_NoDiv",
    ("small_caps", "with_dividends"): "EW_Small_WithDiv",
    ("small_caps", "no_dividends"): "EW_Small_NoDiv",
}


@dataclass
class RegimeStrategyData:
    large_cap_prices: pd.DataFrame
    small_cap_prices: pd.DataFrame
    portfolio_returns: pd.DataFrame
    market_return: pd.Series
    vix_close: pd.Series
    spy_volume: pd.Series
    spy_volume_change: pd.Series
    universe_info: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)


def read_dated_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    return pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()


def slice_dated_frame(
    frame: pd.DataFrame | pd.Series,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame | pd.Series:
    if start_date is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(end_date)]
    return frame


def build_price_index_from_returns(
    return_series: pd.Series,
    base_value: float = 100.0,
    name: str | None = None,
) -> pd.Series:
    cleaned_returns = pd.Series(return_series).dropna().astype(float)
    price_index = base_value * (1 + cleaned_returns).cumprod()
    price_index.name = name or return_series.name or "Synthetic_Price"
    return price_index


def load_price_panel_from_folder(
    folder_path: str | Path,
    value_column: str = "Close",
) -> pd.DataFrame:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Price folder does not exist: {folder}")

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No per-ticker CSV files found in {folder}")

    series_list: list[pd.Series] = []
    for csv_file in csv_files:
        ticker_frame = pd.read_csv(csv_file, parse_dates=["Date"])
        if value_column not in ticker_frame.columns:
            raise KeyError(f"Missing '{value_column}' column in {csv_file}")
        ticker_series = (
            ticker_frame[["Date", value_column]]
            .dropna(subset=[value_column])
            .drop_duplicates(subset=["Date"])
            .set_index("Date")[value_column]
            .sort_index()
            .rename(csv_file.stem)
        )
        if not ticker_series.empty:
            series_list.append(ticker_series)

    if not series_list:
        raise ValueError(f"No usable '{value_column}' series found in {folder}")

    return pd.concat(series_list, axis=1).sort_index()


def infer_default_portfolio_column(
    asset_universe: PriceUniverse,
    dividend_policy: DividendPolicy,
) -> str:
    return DEFAULT_PORTFOLIO_COLUMNS[(asset_universe, dividend_policy)]


def load_cached_dataset_bundle(
    data_dir: str | Path = "data",
    asset_universe: PriceUniverse = "large_caps",
    dividend_policy: DividendPolicy = "with_dividends",
    portfolio_column: str | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> DatasetBundle:
    data_path = Path(data_dir)
    resolved_portfolio_column = portfolio_column or infer_default_portfolio_column(
        asset_universe=asset_universe,
        dividend_policy=dividend_policy,
    )

    asset_prices = read_dated_csv(data_path / PRICE_PANEL_FILES[(asset_universe, dividend_policy)])
    asset_prices = slice_dated_frame(asset_prices, start_date=start_date, end_date=end_date)

    portfolio_returns = read_dated_csv(data_path / "portfolio_returns_equal_weight.csv")
    portfolio_returns = slice_dated_frame(portfolio_returns, start_date=start_date, end_date=end_date)
    if resolved_portfolio_column not in portfolio_returns.columns:
        raise KeyError(
            f"Portfolio column '{resolved_portfolio_column}' was not found in "
            f"{data_path / 'portfolio_returns_equal_weight.csv'}."
        )

    vix_table = read_dated_csv(data_path / "vix_regime_classification.csv")
    vix_table = slice_dated_frame(vix_table, start_date=start_date, end_date=end_date)
    if "VIX_Close" not in vix_table.columns:
        raise KeyError(f"Missing 'VIX_Close' column in {data_path / 'vix_regime_classification.csv'}.")

    spy_volume_features = read_dated_csv(data_path / "spy_volume_features.csv")
    spy_volume_features = slice_dated_frame(spy_volume_features, start_date=start_date, end_date=end_date)
    missing_spy_columns = [
        column for column in ("SPY_Volume", "SPY_Volume_Change") if column not in spy_volume_features.columns
    ]
    if missing_spy_columns:
        raise KeyError(
            f"Missing SPY feature columns {missing_spy_columns} in {data_path / 'spy_volume_features.csv'}."
        )

    universe_info = pd.read_csv(data_path / "universe_info.csv")
    benchmark_returns = portfolio_returns[resolved_portfolio_column].rename(resolved_portfolio_column)
    benchmark_prices = build_price_index_from_returns(
        benchmark_returns,
        base_value=100.0,
        name=f"{resolved_portfolio_column}_Price",
    )

    bundle = DatasetBundle(
        asset_prices=asset_prices,
        asset_volumes=None,
        benchmark_prices=benchmark_prices,
        benchmark_volumes=spy_volume_features["SPY_Volume"].rename("SPY_Volume"),
        vix_close=vix_table["VIX_Close"].rename("VIX_Close"),
        portfolio_returns=portfolio_returns,
        metadata={
            "asset_universe": asset_universe,
            "dividend_policy": dividend_policy,
            "portfolio_column": resolved_portfolio_column,
            "universe_info": universe_info,
            "spy_volume_change": spy_volume_features["SPY_Volume_Change"].rename("SPY_Volume_Change"),
            "data_dir": str(data_path),
            "notes": [
                "asset_volumes are not available in the cached data folder yet.",
                "benchmark_prices are reconstructed from the selected portfolio return column.",
            ],
        },
    )
    bundle.validate()
    return bundle


def extract_market_inputs(
    bundle: DatasetBundle,
    portfolio_column: str | None = None,
) -> dict[str, pd.Series]:
    resolved_portfolio_column = portfolio_column or bundle.metadata.get("portfolio_column")
    if resolved_portfolio_column is None:
        raise ValueError("A portfolio column is required to extract the market return series.")
    if not isinstance(bundle.portfolio_returns, pd.DataFrame):
        raise TypeError("bundle.portfolio_returns must be a DataFrame for market input extraction.")
    if resolved_portfolio_column not in bundle.portfolio_returns.columns:
        raise KeyError(f"Portfolio column '{resolved_portfolio_column}' is not available in the dataset bundle.")

    spy_volume_change = bundle.metadata.get("spy_volume_change")
    if spy_volume_change is not None:
        spy_volume_change = pd.Series(spy_volume_change).rename("SPY_Volume_Change")

    return {
        "portfolio_return": bundle.portfolio_returns[resolved_portfolio_column].rename("Portfolio_Return"),
        "vix_close": bundle.vix_close.rename("VIX_Close") if bundle.vix_close is not None else None,
        "market_volume_change": spy_volume_change,
    }


def build_traded_price_series(
    bundle: DatasetBundle,
    source: TradedPriceSource = "portfolio_returns",
    portfolio_column: str | None = None,
    asset_ticker: str | None = None,
    base_value: float = 100.0,
) -> pd.Series:
    if source == "portfolio_returns":
        resolved_portfolio_column = portfolio_column or bundle.metadata.get("portfolio_column")
        if resolved_portfolio_column is None:
            raise ValueError("A portfolio column is required when source='portfolio_returns'.")
        if not isinstance(bundle.portfolio_returns, pd.DataFrame):
            raise TypeError("bundle.portfolio_returns must be a DataFrame when source='portfolio_returns'.")
        if resolved_portfolio_column not in bundle.portfolio_returns.columns:
            raise KeyError(
                f"Portfolio column '{resolved_portfolio_column}' is not available in the dataset bundle."
            )
        return build_price_index_from_returns(
            bundle.portfolio_returns[resolved_portfolio_column],
            base_value=base_value,
            name=f"{resolved_portfolio_column}_Price",
        )

    if asset_ticker is None:
        raise ValueError("asset_ticker must be provided when source='asset_prices'.")
    if asset_ticker not in bundle.asset_prices.columns:
        raise KeyError(f"Ticker '{asset_ticker}' is not available in bundle.asset_prices.")
    return bundle.asset_prices[asset_ticker].dropna().rename(asset_ticker)


def infer_market_return_series(
    portfolio_returns: pd.DataFrame,
    dividend_policy: DividendPolicy,
    market_return_mode: MarketReturnMode = "average",
) -> pd.Series:
    large_column = DEFAULT_PORTFOLIO_COLUMNS[("large_caps", dividend_policy)]
    small_column = DEFAULT_PORTFOLIO_COLUMNS[("small_caps", dividend_policy)]
    missing_columns = [column for column in (large_column, small_column) if column not in portfolio_returns.columns]
    if missing_columns:
        raise KeyError(f"Missing market return columns in portfolio return table: {missing_columns}")

    if market_return_mode == "large_caps":
        return portfolio_returns[large_column].rename("Market_Portfolio_Return")
    if market_return_mode == "small_caps":
        return portfolio_returns[small_column].rename("Market_Portfolio_Return")

    return portfolio_returns[[large_column, small_column]].mean(axis=1).rename("Market_Portfolio_Return")


def load_regime_strategy_data(
    data_dir: str | Path = "data",
    dividend_policy: DividendPolicy = "no_dividends",
    market_return_mode: MarketReturnMode = "average",
    universe_price_source: UniversePriceSource | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> RegimeStrategyData:
    data_path = Path(data_dir)
    resolved_universe_price_source = universe_price_source
    if resolved_universe_price_source is None:
        resolved_universe_price_source = "folders" if dividend_policy == "no_dividends" else "wide_csv"

    if resolved_universe_price_source == "folders":
        large_cap_prices = load_price_panel_from_folder(data_path / "large_caps", value_column="Close")
        small_cap_prices = load_price_panel_from_folder(data_path / "small_caps", value_column="Close")
    else:
        large_cap_prices = read_dated_csv(data_path / PRICE_PANEL_FILES[("large_caps", dividend_policy)])
        small_cap_prices = read_dated_csv(data_path / PRICE_PANEL_FILES[("small_caps", dividend_policy)])

    portfolio_returns = read_dated_csv(data_path / "portfolio_returns_equal_weight.csv")
    vix_table = read_dated_csv(data_path / "vix_regime_classification.csv")
    spy_volume_features = read_dated_csv(data_path / "spy_volume_features.csv")
    universe_info = pd.read_csv(data_path / "universe_info.csv")

    large_cap_prices = slice_dated_frame(large_cap_prices, start_date=start_date, end_date=end_date)
    small_cap_prices = slice_dated_frame(small_cap_prices, start_date=start_date, end_date=end_date)
    portfolio_returns = slice_dated_frame(portfolio_returns, start_date=start_date, end_date=end_date)
    vix_table = slice_dated_frame(vix_table, start_date=start_date, end_date=end_date)
    spy_volume_features = slice_dated_frame(spy_volume_features, start_date=start_date, end_date=end_date)

    if "VIX_Close" not in vix_table.columns:
        raise KeyError(f"Missing 'VIX_Close' column in {data_path / 'vix_regime_classification.csv'}.")

    missing_spy_columns = [
        column for column in ("SPY_Volume", "SPY_Volume_Change") if column not in spy_volume_features.columns
    ]
    if missing_spy_columns:
        raise KeyError(
            f"Missing SPY feature columns {missing_spy_columns} in {data_path / 'spy_volume_features.csv'}."
        )

    market_return = infer_market_return_series(
        portfolio_returns=portfolio_returns,
        dividend_policy=dividend_policy,
        market_return_mode=market_return_mode,
    )

    return RegimeStrategyData(
        large_cap_prices=large_cap_prices,
        small_cap_prices=small_cap_prices,
        portfolio_returns=portfolio_returns,
        market_return=market_return,
        vix_close=vix_table["VIX_Close"].rename("VIX_Close"),
        spy_volume=spy_volume_features["SPY_Volume"].rename("SPY_Volume"),
        spy_volume_change=spy_volume_features["SPY_Volume_Change"].rename("SPY_Volume_Change"),
        universe_info=universe_info,
        metadata={
            "data_dir": str(data_path),
            "dividend_policy": dividend_policy,
            "market_return_mode": market_return_mode,
            "universe_price_source": resolved_universe_price_source,
            "low_vol_universe": "small_caps",
            "high_vol_universe": "large_caps",
            "notes": [
                "The raw teammate-generated vb/voo prices_volumes files are intentionally ignored here.",
                "When universe_price_source='folders', the strategy universe prices come from data/large_caps and data/small_caps.",
                "The HMM market return still comes from data/portfolio_returns_equal_weight.csv.",
            ],
        },
    )
