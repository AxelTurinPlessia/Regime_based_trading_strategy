from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DatasetBundle:
    asset_prices: pd.DataFrame
    asset_volumes: pd.DataFrame | None = None
    benchmark_prices: pd.Series | None = None
    benchmark_volumes: pd.Series | None = None
    vix_close: pd.Series | None = None
    portfolio_returns: pd.DataFrame | pd.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.asset_prices.empty:
            raise ValueError("asset_prices cannot be empty.")
        if not isinstance(self.asset_prices.index, pd.DatetimeIndex):
            raise TypeError("asset_prices must use a DatetimeIndex.")
        if self.asset_volumes is not None and not self.asset_volumes.index.equals(self.asset_prices.index):
            raise ValueError("asset_volumes must share the same index as asset_prices.")
        if self.benchmark_prices is not None and not isinstance(self.benchmark_prices.index, pd.DatetimeIndex):
            raise TypeError("benchmark_prices must use a DatetimeIndex.")
        if self.vix_close is not None and not isinstance(self.vix_close.index, pd.DatetimeIndex):
            raise TypeError("vix_close must use a DatetimeIndex.")


def build_dataset_bundle_skeleton() -> dict[str, Any]:
    return {
        "required": {
            "asset_prices": "Wide DataFrame indexed by trading date, one column per asset.",
        },
        "optional": {
            "asset_volumes": "Wide DataFrame aligned with asset_prices if volume-based features are needed.",
            "benchmark_prices": "Series for market-level return and realized-volatility features.",
            "benchmark_volumes": "Series for market-level volume change features.",
            "vix_close": "Series aligned on trading dates if VIX or implied-vol inputs are part of the model.",
            "portfolio_returns": "Series or DataFrame if regime switching is driven by an existing portfolio instead of raw prices.",
            "metadata": "Universe definition, survivorship policy, rebalancing calendar, sector tags, region tags, etc.",
        },
        "notes": [
            "Keep the final data adapter outside the strategy code so the rest of the pipeline stays dataset-agnostic.",
            "Decide explicitly how missing dates, delistings, splits, and survivorship will be handled before wiring the final loader.",
            "Prefer returning clean pandas objects with a DatetimeIndex; the HMM and strategy modules in this repo are written to consume those directly.",
        ],
    }


def load_dataset_bundle(*args: Any, **kwargs: Any) -> DatasetBundle:
    raise NotImplementedError(
        "Dataset loading is project-specific and has intentionally not been implemented yet. "
        "Create a project adapter that returns a validated DatasetBundle."
    )
