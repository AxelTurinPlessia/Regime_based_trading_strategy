from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def classify_vix_regime(
    vix_data: pd.DataFrame,
    lookback_days: int = 25,
    threshold_pct: float = 0.10,
) -> pd.DataFrame:
    vix_close = vix_data["Close"].rename("VIX_Close")
    prior_average = vix_close.shift(1).rolling(
        lookback_days,
        min_periods=lookback_days,
    ).mean()
    high_vol_threshold = (1 + threshold_pct) * prior_average

    return pd.DataFrame(
        {
            "VIX_Close": vix_close,
            "Prior_25D_Avg": prior_average,
            "High_Vol_Threshold": high_vol_threshold,
            "Regime": np.where(
                prior_average.notna() & (vix_close > high_vol_threshold),
                "high_vol",
                "low_vol",
            ),
        }
    )


def compute_log_return(price: pd.Series, name: str | None = None) -> pd.Series:
    output_name = name or price.name
    return np.log(price / price.shift(1)).rename(output_name)


def compute_realized_volatility(
    return_series: pd.Series,
    window: int = 21,
    annualization_factor: float = 252.0,
    name: str | None = None,
) -> pd.Series:
    output_name = name or return_series.name
    return (
        return_series.rolling(window).std(ddof=0).mul(np.sqrt(annualization_factor)).rename(output_name)
    )


def compute_log_volume_change(volume: pd.Series, name: str | None = None) -> pd.Series:
    output_name = name or volume.name
    return np.log(volume.replace(0, np.nan)).diff().rename(output_name)


def assemble_feature_table(features: dict[str, pd.Series]) -> pd.DataFrame:
    if not features:
        raise ValueError("At least one feature series is required.")

    renamed_features = [series.rename(name) for name, series in features.items()]
    feature_table = pd.concat(renamed_features, axis=1).dropna()
    if feature_table.empty:
        raise ValueError("No aligned HMM features are available after dropping missing values.")
    return feature_table


def build_market_hmm_features(
    portfolio_return: pd.Series,
    vix_close: pd.Series,
    volume_change: pd.Series | None = None,
    realized_vol_window: int = 21,
    extra_features: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    features: dict[str, pd.Series] = {
        "Portfolio_Return": portfolio_return.rename("Portfolio_Return"),
        "Abs_Return": portfolio_return.abs().rename("Abs_Return"),
        "Realized_Volatility": compute_realized_volatility(
            portfolio_return,
            window=realized_vol_window,
            name="Realized_Volatility",
        ),
        "VIX_Close": vix_close.rename("VIX_Close"),
    }
    if volume_change is not None:
        features["Market_Volume_Change"] = volume_change.rename("Market_Volume_Change")
    if extra_features:
        features.update(extra_features)
    return assemble_feature_table(features)


def build_asset_hmm_features(
    asset_price: pd.Series,
    asset_volume: pd.Series | None = None,
    market_price: pd.Series | None = None,
    market_volume: pd.Series | None = None,
    vix_close: pd.Series | None = None,
    asset_name: str = "Asset",
    realized_vol_window: int = 21,
    extra_features: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    asset_return = compute_log_return(asset_price, name=f"{asset_name}_Return")
    features: dict[str, pd.Series] = {
        f"{asset_name}_Return": asset_return,
        f"{asset_name}_Abs_Return": asset_return.abs().rename(f"{asset_name}_Abs_Return"),
        f"{asset_name}_Realized_Volatility": compute_realized_volatility(
            asset_return,
            window=realized_vol_window,
            name=f"{asset_name}_Realized_Volatility",
        ),
    }
    if asset_volume is not None:
        features[f"{asset_name}_Volume_Change"] = compute_log_volume_change(
            asset_volume,
            name=f"{asset_name}_Volume_Change",
        )
    if market_price is not None:
        market_return = compute_log_return(market_price, name="Market_Return")
        features["Market_Return"] = market_return
        features["Market_Realized_Volatility"] = compute_realized_volatility(
            market_return,
            window=realized_vol_window,
            name="Market_Realized_Volatility",
        )
    if market_volume is not None:
        features["Market_Volume_Change"] = compute_log_volume_change(
            market_volume,
            name="Market_Volume_Change",
        )
    if vix_close is not None:
        features["VIX_Close"] = vix_close.rename("VIX_Close")
    if extra_features:
        features.update(extra_features)
    return assemble_feature_table(features)


def standardize_feature_frames(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray | None]:
    feature_means = train_features.mean()
    feature_stds = train_features.std(ddof=0).replace(0, 1.0)
    x_train = ((train_features - feature_means) / feature_stds).to_numpy()
    x_test = None if test_features is None else ((test_features - feature_means) / feature_stds).to_numpy()
    return feature_means, feature_stds, x_train, x_test


def logsumexp(values: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    stable_values = values - max_values
    summed = np.sum(np.exp(stable_values), axis=axis, keepdims=True)
    output = max_values + np.log(summed)
    if not keepdims:
        output = np.squeeze(output, axis=axis)
    return output


def initialize_hmm_parameters(
    x: np.ndarray,
    volatility_feature_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_obs, n_features = x.shape
    if volatility_feature_indices is None:
        volatility_feature_indices = list(range(n_features))

    split_score = x[:, volatility_feature_indices].mean(axis=1)
    split_order = np.argsort(split_score)
    initial_states = np.zeros(n_obs, dtype=int)
    initial_states[split_order[n_obs // 2 :]] = 1

    means = []
    variances = []
    for state in range(2):
        state_obs = x[initial_states == state]
        if len(state_obs) == 0:
            state_obs = x
        means.append(state_obs.mean(axis=0))
        variances.append(np.clip(state_obs.var(axis=0), 1e-4, None))

    startprob = np.array([0.5, 0.5], dtype=float)
    transmat = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)
    return startprob, transmat, np.vstack(means), np.vstack(variances)


def gaussian_logpdf_diag(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    log_det = np.sum(np.log(2 * np.pi * variances), axis=1)
    squared_mahalanobis = np.sum(
        ((x[:, None, :] - means[None, :, :]) ** 2) / variances[None, :, :],
        axis=2,
    )
    return -0.5 * (log_det[None, :] + squared_mahalanobis)


def forward_backward(
    log_startprob: np.ndarray,
    log_transmat: np.ndarray,
    emission_log_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    n_obs, n_states = emission_log_prob.shape

    log_alpha = np.empty((n_obs, n_states))
    log_alpha[0] = log_startprob + emission_log_prob[0]
    for step in range(1, n_obs):
        log_alpha[step] = emission_log_prob[step] + logsumexp(
            log_alpha[step - 1][:, None] + log_transmat,
            axis=0,
        )

    log_beta = np.zeros((n_obs, n_states))
    for step in range(n_obs - 2, -1, -1):
        log_beta[step] = logsumexp(
            log_transmat + emission_log_prob[step + 1][None, :] + log_beta[step + 1][None, :],
            axis=1,
        )

    log_likelihood = float(logsumexp(log_alpha[-1], axis=0))
    log_gamma = log_alpha + log_beta - log_likelihood
    gamma = np.exp(log_gamma)
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi = np.empty((max(n_obs - 1, 0), n_states, n_states))
    for step in range(n_obs - 1):
        log_xi_step = (
            log_alpha[step][:, None]
            + log_transmat
            + emission_log_prob[step + 1][None, :]
            + log_beta[step + 1][None, :]
            - log_likelihood
        )
        xi[step] = np.exp(log_xi_step)
        xi[step] /= xi[step].sum()

    return gamma, xi, log_likelihood


def fit_gaussian_hmm(
    x: np.ndarray,
    volatility_feature_indices: list[int] | None = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    min_covar: float = 1e-4,
) -> dict[str, Any]:
    startprob, transmat, means, variances = initialize_hmm_parameters(
        x,
        volatility_feature_indices=volatility_feature_indices,
    )
    prev_log_likelihood = -np.inf

    for iteration in range(1, max_iter + 1):
        log_startprob = np.log(np.clip(startprob, 1e-12, None))
        log_transmat = np.log(np.clip(transmat, 1e-12, None))
        emission_log_prob = gaussian_logpdf_diag(x, means, variances)
        gamma, xi, log_likelihood = forward_backward(log_startprob, log_transmat, emission_log_prob)

        startprob = gamma[0]
        transmat = xi.sum(axis=0)
        transmat /= np.clip(transmat.sum(axis=1, keepdims=True), 1e-12, None)

        weights = np.clip(gamma.sum(axis=0), 1e-12, None)
        means = (gamma.T @ x) / weights[:, None]
        centered = x[:, None, :] - means[None, :, :]
        variances = (gamma[:, :, None] * centered**2).sum(axis=0) / weights[:, None]
        variances = np.clip(variances, min_covar, None)

        if log_likelihood - prev_log_likelihood < tol:
            break
        prev_log_likelihood = log_likelihood

    return {
        "startprob": startprob,
        "transmat": transmat,
        "means": means,
        "variances": variances,
        "gamma": gamma,
        "xi": xi,
        "log_likelihood": log_likelihood,
        "iterations": iteration,
    }


def forward_filter_probabilities(
    startprob: np.ndarray,
    transmat: np.ndarray,
    emission_log_prob: np.ndarray,
) -> tuple[np.ndarray, float]:
    n_obs, _ = emission_log_prob.shape
    log_startprob = np.log(np.clip(startprob, 1e-12, None))
    log_transmat = np.log(np.clip(transmat, 1e-12, None))

    log_filtered = np.empty_like(emission_log_prob)
    log_norm_constants = np.empty(n_obs)

    log_joint = log_startprob + emission_log_prob[0]
    log_norm_constants[0] = logsumexp(log_joint, axis=0)
    log_filtered[0] = log_joint - log_norm_constants[0]

    for step in range(1, n_obs):
        log_pred = logsumexp(log_filtered[step - 1][:, None] + log_transmat, axis=0)
        log_joint = log_pred + emission_log_prob[step]
        log_norm_constants[step] = logsumexp(log_joint, axis=0)
        log_filtered[step] = log_joint - log_norm_constants[step]

    return np.exp(log_filtered), float(log_norm_constants.sum())


def walk_forward_one_step_ahead_forecasts(
    x: np.ndarray,
    initial_filtered_prob: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_obs = x.shape[0]
    n_states = transmat.shape[0]

    current_day_forecast = np.empty((n_obs, n_states))
    filtered_probabilities = np.empty((n_obs, n_states))
    next_day_forecast = np.empty((n_obs, n_states))

    previous_filtered = initial_filtered_prob.copy()
    for step in range(n_obs):
        current_day_forecast[step] = previous_filtered @ transmat
        emission_log_prob_step = gaussian_logpdf_diag(x[step : step + 1], means, variances)[0]
        log_filtered_step = np.log(np.clip(current_day_forecast[step], 1e-12, None)) + emission_log_prob_step
        log_filtered_step -= logsumexp(log_filtered_step, axis=0)
        filtered_probabilities[step] = np.exp(log_filtered_step)
        next_day_forecast[step] = filtered_probabilities[step] @ transmat
        previous_filtered = filtered_probabilities[step]

    return current_day_forecast, filtered_probabilities, next_day_forecast


def compute_state_feature_means(gamma: np.ndarray, feature_table: pd.DataFrame) -> pd.DataFrame:
    weights = np.clip(gamma.sum(axis=0), 1e-12, None)
    weighted_means = (gamma.T @ feature_table.to_numpy()) / weights[:, None]
    return pd.DataFrame(weighted_means, index=[0, 1], columns=feature_table.columns)


def label_states_from_volatility(
    state_feature_means: pd.DataFrame,
    volatility_columns: list[str],
) -> tuple[int, int, dict[int, str], pd.DataFrame]:
    missing_columns = [column for column in volatility_columns if column not in state_feature_means.columns]
    if missing_columns:
        raise KeyError(f"Missing volatility columns for state labeling: {missing_columns}")

    volatility_score = state_feature_means[volatility_columns].mean(axis=1)
    high_vol_state = int(volatility_score.idxmax())
    low_vol_state = 1 - high_vol_state
    state_labels = {
        low_vol_state: "low_vol",
        high_vol_state: "high_vol",
    }
    labeled_summary = state_feature_means.rename(index=state_labels).loc[["low_vol", "high_vol"]]
    return low_vol_state, high_vol_state, state_labels, labeled_summary


def build_transition_matrix(transmat: np.ndarray, state_labels: dict[int, str]) -> pd.DataFrame:
    return pd.DataFrame(
        transmat,
        index=[state_labels[0], state_labels[1]],
        columns=[state_labels[0], state_labels[1]],
    ).loc[["low_vol", "high_vol"], ["low_vol", "high_vol"]]


def build_probability_table(
    gamma: np.ndarray,
    index: pd.Index,
    low_vol_state: int,
    high_vol_state: int,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "Prob_Low_Vol": gamma[:, low_vol_state],
            "Prob_High_Vol": gamma[:, high_vol_state],
        },
        index=index,
    )
    output["Most_Likely_Regime"] = np.where(
        output["Prob_High_Vol"] >= output["Prob_Low_Vol"],
        "high_vol",
        "low_vol",
    )
    return output


def fit_full_sample_hmm(
    feature_table: pd.DataFrame,
    volatility_columns: list[str],
) -> dict[str, Any]:
    _, _, x_scaled, _ = standardize_feature_frames(feature_table)
    volatility_feature_indices = [feature_table.columns.get_loc(column) for column in volatility_columns]
    model = fit_gaussian_hmm(x_scaled, volatility_feature_indices=volatility_feature_indices)

    state_feature_means = compute_state_feature_means(model["gamma"], feature_table)
    low_vol_state, high_vol_state, state_labels, state_summary = label_states_from_volatility(
        state_feature_means,
        volatility_columns=volatility_columns,
    )
    probabilities = build_probability_table(
        model["gamma"],
        feature_table.index,
        low_vol_state=low_vol_state,
        high_vol_state=high_vol_state,
    )

    return {
        "model": model,
        "features": feature_table,
        "state_feature_means": state_feature_means,
        "state_summary": state_summary,
        "transition_matrix": build_transition_matrix(model["transmat"], state_labels),
        "probabilities": probabilities,
        "output": feature_table.join(probabilities),
        "high_vol_state": high_vol_state,
        "low_vol_state": low_vol_state,
        "state_labels": state_labels,
    }


def fit_oos_hmm(
    feature_table: pd.DataFrame,
    split_date: str | pd.Timestamp,
    volatility_columns: list[str],
) -> dict[str, Any]:
    split_timestamp = pd.Timestamp(split_date)
    train_features = feature_table.loc[feature_table.index < split_timestamp].copy()
    test_features = feature_table.loc[feature_table.index >= split_timestamp].copy()

    if train_features.empty:
        raise ValueError("Training sample is empty. Choose a later split date.")
    if test_features.empty:
        raise ValueError("Test sample is empty. Choose an earlier split date.")

    feature_means, feature_stds, x_train, x_test = standardize_feature_frames(train_features, test_features)
    volatility_feature_indices = [train_features.columns.get_loc(column) for column in volatility_columns]
    model = fit_gaussian_hmm(x_train, volatility_feature_indices=volatility_feature_indices)

    train_state_feature_means = compute_state_feature_means(model["gamma"], train_features)
    low_vol_state, high_vol_state, state_labels, state_summary = label_states_from_volatility(
        train_state_feature_means,
        volatility_columns=volatility_columns,
    )

    train_emission_log_prob = gaussian_logpdf_diag(x_train, model["means"], model["variances"])
    train_filtered_probabilities, train_filtered_log_likelihood = forward_filter_probabilities(
        model["startprob"],
        model["transmat"],
        train_emission_log_prob,
    )
    last_train_filtered_probability = train_filtered_probabilities[-1]

    current_day_forecast, filtered_probabilities, next_day_forecast = walk_forward_one_step_ahead_forecasts(
        x_test,
        last_train_filtered_probability,
        model["transmat"],
        model["means"],
        model["variances"],
    )

    forecasts = test_features.join(
        pd.DataFrame(
            {
                "Forecast_Prob_Low_Vol": current_day_forecast[:, low_vol_state],
                "Forecast_Prob_High_Vol": current_day_forecast[:, high_vol_state],
                "Filtered_Prob_Low_Vol": filtered_probabilities[:, low_vol_state],
                "Filtered_Prob_High_Vol": filtered_probabilities[:, high_vol_state],
                "Next_Day_Forecast_Prob_Low_Vol": next_day_forecast[:, low_vol_state],
                "Next_Day_Forecast_Prob_High_Vol": next_day_forecast[:, high_vol_state],
            },
            index=test_features.index,
        )
    )
    forecasts["Forecast_Regime"] = np.where(
        forecasts["Forecast_Prob_High_Vol"] >= forecasts["Forecast_Prob_Low_Vol"],
        "high_vol",
        "low_vol",
    )
    forecasts["Filtered_Regime"] = np.where(
        forecasts["Filtered_Prob_High_Vol"] >= forecasts["Filtered_Prob_Low_Vol"],
        "high_vol",
        "low_vol",
    )

    summary = pd.Series(
        {
            "Train_Start": train_features.index.min().date(),
            "Train_End": train_features.index.max().date(),
            "Test_Start": test_features.index.min().date(),
            "Test_End": test_features.index.max().date(),
            "Train_Observations": len(train_features),
            "Test_Observations": len(test_features),
            "Train_Log_Likelihood": model["log_likelihood"],
            "Filtered_Train_Log_Likelihood": train_filtered_log_likelihood,
        }
    )

    return {
        "model": model,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "train_features": train_features,
        "test_features": test_features,
        "train_state_feature_means": train_state_feature_means,
        "state_summary": state_summary,
        "transition_matrix": build_transition_matrix(model["transmat"], state_labels),
        "train_filtered_probabilities": train_filtered_probabilities,
        "forecasts": forecasts,
        "summary": summary,
        "high_vol_state": high_vol_state,
        "low_vol_state": low_vol_state,
        "state_labels": state_labels,
    }

