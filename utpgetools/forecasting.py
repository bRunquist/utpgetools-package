"""
Well Production Forecasting & ML Prediction Module — Three-Stream Edition
=========================================================================

This module provides an end-to-end pipeline for forecasting oil, water, and
gas well production using decline curve analysis (DCA) and machine learning.
It is designed for the Midland Basin horizontal-well dataset exported from
Enverus but is generalizable to any well/production CSV pair whose column
names are supplied by the caller.

All three fluid streams (oil, water, gas) are modelled independently —
each gets its own DCA fits, ML models, and P10/P50/P90 probabilistic
forecasts.

High-level workflow executed by ``predict_new_well()``:
    1. Load, clean, and merge well metadata with monthly production data.
    2. For each fluid stream (oil, water, gas):
       a. Fit multiple DCA models (Arps hyperbolic, modified hyperbolic,
          stretched exponential, Duong) to every qualifying well.
       b. Select the best-fit model per well and discard statistical outliers.
       c. Generate a correlation matrix of well-completion parameters vs. DCA
          forecast parameters.
       d. Train and benchmark several scikit-learn regression models (linear,
          polynomial, ridge, lasso, random forest, gradient boosting) to
          predict DCA parameters from well attributes.
    3. Use the best ML model per stream to predict DCA parameters for a
       proposed well, then generate P10 / P50 / P90 production forecasts
       for oil, water, and gas.

Key public function:
    ``predict_new_well``  – single entry-point that runs the full pipeline.

Batch / reusable API:
    ``build_forecast_pipeline``  – run Steps 1-2 once, return trained state.
    ``predict_from_pipeline``    – predict one new well from saved state.
    ``predict_new_wells``        – batch-predict multiple wells in one call.

Custom features:
    The ``well_features`` parameter accepts arbitrary column names.  If your
    wells CSV contains proprietary data (core porosity, permeability, TOC,
    bottom-hole pressure, etc.), include those names and the ML model will
    incorporate them alongside standard completion attributes.

Dependencies:
    numpy, pandas, scipy, scikit-learn, matplotlib, seaborn
"""

from __future__ import annotations

import warnings
import time
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import zscore
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving to disk
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    """Formatted timestamp for status messages."""
    return time.strftime("%H:%M:%S")


def _print_status(msg: str) -> None:
    """Print a timestamped status message."""
    print(f"[{_ts()}] {msg}")


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

@dataclass
class DCAResult:
    """Stores the result of a single DCA fit for one well.

    Attributes:
        api_uwi: Well identifier.
        model_name: Name of the DCA model that was fit.
        params: Dictionary of fitted model parameters.
        r_squared: Coefficient of determination of the fit.
        rmse: Root-mean-square error of the fit (monthly rate units).
        eur_bbl: Estimated Ultimate Recovery to the forecast horizon.
        forecast_months: Array of month indices used in forecast.
        forecast_rates: Array of forecasted monthly rates (bbl or mcf).
    """
    api_uwi: str = ""
    model_name: str = ""
    params: Dict[str, float] = field(default_factory=dict)
    r_squared: float = np.nan
    rmse: float = np.nan
    eur_bbl: float = np.nan
    forecast_months: np.ndarray = field(default_factory=lambda: np.array([]))
    forecast_rates: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class StreamForecast:
    """Forecast results for one fluid stream (oil, water, or gas).

    Attributes:
        stream: Stream name ('oil', 'water', or 'gas').
        unit: Production unit ('BBL' for oil/water, 'MCF' for gas).
        predicted_dca_params: Dict of DCA parameters predicted by ML.
        forecast_p10: DataFrame with month, rate, cumulative for P10.
        forecast_p50: DataFrame with month, rate, cumulative for P50.
        forecast_p90: DataFrame with month, rate, cumulative for P90.
        eur_p10: EUR at P10.
        eur_p50: EUR at P50.
        eur_p90: EUR at P90.
        best_ml_model_name: Name of the best ML model for this stream.
        model_comparison: ML model comparison for this stream.
        well_dca_results: DCA results for training wells (this stream).
    """
    stream: str = ""
    unit: str = ""
    predicted_dca_params: Dict[str, float] = field(default_factory=dict)
    forecast_p10: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast_p50: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast_p90: pd.DataFrame = field(default_factory=pd.DataFrame)
    eur_p10: float = 0.0
    eur_p50: float = 0.0
    eur_p90: float = 0.0
    best_ml_model_name: str = ""
    model_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    well_dca_results: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class WellPrediction:
    """Prediction results for a single well from a trained pipeline.

    Attributes:
        well_id: User-supplied label for this well (default: auto-generated).
        predicted_dca_params: Dict of predicted DCA parameters (primary stream).
        forecast_p10: DataFrame with month, rate, cumulative for P10 (primary).
        forecast_p50: DataFrame with month, rate, cumulative for P50 (primary).
        forecast_p90: DataFrame with month, rate, cumulative for P90 (primary).
        eur_p10: EUR at P10 (primary stream).
        eur_p50: EUR at P50 (primary stream).
        eur_p90: EUR at P90 (primary stream).
        stream_forecasts: Dict of per-stream StreamForecast objects.
    """
    well_id: str = ""
    predicted_dca_params: Dict[str, float] = field(default_factory=dict)
    forecast_p10: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast_p50: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast_p90: pd.DataFrame = field(default_factory=pd.DataFrame)
    eur_p10: float = 0.0
    eur_p50: float = 0.0
    eur_p90: float = 0.0
    stream_forecasts: Dict[str, StreamForecast] = field(default_factory=dict)


@dataclass
class BatchPipelineResults:
    """Container for batch prediction of multiple new wells.

    Attributes:
        well_dca_results: DCA fit results for all training wells (primary).
        model_comparison: ML model comparison table (primary stream).
        best_ml_model_name: Name of the selected ML model (primary).
        best_ml_model: The fitted scikit-learn estimator (primary).
        correlation_matrix: Feature-target correlation matrix (primary).
        feature_importance: Feature importances (if tree-based model).
        figures_dir: Path where shared diagnostic figures were saved.
        production_unit: 'BBL' or 'MCF' (primary stream).
        training_features: List of feature names used by the ML model.
        predictions: List of per-well WellPrediction objects.
        summary: Summary DataFrame with EUR P10/P50/P90 for all wells/streams.
    """
    well_dca_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_ml_model_name: str = ""
    best_ml_model: Any = None
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    figures_dir: str = ""
    production_unit: str = "BBL"
    training_features: List[str] = field(default_factory=list)
    predictions: List[WellPrediction] = field(default_factory=list)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class _StreamState:
    """Trained model artifacts for a single production stream (internal)."""
    stream: str = ""
    rate_col: str = ""
    unit: str = "BBL"
    dca_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    dca_for_ml: pd.DataFrame = field(default_factory=pd.DataFrame)
    most_common_model: str = ""
    model_specific_targets: List[str] = field(default_factory=list)
    # Universal model (qi + eur)
    best_model_univ: Any = None
    best_name_univ: str = ""
    scaler_univ: Any = None
    log_applied_univ: bool = False
    log_cols_univ: List[str] = field(default_factory=list)
    used_features_univ: List[str] = field(default_factory=list)
    used_targets_univ: List[str] = field(default_factory=list)
    X_univ: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Model-specific model
    best_model_spec: Any = None
    scaler_spec: Any = None
    log_applied_spec: bool = False
    log_cols_spec: List[str] = field(default_factory=list)
    used_features_spec: List[str] = field(default_factory=list)
    used_targets_spec: List[str] = field(default_factory=list)
    X_spec: Optional[pd.DataFrame] = None
    # Reporting artifacts
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    corr_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    importance_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_model_name: str = ""


@dataclass
class _PipelineState:
    """Internal container for trained pipeline artifacts (not public API)."""
    cmap: Dict[str, str] = field(default_factory=dict)
    primary_stream: str = "oil"
    unit: str = "BBL"
    forecast_months: int = 360
    figures_dir: str = ""
    wells_filtered: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_formation: Optional[str] = None
    # Per-stream trained states
    stream_states: Dict[str, _StreamState] = field(default_factory=dict)
    # --- Backward-compat aliases (primary stream) ---
    dca_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    dca_for_ml: pd.DataFrame = field(default_factory=pd.DataFrame)
    most_common_model: str = ""
    model_specific_targets: List[str] = field(default_factory=list)
    best_model_univ: Any = None
    best_name_univ: str = ""
    scaler_univ: Any = None
    log_applied_univ: bool = False
    log_cols_univ: List[str] = field(default_factory=list)
    used_features_univ: List[str] = field(default_factory=list)
    used_targets_univ: List[str] = field(default_factory=list)
    X_univ: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_model_spec: Any = None
    scaler_spec: Any = None
    log_applied_spec: bool = False
    log_cols_spec: List[str] = field(default_factory=list)
    used_features_spec: List[str] = field(default_factory=list)
    used_targets_spec: List[str] = field(default_factory=list)
    X_spec: Optional[pd.DataFrame] = None
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    corr_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    importance_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_model_name: str = ""


@dataclass
class PipelineResults:
    """Container for all outputs from ``predict_new_well()``.

    Attributes:
        well_dca_results: DataFrame of best-fit DCA parameters per well (primary).
        model_comparison: DataFrame comparing ML model performance (primary).
        best_ml_model_name: Name of the selected best ML model (primary).
        best_ml_model: The fitted scikit-learn estimator (primary).
        correlation_matrix: DataFrame of feature-target correlations (primary).
        predicted_dca_params: Dict of DCA parameters predicted (primary stream).
        forecast_p10: DataFrame with month, rate, and cumulative for P10 (primary).
        forecast_p50: DataFrame for P50 (primary).
        forecast_p90: DataFrame for P90 (primary).
        eur_p10: float – EUR at P10 (primary).
        eur_p50: float – EUR at P50 (primary).
        eur_p90: float – EUR at P90 (primary).
        figures_dir: Path where figures were saved.
        feature_importance: DataFrame of feature importances (if available).
        production_unit: str – 'BBL' or 'MCF' depending on primary stream.
        stream_forecasts: Dict mapping stream name -> StreamForecast for all
            three fluid streams (oil, water, gas).  Each contains independent
            DCA+ML forecasts.
    """
    well_dca_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_ml_model_name: str = ""
    best_ml_model: Any = None
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    predicted_dca_params: Dict[str, float] = field(default_factory=dict)
    forecast_p10: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast_p50: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast_p90: pd.DataFrame = field(default_factory=pd.DataFrame)
    eur_p10: float = 0.0
    eur_p50: float = 0.0
    eur_p90: float = 0.0
    figures_dir: str = ""
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    production_unit: str = "BBL"
    stream_forecasts: Dict[str, StreamForecast] = field(default_factory=dict)


# ===================================================================
# DCA MODEL DEFINITIONS
# ===================================================================

def _arps_hyperbolic(t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
    """Arps hyperbolic decline: q(t) = qi / (1 + b*di*t)^(1/b).

    Parameters:
        t: Time in months (starting at 1).
        qi: Initial rate.
        di: Initial decline rate (1/month, nominal).
        b: Arps b-factor (0 < b <= 2).

    Returns:
        Array of rates at each time step.
    """
    b = np.clip(b, 1e-6, 2.0)
    di = np.clip(di, 1e-8, 5.0)
    qi = max(qi, 1e-3)
    return qi / (1.0 + b * di * t) ** (1.0 / b)


def _arps_exponential(t: np.ndarray, qi: float, di: float) -> np.ndarray:
    """Arps exponential decline: q(t) = qi * exp(-di*t).

    Parameters:
        t: Time in months.
        qi: Initial rate.
        di: Decline rate (1/month).

    Returns:
        Array of rates.
    """
    di = np.clip(di, 1e-8, 5.0)
    qi = max(qi, 1e-3)
    return qi * np.exp(-di * t)


def _modified_hyperbolic(t: np.ndarray, qi: float, di: float, b: float,
                         d_min: float) -> np.ndarray:
    """Modified (capped) hyperbolic decline.

    Switches from hyperbolic to exponential when the instantaneous decline
    rate drops to ``d_min``.

    Parameters:
        t: Time in months.
        qi: Initial rate.
        di: Initial decline (1/month, nominal).
        b: Arps b-factor.
        d_min: Minimum (terminal) decline rate (1/month).

    Returns:
        Array of rates.
    """
    b = np.clip(b, 1e-6, 2.0)
    di = np.clip(di, 1e-8, 5.0)
    d_min = np.clip(d_min, 1e-6, di)
    qi = max(qi, 1e-3)

    # Time at which we switch to exponential
    if b * di > 1e-10:
        t_switch = (1.0 / (b * di)) * ((di / d_min) - 1.0)
    else:
        t_switch = 1e12  # effectively never switch

    q = np.empty_like(t, dtype=float)
    hyp_mask = t <= t_switch
    exp_mask = ~hyp_mask

    q[hyp_mask] = qi / (1.0 + b * di * t[hyp_mask]) ** (1.0 / b)

    if np.any(exp_mask):
        q_switch = qi / (1.0 + b * di * t_switch) ** (1.0 / b)
        q[exp_mask] = q_switch * np.exp(-d_min * (t[exp_mask] - t_switch))

    return q


def _stretched_exponential(t: np.ndarray, qi: float, tau: float,
                           n: float) -> np.ndarray:
    """Stretched exponential decline (SEPD).

    q(t) = qi * exp(-(t/tau)^n)

    Parameters:
        t: Time in months.
        qi: Initial rate.
        tau: Characteristic time constant.
        n: Exponent (0 < n <= 1 typical for unconventional).

    Returns:
        Array of rates.
    """
    tau = max(tau, 1e-3)
    n = np.clip(n, 0.01, 2.0)
    qi = max(qi, 1e-3)
    return qi * np.exp(-((t / tau) ** n))


def _duong(t: np.ndarray, qi: float, a: float, m: float) -> np.ndarray:
    """Duong decline model for unconventional wells.

    q(t) = qi * t^(-m) * exp(a/(1-m) * (t^(1-m) - 1))

    Parameters:
        t: Time in months (must be >= 1).
        qi: Rate scaling factor.
        a: Decline constant.
        m: Decline exponent (typically 1 < m < 2).

    Returns:
        Array of rates.
    """
    a = np.clip(a, 0.01, 10.0)
    m = np.clip(m, 1.001, 3.0)
    qi = max(qi, 1e-3)
    t = np.maximum(t, 0.1)
    return qi * (t ** (-m)) * np.exp(a / (1 - m) * (t ** (1 - m) - 1))


# ===================================================================
# DCA FITTING
# ===================================================================

# Mapping of model names -> (function, parameter names, initial guesses, bounds)
_DCA_MODELS: Dict[str, dict] = {
    "arps_hyperbolic": {
        "func": _arps_hyperbolic,
        "param_names": ["qi", "di", "b"],
        "p0_func": lambda qi_est: [qi_est, 0.10, 1.0],
        "bounds": ([0, 1e-8, 0.01], [np.inf, 5.0, 2.0]),
    },
    "arps_exponential": {
        "func": _arps_exponential,
        "param_names": ["qi", "di"],
        "p0_func": lambda qi_est: [qi_est, 0.05],
        "bounds": ([0, 1e-8], [np.inf, 5.0]),
    },
    "modified_hyperbolic": {
        "func": _modified_hyperbolic,
        "param_names": ["qi", "di", "b", "d_min"],
        "p0_func": lambda qi_est: [qi_est, 0.10, 1.0, 0.005],
        "bounds": ([0, 1e-8, 0.01, 1e-6], [np.inf, 5.0, 2.0, 0.5]),
    },
    "stretched_exponential": {
        "func": _stretched_exponential,
        "param_names": ["qi", "tau", "n"],
        "p0_func": lambda qi_est: [qi_est, 20.0, 0.5],
        "bounds": ([0, 0.01, 0.01], [np.inf, 1000.0, 2.0]),
    },
    "duong": {
        "func": _duong,
        "param_names": ["qi", "a", "m"],
        "p0_func": lambda qi_est: [qi_est * 10, 1.5, 1.2],
        "bounds": ([0, 0.01, 1.001], [np.inf, 10.0, 3.0]),
    },
}


def _fit_single_model(
    t: np.ndarray,
    q: np.ndarray,
    model_name: str,
    qi_est: float,
) -> Optional[DCAResult]:
    """Attempt to fit one DCA model to a single well's production.

    Parameters:
        t: Month indices (1, 2, 3, …).
        q: Corresponding monthly production rates.
        model_name: Key into ``_DCA_MODELS``.
        qi_est: Rough estimate of initial rate (for initial guesses).

    Returns:
        A ``DCAResult`` if the fit converges, else ``None``.
    """
    spec = _DCA_MODELS[model_name]
    func = spec["func"]
    p0 = spec["p0_func"](qi_est)
    bounds = spec["bounds"]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                func, t, q, p0=p0, bounds=bounds, maxfev=5000,
                method="trf",
            )
        q_pred = func(t, *popt)
        ss_res = np.sum((q - q_pred) ** 2)
        ss_tot = np.sum((q - np.mean(q)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = np.sqrt(np.mean((q - q_pred) ** 2))

        params = dict(zip(spec["param_names"], popt))
        return DCAResult(
            model_name=model_name,
            params=params,
            r_squared=r2,
            rmse=rmse,
        )
    except (RuntimeError, ValueError, OverflowError, MemoryError):
        return None
    except Exception:
        return None


def _fit_all_models(
    t: np.ndarray,
    q: np.ndarray,
    api_uwi: str,
) -> List[DCAResult]:
    """Fit all DCA models to one well and return list of successful fits.

    Parameters:
        t: Month indices.
        q: Monthly production rates.
        api_uwi: Well identifier (for labelling).

    Returns:
        List of ``DCAResult`` objects (may be empty).
    """
    qi_est = float(np.max(q[:min(6, len(q))]))  # peak in first 6 months
    results = []
    for name in _DCA_MODELS:
        res = _fit_single_model(t, q, name, qi_est)
        if res is not None:
            res.api_uwi = api_uwi
            results.append(res)
    return results


def _select_best_model(results: List[DCAResult]) -> Optional[DCAResult]:
    """Pick the best fit from a list of DCA results by R².

    If R² values are close (within 0.02) the model with fewer parameters
    is preferred (parsimony).

    Parameters:
        results: List of ``DCAResult`` from ``_fit_all_models()``.

    Returns:
        The best ``DCAResult``, or ``None`` if the list is empty.
    """
    if not results:
        return None

    results_sorted = sorted(results, key=lambda r: r.r_squared, reverse=True)
    best = results_sorted[0]

    # Prefer simpler model if R² is within tolerance
    for r in results_sorted[1:]:
        if best.r_squared - r.r_squared < 0.02:
            if len(r.params) < len(best.params):
                best = r
                break
    return best


def _generate_forecast(
    dca_result: DCAResult,
    forecast_months: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate a rate forecast from fitted DCA parameters.

    Parameters:
        dca_result: A fitted ``DCAResult``.
        forecast_months: Total number of months to forecast.

    Returns:
        Tuple of (month_array, rate_array, eur).
    """
    spec = _DCA_MODELS[dca_result.model_name]
    func = spec["func"]
    t_forecast = np.arange(1, forecast_months + 1, dtype=float)
    q_forecast = func(t_forecast, *[dca_result.params[p] for p in spec["param_names"]])
    q_forecast = np.maximum(q_forecast, 0.0)
    eur = float(np.sum(q_forecast))
    return t_forecast, q_forecast, eur


# ===================================================================
# DATA LOADING & CLEANING
# ===================================================================

def _load_and_clean(
    well_data_path: str,
    prod_data_path: str,
    col_map: Dict[str, str],
    min_months: int = 12,
    max_prod_months_per_well: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load well and production CSVs, clean, and return DataFrames.

    Parameters:
        well_data_path: Path to wells CSV.
        prod_data_path: Path to production CSV.
        col_map: Mapping of canonical column names to actual column names
                 in the input data.
        min_months: Minimum number of producing months for a well to be
                    included in the analysis.
        max_prod_months_per_well: If set, truncate each well's production
                    history to at most this many months.  Useful for
                    speeding up DCA fitting during testing.

    Returns:
        Tuple of (wells_df, production_df) after cleaning.
    """
    _print_status("Loading well data...")
    wells = pd.read_csv(well_data_path, low_memory=False).copy()
    _print_status(f"  Loaded {len(wells)} wells with {wells.shape[1]} columns.")

    _print_status("Loading production data...")
    prod = pd.read_csv(prod_data_path, low_memory=False).copy()
    _print_status(f"  Loaded {len(prod)} production records.")

    # Standardize join key
    api_col_w = col_map.get("api_uwi_well", "API_UWI")
    api_col_p = col_map.get("api_uwi_prod", "API_UWI")

    # Strip formatting from API to ensure match
    wells["_api_clean"] = wells[api_col_w].astype(str).str.replace("-", "", regex=False).str.strip()
    prod["_api_clean"] = prod[api_col_p].astype(str).str.replace("-", "", regex=False).str.strip()

    # Filter to wells that have production history
    producing_apis = set(prod["_api_clean"].unique())
    wells = wells[wells["_api_clean"].isin(producing_apis)].copy()
    _print_status(f"  {len(wells)} wells have matching production records.")

    # Parse production month
    prod_month_col = col_map.get("producing_month", "ProducingMonth")
    prod[prod_month_col] = pd.to_datetime(prod[prod_month_col], errors="coerce")
    prod = prod.dropna(subset=[prod_month_col])

    # Filter wells with enough history
    total_prod_months_col = col_map.get("total_prod_months", "TotalProdMonths")
    # Use count() not max() so wells with sparse data (e.g. month 1,2,50)
    # are not over-credited.  A well needs *count* >= min_months of real records.
    month_counts = prod.groupby("_api_clean")[total_prod_months_col].count()
    valid_apis = set(month_counts[month_counts >= min_months].index)
    wells = wells[wells["_api_clean"].isin(valid_apis)].copy()
    prod = prod[prod["_api_clean"].isin(valid_apis)].copy()
    _print_status(f"  {len(wells)} wells with >= {min_months} months of production.")

    # Truncate production history per well if requested
    if max_prod_months_per_well is not None and max_prod_months_per_well > 0:
        prod = prod[
            prod[total_prod_months_col] <= max_prod_months_per_well
        ].copy()
        _print_status(
            f"  Truncated production to first {max_prod_months_per_well} "
            f"months per well ({len(prod)} records remain)."
        )

    # Filter to active producers / well types that make sense
    status_col = col_map.get("well_status", "ENVWellStatus")
    if status_col in wells.columns:
        valid_statuses = ["PRODUCING", "INACTIVE PRODUCER", "COMPLETED"]
        wells = wells[wells[status_col].isin(valid_statuses)].copy()
        prod = prod[prod["_api_clean"].isin(set(wells["_api_clean"]))].copy()
        _print_status(f"  {len(wells)} wells after status filter ({', '.join(valid_statuses)}).")

    # Drop duplicate well entries (keep first completion)
    wells = wells.drop_duplicates(subset=["_api_clean"], keep="first").copy()

    _print_status(f"  Final dataset: {len(wells)} wells, {len(prod)} production records.")
    return wells, prod


def _detect_primary_stream(
    wells: pd.DataFrame,
    col_map: Dict[str, str],
    override: Optional[str] = None,
) -> str:
    """Determine whether the primary production stream is oil or gas.

    Uses ``ENVProdWellType`` or ``WHLiquids_PCT`` to classify.  The dominant
    type across the dataset is chosen unless ``override`` is provided.

    Parameters:
        wells: Well metadata DataFrame.
        col_map: Column name mapping.
        override: If 'oil' or 'gas', force that stream.

    Returns:
        'oil' or 'gas'.
    """
    if override and override.lower() in ("oil", "gas"):
        return override.lower()

    well_type_col = col_map.get("prod_well_type", "ENVProdWellType")
    if well_type_col in wells.columns:
        type_counts = wells[well_type_col].astype(str).str.upper().value_counts()
        oil_count = sum(v for k, v in type_counts.items() if "OIL" in k)
        gas_count = sum(v for k, v in type_counts.items()
                        if "GAS" in k and "OIL" not in k)
        if gas_count > oil_count:
            return "gas"
    return "oil"


# Stream configuration: (col_map_key, unit)
_STREAM_CONFIG: Dict[str, Tuple[str, str]] = {
    "oil":   ("oil_rate",   "BBL"),
    "water": ("water_rate", "BBL"),
    "gas":   ("gas_rate",   "MCF"),
}


# ===================================================================
# CORRELATION ANALYSIS
# ===================================================================

_DEFAULT_WELL_FEATURES = [
    "TVD_FT", "LateralLength_FT", "FracStages", "Proppant_LBS",
    "ProppantIntensity_LBSPerFT", "TotalFluidPumped_BBL",
    "FluidIntensity_BBLPerFT", "AverageStageSpacing_FT",
    "MD_FT", "PerfInterval_FT", "ProppantLoading_LBSPerGAL",
    "WaterIntensity_GALPerFT", "TotalWaterPumped_GAL",
    "Bottom_Hole_Temp_DEGF",
]

_DEFAULT_DCA_TARGETS = ["qi", "di", "b", "eur"]


def _build_correlation_matrix(
    well_features: pd.DataFrame,
    dca_params: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
) -> pd.DataFrame:
    """Compute and return a correlation matrix between well features and DCA params.

    Parameters:
        well_features: DataFrame indexed by API with well completion attributes.
        dca_params: DataFrame indexed by API with best-fit DCA parameters.
        feature_cols: Column names in ``well_features`` to include.
        target_cols: Column names in ``dca_params`` to include.

    Returns:
        DataFrame of Pearson correlation coefficients.
    """
    avail_feat = [c for c in feature_cols if c in well_features.columns]
    avail_targ = [c for c in target_cols if c in dca_params.columns]
    merged = well_features[avail_feat].join(dca_params[avail_targ], how="inner")
    # Use pairwise-complete observations (pandas default) instead of
    # dropna(), which would destroy all rows when any column is all-NaN.
    return merged.corr()


# ===================================================================
# ML TRAINING
# ===================================================================

def _prepare_ml_data(
    wells: pd.DataFrame,
    dca_df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    col_map: Dict[str, str],
    min_feature_coverage: float = 0.3,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Merge well features and DCA targets, engineer features, impute, return X/y.

    This function:
        1. Extracts raw well attributes and one-hot-encodes categorical
           features (e.g. formation interval).
        2. Engineers derived features (ratios, intensities, vintage year).
        3. Drops features with < ``min_feature_coverage`` non-null fraction.
        4. Fills remaining NaN values with column medians.
        5. Removes constant or all-NaN columns.

    Parameters:
        wells: Well metadata DataFrame.
        dca_df: DCA results DataFrame indexed by _api_clean.
        feature_cols: Well attribute columns to use as features.
        target_cols: DCA parameter columns to predict.
        col_map: Column mapping dict.
        min_feature_coverage: Minimum fraction of non-null values a numeric
            feature must have to be retained (default 0.3 = 30 %).

    Returns:
        (X, y, used_feature_cols, used_target_cols)
    """
    # Encode categorical features that might be useful
    interval_col = col_map.get("interval", "ENVInterval")
    extra_cats = []
    if interval_col in wells.columns:
        dummies = pd.get_dummies(wells.set_index("_api_clean")[interval_col],
                                 prefix="interval", dtype=float)
        extra_cats.append(dummies)

    well_feat = wells.set_index("_api_clean")[
        [c for c in feature_cols if c in wells.columns]
    ].copy()

    # ---- Engineered features ----
    # Proppant per frac stage
    if "Proppant_LBS" in well_feat.columns and "FracStages" in well_feat.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            well_feat["ProppantPerStage_LBS"] = (
                well_feat["Proppant_LBS"] / well_feat["FracStages"].replace(0, np.nan)
            )

    # Fluid per frac stage
    if "TotalFluidPumped_BBL" in well_feat.columns and "FracStages" in well_feat.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            well_feat["FluidPerStage_BBL"] = (
                well_feat["TotalFluidPumped_BBL"] / well_feat["FracStages"].replace(0, np.nan)
            )

    # Clusters per stage proxy (lateral length / frac stages)
    if "LateralLength_FT" in well_feat.columns and "FracStages" in well_feat.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            well_feat["StageSpacing_calc_FT"] = (
                well_feat["LateralLength_FT"] / well_feat["FracStages"].replace(0, np.nan)
            )

    # Completion vintage year (wells completed more recently tend to perform better)
    first_prod_col = col_map.get("first_prod_date", "FirstProdDate")
    if first_prod_col in wells.columns:
        wells_idx = wells.set_index("_api_clean")
        fpd = pd.to_datetime(wells_idx[first_prod_col], errors="coerce")
        well_feat["VintageYear"] = fpd.dt.year

    # Log-transformed features for highly skewed attributes
    for col_name in ["Proppant_LBS", "TotalFluidPumped_BBL"]:
        if col_name in well_feat.columns:
            pos_mask = well_feat[col_name] > 0
            if pos_mask.sum() > 0:
                well_feat[f"log_{col_name}"] = np.where(
                    pos_mask, np.log1p(well_feat[col_name]), np.nan
                )

    # Add categorical dummies
    for d in extra_cats:
        well_feat = well_feat.join(d, how="left")

    merged = well_feat.join(dca_df[target_cols], how="inner")

    # Drop rows where targets are missing (required for training)
    merged = merged.dropna(subset=target_cols)

    actual_features = [c for c in merged.columns if c not in target_cols]

    # Drop feature columns with insufficient data coverage
    low_coverage = [
        c for c in actual_features
        if merged[c].notna().mean() < min_feature_coverage
    ]
    if low_coverage:
        merged = merged.drop(columns=low_coverage)
        actual_features = [c for c in actual_features if c not in low_coverage]

    # Fill remaining NaN features with column medians
    for col in actual_features:
        if merged[col].isna().any():
            med = merged[col].median()
            merged[col] = merged[col].fillna(med if pd.notna(med) else 0.0)

    # Final safety: drop any column that is still all-NaN or constant
    drop_cols = [
        c for c in actual_features
        if merged[c].isna().all() or merged[c].nunique() < 2
    ]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
        actual_features = [c for c in actual_features if c not in drop_cols]

    X = merged[actual_features]
    y = merged[target_cols]

    return X, y, actual_features, target_cols


def _train_and_compare_models(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    log_targets: bool = True,
    cv_folds: int = 5,
) -> Tuple[pd.DataFrame, Any, str, StandardScaler, bool, List[str]]:
    """Train multiple ML models and compare using cross-validation.

    Uses k-fold cross-validation for robust R² estimates and a held-out
    test set for final evaluation.  Optionally log-transforms positive
    targets before training.

    Models tested:
        - Linear Regression
        - Polynomial Regression (degree 2)
        - Ridge Regression
        - Lasso Regression
        - Random Forest
        - Gradient Boosting

    Parameters:
        X: Feature matrix.
        y: Target matrix (multi-output: qi, di, b, eur, etc.).
        test_size: Fraction of data to withhold for testing.
        random_state: Seed for reproducibility.
        log_targets: If True, log1p-transform all-positive targets.
        cv_folds: Number of cross-validation folds (default 5).

    Returns:
        (comparison_df, best_model, best_model_name, scaler,
         log_applied, log_cols)
    """
    # Identify columns suitable for log-transform (all positive values)
    log_cols: List[str] = []
    if log_targets:
        for col in y.columns:
            if (y[col] > 0).all():
                log_cols.append(col)

    # Apply log1p transform to suitable target columns
    y_train_raw = y.copy()
    if log_cols:
        for col in log_cols:
            y_train_raw[col] = np.log1p(y_train_raw[col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_train_raw, test_size=test_size, random_state=random_state
    )

    # Also keep raw (un-transformed) test targets for evaluation
    _, _, _, y_test_raw = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Also scale full dataset for cross-validation
    X_all_s = scaler.transform(X)

    models = {
        "Linear Regression": MultiOutputRegressor(LinearRegression()),
        "Polynomial (deg 2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False,
                                         interaction_only=True)),
            ("reg", MultiOutputRegressor(Ridge(alpha=1.0))),
        ]),
        "Ridge Regression": MultiOutputRegressor(Ridge(alpha=1.0)),
        "Lasso Regression": MultiOutputRegressor(Lasso(alpha=0.01, max_iter=5000)),
        "Random Forest": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=300, max_depth=12,
                                  min_samples_leaf=5, min_samples_split=10,
                                  random_state=random_state, n_jobs=-1)
        ),
        "Gradient Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                       learning_rate=0.05, subsample=0.8,
                                       min_samples_leaf=5,
                                       random_state=random_state)
        ),
    }

    records = []
    fitted_models = {}

    # Set up KFold for cross-validation
    kf = KFold(n_splits=min(cv_folds, len(X_train)), shuffle=True,
               random_state=random_state)

    for name, model in models.items():
        _print_status(f"  Training {name}...")
        try:
            t0 = time.time()

            # Cross-validation on training data (in log-space)
            cv_scores = cross_val_score(
                model, X_train_s, y_train,
                cv=kf, scoring="r2",
                error_score=np.nan,
            )
            cv_r2 = float(np.nanmean(cv_scores))

            # Fit on full training set for final model
            model.fit(X_train_s, y_train)
            train_time = time.time() - t0

            y_pred_log = model.predict(X_test_s)

            # Inverse-transform for evaluation in original scale
            y_pred_orig = pd.DataFrame(y_pred_log, columns=y.columns,
                                       index=y_test.index)
            if log_cols:
                for col in log_cols:
                    idx = list(y.columns).index(col)
                    y_pred_orig[col] = np.expm1(y_pred_log[:, idx])

            r2 = r2_score(y_test_raw, y_pred_orig, multioutput="uniform_average")
            mae = mean_absolute_error(y_test_raw, y_pred_orig,
                                      multioutput="uniform_average")
            rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_orig,
                                              multioutput="uniform_average"))

            # Per-target R² for detail
            per_target_r2 = {}
            for i, col in enumerate(y.columns):
                per_target_r2[f"R2_{col}"] = r2_score(
                    y_test_raw.iloc[:, i], y_pred_orig.iloc[:, i]
                )

            records.append({
                "Model": name,
                "R2": r2,
                "CV_R2": cv_r2,
                "MAE": mae,
                "RMSE": rmse,
                "Train_Time_s": train_time,
                **per_target_r2,
            })
            fitted_models[name] = model
        except Exception as e:
            _print_status(f"    {name} failed: {e}")

    comparison = pd.DataFrame(records).sort_values("CV_R2", ascending=False)
    best_name = comparison.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    _print_status(
        f"  Best ML model: {best_name} "
        f"(CV R² = {comparison.iloc[0]['CV_R2']:.4f}, "
        f"Test R² = {comparison.iloc[0]['R2']:.4f})"
    )
    if log_cols:
        _print_status(f"  Log-transformed targets: {log_cols}")

    return comparison, best_model, best_name, scaler, bool(log_cols), log_cols


# ===================================================================
# P10 / P50 / P90 GENERATION
# ===================================================================

def _generate_probabilistic_forecasts(
    predicted_params: Dict[str, float],
    model_name: str,
    forecast_months: int,
    residual_std: Dict[str, float],
    eur_bounds: Optional[Tuple[float, float]] = None,
    analog_wells: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float, float]:
    """Create P10, P50, P90 forecasts via hybrid Monte Carlo + analog wells.

    Strategy:
        - If ``analog_wells`` is provided, the Monte Carlo samples are drawn
          from their actual DCA parameter distributions (parametric bootstrap
          of similar wells).  This grounds the uncertainty in real data.
        - Otherwise, falls back to perturbing the point estimate using IQR.

    EUR samples outside ``eur_bounds`` (if given) are discarded to suppress
    physically implausible realizations.

    Parameters:
        predicted_params: DCA parameters predicted by ML model.
        model_name: DCA model name (e.g. 'duong').
        forecast_months: Number of months to forecast.
        residual_std: Uncertainty scale per DCA parameter (IQR or std).
        eur_bounds: Optional (low, high) credible EUR range from training.
        analog_wells: Optional DataFrame of DCA params from similar wells;
            columns must include the DCA parameter names for ``model_name``.

    Returns:
        (df_p10, df_p50, df_p90, eur_p10, eur_p50, eur_p90)
    """
    spec = _DCA_MODELS[model_name]
    func = spec["func"]
    param_names = spec["param_names"]
    t = np.arange(1, forecast_months + 1, dtype=float)

    # P50 = point estimate (use model median as fallback, not 0.0)
    p50_vals = []
    for p in param_names:
        val = predicted_params.get(p)
        if val is None or np.isnan(val):
            _print_status(f"  WARNING: DCA param '{p}' missing, using safe default.")
            val = {"qi": 100.0, "di": 0.01, "b": 1.0, "d_min": 0.001,
                   "tau": 1.0, "n": 1.0, "a": 1.0, "m": 1.1}.get(p, 1.0)
        p50_vals.append(val)
    q_p50 = np.maximum(func(t, *p50_vals), 0.0)
    eur_p50_deterministic = float(np.sum(q_p50))

    # Monte Carlo approach
    n_samples = 1000
    n_failed = 0
    rng = np.random.default_rng(42)
    q_samples = np.zeros((n_samples, forecast_months))

    use_analogs = (
        analog_wells is not None
        and len(analog_wells) >= 20
        and all(p in analog_wells.columns for p in param_names)
    )

    for i in range(n_samples):
        if use_analogs:
            # Resample from analog wells (parametric bootstrap)
            row = analog_wells.sample(n=1, random_state=rng.integers(1e9))
            sample_params = []
            for p in param_names:
                val = float(row[p].values[0])
                # Add small jitter (5% of IQR) to avoid identical duplicates
                iqr_val = residual_std.get(p, abs(val) * 0.1)
                val += rng.normal(0, max(iqr_val * 0.05, 1e-6))
                sample_params.append(val)
        else:
            # Perturbation-based fallback
            sample_params = []
            for p in param_names:
                mean_val = predicted_params.get(p, 0.0)
                iqr_val = residual_std.get(p, abs(mean_val) * 0.1)
                # IQR / 1.35 ≈ σ for normally distributed data
                std_val = iqr_val / 1.35
                sampled = rng.normal(mean_val, max(std_val, 1e-6))
                sample_params.append(sampled)

        # Enforce physical bounds on all samples
        bounded_params = []
        for p, val in zip(param_names, sample_params):
            if p == "qi":
                val = max(val, 1.0)
            elif p == "di":
                val = np.clip(val, 1e-6, 5.0)
            elif p == "b":
                val = np.clip(val, 0.01, 2.0)
            elif p == "d_min":
                val = np.clip(val, 1e-6, 0.5)
            elif p == "tau":
                val = max(val, 0.1)
            elif p == "n":
                val = np.clip(val, 0.01, 2.0)
            elif p == "a":
                val = np.clip(val, 0.01, 10.0)
            elif p == "m":
                val = np.clip(val, 1.001, 3.0)
            bounded_params.append(val)

        try:
            q_s = func(t, *bounded_params)
            q_s = np.maximum(q_s, 0.0)
            if np.any(np.isnan(q_s)) or np.any(np.isinf(q_s)):
                q_samples[i, :] = q_p50
            else:
                q_samples[i, :] = q_s
        except Exception:
            q_samples[i, :] = q_p50
            n_failed += 1

    if n_failed > n_samples * 0.5:
        _print_status(
            f"  WARNING: {n_failed}/{n_samples} MC samples failed — "
            f"P10/P90 uncertainty may be unreliable."
        )

    q_p10_arr = np.percentile(q_samples, 90, axis=0)  # P10 = high case
    q_p90_arr = np.percentile(q_samples, 10, axis=0)  # P90 = low case

    eur_samples = np.sum(q_samples, axis=1)

    # Clip EUR samples to training data bounds if available
    if eur_bounds is not None:
        eur_lo, eur_hi = eur_bounds
        valid = (eur_samples >= eur_lo) & (eur_samples <= eur_hi)
        if valid.sum() >= 100:
            q_samples_valid = q_samples[valid]
            q_p10_arr = np.percentile(q_samples_valid, 90, axis=0)
            q_p90_arr = np.percentile(q_samples_valid, 10, axis=0)
            eur_samples = np.sum(q_samples_valid, axis=1)

    eur_p10 = float(np.percentile(eur_samples, 90))
    eur_p90 = float(np.percentile(eur_samples, 10))
    # Use MC median for P50 to be consistent with MC-derived P10/P90.
    # Fall back to deterministic if MC distribution is degenerate.
    eur_p50_mc = float(np.percentile(eur_samples, 50))
    eur_p50 = eur_p50_mc if eur_p50_mc > 0 else eur_p50_deterministic
    q_p50 = np.percentile(q_samples, 50, axis=0)  # MC median rate profile

    def _make_df(q_arr, label):
        return pd.DataFrame({
            "Month": np.arange(1, forecast_months + 1),
            f"Rate_{label}": q_arr,
            f"Cumulative_{label}": np.cumsum(q_arr),
        })

    return (
        _make_df(q_p10_arr, "P10"),
        _make_df(q_p50, "P50"),
        _make_df(q_p90_arr, "P90"),
        eur_p10, eur_p50, eur_p90,
    )


# ===================================================================
# PLOTTING
# ===================================================================

def _save_correlation_heatmap(
    corr: pd.DataFrame,
    figures_dir: str,
    feature_cols: List[str],
    target_cols: List[str],
    stream_name: str = "",
) -> None:
    """Save a correlation heatmap of features vs DCA targets.

    Parameters:
        corr: Full correlation matrix.
        figures_dir: Directory to save figure.
        feature_cols: Feature column names.
        target_cols: Target column names.
        stream_name: Stream label for the title/filename (e.g. 'oil').
    """
    # Extract the cross-correlation block
    avail_feat = [c for c in feature_cols if c in corr.columns]
    avail_targ = [c for c in target_cols if c in corr.columns]
    sub = corr.loc[avail_feat, avail_targ]

    # Drop rows/columns that are entirely NaN (no valid correlations)
    sub = sub.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if sub.empty:
        return  # nothing to plot

    suffix = f" — {stream_name.upper()}" if stream_name else ""
    fig, ax = plt.subplots(figsize=(10, max(6, len(sub) * 0.45)))
    # Use empty string for NaN cells instead of showing "nan"
    sns.heatmap(sub, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, vmin=-1, vmax=1, linewidths=0.5, mask=sub.isna())
    ax.set_title(f"Correlation: Well Features vs DCA Parameters{suffix}", fontsize=13)
    plt.tight_layout()
    fname = f"correlation_heatmap_{stream_name}.png" if stream_name else "correlation_heatmap.png"
    fig.savefig(os.path.join(figures_dir, fname), dpi=150)
    plt.close(fig)


def _save_model_comparison_chart(
    comparison: pd.DataFrame,
    figures_dir: str,
    stream_name: str = "",
) -> None:
    """Grouped bar chart comparing ML model Test R² and CV R² scores.

    Parameters:
        comparison: DataFrame from ``_train_and_compare_models``.
        figures_dir: Directory to save figure.
        stream_name: Stream label for title/filename.
    """
    suffix = f" — {stream_name.upper()}" if stream_name else ""
    fig, ax = plt.subplots(figsize=(10, 6))
    models = comparison["Model"].values
    y_pos = np.arange(len(models))
    has_cv = "CV_R2" in comparison.columns

    # Cap x-axis at -2 so extreme outliers (e.g. Polynomial) don't compress
    # all other bars into an unreadable sliver.
    X_MIN_FLOOR = -2.0

    if has_cv:
        bar_h = 0.35
        cv_colors = sns.color_palette("YlOrBr", len(models))
        bars_cv = ax.barh(
            y_pos + bar_h / 2, comparison["CV_R2"], bar_h,
            label="CV R² (5-fold) — used for selection",
            color=cv_colors, edgecolor="black", linewidth=0.4,
        )
        test_colors = sns.color_palette("Blues_d", len(models))
        bars_test = ax.barh(
            y_pos - bar_h / 2, comparison["R2"], bar_h,
            label="Test R² (single 80/20 split)",
            color=test_colors, alpha=0.65,
        )
        for bar, val in zip(bars_cv, comparison["CV_R2"]):
            ax.text(
                max(bar.get_width(), 0) + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, fontweight="bold",
            )
        for bar, val in zip(bars_test, comparison["R2"]):
            ax.text(
                max(bar.get_width(), 0) + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="#444",
            )
        raw_min = min(0, comparison["R2"].min(), comparison["CV_R2"].min())
        x_min = max(raw_min - 0.05, X_MIN_FLOOR)
    else:
        bar_h = 0.5
        colors = sns.color_palette("viridis", len(models))
        bars_test = ax.barh(y_pos, comparison["R2"], bar_h, color=colors)
        for bar, val in zip(bars_test, comparison["R2"]):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9,
            )
        raw_min = min(0, comparison["R2"].min())
        x_min = max(raw_min - 0.05, X_MIN_FLOOR)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel("R² (higher is better)")
    ax.set_title(f"ML Model Comparison – Multi-output R²{suffix}")
    ax.set_xlim(x_min, 1.0)
    if has_cv:
        ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fname = f"ml_model_comparison_{stream_name}.png" if stream_name else "ml_model_comparison.png"
    fig.savefig(os.path.join(figures_dir, fname), dpi=150)
    plt.close(fig)


def _save_forecast_plot(
    df_p10: pd.DataFrame,
    df_p50: pd.DataFrame,
    df_p90: pd.DataFrame,
    unit: str,
    figures_dir: str,
    filename: str = "forecast_p10_p50_p90.png",
    stream_name: str = "",
) -> None:
    """Plot and save the P10/P50/P90 production forecast.

    Parameters:
        df_p10: P10 forecast DataFrame.
        df_p50: P50 forecast DataFrame.
        df_p90: P90 forecast DataFrame.
        unit: Production unit (BBL or MCF).
        figures_dir: Directory to save figure.
        filename: Output filename.
        stream_name: Stream label for title.
    """
    suffix = f" — {stream_name.upper()}" if stream_name else ""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Rate plot
    ax1.plot(df_p50["Month"], df_p50["Rate_P50"], "b-", lw=2, label="P50")
    ax1.plot(df_p10["Month"], df_p10["Rate_P10"], "g--", lw=1.2, label="P10 (high)")
    ax1.plot(df_p90["Month"], df_p90["Rate_P90"], "r--", lw=1.2, label="P90 (low)")
    ax1.fill_between(df_p50["Month"], df_p90["Rate_P90"], df_p10["Rate_P10"],
                      alpha=0.15, color="blue")
    ax1.set_xlabel("Months on Production")
    ax1.set_ylabel(f"Monthly Rate ({unit})")
    ax1.set_title(f"Forecasted Production Rate{suffix}")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=0.1)  # floor for log scale
    ax1.grid(True, alpha=0.3)

    # Cumulative plot
    ax2.plot(df_p50["Month"], df_p50["Cumulative_P50"], "b-", lw=2, label="P50")
    ax2.plot(df_p10["Month"], df_p10["Cumulative_P10"], "g--", lw=1.2, label="P10")
    ax2.plot(df_p90["Month"], df_p90["Cumulative_P90"], "r--", lw=1.2, label="P90")
    ax2.fill_between(df_p50["Month"], df_p90["Cumulative_P90"],
                      df_p10["Cumulative_P10"], alpha=0.15, color="blue")
    ax2.set_xlabel("Months on Production")
    ax2.set_ylabel(f"Cumulative Production ({unit})")
    ax2.set_title(f"Forecasted Cumulative Production{suffix}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, filename), dpi=150)
    plt.close(fig)


def _save_multistream_forecast_plot(
    stream_forecasts: Dict[str, StreamForecast],
    figures_dir: str,
    filename: str = "forecast_all_streams.png",
) -> None:
    """Plot P50 forecasts for all three fluid streams on one figure.

    Parameters:
        stream_forecasts: Dict mapping stream name to StreamForecast.
        figures_dir: Directory to save figure.
        filename: Output filename.
    """
    streams_present = [s for s in ["oil", "water", "gas"] if s in stream_forecasts]
    if not streams_present:
        return

    n_streams = len(streams_present)
    fig, axes = plt.subplots(2, n_streams, figsize=(7 * n_streams, 10))
    if n_streams == 1:
        axes = axes.reshape(2, 1)

    colors = {"oil": "green", "water": "blue", "gas": "red"}

    for col_idx, stream_name in enumerate(streams_present):
        sf = stream_forecasts[stream_name]
        ax_rate = axes[0, col_idx]
        ax_cum = axes[1, col_idx]
        clr = colors.get(stream_name, "black")

        # Rate plot
        ax_rate.plot(sf.forecast_p50["Month"], sf.forecast_p50["Rate_P50"],
                     color=clr, lw=2, label="P50")
        ax_rate.plot(sf.forecast_p10["Month"], sf.forecast_p10["Rate_P10"],
                     color=clr, ls="--", lw=1, alpha=0.6, label="P10")
        ax_rate.plot(sf.forecast_p90["Month"], sf.forecast_p90["Rate_P90"],
                     color=clr, ls=":", lw=1, alpha=0.6, label="P90")
        ax_rate.fill_between(
            sf.forecast_p50["Month"],
            sf.forecast_p90["Rate_P90"],
            sf.forecast_p10["Rate_P10"],
            alpha=0.12, color=clr,
        )
        ax_rate.set_title(f"{stream_name.upper()} Rate ({sf.unit}/mo)")
        ax_rate.set_xlabel("Month")
        ax_rate.set_ylabel(f"Rate ({sf.unit})")
        ax_rate.set_yscale("log")
        ax_rate.set_ylim(bottom=0.1)  # floor for log scale
        ax_rate.legend(fontsize=8)
        ax_rate.grid(True, alpha=0.3)

        # Cumulative plot
        ax_cum.plot(sf.forecast_p50["Month"], sf.forecast_p50["Cumulative_P50"],
                    color=clr, lw=2, label=f"P50 EUR={sf.eur_p50:,.0f}")
        ax_cum.plot(sf.forecast_p10["Month"], sf.forecast_p10["Cumulative_P10"],
                    color=clr, ls="--", lw=1, alpha=0.6,
                    label=f"P10 EUR={sf.eur_p10:,.0f}")
        ax_cum.plot(sf.forecast_p90["Month"], sf.forecast_p90["Cumulative_P90"],
                    color=clr, ls=":", lw=1, alpha=0.6,
                    label=f"P90 EUR={sf.eur_p90:,.0f}")
        ax_cum.fill_between(
            sf.forecast_p50["Month"],
            sf.forecast_p90["Cumulative_P90"],
            sf.forecast_p10["Cumulative_P10"],
            alpha=0.12, color=clr,
        )
        ax_cum.set_title(f"{stream_name.upper()} Cumulative ({sf.unit})")
        ax_cum.set_xlabel("Month")
        ax_cum.set_ylabel(f"Cumulative ({sf.unit})")
        ax_cum.legend(fontsize=8)
        ax_cum.grid(True, alpha=0.3)
        ax_cum.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )

    plt.suptitle("Three-Stream Production Forecast", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_dca_model_selection_chart(
    dca_df: pd.DataFrame,
    figures_dir: str,
    stream_name: str = "",
) -> None:
    """Pie/bar chart showing which DCA model was selected for each well.

    Parameters:
        dca_df: DataFrame with a 'best_model' column.
        figures_dir: Directory to save figure.
        stream_name: Stream label for title/filename.
    """
    suffix = f" — {stream_name.upper()}" if stream_name else ""
    counts = dca_df["best_model"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Set2", len(counts))
    x_pos = range(len(counts))
    ax.bar(x_pos, counts.values, color=colors)
    ax.set_ylabel("Number of Wells")
    ax.set_title(f"Best-Fit DCA Model Distribution{suffix}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    for i, (name, val) in enumerate(counts.items()):
        ax.text(i, val + 0.5, str(val), ha="center", fontsize=9)
    plt.tight_layout()
    fname = f"dca_model_distribution_{stream_name}.png" if stream_name else "dca_model_distribution.png"
    fig.savefig(os.path.join(figures_dir, fname), dpi=150)
    plt.close(fig)


def _save_feature_importance_plot(
    importance_df: pd.DataFrame,
    figures_dir: str,
    stream_name: str = "",
) -> None:
    """Horizontal bar chart of feature importances.

    Parameters:
        importance_df: DataFrame with 'feature' and 'importance' columns.
        figures_dir: Directory to save figure.
        stream_name: Stream label for title/filename.
    """
    suffix = f" — {stream_name.upper()}" if stream_name else ""
    top = importance_df.head(20)
    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.35)))
    ax.barh(top["feature"], top["importance"], color=sns.color_palette("viridis", len(top)))
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 20 Feature Importances{suffix}")
    ax.invert_yaxis()
    plt.tight_layout()
    fname = f"feature_importance_{stream_name}.png" if stream_name else "feature_importance.png"
    fig.savefig(os.path.join(figures_dir, fname), dpi=150)
    plt.close(fig)


def _save_eur_distribution_plot(
    dca_df: pd.DataFrame,
    unit: str,
    figures_dir: str,
    stream_name: str = "",
) -> None:
    """Histogram of EUR distribution across all fitted wells.

    Parameters:
        dca_df: DataFrame with 'eur' column.
        unit: Production unit label.
        figures_dir: Directory to save figure.
        stream_name: Stream label for title/filename.
    """
    suffix = f" — {stream_name.upper()}" if stream_name else ""
    fig, ax = plt.subplots(figsize=(10, 5))
    eur_vals = dca_df["eur"].dropna()

    p99 = eur_vals.quantile(0.99)
    clipped = eur_vals[eur_vals <= p99]
    n_clipped = len(eur_vals) - len(clipped)

    ax.hist(clipped, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(eur_vals.median(), color="red", ls="--", lw=1.5,
               label=f"Median: {eur_vals.median():,.0f} {unit}")
    ax.axvline(eur_vals.mean(), color="orange", ls="--", lw=1.5,
               label=f"Mean: {eur_vals.mean():,.0f} {unit}")
    p10 = eur_vals.quantile(0.10)
    p90 = eur_vals.quantile(0.90)
    ax.axvline(p10, color="gray", ls=":", lw=1.2, label=f"P90: {p10:,.0f} {unit}")
    ax.axvline(p90, color="gray", ls="-.", lw=1.2, label=f"P10: {p90:,.0f} {unit}")
    ax.set_xlabel(f"EUR ({unit})")
    ax.set_ylabel("Well Count")
    title = f"EUR Distribution{suffix}"
    if n_clipped:
        title += f"  (x-axis clipped at P99; {n_clipped} wells beyond)"
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    fname = f"eur_distribution_{stream_name}.png" if stream_name else "eur_distribution.png"
    fig.savefig(os.path.join(figures_dir, fname), dpi=150)
    plt.close(fig)


def _save_actual_vs_predicted_plot(
    y_test: pd.DataFrame,
    y_pred: pd.DataFrame,
    target_cols: List[str],
    figures_dir: str,
    stream_name: str = "",
) -> None:
    """Save a scatter plot of actual vs predicted for each target variable.

    Parameters:
        y_test: Actual target values (un-transformed).
        y_pred: Predicted target values (un-transformed).
        target_cols: Column names being compared.
        figures_dir: Directory to save figure.
        stream_name: Stream label for title/filename.
    """
    suffix = f" — {stream_name.upper()}" if stream_name else ""
    n = len(target_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, target_cols):
        actual = y_test[col].values
        predicted = y_pred[col].values
        ax.scatter(actual, predicted, alpha=0.3, s=15, color="steelblue")
        lims = [
            min(actual.min(), predicted.min()),
            max(actual.max(), predicted.max()),
        ]
        ax.plot(lims, lims, "r--", lw=1, label="Perfect")
        ax.set_xlabel(f"Actual {col}")
        ax.set_ylabel(f"Predicted {col}")
        r2 = r2_score(actual, predicted)
        ax.set_title(f"{col}  (R² = {r2:.3f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Actual vs Predicted — ML Model Diagnostic{suffix}", fontsize=13, y=1.02)
    plt.tight_layout()
    fname = f"actual_vs_predicted_{stream_name}.png" if stream_name else "actual_vs_predicted.png"
    fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# DEFAULT COLUMN MAP
# ===================================================================

DEFAULT_COL_MAP: Dict[str, str] = {
    # Join keys
    "api_uwi_well": "API_UWI",
    "api_uwi_prod": "API_UWI",
    # Production columns
    "producing_month": "ProducingMonth",
    "total_prod_months": "TotalProdMonths",
    "oil_rate": "LiquidsProd_BBL",
    "gas_rate": "GasProd_MCF",
    "water_rate": "WaterProd_BBL",
    "boe_rate": "Prod_BOE",
    # Well metadata
    "well_status": "ENVWellStatus",
    "prod_well_type": "ENVProdWellType",
    "well_type": "ENVWellType",
    "interval": "ENVInterval",
    "trajectory": "Trajectory",
    "first_prod_date": "FirstProdDate",
}


# ===================================================================
# SMART IMPUTATION FOR NEW WELLS
# ===================================================================

# Each rule is (product, factor1, factor2) meaning product = factor1 × factor2.
# Given any two of the three, the third can be derived.
# Rules are evaluated iteratively until no new values can be resolved.
_IMPUTATION_RULES: List[Tuple[str, str, str]] = [
    # Total proppant = proppant intensity × lateral length
    ("Proppant_LBS", "ProppantIntensity_LBSPerFT", "LateralLength_FT"),
    # Total fluid pumped = fluid intensity × lateral length
    ("TotalFluidPumped_BBL", "FluidIntensity_BBLPerFT", "LateralLength_FT"),
    # Total water pumped = water intensity × lateral length
    ("TotalWaterPumped_GAL", "WaterIntensity_GALPerFT", "LateralLength_FT"),
    # Total proppant = proppant per stage × number of stages
    ("Proppant_LBS", "ProppantPerStage_LBS", "FracStages"),
    # Total fluid pumped = fluid per stage × number of stages
    ("TotalFluidPumped_BBL", "FluidPerStage_BBL", "FracStages"),
    # Lateral length = stage spacing × number of stages
    ("LateralLength_FT", "StageSpacing_calc_FT", "FracStages"),
    # Lateral length = average stage spacing × number of stages
    ("LateralLength_FT", "AverageStageSpacing_FT", "FracStages"),
    # Total proppant = proppant loading × total water
    ("Proppant_LBS", "ProppantLoading_LBSPerGAL", "TotalWaterPumped_GAL"),
]

# Log-transform relationships: log_feature = log1p(raw_feature)
_LOG_FEATURE_PAIRS: List[Tuple[str, str]] = [
    ("log_Proppant_LBS", "Proppant_LBS"),
    ("log_TotalFluidPumped_BBL", "TotalFluidPumped_BBL"),
]


def _smart_impute_features(
    new_well_df: pd.DataFrame,
    needed_features: List[str],
    training_X: pd.DataFrame,
) -> pd.DataFrame:
    """Fill missing ML features using physics-aware imputation.

    Instead of blindly substituting the training median for every missing
    feature, this function:

    1. **Propagates known values** through engineering relationships.  For
       example, if the user supplies ``LateralLength_FT`` but not
       ``Proppant_LBS``, the function computes
       ``Proppant_LBS = median(ProppantIntensity_LBSPerFT) × LateralLength_FT``.
       Intensity / per-foot features are relatively stable across wells of
       different sizes, so imputing the intensity and back-calculating the
       absolute quantity is far more representative than imputing the
       absolute quantity directly.

    2. **Derives log-transforms** from their raw counterparts when available.

    3. Falls back to the training-set median only for features that cannot
       be resolved from the relationships above.

    Parameters:
        new_well_df: Single-row DataFrame of the new well (after feature
            engineering by ``_engineer_new_well_features``).
        needed_features: Ordered list of feature column names the ML model
            expects.
        training_X: Training feature matrix (for computing medians).

    Returns:
        Single-row DataFrame aligned to ``needed_features``.
    """
    # Build a dict of currently known values
    known: Dict[str, float] = {}
    for col in needed_features:
        if col in new_well_df.columns:
            val = new_well_df[col].values[0]
            if pd.notna(val):
                known[col] = float(val)

    # Pre-compute training medians (only for needed features)
    medians: Dict[str, float] = {}
    for col in needed_features:
        if col in training_X.columns:
            med = training_X[col].median()
            medians[col] = float(med) if pd.notna(med) else 0.0
        else:
            medians[col] = 0.0

    # Also compute medians for features that appear in rules but might
    # not be in the needed list (they are intermediaries for derivation).
    all_rule_features = set()
    for prod, f1, f2 in _IMPUTATION_RULES:
        all_rule_features.update([prod, f1, f2])
    for col in all_rule_features:
        if col not in medians:
            if col in training_X.columns:
                med = training_X[col].median()
                medians[col] = float(med) if pd.notna(med) else 0.0
            elif col in new_well_df.columns:
                val = new_well_df[col].values[0]
                if pd.notna(val):
                    medians[col] = float(val)

    # Iterative constraint propagation: keep applying rules until stable.
    # Each rule (P, F1, F2) means P = F1 × F2.  If any one is unknown it
    # can be derived from the other two.  We use median intensity as a
    # fill for the ratio/intensity factor when the user provides the
    # absolute or dimensional feature.
    MAX_ITER = 5
    for _ in range(MAX_ITER):
        progress = False

        for product, factor1, factor2 in _IMPUTATION_RULES:
            p_known = product in known
            f1_known = factor1 in known
            f2_known = factor2 in known

            if p_known and f1_known and f2_known:
                continue  # all three known, nothing to do

            if not p_known and f1_known and f2_known:
                val = known[factor1] * known[factor2]
                if np.isfinite(val) and val > 0:
                    known[product] = val
                    progress = True
            elif p_known and not f1_known and f2_known:
                if known[factor2] != 0:
                    val = known[product] / known[factor2]
                    if np.isfinite(val) and val > 0:
                        known[factor1] = val
                        progress = True
            elif p_known and f1_known and not f2_known:
                if known[factor1] != 0:
                    val = known[product] / known[factor1]
                    if np.isfinite(val) and val > 0:
                        known[factor2] = val
                        progress = True
            # Exactly one known: use median for the ratio/intensity factor
            # and derive the remaining one.  Prefer computing the product
            # (absolute feature) from a known dimension + median intensity.
            elif not p_known and not f1_known and f2_known:
                # factor2 known (e.g. LateralLength_FT), factor1 missing
                # (e.g. ProppantIntensity), product missing (Proppant_LBS).
                # Fill factor1 with median, then compute product.
                if factor1 in medians and medians[factor1] > 0:
                    known[factor1] = medians[factor1]
                    val = known[factor1] * known[factor2]
                    if np.isfinite(val) and val > 0:
                        known[product] = val
                        progress = True
            elif not p_known and f1_known and not f2_known:
                # Symmetric case
                if factor2 in medians and medians[factor2] > 0:
                    known[factor2] = medians[factor2]
                    val = known[factor1] * known[factor2]
                    if np.isfinite(val) and val > 0:
                        known[product] = val
                        progress = True

        # Derive log-transform features
        for log_feat, raw_feat in _LOG_FEATURE_PAIRS:
            if log_feat not in known and raw_feat in known and known[raw_feat] > 0:
                known[log_feat] = float(np.log1p(known[raw_feat]))
                progress = True
            elif raw_feat not in known and log_feat in known:
                known[raw_feat] = float(np.expm1(known[log_feat]))
                progress = True

        if not progress:
            break

    # Assemble final feature vector
    row: Dict[str, float] = {}
    for col in needed_features:
        if col in known:
            row[col] = known[col]
        elif col.startswith("interval_"):
            row[col] = 0.0  # one-hot dummies: absent = 0
        else:
            row[col] = medians.get(col, 0.0)
    return pd.DataFrame([row], columns=needed_features).astype(float)


# ===================================================================
# INTERNAL PIPELINE HELPERS
# ===================================================================

def _engineer_new_well_features(
    new_well_df: pd.DataFrame,
    cmap: Dict[str, str],
) -> pd.DataFrame:
    """Apply the same feature engineering as ``_prepare_ml_data()`` to a
    new well DataFrame so its feature vector is consistent with training.
    """
    df = new_well_df.copy()

    # Proppant per frac stage
    if "Proppant_LBS" in df.columns and "FracStages" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["ProppantPerStage_LBS"] = (
                df["Proppant_LBS"] / df["FracStages"].replace(0, np.nan)
            )

    # Fluid per frac stage
    if "TotalFluidPumped_BBL" in df.columns and "FracStages" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["FluidPerStage_BBL"] = (
                df["TotalFluidPumped_BBL"] / df["FracStages"].replace(0, np.nan)
            )

    # Stage spacing proxy
    if "LateralLength_FT" in df.columns and "FracStages" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["StageSpacing_calc_FT"] = (
                df["LateralLength_FT"] / df["FracStages"].replace(0, np.nan)
            )

    # Completion vintage year
    first_prod_col = cmap.get("first_prod_date", "FirstProdDate")
    if first_prod_col in df.columns:
        fpd = pd.to_datetime(df[first_prod_col], errors="coerce")
        df["VintageYear"] = fpd.dt.year

    # Log-transformed features
    for col_name in ["Proppant_LBS", "TotalFluidPumped_BBL"]:
        if col_name in df.columns:
            val = df[col_name].values[0]
            if pd.notna(val) and val > 0:
                df[f"log_{col_name}"] = np.log1p(val)

    # Encode interval dummies
    interval_col = cmap.get("interval", "ENVInterval")
    if interval_col in df.columns:
        dummies = pd.get_dummies(df[interval_col], prefix="interval", dtype=float)
        df = pd.concat([df, dummies], axis=1)

    return df


# ===================================================================
# PER-STREAM DCA + ML HELPER
# ===================================================================

def _fit_and_train_stream(
    wells: pd.DataFrame,
    prod: pd.DataFrame,
    prod_grouped: Dict[str, pd.DataFrame],
    all_apis: np.ndarray,
    stream_name: str,
    rate_col: str,
    unit: str,
    total_prod_months_col: str,
    forecast_months: int,
    figures_dir: str,
    well_features: List[str],
    cmap: Dict[str, str],
    test_size: float,
    random_state: int,
    outlier_z_threshold: float,
    wells_filtered_out: pd.DataFrame,
    min_dca_r2: float = 0.3,
    min_wells_for_ml: int = 30,
    min_peak_rate: float = 0.0,
) -> Optional[_StreamState]:
    """Fit DCA models and train ML for a single production stream.

    This encapsulates Steps 2-5 of the pipeline for one fluid stream:
        2. Fit DCA models to all wells using the stream's rate column.
        3. Remove outlier wells by z-score on DCA parameters.
        4. Build correlation matrix (well features vs DCA targets).
        5. Train and compare ML models to predict DCA params.

    Parameters:
        wells: Cleaned wells DataFrame.
        prod: Cleaned production DataFrame.
        prod_grouped: Dict of API -> sorted production DataFrame.
        all_apis: Array of well APIs to process.
        stream_name: 'oil', 'water', or 'gas'.
        rate_col: Production column name in the production DataFrame.
        unit: 'BBL' or 'MCF'.
        total_prod_months_col: Column name for month number.
        forecast_months: Number of months for EUR calculation.
        figures_dir: Directory for output figures.
        well_features: Well attribute column names.
        cmap: Column mapping dict.
        test_size: ML test split fraction.
        random_state: Random seed.
        outlier_z_threshold: Z-score threshold for outlier removal.
        wells_filtered_out: Wells DataFrame for ML data preparation.
        min_dca_r2: Minimum R² for accepting a DCA fit (default 0.3).
        min_wells_for_ml: Minimum wells needed for ML training (default 30).
        min_peak_rate: Minimum peak rate to attempt fitting a well.

    Returns:
        A ``_StreamState`` if sufficient data, else ``None``.
    """
    stream_label = stream_name.upper()

    # ------------------------------------------------------------------
    # STEP 2: Fit DCA models
    # ------------------------------------------------------------------
    _print_status(f"  [{stream_label}] Fitting DCA models ({rate_col}, {unit})...")

    # Check that the rate column exists in the production data
    if rate_col not in prod.columns:
        _print_status(f"  [{stream_label}] Column '{rate_col}' not found in production data. Skipping.")
        return None

    dca_records = []
    n_wells = len(all_apis)
    log_interval = max(1, n_wells // 20)
    t_step_start = time.time()

    for idx, api in enumerate(all_apis):
        if (idx + 1) % log_interval == 0 or idx == 0:
            elapsed = time.time() - t_step_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_wells - idx - 1) / rate / 60 if rate > 0 else 0
            _print_status(
                f"  [{stream_label}] Fitting well {idx + 1}/{n_wells}"
                f" ({(idx + 1) / n_wells * 100:.0f}%)"
                f" | ~{remaining:.1f} min remaining"
            )
        well_prod = prod_grouped.get(api)
        if well_prod is None:
            continue
        if rate_col not in well_prod.columns:
            continue
        t = well_prod[total_prod_months_col].values.astype(float)
        q = well_prod[rate_col].values.astype(float)

        # Drop NaN rate months (preserve t/q alignment) instead of
        # injecting artificial zeros that distort the decline fit.
        valid_mask = np.isfinite(q)
        t = t[valid_mask]
        q = q[valid_mask]

        if len(t) < 6 or np.max(q) <= min_peak_rate:
            continue
        first_nonzero = np.argmax(q > 0)
        t, q = t[first_nonzero:], q[first_nonzero:]
        if len(t) < 6:
            continue
        t = t - t[0] + 1
        results = _fit_all_models(t, q, api)
        best = _select_best_model(results)
        if best is None or best.r_squared < min_dca_r2:
            continue
        _, _, eur = _generate_forecast(best, forecast_months)
        record = {
            "_api_clean": api, "best_model": best.model_name,
            "r_squared": best.r_squared, "rmse": best.rmse, "eur": eur,
        }
        for pname, pval in best.params.items():
            record[pname] = pval
        dca_records.append(record)

    if len(dca_records) < 10:
        _print_status(
            f"  [{stream_label}] Only {len(dca_records)} wells fit successfully. "
            f"Need >= 10. Skipping this stream."
        )
        return None

    dca_df = pd.DataFrame(dca_records).set_index("_api_clean")
    _print_status(f"  [{stream_label}] Successfully fit {len(dca_df)} out of {n_wells} wells.")

    # ------------------------------------------------------------------
    # STEP 3: Remove outliers
    # ------------------------------------------------------------------
    numeric_dca_cols = [
        c for c in dca_df.columns
        if c not in ("best_model",) and pd.api.types.is_numeric_dtype(dca_df[c])
    ]
    z_scores = dca_df[numeric_dca_cols].apply(zscore, nan_policy="omit")
    outlier_mask = (z_scores.abs() > outlier_z_threshold).any(axis=1)
    n_outliers = outlier_mask.sum()
    dca_df = dca_df[~outlier_mask].copy()
    _print_status(f"  [{stream_label}] Removed {n_outliers} outlier wells. {len(dca_df)} remaining.")

    if len(dca_df) < 10:
        _print_status(f"  [{stream_label}] Too few wells after outlier removal. Skipping.")
        return None

    wells_for_stream = wells_filtered_out[
        wells_filtered_out["_api_clean"].isin(dca_df.index)
    ].copy()

    # ------------------------------------------------------------------
    # STEP 4: Correlation matrix
    # ------------------------------------------------------------------
    well_feat_df = wells_for_stream.set_index("_api_clean")
    # Use ACTUAL numeric DCA columns (not hardcoded list) so model-specific
    # params (e.g. Duong a/m) appear in the heatmap.
    target_cols_for_corr = [
        c for c in dca_df.columns
        if c not in ("best_model",) and pd.api.types.is_numeric_dtype(dca_df[c])
    ]
    avail_features = [c for c in well_features if c in well_feat_df.columns]
    corr_matrix = _build_correlation_matrix(
        well_feat_df, dca_df, avail_features, target_cols_for_corr
    )
    _save_correlation_heatmap(corr_matrix, figures_dir, avail_features,
                              target_cols_for_corr, stream_name=stream_name)

    # ------------------------------------------------------------------
    # STEP 5: Train ML models
    # ------------------------------------------------------------------
    most_common_model = dca_df["best_model"].mode().iloc[0]
    common_params = _DCA_MODELS[most_common_model]["param_names"]
    dca_for_ml = dca_df[dca_df["best_model"] == most_common_model].copy()
    _print_status(
        f"  [{stream_label}] Most common DCA model: '{most_common_model}' "
        f"({len(dca_for_ml)} wells)."
    )

    universal_targets = ["qi", "eur"]
    model_specific_targets = [p for p in common_params if p not in universal_targets]

    # Universal model (qi + eur) on ALL wells
    univ_target_cols = [c for c in universal_targets if c in dca_df.columns]
    X_univ, y_univ, used_features_univ, used_targets_univ = _prepare_ml_data(
        wells_for_stream, dca_df, well_features, univ_target_cols, cmap
    )
    _print_status(
        f"  [{stream_label}] Universal ML dataset: "
        f"{X_univ.shape[0]} samples, {X_univ.shape[1]} features"
    )

    # Graceful degradation: if too few samples for ML, use DCA-only mode
    dca_only_mode = len(X_univ) < min_wells_for_ml

    # -- ML training (or fallback to DCA-only) --------------------------
    comparison_univ = pd.DataFrame()
    best_model_univ = None
    best_name_univ = "DCA_MEDIAN_FALLBACK"
    scaler_univ = None
    log_applied_univ = False
    log_cols_univ: List[str] = []
    importance_df = pd.DataFrame()
    spec_target_cols = [c for c in model_specific_targets if c in dca_for_ml.columns]
    best_model_spec = None
    scaler_spec = None
    used_features_spec: List[str] = []
    used_targets_spec: List[str] = []
    log_applied_spec = False
    log_cols_spec: List[str] = []
    X_spec: Optional[pd.DataFrame] = None

    if dca_only_mode:
        _print_status(
            f"  [{stream_label}] Only {len(X_univ)} samples for ML "
            f"(need >= {min_wells_for_ml}). Using DCA-only median fallback."
        )
    else:
        # Full ML training path
        (comparison_univ, best_model_univ, best_name_univ,
         scaler_univ, log_applied_univ, log_cols_univ) = _train_and_compare_models(
            X_univ, y_univ, test_size=test_size, random_state=random_state
        )

        # Model-specific model for extra DCA params (a, m, di, b, etc.)
        if spec_target_cols:
            X_spec, y_spec, used_features_spec, used_targets_spec = _prepare_ml_data(
                wells_for_stream, dca_for_ml, well_features, spec_target_cols, cmap
            )
            if len(X_spec) >= min_wells_for_ml:
                (_, best_model_spec, _, scaler_spec,
                 log_applied_spec, log_cols_spec) = _train_and_compare_models(
                    X_spec, y_spec, test_size=test_size, random_state=random_state
                )

    # Save diagnostics (always — even in DCA-only mode)
    if not comparison_univ.empty:
        _save_model_comparison_chart(comparison_univ, figures_dir, stream_name=stream_name)
    _save_dca_model_selection_chart(dca_df, figures_dir, stream_name=stream_name)
    _save_eur_distribution_plot(dca_df, unit, figures_dir, stream_name=stream_name)

    # Actual vs Predicted diagnostic (ML path only)
    if best_model_univ is not None and scaler_univ is not None:
        try:
            X_train_ml, X_test_ml, _, y_test_ml_raw = train_test_split(
                X_univ, y_univ, test_size=test_size, random_state=random_state
            )
            X_test_scaled = scaler_univ.transform(X_test_ml)
            y_pred_log = best_model_univ.predict(X_test_scaled)
            y_pred_df = pd.DataFrame(
                y_pred_log, columns=y_univ.columns, index=y_test_ml_raw.index
            )
            if log_cols_univ:
                for col in log_cols_univ:
                    cidx = list(y_univ.columns).index(col)
                    y_pred_df[col] = np.expm1(y_pred_log[:, cidx])
            _save_actual_vs_predicted_plot(
                y_test_ml_raw, y_pred_df, list(y_univ.columns), figures_dir,
                stream_name=stream_name,
            )
        except Exception:
            pass

        # Feature importance (ML path only)
        try:
            if hasattr(best_model_univ, "estimators_"):
                importances = np.zeros(X_univ.shape[1])
                for est in best_model_univ.estimators_:
                    if hasattr(est, "feature_importances_"):
                        importances += est.feature_importances_
                importances /= len(best_model_univ.estimators_)
                importance_df = pd.DataFrame({
                    "feature": used_features_univ,
                    "importance": importances,
                }).sort_values("importance", ascending=False)
                _save_feature_importance_plot(importance_df, figures_dir,
                                             stream_name=stream_name)
        except Exception:
            pass

    if dca_only_mode:
        _print_status(
            f"  [{stream_label}] DCA-only mode. Median params will be used for prediction."
        )
    else:
        _print_status(f"  [{stream_label}] ML training complete. Best model: {best_name_univ}")

    return _StreamState(
        stream=stream_name,
        rate_col=rate_col,
        unit=unit,
        dca_df=dca_df,
        dca_for_ml=dca_for_ml,
        most_common_model=most_common_model,
        model_specific_targets=model_specific_targets,
        best_model_univ=best_model_univ,
        best_name_univ=best_name_univ,
        scaler_univ=scaler_univ,
        log_applied_univ=log_applied_univ,
        log_cols_univ=log_cols_univ,
        used_features_univ=used_features_univ,
        used_targets_univ=used_targets_univ,
        X_univ=X_univ,
        best_model_spec=best_model_spec,
        scaler_spec=scaler_spec,
        log_applied_spec=log_applied_spec,
        log_cols_spec=log_cols_spec,
        used_features_spec=used_features_spec,
        used_targets_spec=used_targets_spec,
        X_spec=X_spec,
        comparison=comparison_univ,
        corr_matrix=corr_matrix,
        importance_df=importance_df,
        best_model_name=best_name_univ,
    )


# ===================================================================
# PIPELINE STATE BUILDER
# ===================================================================

def _build_pipeline_state(
    well_data_path: str,
    prod_data_path: str,
    forecast_months: int,
    figures_dir: str = "./forecast_output",
    col_map: Optional[Dict[str, str]] = None,
    well_features: Optional[List[str]] = None,
    primary_stream_override: Optional[str] = None,
    min_producing_months: int = 12,
    test_size: float = 0.2,
    random_state: int = 42,
    outlier_z_threshold: float = 3.0,
    max_wells: Optional[int] = None,
    target_formation: Optional[str] = None,
    max_prod_months_per_well: Optional[int] = None,
) -> _PipelineState:
    """Execute pipeline: load data, then fit DCA + train ML for all 3 streams.

    Returns a ``_PipelineState`` whose artifacts can be fed to
    ``_predict_well_from_state()`` for any number of new wells without
    repeating the expensive training.
    """
    cmap = dict(DEFAULT_COL_MAP)
    if col_map:
        cmap.update(col_map)
    if well_features is None:
        well_features = list(_DEFAULT_WELL_FEATURES)
    os.makedirs(figures_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: Load & clean
    # ------------------------------------------------------------------
    _print_status("=" * 60)
    _print_status("STEP 1: Loading and cleaning data")
    _print_status("=" * 60)
    wells, prod = _load_and_clean(
        well_data_path, prod_data_path, cmap, min_producing_months,
        max_prod_months_per_well=max_prod_months_per_well,
    )
    if len(wells) == 0:
        raise ValueError("No wells remain after cleaning.  Check data & column mappings.")
    primary_stream = _detect_primary_stream(wells, cmap, primary_stream_override)
    _print_status(f"  Primary production stream: {primary_stream.upper()}")

    # Optional formation filter
    interval_col = cmap.get("interval", "ENVInterval")
    _MIN_FORMATION_WELLS = 200
    if target_formation is not None and interval_col in wells.columns:
        n_form = (wells[interval_col] == target_formation).sum()
        if n_form >= _MIN_FORMATION_WELLS:
            wells = wells[wells[interval_col] == target_formation].copy()
            prod = prod[prod["_api_clean"].isin(set(wells["_api_clean"]))].copy()
            _print_status(
                f"  Formation filter: {target_formation} "
                f"({n_form} wells, threshold {_MIN_FORMATION_WELLS}). Applied."
            )
        else:
            _print_status(
                f"  Formation filter: {target_formation} has only {n_form} wells "
                f"(need >= {_MIN_FORMATION_WELLS}). Ignoring filter — using all formations."
            )
            target_formation = None
    elif target_formation is not None:
        _print_status(f"  Formation filter column '{interval_col}' not found. Ignoring.")
        target_formation = None

    # Pre-group production data by well
    total_prod_months_col = cmap["total_prod_months"]
    _print_status("  Pre-grouping production data by well...")
    prod_grouped = {
        api: grp.sort_values(total_prod_months_col)
        for api, grp in prod.groupby("_api_clean")
    }
    all_apis = wells["_api_clean"].unique()
    if max_wells is not None and len(all_apis) > max_wells:
        rng = np.random.default_rng(random_state)
        all_apis = rng.choice(all_apis, size=max_wells, replace=False)
        _print_status(f"  Sampled {max_wells} wells (max_wells parameter).")

    # ------------------------------------------------------------------
    # STEPS 2-5: Fit DCA + Train ML for each fluid stream
    # ------------------------------------------------------------------
    stream_states: Dict[str, _StreamState] = {}

    for stream_name, (rate_key, unit) in _STREAM_CONFIG.items():
        rate_col = cmap.get(rate_key, rate_key)
        _print_status("=" * 60)
        _print_status(f"STREAM: {stream_name.upper()} — DCA fitting + ML training")
        _print_status("=" * 60)

        ss = _fit_and_train_stream(
            wells=wells,
            prod=prod,
            prod_grouped=prod_grouped,
            all_apis=all_apis,
            stream_name=stream_name,
            rate_col=rate_col,
            unit=unit,
            total_prod_months_col=total_prod_months_col,
            forecast_months=forecast_months,
            figures_dir=figures_dir,
            well_features=well_features,
            cmap=cmap,
            test_size=test_size,
            random_state=random_state,
            outlier_z_threshold=outlier_z_threshold,
            wells_filtered_out=wells,
        )
        if ss is not None:
            stream_states[stream_name] = ss
            _print_status(f"  {stream_name.upper()} stream: READY")
        else:
            _print_status(f"  {stream_name.upper()} stream: SKIPPED (insufficient data)")

    if not stream_states:
        raise ValueError(
            "No fluid streams had enough data for DCA + ML training. "
            "Check your data and column mappings."
        )

    # Use primary stream for backward-compat fields
    primary_ss = stream_states.get(primary_stream)
    if primary_ss is None:
        # Fall back to first available stream
        original_primary = primary_stream
        primary_stream = next(iter(stream_states))
        primary_ss = stream_states[primary_stream]
        _print_status(
            f"  Primary stream '{original_primary}' not available, "
            f"falling back to '{primary_stream}'."
        )

    return _PipelineState(
        cmap=cmap,
        primary_stream=primary_stream,
        unit=primary_ss.unit,
        forecast_months=forecast_months,
        figures_dir=os.path.abspath(figures_dir),
        wells_filtered=wells,
        target_formation=target_formation,
        stream_states=stream_states,
        # Backward-compat fields from primary stream
        dca_df=primary_ss.dca_df,
        dca_for_ml=primary_ss.dca_for_ml,
        most_common_model=primary_ss.most_common_model,
        model_specific_targets=primary_ss.model_specific_targets,
        best_model_univ=primary_ss.best_model_univ,
        best_name_univ=primary_ss.best_name_univ,
        scaler_univ=primary_ss.scaler_univ,
        log_applied_univ=primary_ss.log_applied_univ,
        log_cols_univ=primary_ss.log_cols_univ,
        used_features_univ=primary_ss.used_features_univ,
        used_targets_univ=primary_ss.used_targets_univ,
        X_univ=primary_ss.X_univ,
        best_model_spec=primary_ss.best_model_spec,
        scaler_spec=primary_ss.scaler_spec,
        log_applied_spec=primary_ss.log_applied_spec,
        log_cols_spec=primary_ss.log_cols_spec,
        used_features_spec=primary_ss.used_features_spec,
        used_targets_spec=primary_ss.used_targets_spec,
        X_spec=primary_ss.X_spec,
        comparison=primary_ss.comparison,
        corr_matrix=primary_ss.corr_matrix,
        importance_df=primary_ss.importance_df,
        best_model_name=primary_ss.best_model_name,
    )


# ===================================================================
# PER-STREAM PREDICTION HELPER
# ===================================================================

def _predict_stream_for_well(
    ss: _StreamState,
    new_well_df: pd.DataFrame,
    cmap: Dict[str, str],
    forecast_months: int,
    label: str = "",
) -> StreamForecast:
    """Predict DCA parameters and generate forecasts for one stream.

    Parameters:
        ss: Trained stream state.
        new_well_df: Engineered new-well DataFrame (single row).
        cmap: Column mapping.
        forecast_months: Forecast horizon.
        label: Label for log messages.

    Returns:
        StreamForecast for this stream.
    """
    stream_label = ss.stream.upper()

    # ------------------------------------------------------------------
    # DCA-only fallback: when ML models aren't available, use median DCA
    # params from the training set.
    # ------------------------------------------------------------------
    dca_only_mode = (ss.best_model_univ is None or ss.scaler_univ is None)
    new_X_scaled = None  # only set when ML is available

    if dca_only_mode:
        _print_status(f"  [{stream_label}] Using DCA-only median params (no ML model){label}")
        predicted_dca: Dict[str, float] = {}
        # Use median of all DCA params from the model-specific training set
        for col in _DCA_MODELS[ss.most_common_model]["param_names"]:
            if col in ss.dca_for_ml.columns:
                predicted_dca[col] = float(ss.dca_for_ml[col].median())
        # Use median EUR from the SAME model-specific subset for consistency
        if "eur" in ss.dca_for_ml.columns:
            predicted_dca["eur"] = float(ss.dca_for_ml["eur"].median())
        elif "eur" in ss.dca_df.columns:
            predicted_dca["eur"] = float(ss.dca_df["eur"].median())
    else:
        # Align to universal training features using physics-aware imputation.
        # Instead of filling every missing feature with the training median,
        # we propagate user-supplied values through engineering relationships
        # (e.g. Proppant_LBS = ProppantIntensity_LBSPerFT × LateralLength_FT)
        # so that absolute features scale correctly with the proposed well's
        # dimensions.  See ``_smart_impute_features`` for details.
        new_X = _smart_impute_features(
            new_well_df, ss.used_features_univ, ss.X_univ,
        )

        # Predict universal targets (qi + eur)
        new_X_scaled = ss.scaler_univ.transform(new_X)
        pred_univ_raw = ss.best_model_univ.predict(new_X_scaled)[0]
        predicted_dca: Dict[str, float] = {}
        for i, col in enumerate(ss.used_targets_univ):
            val = float(pred_univ_raw[i])
            if ss.log_applied_univ and col in ss.log_cols_univ:
                val = float(np.expm1(val))
            predicted_dca[col] = val

        # Predict model-specific targets
        if ss.best_model_spec is not None and ss.scaler_spec is not None:
            spec_training_X = ss.X_spec if ss.X_spec is not None else ss.X_univ
            new_X_spec = _smart_impute_features(
                new_well_df, ss.used_features_spec, spec_training_X,
            )
            new_X_spec_scaled = ss.scaler_spec.transform(new_X_spec)
            pred_spec_raw = ss.best_model_spec.predict(new_X_spec_scaled)[0]
            for i, col in enumerate(ss.used_targets_spec):
                val = float(pred_spec_raw[i])
                if ss.log_applied_spec and col in ss.log_cols_spec:
                    val = float(np.expm1(val))
                predicted_dca[col] = val
        else:
            for col in ss.model_specific_targets:
                if col in ss.dca_for_ml.columns:
                    predicted_dca[col] = float(ss.dca_for_ml[col].median())

    _print_status(f"  [{stream_label}] Predicted DCA{label}: {predicted_dca}")

    # Param IQR for Monte Carlo
    param_iqr = {}
    for col in _DCA_MODELS[ss.most_common_model]["param_names"]:
        if col in ss.dca_for_ml.columns:
            vals = ss.dca_for_ml[col].dropna()
            param_iqr[col] = float(vals.quantile(0.75) - vals.quantile(0.25))
        else:
            param_iqr[col] = abs(predicted_dca.get(col, 1.0)) * 0.2

    # EUR bounds
    eur_vals = ss.dca_df["eur"].dropna()
    eur_bounds = (float(eur_vals.quantile(0.05)), float(eur_vals.quantile(0.95)))

    # Analog wells via KNN (ML path only)
    analog_df = None
    if not dca_only_mode and new_X_scaled is not None:
        try:
            from sklearn.neighbors import NearestNeighbors
            k_analogs = min(100, len(ss.X_univ) // 2)
            if k_analogs >= 20:
                nn = NearestNeighbors(n_neighbors=k_analogs, metric="euclidean")
                X_univ_scaled = ss.scaler_univ.transform(ss.X_univ)
                nn.fit(X_univ_scaled)
                _, nn_indices = nn.kneighbors(new_X_scaled)
                analog_apis = ss.X_univ.index[nn_indices[0]]
                analog_df = ss.dca_for_ml.loc[
                    ss.dca_for_ml.index.isin(analog_apis)
                ].copy()
                if len(analog_df) < 20:
                    analog_df = None
                else:
                    _print_status(
                        f"  [{stream_label}] Using {len(analog_df)} analog wells"
                        f" for probabilistic forecast{label}."
                    )
        except Exception:
            analog_df = None
    else:
        # DCA-only mode: use all DCA wells as analogs
        analog_df = ss.dca_for_ml.copy() if len(ss.dca_for_ml) >= 5 else None
        if analog_df is not None:
            _print_status(
                f"  [{stream_label}] Using all {len(analog_df)} DCA wells"
                f" as analogs for probabilistic forecast{label}."
            )

    df_p10, df_p50, df_p90, eur_p10, eur_p50, eur_p90 = (
        _generate_probabilistic_forecasts(
            predicted_dca, ss.most_common_model,
            forecast_months, param_iqr, eur_bounds,
            analog_wells=analog_df,
        )
    )

    _print_status(f"  [{stream_label}] EUR P10: {eur_p10:,.0f} | P50: {eur_p50:,.0f} | P90: {eur_p90:,.0f} {ss.unit}")

    return StreamForecast(
        stream=ss.stream,
        unit=ss.unit,
        predicted_dca_params=predicted_dca,
        forecast_p10=df_p10,
        forecast_p50=df_p50,
        forecast_p90=df_p90,
        eur_p10=eur_p10,
        eur_p50=eur_p50,
        eur_p90=eur_p90,
        best_ml_model_name=ss.best_model_name,
        model_comparison=ss.comparison,
        well_dca_results=ss.dca_df.reset_index(),
    )


def _predict_well_from_state(
    state: _PipelineState,
    new_well: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    well_id: str = "",
    save_figure: bool = True,
    figure_filename: str = "forecast_p10_p50_p90.png",
) -> WellPrediction:
    """Predict DCA parameters and P10/P50/P90 for all streams of a new well.

    Parameters:
        state: Trained pipeline from ``_build_pipeline_state()``.
        new_well: Proposed well attributes (dict, Series, single-row DF).
        well_id: Optional label for this well.
        save_figure: Whether to save the forecast figure.
        figure_filename: Base filename for the forecast figure.

    Returns:
        WellPrediction with DCA params and probabilistic forecasts for all streams.
    """
    # Convert to DataFrame
    if isinstance(new_well, dict):
        new_well_df = pd.DataFrame([new_well])
    elif isinstance(new_well, pd.Series):
        new_well_df = pd.DataFrame([new_well.to_dict()])
    elif isinstance(new_well, pd.DataFrame):
        new_well_df = new_well.copy()
    else:
        raise ValueError("new_well must be a dict, Series, or single-row DataFrame.")

    # Apply feature engineering
    new_well_df = _engineer_new_well_features(new_well_df, state.cmap)

    label = f" ({well_id})" if well_id else ""

    # Predict each stream
    stream_forecasts: Dict[str, StreamForecast] = {}
    for stream_name, ss in state.stream_states.items():
        _print_status(f"  Predicting {stream_name.upper()} stream{label}...")
        sf = _predict_stream_for_well(
            ss, new_well_df, state.cmap, state.forecast_months, label=label,
        )
        stream_forecasts[stream_name] = sf

        # Save per-stream forecast plot
        if save_figure:
            base, ext = os.path.splitext(figure_filename)
            stream_fname = f"{base}_{stream_name}{ext}"
            _save_forecast_plot(
                sf.forecast_p10, sf.forecast_p50, sf.forecast_p90,
                sf.unit, state.figures_dir,
                filename=stream_fname, stream_name=stream_name,
            )
            _print_status(f"  Saved {stream_fname}")

    # Save combined multi-stream plot
    if save_figure and len(stream_forecasts) > 1:
        base, ext = os.path.splitext(figure_filename)
        multi_fname = f"{base}_all_streams{ext}"
        _save_multistream_forecast_plot(
            stream_forecasts, state.figures_dir, filename=multi_fname,
        )
        _print_status(f"  Saved {multi_fname}")

    # Primary stream for backward compat
    primary_sf = stream_forecasts.get(state.primary_stream)
    if primary_sf is None:
        primary_sf = next(iter(stream_forecasts.values()))

    return WellPrediction(
        well_id=well_id,
        predicted_dca_params=primary_sf.predicted_dca_params,
        forecast_p10=primary_sf.forecast_p10,
        forecast_p50=primary_sf.forecast_p50,
        forecast_p90=primary_sf.forecast_p90,
        eur_p10=primary_sf.eur_p10,
        eur_p50=primary_sf.eur_p50,
        eur_p90=primary_sf.eur_p90,
        stream_forecasts=stream_forecasts,
    )


# ===================================================================
# MAIN ENTRY POINT
# ===================================================================

def predict_new_well(
    well_data_path: str,
    prod_data_path: str,
    new_well: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    forecast_months: int,
    figures_dir: str = "./forecast_output",
    col_map: Optional[Dict[str, str]] = None,
    well_features: Optional[List[str]] = None,
    primary_stream_override: Optional[str] = None,
    min_producing_months: int = 12,
    test_size: float = 0.2,
    random_state: int = 42,
    outlier_z_threshold: float = 3.0,
    max_wells: Optional[int] = None,
    target_formation: Optional[str] = None,
    max_prod_months_per_well: Optional[int] = None,
) -> PipelineResults:
    """Run the full forecasting pipeline and predict production for a new well.

    This function models **all three fluid streams** (oil, water, gas)
    independently.  Each stream gets its own DCA fits, ML models, and
    P10/P50/P90 forecasts.

    Workflow:
        1. Loads and cleans the well + production data.
        2. For each fluid stream (oil, water, gas):
           a. Fits five DCA models to every qualifying well.
           b. Removes outlier wells by z-score on DCA parameters.
           c. Computes a correlation matrix (well attributes → DCA parameters).
           d. Trains six ML models and selects the best.
        3. Predicts DCA parameters for the new well and generates P10/P50/P90
           production forecasts for all three streams.
        4. Saves diagnostic figures to ``figures_dir``.

    Parameters:
        well_data_path (str):
            Path to the wells CSV file.
        prod_data_path (str):
            Path to the monthly production CSV file.
        new_well (dict | DataFrame | Series):
            Proposed well attributes.
        forecast_months (int):
            Total number of months to forecast (e.g. 360 for 30 years).
        figures_dir (str):
            Directory where output figures will be saved.
        col_map (dict, optional):
            Override column-name mapping.
        well_features (list[str], optional):
            List of numeric well-attribute column names to use as ML features.
        primary_stream_override (str, optional):
            Force ``'oil'`` or ``'gas'`` as the primary production stream.
        min_producing_months (int):
            Minimum producing months a well must have (default 12).
        test_size (float):
            Fraction of data for ML testing (default 0.2).
        random_state (int):
            Random seed (default 42).
        outlier_z_threshold (float):
            Z-score threshold for outlier removal (default 3.0).
        max_wells (int, optional):
            If set, randomly sample this many wells for DCA fitting.
        target_formation (str, optional):
            Restrict training to a single formation.
        max_prod_months_per_well (int, optional):
            If set, truncate each well's production history to this many
            months.  Useful for speeding up DCA fitting during testing.

    Returns:
        PipelineResults:
            A dataclass with all outputs including per-stream forecasts.
            Access ``results.stream_forecasts["oil"]``,
            ``results.stream_forecasts["water"]``, and
            ``results.stream_forecasts["gas"]`` for individual stream
            results.  Top-level fields (eur_p10, forecast_p50, etc.)
            correspond to the primary stream for backward compatibility.

    Examples:
        >>> results = predict_new_well(
        ...     well_data_path="wells.csv",
        ...     prod_data_path="production.csv",
        ...     new_well={"TVD_FT": 8500, "LateralLength_FT": 10000},
        ...     forecast_months=360,
        ... )
        >>> # Primary stream (backward compat)
        >>> print(f"Primary EUR P50: {results.eur_p50:,.0f}")
        >>> # Per-stream access
        >>> for stream, sf in results.stream_forecasts.items():
        ...     print(f"{stream}: EUR P50 = {sf.eur_p50:,.0f} {sf.unit}")
    """
    pipeline_start = time.time()

    # Resolve "auto" formation filter
    _resolved_formation = target_formation
    if target_formation == "auto":
        _cmap = dict(DEFAULT_COL_MAP)
        if col_map:
            _cmap.update(col_map)
        _interval_col = _cmap.get("interval", "ENVInterval")
        _nw = new_well if isinstance(new_well, dict) else (
            new_well.to_dict() if isinstance(new_well, pd.Series)
            else new_well.iloc[0].to_dict() if isinstance(new_well, pd.DataFrame)
            else {}
        )
        _resolved_formation = _nw.get(_interval_col) or _nw.get("ENVInterval")
        if _resolved_formation:
            _print_status(f"  Auto-detected formation: {_resolved_formation}")
        else:
            _resolved_formation = None

    state = _build_pipeline_state(
        well_data_path=well_data_path,
        prod_data_path=prod_data_path,
        forecast_months=forecast_months,
        figures_dir=figures_dir,
        col_map=col_map,
        well_features=well_features,
        primary_stream_override=primary_stream_override,
        min_producing_months=min_producing_months,
        test_size=test_size,
        random_state=random_state,
        outlier_z_threshold=outlier_z_threshold,
        max_wells=max_wells,
        target_formation=_resolved_formation,
        max_prod_months_per_well=max_prod_months_per_well,
    )

    # ------------------------------------------------------------------
    # STEP 6: Predict new well (all streams)
    # ------------------------------------------------------------------
    _print_status("=" * 60)
    _print_status("STEP 6/6: Predicting new well performance (all streams)")
    _print_status("=" * 60)

    pred = _predict_well_from_state(state, new_well)

    elapsed = time.time() - pipeline_start
    _print_status("=" * 60)
    _print_status(f"PIPELINE COMPLETE — elapsed {elapsed / 60:.1f} minutes")
    for sn, sf in pred.stream_forecasts.items():
        _print_status(
            f"  {sn.upper()}: EUR P10={sf.eur_p10:,.0f} | "
            f"P50={sf.eur_p50:,.0f} | P90={sf.eur_p90:,.0f} {sf.unit}"
        )
    _print_status(f"Figures saved to: {state.figures_dir}")
    _print_status("=" * 60)

    # Primary stream for backward compat
    primary_ss = state.stream_states.get(state.primary_stream)
    if primary_ss is None:
        primary_ss = next(iter(state.stream_states.values()))

    return PipelineResults(
        well_dca_results=primary_ss.dca_df.reset_index(),
        model_comparison=primary_ss.comparison,
        best_ml_model_name=primary_ss.best_model_name,
        best_ml_model=primary_ss.best_model_univ,
        correlation_matrix=primary_ss.corr_matrix,
        predicted_dca_params=pred.predicted_dca_params,
        forecast_p10=pred.forecast_p10,
        forecast_p50=pred.forecast_p50,
        forecast_p90=pred.forecast_p90,
        eur_p10=pred.eur_p10,
        eur_p50=pred.eur_p50,
        eur_p90=pred.eur_p90,
        figures_dir=state.figures_dir,
        feature_importance=primary_ss.importance_df,
        production_unit=primary_ss.unit,
        stream_forecasts=pred.stream_forecasts,
    )


# ===================================================================
# BATCH / REUSABLE PIPELINE API
# ===================================================================

def build_forecast_pipeline(
    well_data_path: str,
    prod_data_path: str,
    forecast_months: int = 360,
    figures_dir: str = "./forecast_output",
    col_map: Optional[Dict[str, str]] = None,
    well_features: Optional[List[str]] = None,
    primary_stream_override: Optional[str] = None,
    min_producing_months: int = 12,
    test_size: float = 0.2,
    random_state: int = 42,
    outlier_z_threshold: float = 3.0,
    max_wells: Optional[int] = None,
    target_formation: Optional[str] = None,
    max_prod_months_per_well: Optional[int] = None,
) -> _PipelineState:
    """Build a trained forecasting pipeline (all three streams) for reuse.

    Runs the expensive data loading, DCA fitting, outlier removal,
    correlation analysis, and ML model training **once** for each fluid
    stream.  The returned object can then be passed to
    ``predict_from_pipeline()`` for any number of new wells.

    Parameters:
        Same as ``predict_new_well()`` except ``new_well`` is not needed.

    Returns:
        Trained pipeline state object (pass to ``predict_from_pipeline``
        or ``predict_new_wells``).

    Example::

        pipeline = build_forecast_pipeline("wells.csv", "production.csv")
        result_A = predict_from_pipeline(pipeline, well_A)
        result_B = predict_from_pipeline(pipeline, well_B)
    """
    pipeline_start = time.time()
    state = _build_pipeline_state(
        well_data_path=well_data_path,
        prod_data_path=prod_data_path,
        forecast_months=forecast_months,
        figures_dir=figures_dir,
        col_map=col_map,
        well_features=well_features,
        primary_stream_override=primary_stream_override,
        min_producing_months=min_producing_months,
        test_size=test_size,
        random_state=random_state,
        outlier_z_threshold=outlier_z_threshold,
        max_wells=max_wells,
        target_formation=target_formation,
        max_prod_months_per_well=max_prod_months_per_well,
    )
    elapsed = time.time() - pipeline_start
    _print_status("=" * 60)
    _print_status(f"PIPELINE BUILD COMPLETE — elapsed {elapsed / 60:.1f} minutes")
    _print_status(f"  Streams trained: {list(state.stream_states.keys())}")
    _print_status(f"  Primary stream: {state.primary_stream}")
    if state.target_formation:
        _print_status(f"  Formation filter: {state.target_formation}")
    _print_status("  Ready to predict new wells.")
    _print_status("=" * 60)
    return state


def predict_from_pipeline(
    pipeline: _PipelineState,
    new_well: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    forecast_months: Optional[int] = None,
    well_id: str = "",
    save_figure: bool = True,
    figure_filename: str = "forecast_p10_p50_p90.png",
) -> PipelineResults:
    """Predict production for a single new well using a pre-built pipeline.

    Predicts all three fluid streams (oil, water, gas).

    Parameters:
        pipeline: Trained pipeline from ``build_forecast_pipeline()``.
        new_well: Proposed well attributes (dict, Series, or DataFrame).
        forecast_months: Override forecast horizon (default: same as build).
        well_id: Optional label for this well.
        save_figure: Whether to save forecast plot (default True).
        figure_filename: Custom filename for the forecast figure.

    Returns:
        PipelineResults with per-stream forecasts in ``stream_forecasts``.
    """
    # Use a shallow copy to avoid mutating the shared pipeline object
    import copy
    state = copy.copy(pipeline)
    if forecast_months is not None:
        state.forecast_months = forecast_months

    pred = _predict_well_from_state(
        state, new_well, well_id=well_id,
        save_figure=save_figure, figure_filename=figure_filename,
    )

    primary_ss = pipeline.stream_states.get(pipeline.primary_stream)
    if primary_ss is None:
        primary_ss = next(iter(pipeline.stream_states.values()))

    return PipelineResults(
        well_dca_results=primary_ss.dca_df.reset_index(),
        model_comparison=primary_ss.comparison,
        best_ml_model_name=primary_ss.best_model_name,
        best_ml_model=primary_ss.best_model_univ,
        correlation_matrix=primary_ss.corr_matrix,
        predicted_dca_params=pred.predicted_dca_params,
        forecast_p10=pred.forecast_p10,
        forecast_p50=pred.forecast_p50,
        forecast_p90=pred.forecast_p90,
        eur_p10=pred.eur_p10,
        eur_p50=pred.eur_p50,
        eur_p90=pred.eur_p90,
        figures_dir=pipeline.figures_dir,
        feature_importance=primary_ss.importance_df,
        production_unit=primary_ss.unit,
        stream_forecasts=pred.stream_forecasts,
    )


def predict_new_wells(
    well_data_path: str,
    prod_data_path: str,
    new_wells: List[Union[Dict[str, Any], pd.DataFrame, pd.Series]],
    forecast_months: int = 360,
    figures_dir: str = "./forecast_output",
    col_map: Optional[Dict[str, str]] = None,
    well_features: Optional[List[str]] = None,
    primary_stream_override: Optional[str] = None,
    min_producing_months: int = 12,
    test_size: float = 0.2,
    random_state: int = 42,
    outlier_z_threshold: float = 3.0,
    max_wells: Optional[int] = None,
    per_well_figures: bool = True,
    target_formation: Optional[str] = None,
    max_prod_months_per_well: Optional[int] = None,
) -> BatchPipelineResults:
    """Predict production for **multiple** new wells in one pipeline run.

    Builds the trained pipeline once (all three streams), then predicts
    for each well in ``new_wells``.

    Parameters:
        well_data_path: Path to wells CSV.
        prod_data_path: Path to production CSV.
        new_wells: List of well dicts / Series / DataFrames.
        forecast_months: Forecast horizon in months (default 360 = 30 yr).
        figures_dir: Output directory for figures.
        col_map: Optional column-name overrides.
        well_features: Optional custom feature list.
        primary_stream_override: Force ``'oil'`` or ``'gas'``.
        min_producing_months: Minimum months for training wells.
        test_size: ML test split fraction.
        random_state: Random seed.
        outlier_z_threshold: Z-score outlier threshold.
        max_wells: Cap on training well count.
        per_well_figures: Save individual forecast plots per well.
        target_formation: Restrict training to a formation.
        max_prod_months_per_well: Truncate per-well production history.

    Returns:
        BatchPipelineResults with per-well predictions (including all streams).

    Example::

        wells = [
            {"well_id": "Pad-A-1", "TVD_FT": 8500, "LateralLength_FT": 10000},
            {"well_id": "Pad-A-2", "TVD_FT": 9200, "LateralLength_FT": 7500},
        ]
        batch = predict_new_wells("wells.csv", "production.csv", new_wells=wells)
        for pred in batch.predictions:
            for stream, sf in pred.stream_forecasts.items():
                print(f"{pred.well_id} {stream}: EUR P50 = {sf.eur_p50:,.0f}")
    """
    pipeline_start = time.time()
    _print_status("=" * 60)
    _print_status(f"BATCH PREDICTION — {len(new_wells)} wells")
    _print_status("=" * 60)

    state = _build_pipeline_state(
        well_data_path=well_data_path,
        prod_data_path=prod_data_path,
        forecast_months=forecast_months,
        figures_dir=figures_dir,
        col_map=col_map,
        well_features=well_features,
        primary_stream_override=primary_stream_override,
        min_producing_months=min_producing_months,
        test_size=test_size,
        random_state=random_state,
        outlier_z_threshold=outlier_z_threshold,
        max_wells=max_wells,
        target_formation=target_formation,
        max_prod_months_per_well=max_prod_months_per_well,
    )

    predictions: List[WellPrediction] = []
    for i, well in enumerate(new_wells):
        well_label = f"Well_{i + 1}"
        if isinstance(well, dict) and "well_id" in well:
            well_label = str(well["well_id"])
        elif isinstance(well, pd.Series) and "well_id" in well.index:
            well_label = str(well["well_id"])
        elif isinstance(well, pd.DataFrame) and "well_id" in well.columns:
            well_label = str(well["well_id"].iloc[0])

        _print_status("=" * 60)
        _print_status(f"PREDICTING {well_label} ({i + 1}/{len(new_wells)})")
        _print_status("=" * 60)

        fig_name = f"forecast_{well_label}.png" if per_well_figures else ""
        pred = _predict_well_from_state(
            state, well,
            well_id=well_label,
            save_figure=per_well_figures,
            figure_filename=fig_name,
        )
        predictions.append(pred)

    # Build summary table (with per-stream EUR columns)
    summary_records = []
    for pred in predictions:
        rec = {
            "well_id": pred.well_id,
            "eur_p10": pred.eur_p10,
            "eur_p50": pred.eur_p50,
            "eur_p90": pred.eur_p90,
        }
        for sn, sf in pred.stream_forecasts.items():
            rec[f"{sn}_eur_p10"] = sf.eur_p10
            rec[f"{sn}_eur_p50"] = sf.eur_p50
            rec[f"{sn}_eur_p90"] = sf.eur_p90
        rec.update({f"pred_{k}": v for k, v in pred.predicted_dca_params.items()})
        summary_records.append(rec)
    summary = pd.DataFrame(summary_records)

    elapsed = time.time() - pipeline_start
    _print_status("=" * 60)
    _print_status(f"BATCH COMPLETE — {len(predictions)} wells in {elapsed / 60:.1f} minutes")
    _print_status(f"Figures saved to: {state.figures_dir}")
    _print_status("=" * 60)

    primary_ss = state.stream_states.get(state.primary_stream)
    if primary_ss is None:
        primary_ss = next(iter(state.stream_states.values()))

    return BatchPipelineResults(
        well_dca_results=primary_ss.dca_df.reset_index(),
        model_comparison=primary_ss.comparison,
        best_ml_model_name=primary_ss.best_model_name,
        best_ml_model=primary_ss.best_model_univ,
        correlation_matrix=primary_ss.corr_matrix,
        feature_importance=primary_ss.importance_df,
        figures_dir=state.figures_dir,
        production_unit=primary_ss.unit,
        training_features=list(primary_ss.used_features_univ),
        predictions=predictions,
        summary=summary,
    )
