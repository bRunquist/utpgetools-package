# utpgetools

A collection of utility tools for your UT PGE projects.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/bRunquist/utpgetools.git
```

Or, if published on PyPI (it is not right now and likely will not be):

```bash
pip install utpgetools
```

## Modules

| Module | Description |
|--------|-------------|
| `artificial_lift` | Artificial lift systems analysis (VLP, IPR, gas lift, plunger lift, PCP) |
| `facilities` | Oil & gas processing facilities design and analysis |
| `forecasting` | DCA-based production forecasting with ML prediction for new wells |
| `formation_evaluation` | Petrophysical analysis *(under development)* |
| `general` | General-purpose math & utility functions |
| `geomechanics` | Stress analysis, Mohr's circle, fault stability |
| `geostats` | Geostatistical analysis *(under development)* |
| `numerical_methods` | Computational tools *(under development)* |
| `production` | Production engineering *(under development)* |
| `utilities_package` | Fluid properties & multiphase flow calculations |

## Quick Start — Production Forecasting

The `forecasting` module provides a one-function pipeline that fits DCA
models to historical wells, trains ML regressors, and predict P10/P50/P90
production forecasts for a proposed new well.

```python
from utpgetools.forecasting import predict_new_well

new_well = {
    "TVD_FT": 8500,
    "LateralLength_FT": 10000,
    "FracStages": 40,
    "Proppant_LBS": 12_000_000,
    "ProppantIntensity_LBSPerFT": 1200,
    "TotalFluidPumped_BBL": 400_000,
    "FluidIntensity_BBLPerFT": 40,
    "ENVInterval": "WOLFCAMP A",
}

results = predict_new_well(
    well_data_path="wells.csv",
    prod_data_path="production.csv",
    new_well=new_well,
    forecast_months=360,
    figures_dir="./forecast_output",
)

print(f"EUR P50: {results.eur_p50:,.0f} {results.production_unit}")
```

### What you get back

`predict_new_well()` returns a `PipelineResults` dataclass containing:

| Attribute | Type | Description |
|-----------|------|-------------|
| `eur_p10` / `eur_p50` / `eur_p90` | `float` | Estimated Ultimate Recovery at P10, P50, P90 |
| `forecast_p10` / `forecast_p50` / `forecast_p90` | `DataFrame` | Monthly forecasts (columns: `month`, `rate`, `cumulative`) |
| `predicted_dca_params` | `dict` | DCA parameters predicted for the new well |
| `well_dca_results` | `DataFrame` | Best-fit DCA results for every training well |
| `model_comparison` | `DataFrame` | ML model benchmark (R², MAE, RMSE, CV R²) |
| `best_ml_model_name` | `str` | Name of the winning ML model |
| `best_ml_model` | sklearn estimator | The fitted model (can be pickled) |
| `correlation_matrix` | `DataFrame` | Feature-to-DCA-parameter correlations |
| `feature_importance` | `DataFrame` | Feature importances (if tree-based) |
| `figures_dir` | `str` | Path to the 7 saved diagnostic PNGs |
| `production_unit` | `str` | `"BBL"` or `"MCF"` |

### Diagnostic figures

The pipeline saves 7 figures to `figures_dir`:

1. **correlation_heatmap.png** — well features vs DCA parameters
2. **ml_model_comparison.png** — R² comparison across 6 ML models
3. **actual_vs_predicted.png** — scatter of actual vs predicted targets
4. **feature_importance.png** — top features driving predictions
5. **dca_model_distribution.png** — which DCA model was selected per well
6. **eur_distribution.png** — EUR histogram with P10/P50/P90 annotations
7. **forecast_p10_p50_p90.png** — P10/P50/P90 production forecast curves

### Vertical wells

The pipeline also works for conventional vertical wells — see
[examples/forecasting_example.py](examples/forecasting_example.py) for
details.  Horizontal-specific features (lateral length, frac stages, etc.)
are automatically dropped when they're null/zero in the training data.

### More examples

See [examples/forecasting_example.py](examples/forecasting_example.py)
for complete usage examples including:
- Custom column mappings for non-Enverus data
- Vertical well forecasting
- Quick development runs with `max_wells`
- Exporting and reusing results
- Forcing oil vs gas stream detection

## Other Usage

```python
from utpgetools import hello_world

print(hello_world())
```

## Development

- Clone the repo
- Install dependencies (`pip install -e .`)
- Modify/add your tools in `utpgetools/`

## License

Custom Academic Use License