# Advanced Forecasting Metrics

This module implements comprehensive metrics for evaluating and improving forecasting models over time. The metrics focus on forecasting accuracy, calibration, coverage, and comparative performance.

## Overview

The metrics implementation includes:

- **Brier Score**: Measures the accuracy of probabilistic predictions
- **Calibration**: Assesses how well predicted probabilities match observed frequencies
- **Peer Score**: Compares model performance relative to other models
- **Coverage**: Evaluates how often actual outcomes fall within confidence intervals
- **Time-weighted metrics**: Emphasizes recent performance for adaptive learning

These metrics are used throughout the system to evaluate forecaster performance, create adaptive ensembles, and provide insights into model behavior.

## Key Features

### Core Metrics

- `brier_score_df`: Calculate Brier score for probabilistic forecasts
- `calibration_curve_df`: Generate calibration curves with binned predictions
- `calibration_error_df`: Quantify calibration error as mean squared difference
- `coverage_df`: Measure confidence interval coverage rate
- `peer_score_df`: Compare forecasters through relative performance
- `sharpness_df`: Assess decisiveness of predictions through variance
  
### Advanced Metrics

- `time_weighted_brier_score`: Apply exponential decay to weight recent forecasts more heavily
- `model_performance_over_time`: Track metrics across time windows
- `aggregate_metrics`: Generate comprehensive performance summary

## Usage Examples

### Basic Metrics Calculation

```python
from forecasting_tools.forecast_helpers.metrics import brier_score_df, calibration_curve_df

# Calculate Brier score for a dataframe of predictions
score = brier_score_df(df, prediction_col='prediction', outcome_col='outcome')

# Generate calibration curve
prob_pred, prob_true, bin_total = calibration_curve_df(df, n_bins=10)
```

### Advanced Evaluation

```python
from forecasting_tools.forecast_helpers.metrics import aggregate_metrics, time_weighted_brier_score

# Comprehensive performance evaluation
metrics_df = aggregate_metrics(df, model_col='model')

# Time-weighted performance (emphasizing recent forecasts)
tw_score = time_weighted_brier_score(df, half_life_days=30)
```

### Visualization with the Metrics Dashboard

The metrics are visualized through:

1. **Metrics Dashboard** (`3_Metrics_Dashboard.py`): Shows overall performance metrics
2. **Advanced Metrics Dashboard** (`6_Advanced_Metrics.py`): Provides detailed analysis
3. **Adaptive Ensemble** (`7_Adaptive_Ensemble.py`): Uses metrics for adaptive weighting

## Adaptive Ensemble Implementation

The `AdaptiveEnsembleForecaster` leverages these metrics to dynamically adjust forecaster weights based on historical performance:

- **Equal Weights**: Simple average of all forecasters
- **Dynamic Weights**: Weights updated based on time-weighted Brier scores
- **Stacking**: Meta-model learns optimal combinations from historical data

The ensemble also:
- Records historical performance for continuous improvement
- Applies calibration correction based on past data
- Generates confidence intervals using bootstrapping
- Provides detailed explanations of how forecasts were derived

## Integration

The metrics module is integrated with:

1. **Forecasting Pipeline**: Automatically tracks and records performance
2. **Ensemble Methods**: Powers adaptive weight adjustment
3. **Dashboards**: Visualizes performance for user understanding
4. **Active Learning**: Identifies areas for improvement

## Performance Considerations

- Time-weighted metrics require datetime information
- Calibration curves need sufficient data points in each bin (n_bins parameter)
- Peer scores are most meaningful with multiple diverse forecasters
- Coverage metrics require confidence interval data

## Future Directions

- Implementing proper scoring rules beyond Brier score (e.g., logarithmic scoring)
- Adding domain-specific calibration metrics
- Supporting multi-class and continuous forecast evaluation
- Expanding time-series specific metrics 