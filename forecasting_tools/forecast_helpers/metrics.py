import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from typing import Tuple, List, Dict, Optional, Union


def brier_score_df(df: pd.DataFrame, prediction_col: str = 'prediction', outcome_col: str = 'outcome') -> float:
    """
    Calculate Brier score for a dataframe of predictions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing predictions and outcomes
    prediction_col : str
        Column name for probability predictions (between 0 and 1)
    outcome_col : str
        Column name for binary outcomes (0 or 1)
        
    Returns:
    --------
    float
        Brier score (lower is better)
    """
    if df.empty or prediction_col not in df.columns or outcome_col not in df.columns:
        return np.nan
    
    # Ensure predictions are between 0 and 1
    predictions = np.clip(df[prediction_col].values, 0, 1)
    outcomes = df[outcome_col].values
    
    # Calculate Brier score
    return brier_score_loss(outcomes, predictions)


def calibration_curve_df(df: pd.DataFrame, prediction_col: str = 'prediction', 
                       outcome_col: str = 'outcome', n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate calibration curve for model predictions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing predictions and outcomes
    prediction_col : str
        Column name for probability predictions
    outcome_col : str
        Column name for binary outcomes
    n_bins : int
        Number of bins for calibration curve
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        prob_pred: mean predicted probability in each bin
        prob_true: fraction of positive samples in each bin
        bin_total: number of samples in each bin
    """
    if df.empty or prediction_col not in df.columns or outcome_col not in df.columns:
        return np.array([]), np.array([]), np.array([])
    
    # Ensure predictions are between 0 and 1
    predictions = np.clip(df[prediction_col].values, 0, 1)
    outcomes = df[outcome_col].values
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(outcomes, predictions, n_bins=n_bins, strategy='quantile')
    
    # Calculate number of samples in each bin
    bin_edges = np.percentile(predictions, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(predictions, bin_edges[1:-1])
    bin_total = np.bincount(bin_indices, minlength=n_bins)
    
    return prob_pred, prob_true, bin_total


def coverage_df(df: pd.DataFrame, lower_col: str = 'lower', upper_col: str = 'upper', 
              outcome_col: str = 'outcome') -> float:
    """
    Calculate coverage of confidence intervals.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing confidence intervals and outcomes
    lower_col : str
        Column name for lower bound of confidence interval
    upper_col : str
        Column name for upper bound of confidence interval
    outcome_col : str
        Column name for actual outcomes
        
    Returns:
    --------
    float
        Coverage (fraction of outcomes within confidence intervals)
    """
    if df.empty or lower_col not in df.columns or upper_col not in df.columns or outcome_col not in df.columns:
        return np.nan
    
    # Calculate coverage
    within_interval = (df[outcome_col] >= df[lower_col]) & (df[outcome_col] <= df[upper_col])
    coverage = within_interval.mean()
    
    return coverage


def peer_score_df(df: pd.DataFrame, model_col: str = 'model', prediction_col: str = 'prediction',
                outcome_col: str = 'outcome') -> pd.DataFrame:
    """
    Calculate peer scores for models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing model predictions and outcomes
    model_col : str
        Column name for model identifiers
    prediction_col : str
        Column name for probability predictions
    outcome_col : str
        Column name for binary outcomes
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with peer scores for each model
    """
    if df.empty or model_col not in df.columns:
        return pd.DataFrame()
    
    models = df[model_col].unique()
    scores = []
    
    # Calculate Brier score for each model
    for model in models:
        model_df = df[df[model_col] == model]
        brier = brier_score_df(model_df, prediction_col, outcome_col)
        scores.append((model, brier))
    
    # Create DataFrame with scores
    score_df = pd.DataFrame(scores, columns=['Model', 'Brier Score'])
    
    # Calculate peer score as difference from mean
    score_df['Peer Score'] = score_df['Brier Score'] - score_df['Brier Score'].mean()
    
    # Sort by peer score (lower is better)
    return score_df.sort_values('Peer Score')


def calibration_error_df(df: pd.DataFrame, prediction_col: str = 'prediction', 
                        outcome_col: str = 'outcome', n_bins: int = 10) -> float:
    """
    Calculate calibration error (mean squared difference between predicted and true probabilities).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing predictions and outcomes
    prediction_col : str
        Column name for probability predictions
    outcome_col : str
        Column name for binary outcomes
    n_bins : int
        Number of bins for calibration curve
        
    Returns:
    --------
    float
        Calibration error (lower is better)
    """
    prob_pred, prob_true, bin_total = calibration_curve_df(df, prediction_col, outcome_col, n_bins)
    
    if len(prob_pred) == 0:
        return np.nan
    
    # Calculate calibration error (mean squared difference)
    cal_error = np.mean((prob_pred - prob_true) ** 2)
    
    return cal_error


def sharpness_df(df: pd.DataFrame, prediction_col: str = 'prediction') -> float:
    """
    Calculate sharpness of predictions (variance of predicted probabilities).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing predictions
    prediction_col : str
        Column name for probability predictions
        
    Returns:
    --------
    float
        Sharpness (higher is better for decisive predictions)
    """
    if df.empty or prediction_col not in df.columns:
        return np.nan
    
    # Calculate variance of predicted probabilities
    predictions = np.clip(df[prediction_col].values, 0, 1)
    sharpness = np.var(predictions)
    
    return sharpness


def time_weighted_brier_score(df: pd.DataFrame, prediction_col: str = 'prediction', 
                              outcome_col: str = 'outcome', time_col: str = 'timestamp',
                              half_life_days: float = 30.0) -> float:
    """
    Calculate time-weighted Brier score with exponential decay.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing predictions, outcomes, and timestamps
    prediction_col : str
        Column name for probability predictions
    outcome_col : str
        Column name for binary outcomes
    time_col : str
        Column name for timestamps
    half_life_days : float
        Half-life for exponential decay in days
        
    Returns:
    --------
    float
        Time-weighted Brier score (lower is better)
    """
    if df.empty or prediction_col not in df.columns or outcome_col not in df.columns or time_col not in df.columns:
        return np.nan
    
    # Convert timestamps to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort by time
    df = df.sort_values(time_col)
    
    # Calculate age in days
    latest_time = df[time_col].max()
    df['age_days'] = (latest_time - df[time_col]).dt.total_seconds() / (24 * 3600)
    
    # Calculate weights based on exponential decay
    half_life_seconds = half_life_days * 24 * 3600
    lambda_param = np.log(2) / half_life_days
    df['weight'] = np.exp(-lambda_param * df['age_days'])
    
    # Normalize weights to sum to 1
    df['weight'] = df['weight'] / df['weight'].sum()
    
    # Calculate squared errors
    df['squared_error'] = (df[prediction_col] - df[outcome_col]) ** 2
    
    # Calculate weighted Brier score
    weighted_brier = np.sum(df['weight'] * df['squared_error'])
    
    return weighted_brier


def model_performance_over_time(df: pd.DataFrame, model_col: str = 'model', prediction_col: str = 'prediction',
                              outcome_col: str = 'outcome', time_col: str = 'timestamp',
                              window_size: str = '30D') -> pd.DataFrame:
    """
    Calculate model performance metrics over time using rolling windows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing model predictions, outcomes, and timestamps
    model_col : str
        Column name for model identifiers
    prediction_col : str
        Column name for probability predictions
    outcome_col : str
        Column name for binary outcomes
    time_col : str
        Column name for timestamps
    window_size : str
        Size of rolling window (pandas offset string)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with performance metrics over time for each model
    """
    if df.empty or model_col not in df.columns or time_col not in df.columns:
        return pd.DataFrame()
    
    # Convert timestamps to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort by time
    df = df.sort_values(time_col)
    
    # Group by model and time
    grouped = df.groupby([pd.Grouper(key=time_col, freq=window_size), model_col])
    
    # Calculate metrics for each group
    results = []
    for (time, model), group in grouped:
        if len(group) >= 5:  # Require at least 5 samples for reliable metrics
            brier = brier_score_df(group, prediction_col, outcome_col)
            cal_error = calibration_error_df(group, prediction_col, outcome_col)
            sharp = sharpness_df(group, prediction_col)
            results.append({
                'Time': time,
                'Model': model,
                'Brier Score': brier,
                'Calibration Error': cal_error,
                'Sharpness': sharp,
                'Sample Size': len(group)
            })
    
    return pd.DataFrame(results)


def aggregate_metrics(df: pd.DataFrame, model_col: str = 'model', prediction_col: str = 'prediction',
                    outcome_col: str = 'outcome', lower_col: str = 'lower', upper_col: str = 'upper') -> pd.DataFrame:
    """
    Calculate comprehensive performance metrics for models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing model predictions, outcomes, and confidence intervals
    model_col : str
        Column name for model identifiers
    prediction_col : str
        Column name for probability predictions
    outcome_col : str
        Column name for binary outcomes
    lower_col : str
        Column name for lower bound of confidence interval
    upper_col : str
        Column name for upper bound of confidence interval
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with performance metrics for each model
    """
    if df.empty or model_col not in df.columns:
        return pd.DataFrame()
    
    models = df[model_col].unique()
    results = []
    
    for model in models:
        model_df = df[df[model_col] == model]
        
        # Calculate basic metrics
        brier = brier_score_df(model_df, prediction_col, outcome_col)
        cal_error = calibration_error_df(model_df, prediction_col, outcome_col)
        sharp = sharpness_df(model_df, prediction_col)
        
        # Calculate coverage if confidence intervals are available
        has_intervals = (lower_col in model_df.columns and upper_col in model_df.columns)
        coverage = coverage_df(model_df, lower_col, upper_col, outcome_col) if has_intervals else np.nan
        
        # Calculate log score
        predictions = np.clip(model_df[prediction_col].values, 0.001, 0.999)  # Avoid log(0)
        outcomes = model_df[outcome_col].values
        log_score = np.mean(outcomes * np.log(predictions) + (1 - outcomes) * np.log(1 - predictions))
        
        results.append({
            'Model': model,
            'Sample Size': len(model_df),
            'Brier Score': brier,
            'Log Score': log_score,
            'Calibration Error': cal_error,
            'Sharpness': sharp,
            'Coverage': coverage if has_intervals else None
        })
    
    # Calculate peer scores
    result_df = pd.DataFrame(results)
    result_df['Peer Score'] = result_df['Brier Score'] - result_df['Brier Score'].mean()
    
    # Sort by Brier score (lower is better)
    return result_df.sort_values('Brier Score') 