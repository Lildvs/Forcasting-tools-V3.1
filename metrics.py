"""
Metrics module for evaluating forecasting performance.

This module provides functions to calculate common forecasting metrics including:
- Brier Score: Measures the accuracy of probabilistic predictions
- Calibration Curve: Assesses how well calibrated the predictions are
- Coverage: Evaluates how well confidence intervals cover actual outcomes
- Peer Score: Compares a forecaster's performance relative to peers

All functions support both individual predictions and pandas DataFrame inputs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Sequence


def brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Calculate the Brier score for a set of binary predictions.
    
    The Brier score is the mean squared error between predictions and outcomes.
    Lower scores indicate better forecasts (0 is perfect, 1 is worst).
    
    Args:
        predictions: Array of prediction probabilities (0 to 1)
        outcomes: Array of binary outcomes (0 or 1)
        
    Returns:
        Brier score (float between 0 and 1)
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    
    if len(predictions) != len(outcomes):
        raise ValueError("Predictions and outcomes must have the same length")
    
    if len(predictions) == 0:
        raise ValueError("At least one prediction is required")
    
    # Calculate mean squared error
    return np.mean((predictions - outcomes) ** 2)


def brier_score_df(df: pd.DataFrame, 
                  prediction_col: str = 'prediction', 
                  outcome_col: str = 'outcome') -> float:
    """
    Calculate Brier score from a pandas DataFrame.
    
    Args:
        df: DataFrame containing predictions and outcomes
        prediction_col: Name of the column containing prediction probabilities
        outcome_col: Name of the column containing binary outcomes
        
    Returns:
        Brier score (float between 0 and 1)
    """
    if prediction_col not in df.columns:
        raise ValueError(f"Column '{prediction_col}' not found in DataFrame")
    if outcome_col not in df.columns:
        raise ValueError(f"Column '{outcome_col}' not found in DataFrame")
    
    predictions = df[prediction_col].values
    outcomes = df[outcome_col].values
    
    return brier_score(predictions, outcomes)


def calibration_curve(predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate calibration curve for a set of binary predictions.
    
    Calibration curves plot the observed frequency against the predicted probability.
    Well-calibrated forecasts have points close to the diagonal.
    
    Args:
        predictions: Array of prediction probabilities (0 to 1)
        outcomes: Array of binary outcomes (0 or 1)
        n_bins: Number of bins to use for calibration curve
        
    Returns:
        Tuple of (bin_centers, bin_frequencies, bin_counts):
            - bin_centers: midpoint of each bin
            - bin_frequencies: observed frequency in each bin
            - bin_counts: number of samples in each bin
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    
    if len(predictions) != len(outcomes):
        raise ValueError("Predictions and outcomes must have the same length")
    
    if len(predictions) == 0:
        raise ValueError("At least one prediction is required")
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bins) - 1
    
    # Handle edge case where prediction = 1
    bin_indices[bin_indices == n_bins] = n_bins - 1
    
    # Calculate bin frequencies and counts
    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(len(predictions)):
        bin_idx = bin_indices[i]
        bin_sums[bin_idx] += outcomes[i]
        bin_counts[bin_idx] += 1
    
    # Calculate bin centers and frequencies
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_frequencies = np.zeros(n_bins)
    
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_frequencies[i] = bin_sums[i] / bin_counts[i]
    
    return bin_centers, bin_frequencies, bin_counts


def calibration_curve_df(df: pd.DataFrame,
                         prediction_col: str = 'prediction', 
                         outcome_col: str = 'outcome',
                         n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate calibration curve from a pandas DataFrame.
    
    Args:
        df: DataFrame containing predictions and outcomes
        prediction_col: Name of the column containing prediction probabilities
        outcome_col: Name of the column containing binary outcomes
        n_bins: Number of bins to use for calibration curve
        
    Returns:
        Tuple of (bin_centers, bin_frequencies, bin_counts)
    """
    if prediction_col not in df.columns:
        raise ValueError(f"Column '{prediction_col}' not found in DataFrame")
    if outcome_col not in df.columns:
        raise ValueError(f"Column '{outcome_col}' not found in DataFrame")
    
    predictions = df[prediction_col].values
    outcomes = df[outcome_col].values
    
    return calibration_curve(predictions, outcomes, n_bins)


def coverage(confidence_intervals: List[List[float]], outcomes: np.ndarray) -> float:
    """
    Calculate coverage for a set of confidence intervals.
    
    Coverage measures the fraction of outcomes that fall within the predicted confidence intervals.
    
    Args:
        confidence_intervals: List of confidence intervals [lower_bound, upper_bound]
        outcomes: Array of outcomes
        
    Returns:
        Coverage (fraction of outcomes within intervals)
    """
    outcomes = np.asarray(outcomes, dtype=float)
    
    if len(confidence_intervals) != len(outcomes):
        raise ValueError("Confidence intervals and outcomes must have the same length")
    
    if len(outcomes) == 0:
        raise ValueError("At least one outcome is required")
    
    # Count outcomes within confidence intervals
    in_interval_count = 0
    for i, outcome in enumerate(outcomes):
        lower, upper = confidence_intervals[i]
        if lower <= outcome <= upper:
            in_interval_count += 1
    
    # Calculate coverage
    return in_interval_count / len(outcomes)


def coverage_df(df: pd.DataFrame,
                lower_col: str = 'confidence_interval_lower',
                upper_col: str = 'confidence_interval_upper',
                outcome_col: str = 'outcome') -> float:
    """
    Calculate coverage from a pandas DataFrame.
    
    Args:
        df: DataFrame containing confidence intervals and outcomes
        lower_col: Name of the column containing lower bounds
        upper_col: Name of the column containing upper bounds
        outcome_col: Name of the column containing outcomes
        
    Returns:
        Coverage (fraction of outcomes within intervals)
    """
    required_cols = [lower_col, upper_col, outcome_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    confidence_intervals = [[df.loc[i, lower_col], df.loc[i, upper_col]] for i in df.index]
    outcomes = df[outcome_col].values
    
    return coverage(confidence_intervals, outcomes)


def peer_score(model_predictions: Dict[str, np.ndarray], outcomes: np.ndarray) -> Dict[str, float]:
    """
    Calculate peer scores for multiple forecasting models.
    
    Peer score measures how much better/worse a forecaster is compared to the average.
    Positive scores indicate better than average performance.
    
    Args:
        model_predictions: Dictionary mapping model names to arrays of predictions
        outcomes: Array of binary outcomes (0 or 1)
        
    Returns:
        Dictionary mapping model names to peer scores
    """
    # Validate inputs
    for model_name, predictions in model_predictions.items():
        if len(predictions) != len(outcomes):
            raise ValueError(f"Model {model_name} predictions and outcomes must have the same length")
    
    if not model_predictions:
        raise ValueError("At least one model is required")
    
    # Calculate Brier scores for each model
    brier_scores = {}
    for model_name, predictions in model_predictions.items():
        brier_scores[model_name] = brier_score(predictions, outcomes)
    
    # Calculate average Brier score
    avg_brier = sum(brier_scores.values()) / len(brier_scores)
    
    # Calculate peer scores (deviation from average, negative is better for Brier score)
    peer_scores = {}
    for model_name, score in brier_scores.items():
        # Calculate so that positive is better
        peer_scores[model_name] = avg_brier - score
    
    return peer_scores


def peer_score_df(df: pd.DataFrame,
                 model_col: str = 'model_name',
                 prediction_col: str = 'prediction',
                 outcome_col: str = 'outcome',
                 question_id_col: str = 'question_id') -> Dict[str, float]:
    """
    Calculate peer scores from a pandas DataFrame.
    
    Args:
        df: DataFrame containing model names, predictions, and outcomes
        model_col: Name of the column containing model names
        prediction_col: Name of the column containing prediction probabilities
        outcome_col: Name of the column containing binary outcomes
        question_id_col: Name of the column containing question IDs
        
    Returns:
        Dictionary mapping model names to peer scores
    """
    required_cols = [model_col, prediction_col, outcome_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Get unique models and questions
    models = df[model_col].unique()
    
    # If there's a question_id column, we'll organize by question
    if question_id_col in df.columns:
        questions = df[question_id_col].unique()
        
        # Create dictionary to hold predictions by model
        model_predictions = {model: [] for model in models}
        outcomes_list = []
        
        # For each question, get the prediction from each model
        for q in questions:
            q_df = df[df[question_id_col] == q]
            # Add the outcome once per question (they should all be the same)
            outcomes_list.append(q_df[outcome_col].iloc[0])
            
            # Get each model's prediction for this question
            for model in models:
                model_q_df = q_df[q_df[model_col] == model]
                if len(model_q_df) > 0:
                    # Use the first prediction if multiple exist
                    model_predictions[model].append(model_q_df[prediction_col].iloc[0])
                else:
                    # If a model doesn't have a prediction for this question,
                    # use 0.5 as a placeholder (uninformative prediction)
                    model_predictions[model].append(0.5)
        
        # Convert lists to arrays
        outcomes = np.array(outcomes_list)
        for model in model_predictions:
            model_predictions[model] = np.array(model_predictions[model])
    else:
        # Without question IDs, assume the dataframe is already organized correctly
        # with one row per model per question
        model_predictions = {}
        for model in models:
            model_df = df[df[model_col] == model]
            model_predictions[model] = model_df[prediction_col].values
        
        # Get one outcome per row (assuming they're organized by question)
        outcomes = df[outcome_col].values
    
    # Calculate peer scores
    return peer_score(model_predictions, outcomes)


# Additional utility functions

def calculate_metrics_summary(predictions: np.ndarray, outcomes: np.ndarray) -> Dict[str, float]:
    """
    Calculate a summary of metrics for a set of predictions.
    
    Args:
        predictions: Array of prediction probabilities (0 to 1)
        outcomes: Array of binary outcomes (0 or 1)
        
    Returns:
        Dictionary of metric names and values
    """
    bs = brier_score(predictions, outcomes)
    bin_centers, bin_freqs, bin_counts = calibration_curve(predictions, outcomes)
    
    # Calculate calibration error (mean absolute deviation from diagonal)
    cal_error = np.mean(np.abs(bin_centers - bin_freqs))
    
    return {
        "brier_score": bs,
        "calibration_error": cal_error,
        "sample_size": len(predictions)
    }


def mixing_brier_score(model_weights: Dict[str, float], 
                      model_predictions: Dict[str, np.ndarray], 
                      outcomes: np.ndarray) -> float:
    """
    Calculate Brier score for a weighted mix of models.
    
    Args:
        model_weights: Dictionary mapping model names to weights (should sum to 1)
        model_predictions: Dictionary mapping model names to arrays of predictions
        outcomes: Array of binary outcomes (0 or 1)
        
    Returns:
        Brier score for the ensemble
    """
    # Validate inputs
    weight_sum = sum(model_weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Model weights must sum to 1 (got {weight_sum})")
    
    # Calculate weighted ensemble predictions
    ensemble_predictions = np.zeros_like(outcomes, dtype=float)
    for model_name, weight in model_weights.items():
        if model_name not in model_predictions:
            raise ValueError(f"Model {model_name} not found in model_predictions")
        ensemble_predictions += weight * model_predictions[model_name]
    
    # Calculate Brier score for ensemble
    return brier_score(ensemble_predictions, outcomes) 