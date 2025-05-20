"""
Test script for the metrics module.

This script creates test data and runs the various metrics functions to verify
that they work correctly.
"""

import numpy as np
import pandas as pd
from metrics import (
    brier_score, brier_score_df,
    calibration_curve, calibration_curve_df,
    coverage, coverage_df,
    peer_score, peer_score_df
)

def test_brier_score():
    """Test the brier score calculation."""
    # Create test data
    predictions = np.array([0.7, 0.3, 0.5, 0.9, 0.1])
    outcomes = np.array([1, 0, 1, 1, 0])
    
    # Calculate Brier score
    score = brier_score(predictions, outcomes)
    print(f"Brier Score: {score:.4f}")
    
    # Create a DataFrame
    df = pd.DataFrame({
        'prediction': predictions,
        'outcome': outcomes
    })
    
    # Calculate Brier score using DataFrame
    df_score = brier_score_df(df)
    print(f"Brier Score (DataFrame): {df_score:.4f}")
    
    # Verify they match
    assert np.isclose(score, df_score)
    
    return score

def test_calibration_curve():
    """Test the calibration curve calculation."""
    # Create test data with more samples
    np.random.seed(42)
    n_samples = 100
    
    # Generate predictions (biased toward overconfidence)
    predictions = np.random.beta(2, 2, n_samples)
    
    # Generate outcomes with some relationship to predictions
    # but deliberately miscalibrated
    probs = predictions * 0.8 + 0.1  # Compress range to make overconfident
    outcomes = (np.random.random(n_samples) < probs).astype(int)
    
    # Calculate calibration curve
    bin_centers, bin_freqs, bin_counts = calibration_curve(predictions, outcomes)
    
    print("\nCalibration Curve:")
    for center, freq, count in zip(bin_centers, bin_freqs, bin_counts):
        if not np.isnan(freq):
            print(f"  Bin {center:.2f}: Predicted={center:.2f}, Actual={freq:.2f}, Count={int(count)}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'prediction': predictions,
        'outcome': outcomes
    })
    
    # Calculate calibration using DataFrame
    df_centers, df_freqs, df_counts = calibration_curve_df(df)
    
    # Verify they match
    assert np.array_equal(bin_centers, df_centers)
    assert np.array_equal(bin_freqs[~np.isnan(bin_freqs)], df_freqs[~np.isnan(df_freqs)])
    assert np.array_equal(bin_counts, df_counts)
    
    # Calculate calibration error
    valid_bins = ~np.isnan(bin_freqs) & (bin_counts > 0)
    if np.any(valid_bins):
        calibration_error = np.sqrt(np.mean((bin_centers[valid_bins] - bin_freqs[valid_bins]) ** 2))
        print(f"Calibration Error: {calibration_error:.4f}")
    
    return bin_centers, bin_freqs, bin_counts

def test_coverage():
    """Test the coverage calculation."""
    # Create test data
    predictions = np.array([0.7, 0.3, 0.5, 0.9, 0.1])
    lower_bounds = np.array([0.6, 0.2, 0.3, 0.8, 0.0])
    upper_bounds = np.array([0.8, 0.4, 0.7, 1.0, 0.2])
    outcomes = np.array([0.7, 0.5, 0.6, 0.9, 0.3])
    
    # Create intervals
    intervals = list(zip(lower_bounds, upper_bounds))
    
    # Calculate coverage
    cov = coverage(intervals, outcomes)
    print(f"\nCoverage: {cov:.2f}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'prediction': predictions,
        'lower': lower_bounds,
        'upper': upper_bounds,
        'outcome': outcomes
    })
    
    # Calculate coverage using DataFrame
    df_cov = coverage_df(df)
    print(f"Coverage (DataFrame): {df_cov:.2f}")
    
    # Verify they match
    assert np.isclose(cov, df_cov)
    
    return cov

def test_peer_score():
    """Test the peer score calculation."""
    # Create test data for 3 forecasters on 5 questions
    forecasters = ['model_A', 'model_B', 'model_C']
    questions = ['q1', 'q2', 'q3', 'q4', 'q5']
    
    # Different quality forecasters
    # A: Good (0.8)
    # B: Average (0.5)
    # C: Poor (0.2)
    predictions = {
        'model_A': np.array([0.8, 0.2, 0.9, 0.7, 0.1]),  # Good forecaster
        'model_B': np.array([0.6, 0.4, 0.7, 0.6, 0.3]),  # Medium forecaster
        'model_C': np.array([0.4, 0.6, 0.5, 0.4, 0.5])   # Poor forecaster
    }
    
    # True outcomes
    outcomes = np.array([1, 0, 1, 1, 0])
    
    # Calculate peer scores
    scores = peer_score(predictions, outcomes)
    
    print("\nPeer Scores:")
    for model, score in scores.items():
        print(f"  {model}: {score:.4f}")
    
    # Create DataFrame in the format used by the application
    data = []
    for q_idx, question in enumerate(questions):
        for model in forecasters:
            data.append({
                'question_id': question,
                'model_name': model,
                'prediction': predictions[model][q_idx],
                'outcome': outcomes[q_idx]
            })
    
    df = pd.DataFrame(data)
    
    # Calculate peer scores using DataFrame
    df_scores = peer_score_df(df)
    
    print("\nPeer Scores (DataFrame):")
    for model, score in df_scores.items():
        print(f"  {model}: {score:.4f}")
    
    # Verify they match (within floating point precision)
    for model in forecasters:
        assert np.isclose(scores[model], df_scores[model]), f"Mismatch for {model}"
    
    return scores

if __name__ == "__main__":
    print("Testing Metrics Module")
    print("=====================\n")
    
    # Run tests
    test_brier_score()
    test_calibration_curve()
    test_coverage()
    test_peer_score()
    
    print("\nAll tests completed successfully!") 