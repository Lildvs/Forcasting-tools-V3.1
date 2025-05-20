#!/usr/bin/env python3
"""
Test script for verifying the metrics implementation.
This script creates test data and runs the various metrics functions to verify they work correctly.
"""

import numpy as np
import pandas as pd
from metrics import (
    brier_score, brier_score_df,
    calibration_curve, calibration_curve_df,
    coverage, coverage_df,
    peer_score, peer_score_df,
    mixing_brier_score, calculate_metrics_summary
)

def test_brier_score():
    """Test the Brier score calculation."""
    print("\n--- Testing Brier Score ---")
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
    
    assert np.isclose(score, df_score), "DataFrame and array calculations should match"
    return score

def test_calibration_curve():
    """Test the calibration curve calculation."""
    print("\n--- Testing Calibration Curve ---")
    # Create test data
    predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 
                          0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.9])
    outcomes = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 
                       0, 0, 1, 1, 1, 1, 1, 1, 0, 1])
    
    # Calculate calibration curve with 5 bins
    bin_centers, bin_freqs, bin_counts = calibration_curve(predictions, outcomes, n_bins=5)
    
    print("Calibration Curve:")
    print("Bin Centers:", bin_centers)
    print("Bin Frequencies:", bin_freqs)
    print("Bin Counts:", bin_counts)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'prediction': predictions,
        'outcome': outcomes
    })
    
    # Calculate calibration curve using DataFrame
    df_bin_centers, df_bin_freqs, df_bin_counts = calibration_curve_df(df, n_bins=5)
    
    # Verify results match
    assert np.allclose(bin_centers, df_bin_centers), "DataFrame and array bin centers should match"
    assert np.allclose(bin_freqs, df_bin_freqs), "DataFrame and array bin frequencies should match"
    assert np.allclose(bin_counts, df_bin_counts), "DataFrame and array bin counts should match"
    
    return bin_freqs

def test_coverage():
    """Test the coverage calculation."""
    print("\n--- Testing Coverage ---")
    # Create test data
    outcomes = np.array([0.3, 0.5, 0.7, 0.9, 0.2])
    confidence_intervals = [
        [0.2, 0.4],  # Covers outcome
        [0.6, 0.8],  # Doesn't cover outcome
        [0.6, 0.8],  # Covers outcome
        [0.8, 1.0],  # Covers outcome
        [0.1, 0.3]   # Covers outcome
    ]
    
    # Calculate coverage
    cov = coverage(confidence_intervals, outcomes)
    print(f"Coverage: {cov:.4f}")
    
    # Create a DataFrame
    df = pd.DataFrame({
        'confidence_interval_lower': [interval[0] for interval in confidence_intervals],
        'confidence_interval_upper': [interval[1] for interval in confidence_intervals],
        'outcome': outcomes
    })
    
    # Calculate coverage using DataFrame
    df_cov = coverage_df(df)
    print(f"Coverage (DataFrame): {df_cov:.4f}")
    
    assert np.isclose(cov, df_cov), "DataFrame and array calculations should match"
    return cov

def test_peer_score():
    """Test the peer score calculation."""
    print("\n--- Testing Peer Score ---")
    # Create test data for three models
    outcomes = np.array([1, 0, 1, 1, 0])
    
    model_predictions = {
        "model_A": np.array([0.8, 0.2, 0.6, 0.9, 0.1]),  # Good model
        "model_B": np.array([0.6, 0.4, 0.6, 0.7, 0.3]),  # Average model
        "model_C": np.array([0.5, 0.5, 0.5, 0.5, 0.5])   # Poor model
    }
    
    # Calculate peer scores
    scores = peer_score(model_predictions, outcomes)
    print("Peer Scores:")
    for model, score in scores.items():
        print(f"  {model}: {score:.4f}")
    
    # Create a DataFrame
    data = []
    for model, preds in model_predictions.items():
        for i, (pred, outcome) in enumerate(zip(preds, outcomes)):
            data.append({
                'model_name': model,
                'prediction': pred,
                'outcome': outcome,
                'question_id': i  # Simulate different questions
            })
    
    df = pd.DataFrame(data)
    
    # Calculate peer scores using DataFrame
    df_scores = peer_score_df(df)
    print("Peer Scores (DataFrame):")
    for model, score in df_scores.items():
        print(f"  {model}: {score:.4f}")
    
    # Verify results (approximately)
    for model in scores.keys():
        assert np.isclose(scores[model], df_scores[model], atol=0.1), f"DataFrame and array calculations for {model} should be close"
    
    return scores

def test_mixing_brier_score():
    """Test the mixing Brier score calculation."""
    print("\n--- Testing Mixing Brier Score ---")
    # Create test data
    outcomes = np.array([1, 0, 1, 1, 0])
    
    model_predictions = {
        "model_A": np.array([0.8, 0.2, 0.6, 0.9, 0.1]),  # Good model
        "model_B": np.array([0.6, 0.4, 0.6, 0.7, 0.3]),  # Average model
        "model_C": np.array([0.5, 0.5, 0.5, 0.5, 0.5])   # Poor model
    }
    
    # Try different weights
    weights = {
        "Equal": {"model_A": 1/3, "model_B": 1/3, "model_C": 1/3},
        "Favor A": {"model_A": 0.6, "model_B": 0.3, "model_C": 0.1},
        "Favor B": {"model_A": 0.2, "model_B": 0.6, "model_C": 0.2},
    }
    
    # Calculate individual model scores first
    individual_scores = {}
    for model, preds in model_predictions.items():
        individual_scores[model] = brier_score(preds, outcomes)
    
    print("Individual Model Brier Scores:")
    for model, score in individual_scores.items():
        print(f"  {model}: {score:.4f}")
    
    # Calculate ensemble scores
    print("\nEnsemble Brier Scores:")
    for name, weight_dict in weights.items():
        score = mixing_brier_score(weight_dict, model_predictions, outcomes)
        print(f"  {name} weights: {score:.4f}")
    
    return individual_scores

def main():
    """Run all tests."""
    print("Testing Metrics Implementation")
    print("=============================")
    
    brier = test_brier_score()
    calibration = test_calibration_curve()
    cov = test_coverage()
    peer = test_peer_score()
    mixing = test_mixing_brier_score()
    
    print("\n--- Summary ---")
    print("All tests completed successfully!")

if __name__ == "__main__":
    main() 