#!/usr/bin/env python3
"""
Test script for the Metrics API client.
This script creates test data and runs various API client functions to verify they work correctly.
"""

import time
import numpy as np
from metrics_client import MetricsApiClient

def test_brier_score_api():
    """Test calculating Brier score through the API."""
    print("\n--- Testing Brier Score API ---")
    client = MetricsApiClient()
    
    # Create test data
    predictions = [0.7, 0.3, 0.5, 0.9, 0.1]
    outcomes = [1, 0, 1, 1, 0]
    
    # Calculate Brier score
    try:
        brier = client.calculate_brier_score(
            predictions=predictions,
            outcomes=outcomes,
            save_result=True,
            dataset_name="Test Brier Dataset"
        )
        print(f"Brier Score via API: {brier:.4f}")
        return True
    except Exception as e:
        print(f"Error testing Brier score API: {e}")
        return False

def test_calibration_api():
    """Test calculating calibration curve through the API."""
    print("\n--- Testing Calibration API ---")
    client = MetricsApiClient()
    
    # Create test data
    predictions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 
                 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.9]
    outcomes = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 
              0, 0, 1, 1, 1, 1, 1, 1, 0, 1]
    
    try:
        # Calculate calibration
        calibration = client.calculate_calibration(
            predictions=predictions,
            outcomes=outcomes,
            save_result=True,
            dataset_name="Test Calibration Dataset"
        )
        
        print("Calibration Curve via API:")
        for i, (center, freq) in enumerate(zip(
            calibration["bin_centers"], 
            calibration["bin_frequencies"]
        )):
            if not np.isnan(freq):
                print(f"  Bin {i+1}: {center:.1f} -> {freq:.2f}")
        
        return True
    except Exception as e:
        print(f"Error testing calibration API: {e}")
        return False

def test_peer_score_api():
    """Test calculating peer score through the API."""
    print("\n--- Testing Peer Score API ---")
    client = MetricsApiClient()
    
    # Create test data for three models
    predictions = []
    outcomes = []
    model_names = []
    
    # Model A predictions (good model)
    predictions.extend([0.8, 0.2, 0.6, 0.9, 0.1])
    outcomes.extend([1, 0, 1, 1, 0])
    model_names.extend(["model_A"] * 5)
    
    # Model B predictions (average model)
    predictions.extend([0.6, 0.4, 0.6, 0.7, 0.3])
    outcomes.extend([1, 0, 1, 1, 0])
    model_names.extend(["model_B"] * 5)
    
    # Model C predictions (poor model)
    predictions.extend([0.5, 0.5, 0.5, 0.5, 0.5])
    outcomes.extend([1, 0, 1, 1, 0])
    model_names.extend(["model_C"] * 5)
    
    try:
        # Calculate peer scores
        peer_scores = client.calculate_peer_score(
            predictions=predictions,
            outcomes=outcomes,
            model_names=model_names,
            save_result=True,
            dataset_name="Test Peer Score Dataset"
        )
        
        print("Peer Scores via API:")
        for model, score in peer_scores.items():
            print(f"  {model}: {score:.4f}")
        
        return True
    except Exception as e:
        print(f"Error testing peer score API: {e}")
        return False

def test_dataset_management():
    """Test dataset management functions."""
    print("\n--- Testing Dataset Management ---")
    client = MetricsApiClient()
    
    # Create test data
    predictions = [0.7, 0.3, 0.5, 0.9, 0.1]
    outcomes = [1, 0, 1, 1, 0]
    confidence_intervals = [[0.6, 0.8], [0.2, 0.4], [0.3, 0.7], [0.8, 1.0], [0.0, 0.2]]
    
    try:
        # Create a dataset
        dataset_id = client.create_dataset(
            name="Test Dataset",
            predictions=predictions,
            outcomes=outcomes,
            confidence_intervals=confidence_intervals,
            model_names=["test_model"] * 5,
            question_ids=["q1", "q2", "q3", "q4", "q5"]
        )
        
        print(f"Created dataset with ID: {dataset_id}")
        
        # List datasets
        datasets = client.list_datasets()
        print(f"Number of datasets: {len(datasets)}")
        
        # Get the dataset
        dataset = client.get_dataset(dataset_id)
        print(f"Retrieved dataset: {dataset['name']}")
        
        # Calculate from dataset
        result = client.calculate_from_dataset(
            dataset_id=dataset_id,
            metric_type="brier_score",
            save_result=True
        )
        
        print(f"Calculated Brier score from dataset: {result['value']:.4f}")
        
        return True
    except Exception as e:
        print(f"Error testing dataset management: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Metrics API")
    print("=================")
    
    # Give the API server time to start
    print("Waiting for API server to start...")
    time.sleep(2)
    
    # Test health check endpoint
    client = MetricsApiClient()
    try:
        health = client.health_check()
        print(f"API Status: {health['status']}")
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print("Make sure the API server is running.")
        return
    
    # Run tests
    success = True
    success = success and test_brier_score_api()
    success = success and test_calibration_api()
    success = success and test_peer_score_api()
    success = success and test_dataset_management()
    
    print("\n--- Summary ---")
    if success:
        print("All API tests completed successfully!")
    else:
        print("Some API tests failed, check the output for details.")

if __name__ == "__main__":
    main() 