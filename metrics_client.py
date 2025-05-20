"""
Client library for the Forecasting Tools Metrics API.

This module provides a client for interacting with the Metrics API, 
making it easy to calculate metrics, manage datasets, and analyze results
from external applications.
"""

import json
import requests
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


class MetricsApiClient:
    """Client for the Forecasting Tools Metrics API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the Metrics API client.
        
        Args:
            base_url: Base URL of the Metrics API
        """
        self.base_url = base_url.rstrip("/")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the API.
        
        Returns:
            Health status information
        """
        response = requests.get(f"{self.base_url}/api/health")
        response.raise_for_status()
        return response.json()
    
    def calculate_metric(
        self,
        predictions: List[float],
        outcomes: List[float],
        metric_type: str = "brier_score",
        save_result: bool = False,
        dataset_name: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Calculate a metric for the given predictions and outcomes.
        
        Args:
            predictions: List of prediction probabilities
            outcomes: List of binary outcomes (0 or 1)
            metric_type: Type of metric to calculate (brier_score, calibration, coverage, peer_score)
            save_result: Whether to save the result
            dataset_name: Name for the dataset (if saving)
            metadata: Additional metadata for the calculation
            
        Returns:
            Metric result
        """
        if metadata is None:
            metadata = {}
            
        data = {
            "predictions": predictions,
            "outcomes": outcomes,
            "metric_type": metric_type,
            "save_result": save_result,
            "dataset_name": dataset_name,
            "metadata": metadata
        }
        
        response = requests.post(
            f"{self.base_url}/api/metrics/calculate",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def calculate_brier_score(
        self,
        predictions: List[float],
        outcomes: List[float],
        save_result: bool = False,
        dataset_name: Optional[str] = None
    ) -> float:
        """
        Calculate the Brier score for the given predictions and outcomes.
        
        Args:
            predictions: List of prediction probabilities
            outcomes: List of binary outcomes (0 or 1)
            save_result: Whether to save the result
            dataset_name: Name for the dataset (if saving)
            
        Returns:
            Brier score
        """
        result = self.calculate_metric(
            predictions=predictions,
            outcomes=outcomes,
            metric_type="brier_score",
            save_result=save_result,
            dataset_name=dataset_name
        )
        return result["value"]
    
    def calculate_calibration(
        self,
        predictions: List[float],
        outcomes: List[float],
        save_result: bool = False,
        dataset_name: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Calculate the calibration curve for the given predictions and outcomes.
        
        Args:
            predictions: List of prediction probabilities
            outcomes: List of binary outcomes (0 or 1)
            save_result: Whether to save the result
            dataset_name: Name for the dataset (if saving)
            
        Returns:
            Calibration curve data
        """
        result = self.calculate_metric(
            predictions=predictions,
            outcomes=outcomes,
            metric_type="calibration",
            save_result=save_result,
            dataset_name=dataset_name
        )
        return result["metadata"]["detailed_result"]
    
    def calculate_coverage(
        self,
        predictions: List[float],
        outcomes: List[float],
        confidence_intervals: List[List[float]],
        save_result: bool = False,
        dataset_name: Optional[str] = None
    ) -> float:
        """
        Calculate the coverage for the given predictions and outcomes.
        
        Args:
            predictions: List of prediction probabilities
            outcomes: List of binary outcomes (0 or 1)
            confidence_intervals: List of confidence intervals [lower, upper]
            save_result: Whether to save the result
            dataset_name: Name for the dataset (if saving)
            
        Returns:
            Coverage score
        """
        result = self.calculate_metric(
            predictions=predictions,
            outcomes=outcomes,
            metric_type="coverage",
            save_result=save_result,
            dataset_name=dataset_name,
            metadata={"confidence_intervals": confidence_intervals}
        )
        return result["value"]
    
    def calculate_peer_score(
        self,
        predictions: List[float],
        outcomes: List[float],
        model_names: List[str],
        save_result: bool = False,
        dataset_name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate peer scores for the given predictions and outcomes.
        
        Args:
            predictions: List of prediction probabilities
            outcomes: List of binary outcomes (0 or 1)
            model_names: List of model names corresponding to each prediction
            save_result: Whether to save the result
            dataset_name: Name for the dataset (if saving)
            
        Returns:
            Dictionary of peer scores by model
        """
        result = self.calculate_metric(
            predictions=predictions,
            outcomes=outcomes,
            metric_type="peer_score",
            save_result=save_result,
            dataset_name=dataset_name,
            metadata={"model_names": model_names}
        )
        return result["metadata"]["detailed_result"]
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets.
        
        Returns:
            List of dataset information
        """
        response = requests.get(f"{self.base_url}/api/metrics/datasets")
        response.raise_for_status()
        return response.json()
    
    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get a dataset by ID.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Dataset information
        """
        response = requests.get(f"{self.base_url}/api/metrics/datasets/{dataset_id}")
        response.raise_for_status()
        return response.json()
    
    def create_dataset(
        self,
        name: str,
        predictions: List[float],
        outcomes: List[float],
        confidence_intervals: Optional[List[List[float]]] = None,
        model_names: Optional[List[str]] = None,
        question_ids: Optional[List[str]] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Create a new dataset.
        
        Args:
            name: Name of the dataset
            predictions: List of prediction probabilities
            outcomes: List of binary outcomes (0 or 1)
            confidence_intervals: List of confidence intervals [lower, upper]
            model_names: List of model names corresponding to each prediction
            question_ids: List of question IDs corresponding to each prediction
            metadata: Additional metadata
            
        Returns:
            ID of the created dataset
        """
        if metadata is None:
            metadata = {}
            
        data = {
            "dataset_id": "",  # Will be generated by the API
            "name": name,
            "predictions": predictions,
            "outcomes": outcomes,
            "confidence_intervals": confidence_intervals,
            "model_names": model_names,
            "question_ids": question_ids,
            "metadata": metadata
        }
        
        response = requests.post(
            f"{self.base_url}/api/metrics/datasets",
            json=data
        )
        response.raise_for_status()
        return response.json()["dataset_id"]
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            True if successful
        """
        response = requests.delete(f"{self.base_url}/api/metrics/datasets/{dataset_id}")
        response.raise_for_status()
        return True
    
    def calculate_from_dataset(
        self,
        dataset_id: str,
        metric_type: str = "brier_score",
        save_result: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate a metric using data from a stored dataset.
        
        Args:
            dataset_id: ID of the dataset
            metric_type: Type of metric to calculate
            save_result: Whether to save the result
            
        Returns:
            Metric result
        """
        params = {
            "metric_type": metric_type,
            "save_result": "true" if save_result else "false"
        }
        
        response = requests.post(
            f"{self.base_url}/api/metrics/calculate_from_dataset/{dataset_id}",
            params=params
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    # Example usage
    client = MetricsApiClient()
    
    try:
        # Check API health
        health = client.health_check()
        print(f"API Status: {health['status']}")
        
        # Calculate Brier score
        predictions = [0.7, 0.3, 0.5, 0.9, 0.1]
        outcomes = [1, 0, 1, 1, 0]
        
        brier = client.calculate_brier_score(
            predictions=predictions,
            outcomes=outcomes,
            save_result=True,
            dataset_name="Example Dataset"
        )
        
        print(f"Brier Score: {brier}")
        
        # List datasets
        datasets = client.list_datasets()
        print(f"Available Datasets: {len(datasets)}")
        for ds in datasets:
            print(f"- {ds['name']} ({ds['dataset_id']})")
            
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        print("Make sure the API server is running at http://localhost:8000") 