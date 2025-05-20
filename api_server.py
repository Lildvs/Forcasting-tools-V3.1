"""
FastAPI server for the Forecasting Tools API.

This module implements a REST API for accessing and managing metrics and forecasting data.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import metrics functionality
from metrics import (
    brier_score, brier_score_df,
    calibration_curve, calibration_curve_df,
    coverage, coverage_df,
    peer_score, peer_score_df
)

# Import data helpers
import pandas as pd
import numpy as np

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api_server")

# Create data directory for API storage
API_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_data")
os.makedirs(API_DATA_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Forecasting Tools API",
    description="API for accessing forecasting metrics and data",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Data Models ----

class MetricsResult(BaseModel):
    """Model for metrics results."""
    metric_name: str
    value: float
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BatchMetricsRequest(BaseModel):
    """Model for batch metrics calculation request."""
    predictions: List[float]
    outcomes: List[float]
    metric_type: str = "brier_score"
    save_result: bool = False
    dataset_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MetricsDataset(BaseModel):
    """Model for a metrics dataset."""
    dataset_id: str
    name: str
    predictions: List[float]
    outcomes: List[float]
    confidence_intervals: Optional[List[List[float]]] = None
    model_names: Optional[List[str]] = None
    question_ids: Optional[List[str]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MetricsStorage:
    """Class for storing and retrieving metrics data."""
    
    def __init__(self, data_dir: str = API_DATA_DIR):
        self.data_dir = data_dir
        self.datasets_dir = os.path.join(data_dir, "datasets")
        self.results_dir = os.path.join(data_dir, "results")
        
        # Create directories if they don't exist
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize datasets dictionary
        self.datasets: Dict[str, MetricsDataset] = {}
        
        # Load existing datasets
        self._load_datasets()
    
    def _load_datasets(self):
        """Load existing datasets from disk."""
        for filename in os.listdir(self.datasets_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.datasets_dir, filename), 'r') as f:
                        data = json.load(f)
                        dataset = MetricsDataset(**data)
                        self.datasets[dataset.dataset_id] = dataset
                        logger.info(f"Loaded dataset: {dataset.dataset_id}")
                except Exception as e:
                    logger.error(f"Error loading dataset {filename}: {e}")
    
    def save_dataset(self, dataset: MetricsDataset) -> str:
        """Save a dataset to disk."""
        dataset.last_updated = datetime.now()
        
        # Generate dataset_id if not provided
        if not dataset.dataset_id:
            dataset.dataset_id = f"dataset_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(dataset.predictions)}"
        
        # Update in-memory cache
        self.datasets[dataset.dataset_id] = dataset
        
        # Save to disk
        try:
            with open(os.path.join(self.datasets_dir, f"{dataset.dataset_id}.json"), 'w') as f:
                json.dump(dataset.dict(), f, indent=2, default=str)
            logger.info(f"Saved dataset: {dataset.dataset_id}")
            return dataset.dataset_id
        except Exception as e:
            logger.error(f"Error saving dataset {dataset.dataset_id}: {e}")
            raise
    
    def get_dataset(self, dataset_id: str) -> Optional[MetricsDataset]:
        """Get a dataset by ID."""
        return self.datasets.get(dataset_id)
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets."""
        return [
            {
                "dataset_id": ds.dataset_id,
                "name": ds.name,
                "size": len(ds.predictions),
                "created_at": ds.created_at,
                "last_updated": ds.last_updated,
            }
            for ds in self.datasets.values()
        ]
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset."""
        if dataset_id not in self.datasets:
            return False
        
        # Remove from in-memory cache
        del self.datasets[dataset_id]
        
        # Remove from disk
        try:
            os.remove(os.path.join(self.datasets_dir, f"{dataset_id}.json"))
            logger.info(f"Deleted dataset: {dataset_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting dataset {dataset_id}: {e}")
            return False
    
    def save_result(self, result: MetricsResult) -> None:
        """Save a metrics result."""
        result_id = f"result_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        try:
            with open(os.path.join(self.results_dir, f"{result_id}.json"), 'w') as f:
                json.dump(result.dict(), f, indent=2, default=str)
            logger.info(f"Saved result: {result_id}")
        except Exception as e:
            logger.error(f"Error saving result {result_id}: {e}")
            raise

# Initialize metrics storage
metrics_storage = MetricsStorage()

# ---- API Routes ----

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the Forecasting Tools API"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# ---- Metrics Endpoints ----

@app.post("/api/metrics/calculate")
async def calculate_metrics(request: BatchMetricsRequest):
    """Calculate metrics for a batch of predictions and outcomes."""
    if len(request.predictions) != len(request.outcomes):
        raise HTTPException(status_code=400, detail="Predictions and outcomes must have the same length")
    
    if not request.predictions:
        raise HTTPException(status_code=400, detail="No predictions provided")
    
    try:
        # Convert to numpy arrays
        predictions = np.array(request.predictions)
        outcomes = np.array(request.outcomes)
        
        # Calculate the requested metric
        result = None
        if request.metric_type == "brier_score":
            result = brier_score(predictions, outcomes)
        elif request.metric_type == "calibration":
            bin_centers, bin_freqs, bin_counts = calibration_curve(predictions, outcomes)
            # Convert numpy arrays to lists, handling NaN values
            bin_centers_list = bin_centers.tolist()
            bin_freqs_list = [float(f) if np.isfinite(f) else None for f in bin_freqs]
            bin_counts_list = bin_counts.tolist()
            
            # Calculate calibration error (mean absolute deviation from diagonal)
            # Filter out NaN values for calibration error calculation
            valid_indices = np.isfinite(bin_freqs)
            if np.any(valid_indices):
                cal_error = float(np.mean(np.abs(bin_centers[valid_indices] - bin_freqs[valid_indices])))
            else:
                cal_error = 0.0
                
            result = {
                "bin_centers": bin_centers_list,
                "bin_frequencies": bin_freqs_list,
                "bin_counts": bin_counts_list
            }
            
            # Set a float value for the MetricsResult
            metrics_value = cal_error
        elif request.metric_type == "coverage":
            # For coverage, we need confidence intervals
            if "confidence_intervals" not in request.metadata:
                raise HTTPException(status_code=400, detail="Confidence intervals required for coverage calculation")
            
            intervals = request.metadata["confidence_intervals"]
            result = coverage(intervals, outcomes)
        elif request.metric_type == "peer_score":
            # For peer score, we need model names
            if "model_names" not in request.metadata:
                raise HTTPException(status_code=400, detail="Model names required for peer score calculation")
            
            model_names = request.metadata["model_names"]
            # Organize predictions by model
            model_predictions = {}
            for i, model in enumerate(model_names):
                if model not in model_predictions:
                    model_predictions[model] = []
                model_predictions[model].append(predictions[i])
            
            # Convert to numpy arrays
            for model in model_predictions:
                model_predictions[model] = np.array(model_predictions[model])
            
            # Calculate peer scores
            peer_scores = peer_score(model_predictions, outcomes)
            result = peer_scores
            
            # For the MetricsResult value, use the average peer score
            if peer_scores:
                metrics_value = float(sum(peer_scores.values())) / len(peer_scores)
            else:
                metrics_value = 0.0
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported metric type: {request.metric_type}")
        
        # Create metrics result
        metrics_result = MetricsResult(
            metric_name=request.metric_type,
            value=result if isinstance(result, float) else (metrics_value if 'metrics_value' in locals() else 0.0),
            metadata={
                "data_points": len(predictions),
                "dataset_name": request.dataset_name,
                "detailed_result": result if not isinstance(result, float) else None,
                **request.metadata
            }
        )
        
        # Save the result if requested
        if request.save_result:
            metrics_storage.save_result(metrics_result)
        
        # Save the dataset if a name is provided
        if request.dataset_name:
            dataset = MetricsDataset(
                dataset_id="",  # Will be generated in save_dataset
                name=request.dataset_name,
                predictions=request.predictions,
                outcomes=request.outcomes,
                metadata=request.metadata
            )
            dataset_id = metrics_storage.save_dataset(dataset)
            metrics_result.metadata["dataset_id"] = dataset_id
        
        return metrics_result
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating metrics: {str(e)}")

@app.get("/api/metrics/datasets")
async def list_datasets():
    """List all datasets."""
    return metrics_storage.list_datasets()

@app.get("/api/metrics/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get a dataset by ID."""
    dataset = metrics_storage.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    return dataset

@app.delete("/api/metrics/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    success = metrics_storage.delete_dataset(dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    return {"message": f"Dataset {dataset_id} deleted successfully"}

@app.post("/api/metrics/datasets")
async def create_dataset(dataset: MetricsDataset):
    """Create a new dataset."""
    try:
        dataset_id = metrics_storage.save_dataset(dataset)
        return {"dataset_id": dataset_id, "message": "Dataset created successfully"}
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating dataset: {str(e)}")

@app.post("/api/metrics/calculate_from_dataset/{dataset_id}")
async def calculate_from_dataset(
    dataset_id: str, 
    metric_type: str = Query("brier_score", description="Type of metric to calculate"),
    save_result: bool = Query(False, description="Whether to save the result")
):
    """Calculate metrics from an existing dataset."""
    dataset = metrics_storage.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    
    try:
        # Create request from dataset
        request = BatchMetricsRequest(
            predictions=dataset.predictions,
            outcomes=dataset.outcomes,
            metric_type=metric_type,
            save_result=save_result,
            dataset_name=dataset.name,
            metadata={
                "dataset_id": dataset_id,
                "confidence_intervals": dataset.confidence_intervals,
                "model_names": dataset.model_names,
                "question_ids": dataset.question_ids,
                **dataset.metadata
            }
        )
        
        # Call calculate_metrics
        return await calculate_metrics(request)
    
    except Exception as e:
        logger.error(f"Error calculating metrics from dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating metrics from dataset: {str(e)}")

# ---- Main Function ----

def start():
    """Start the FastAPI server."""
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    start() 