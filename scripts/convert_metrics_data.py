#!/usr/bin/env python3
"""
Script to convert existing backtest data to the metrics API format.

This script reads backtest results from the BacktestManager and converts them
to datasets in the metrics API storage format.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forecasting_tools.forecast_helpers.backtest_manager import BacktestManager
from api_server import MetricsDataset, MetricsStorage, API_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("convert_metrics_data")

def convert_backtest_to_metrics_datasets(
    backtest_data_dir: Optional[str] = None,
    metrics_data_dir: str = API_DATA_DIR,
    group_by: str = "model_name"
) -> List[str]:
    """
    Convert backtest data to metrics API datasets.
    
    Args:
        backtest_data_dir: Directory containing backtest data
        metrics_data_dir: Directory to save metrics datasets
        group_by: How to group data into datasets ("model_name" or "question_id")
        
    Returns:
        List of created dataset IDs
    """
    logger.info(f"Converting backtest data to metrics datasets")
    
    # Initialize backtest manager and metrics storage
    backtest_manager = BacktestManager(data_dir=backtest_data_dir)
    metrics_storage = MetricsStorage(data_dir=metrics_data_dir)
    
    # Check if results are available
    if backtest_manager.results_df.empty:
        logger.warning("No backtest results found")
        return []
    
    # Get the results DataFrame
    df = backtest_manager.results_df
    
    # Group by the specified column
    dataset_ids = []
    
    if group_by == "model_name":
        # Create one dataset per model
        for model_name, group in df.groupby("model_name"):
            if len(group) < 5:
                logger.warning(f"Skipping model {model_name}: not enough data (only {len(group)} samples)")
                continue
                
            # Extract data
            predictions = group["prediction"].tolist()
            outcomes = group["outcome"].tolist()
            
            # Extract confidence intervals if available
            confidence_intervals = None
            if "confidence_interval_lower" in group.columns and "confidence_interval_upper" in group.columns:
                confidence_intervals = list(zip(
                    group["confidence_interval_lower"].tolist(),
                    group["confidence_interval_upper"].tolist()
                ))
            
            # Extract question IDs
            question_ids = group["question_id"].astype(str).tolist()
            
            # Create dataset
            dataset = MetricsDataset(
                dataset_id="",  # Will be generated in save_dataset
                name=f"Model: {model_name}",
                predictions=predictions,
                outcomes=outcomes,
                confidence_intervals=confidence_intervals,
                model_names=[model_name] * len(predictions),
                question_ids=question_ids,
                created_at=datetime.now(),
                metadata={
                    "source": "backtest_manager",
                    "group_by": "model_name",
                    "model_name": model_name,
                    "question_texts": group["question_text"].tolist() if "question_text" in group.columns else None,
                    "categories": group["category"].tolist() if "category" in group.columns else None,
                    "difficulties": group["difficulty"].tolist() if "difficulty" in group.columns else None,
                }
            )
            
            # Save dataset
            dataset_id = metrics_storage.save_dataset(dataset)
            dataset_ids.append(dataset_id)
            logger.info(f"Created dataset {dataset_id} for model {model_name} with {len(predictions)} samples")
    
    elif group_by == "question_id":
        # Create one dataset per question
        for question_id, group in df.groupby("question_id"):
            if len(group) < 2:
                logger.warning(f"Skipping question {question_id}: not enough models (only {len(group)})")
                continue
                
            # Extract data
            predictions = group["prediction"].tolist()
            outcomes = group["outcome"].tolist()
            
            # Extract confidence intervals if available
            confidence_intervals = None
            if "confidence_interval_lower" in group.columns and "confidence_interval_upper" in group.columns:
                confidence_intervals = list(zip(
                    group["confidence_interval_lower"].tolist(),
                    group["confidence_interval_upper"].tolist()
                ))
            
            # Extract model names
            model_names = group["model_name"].tolist()
            
            # Get question text
            question_text = group["question_text"].iloc[0] if "question_text" in group.columns else f"Question {question_id}"
            
            # Create dataset
            dataset = MetricsDataset(
                dataset_id="",  # Will be generated in save_dataset
                name=f"Question: {question_text[:50]}{'...' if len(question_text) > 50 else ''}",
                predictions=predictions,
                outcomes=outcomes,
                confidence_intervals=confidence_intervals,
                model_names=model_names,
                question_ids=[str(question_id)] * len(predictions),
                created_at=datetime.now(),
                metadata={
                    "source": "backtest_manager",
                    "group_by": "question_id",
                    "question_id": str(question_id),
                    "question_text": question_text,
                    "category": group["category"].iloc[0] if "category" in group.columns else None,
                    "difficulty": group["difficulty"].iloc[0] if "difficulty" in group.columns else None,
                }
            )
            
            # Save dataset
            dataset_id = metrics_storage.save_dataset(dataset)
            dataset_ids.append(dataset_id)
            logger.info(f"Created dataset {dataset_id} for question {question_id} with {len(predictions)} samples")
    
    else:
        # Create one dataset for all data
        predictions = df["prediction"].tolist()
        outcomes = df["outcome"].tolist()
        
        # Extract confidence intervals if available
        confidence_intervals = None
        if "confidence_interval_lower" in df.columns and "confidence_interval_upper" in df.columns:
            confidence_intervals = list(zip(
                df["confidence_interval_lower"].tolist(),
                df["confidence_interval_upper"].tolist()
            ))
        
        # Extract model names and question IDs
        model_names = df["model_name"].tolist()
        question_ids = df["question_id"].astype(str).tolist()
        
        # Create dataset
        dataset = MetricsDataset(
            dataset_id="",  # Will be generated in save_dataset
            name=f"All backtest data",
            predictions=predictions,
            outcomes=outcomes,
            confidence_intervals=confidence_intervals,
            model_names=model_names,
            question_ids=question_ids,
            created_at=datetime.now(),
            metadata={
                "source": "backtest_manager",
                "group_by": "all",
                "num_models": df["model_name"].nunique(),
                "num_questions": df["question_id"].nunique(),
            }
        )
        
        # Save dataset
        dataset_id = metrics_storage.save_dataset(dataset)
        dataset_ids.append(dataset_id)
        logger.info(f"Created dataset {dataset_id} with all {len(predictions)} samples")
    
    return dataset_ids

def main():
    """Run the conversion script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert backtest data to metrics API datasets")
    parser.add_argument("--backtest-dir", help="Directory containing backtest data")
    parser.add_argument("--metrics-dir", default=API_DATA_DIR, help="Directory to save metrics datasets")
    parser.add_argument("--group-by", choices=["model_name", "question_id", "all"], default="model_name",
                        help="How to group data into datasets")
    
    args = parser.parse_args()
    
    dataset_ids = convert_backtest_to_metrics_datasets(
        backtest_data_dir=args.backtest_dir,
        metrics_data_dir=args.metrics_dir,
        group_by=args.group_by
    )
    
    print(f"Created {len(dataset_ids)} datasets:")
    for dataset_id in dataset_ids:
        print(f"- {dataset_id}")

if __name__ == "__main__":
    main() 