import logging
import json
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import asyncio

from forecasting_tools.ai_models.model_interfaces.forecaster_base import ForecasterBase
from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi

logger = logging.getLogger(__name__)

class BacktestManager:
    """
    Manages backtesting of forecasting models on historical data.
    
    This class:
    1. Retrieves resolved historical questions
    2. Runs multiple forecasting models on those questions
    3. Stores prediction results and actual outcomes
    4. Calculates performance metrics like Brier score, calibration, etc.
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        results_file: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the backtest manager.
        
        Args:
            data_dir: Directory for storing backtest data
            results_file: File name for storing backtest results
            cache_dir: Directory for caching historical questions
        """
        self.data_dir = data_dir or "forecasting_tools/data/backtest"
        self.results_file = results_file or "backtest_results.csv"
        self.cache_dir = cache_dir or "forecasting_tools/data/cache"
        self.results_path = os.path.join(self.data_dir, self.results_file)
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing results if available
        self.results_df = self._load_results()
        
        logger.info(f"Initialized BacktestManager with {len(self.results_df)} existing results")
    
    def _load_results(self) -> pd.DataFrame:
        """Load existing backtest results or create a new DataFrame."""
        if os.path.exists(self.results_path):
            try:
                df = pd.read_csv(self.results_path)
                return df
            except Exception as e:
                logger.error(f"Error loading backtest results: {e}")
        
        # Create a new DataFrame with the appropriate columns
        return pd.DataFrame({
            'question_id': [],
            'question_text': [],
            'model_name': [],
            'prediction': [],
            'confidence_interval_lower': [],
            'confidence_interval_upper': [],
            'outcome': [],
            'prediction_time': [],
            'resolution_time': [],
            'category': [],
            'difficulty': [],
            'tags': []
        })
    
    def _save_results(self):
        """Save backtest results to CSV file."""
        try:
            self.results_df.to_csv(self.results_path, index=False)
            logger.info(f"Saved backtest results to {self.results_path}")
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
    
    async def fetch_historical_questions(
        self,
        limit: int = 100,
        categories: Optional[List[str]] = None,
        min_resolution_date: Optional[str] = None,
        max_resolution_date: Optional[str] = None,
        cache_key: Optional[str] = None
    ) -> List[BinaryQuestion]:
        """
        Fetch resolved historical questions from Metaculus.
        
        Args:
            limit: Maximum number of questions to fetch
            categories: List of categories to filter questions by
            min_resolution_date: Minimum resolution date (YYYY-MM-DD)
            max_resolution_date: Maximum resolution date (YYYY-MM-DD)
            cache_key: Key for caching results (if None, no caching)
            
        Returns:
            List of resolved BinaryQuestion objects
        """
        # Check if we have a cached result
        if cache_key:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r') as f:
                        cache_data = json.load(f)
                    
                    questions = []
                    for q_data in cache_data:
                        try:
                            q = BinaryQuestion(**q_data)
                            questions.append(q)
                        except Exception as e:
                            logger.error(f"Error loading cached question: {e}")
                    
                    logger.info(f"Loaded {len(questions)} questions from cache")
                    return questions
                except Exception as e:
                    logger.error(f"Error loading from cache: {e}")
        
        # Fetch questions from Metaculus API
        try:
            # Initialize API
            api = MetaculusApi()
            
            # Fetch resolved binary questions
            questions = await api.get_resolved_questions(
                limit=limit,
                question_type="binary",
                categories=categories,
                min_resolution_date=min_resolution_date,
                max_resolution_date=max_resolution_date
            )
            
            logger.info(f"Fetched {len(questions)} historical questions from Metaculus")
            
            # Cache the results if a cache key is provided
            if cache_key and questions:
                try:
                    # Convert to JSON serializable format
                    cache_data = [q.model_dump() for q in questions]
                    
                    cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
                    with open(cache_path, 'w') as f:
                        json.dump(cache_data, f)
                    
                    logger.info(f"Cached {len(questions)} questions to {cache_path}")
                except Exception as e:
                    logger.error(f"Error caching questions: {e}")
            
            return questions
            
        except Exception as e:
            logger.error(f"Error fetching historical questions: {e}")
            return []
    
    async def run_backtest(
        self,
        forecasters: Dict[str, ForecasterBase],
        questions: Optional[List[BinaryQuestion]] = None,
        num_questions: int = 50,
        categories: Optional[List[str]] = None,
        append_results: bool = True
    ) -> pd.DataFrame:
        """
        Run a backtest using the provided forecasters on historical questions.
        
        Args:
            forecasters: Dictionary of {name: forecaster} to test
            questions: Optional list of questions to use (if None, fetches historical)
            num_questions: Number of questions to use if fetching historical
            categories: List of categories to filter questions by
            append_results: Whether to append to existing results
            
        Returns:
            DataFrame with backtest results
        """
        # Fetch historical questions if not provided
        if questions is None:
            questions = await self.fetch_historical_questions(
                limit=num_questions,
                categories=categories,
                cache_key=f"backtest_q{num_questions}"
            )
        
        if not questions:
            logger.error("No questions available for backtesting")
            return pd.DataFrame()
        
        logger.info(f"Running backtest with {len(forecasters)} forecasters on {len(questions)} questions")
        
        # Create a new results DataFrame
        results = []
        
        # Run each forecaster on each question
        for question in questions:
            question_id = question.id_of_question if hasattr(question, 'id_of_question') else hash(question.question_text)
            outcome = 1 if question.resolve_to_true else 0
            
            # Check if the question has been resolved
            if not hasattr(question, 'resolve_to_true'):
                logger.warning(f"Question {question_id} is not resolved, skipping")
                continue
            
            # Add tags and categories if available
            categories = []
            if hasattr(question, 'categories'):
                categories = question.categories
            
            tags = []
            if hasattr(question, 'tags'):
                tags = question.tags
                
            # Calculate difficulty (placeholder - can be refined)
            community_prediction = 0.5
            if hasattr(question, 'community_prediction'):
                community_prediction = question.community_prediction
            
            # Closer to 0.5 is more difficult
            difficulty = 1 - 2 * abs(community_prediction - 0.5)
            
            # Process each forecaster
            for name, forecaster in forecasters.items():
                try:
                    # Get prediction
                    prediction = await forecaster.predict(question)
                    
                    # Get confidence interval
                    ci_lower, ci_upper = await forecaster.confidence_interval(question)
                    
                    # Store result
                    results.append({
                        'question_id': question_id,
                        'question_text': question.question_text,
                        'model_name': name,
                        'prediction': prediction,
                        'confidence_interval_lower': ci_lower,
                        'confidence_interval_upper': ci_upper,
                        'outcome': outcome,
                        'prediction_time': datetime.now().isoformat(),
                        'resolution_time': question.resolves.isoformat() if hasattr(question, 'resolves') else "",
                        'category': ','.join(str(c) for c in categories),
                        'difficulty': difficulty,
                        'tags': ','.join(str(t) for t in tags)
                    })
                    
                    logger.debug(f"Processed forecaster {name} on question {question_id}")
                    
                except Exception as e:
                    logger.error(f"Error running forecaster {name} on question {question_id}: {e}")
        
        # Create a new DataFrame
        new_results_df = pd.DataFrame(results)
        
        # Combine with existing results or replace them
        if append_results and not self.results_df.empty:
            # Remove any duplicates (same forecaster on same question)
            combined_df = pd.concat([self.results_df, new_results_df])
            
            # Drop duplicates based on question_id and model_name
            self.results_df = combined_df.drop_duplicates(subset=['question_id', 'model_name'], keep='last')
        else:
            self.results_df = new_results_df
        
        # Save the results
        self._save_results()
        
        logger.info(f"Completed backtest with {len(forecasters)} forecasters on {len(questions)} questions")
        return self.results_df
    
    def calculate_metrics(self, min_predictions: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for each forecaster in the results.
        
        Args:
            min_predictions: Minimum number of predictions required to calculate metrics
            
        Returns:
            Dictionary of {forecaster_name: {metric_name: value}}
        """
        if self.results_df.empty:
            logger.warning("No results available to calculate metrics")
            return {}
        
        # Import metrics functions
        from metrics import brier_score, calibration_curve, coverage, peer_score
        
        # Group results by forecaster
        metrics = {}
        for model_name, group in self.results_df.groupby('model_name'):
            # Skip if not enough predictions
            if len(group) < min_predictions:
                logger.warning(f"Skipping metrics for {model_name}: not enough predictions ({len(group)} < {min_predictions})")
                continue
            
            try:
                # Extract predictions and outcomes
                predictions = group['prediction'].values
                outcomes = group['outcome'].values
                
                # Calculate Brier score
                brier = brier_score(predictions, outcomes)
                
                # Calculate calibration curve
                prob_pred, prob_true, bin_total = calibration_curve(predictions, outcomes)
                
                # Calculate calibration error (root mean squared error of calibration curve)
                valid_bins = bin_total > 0
                if np.any(valid_bins):
                    calibration_error = np.sqrt(np.mean((prob_pred[valid_bins] - prob_true[valid_bins]) ** 2))
                else:
                    calibration_error = np.nan
                
                # Calculate coverage (if confidence intervals available)
                if 'confidence_interval_lower' in group.columns and 'confidence_interval_upper' in group.columns:
                    intervals = list(zip(group['confidence_interval_lower'], group['confidence_interval_upper']))
                    coverage_score = coverage(intervals, outcomes)
                else:
                    coverage_score = np.nan
                
                # Calculate sharpness (average width of confidence interval)
                if 'confidence_interval_lower' in group.columns and 'confidence_interval_upper' in group.columns:
                    sharpness = np.mean(group['confidence_interval_upper'] - group['confidence_interval_lower'])
                else:
                    sharpness = np.nan
                
                # Calculate accuracy (correct predictions using threshold of 0.5)
                binary_predictions = (predictions >= 0.5).astype(int)
                accuracy = np.mean(binary_predictions == outcomes)
                
                # Calculate log score
                # Avoid log(0) by clipping
                eps = 1e-15
                p_clip = np.clip(predictions, eps, 1 - eps)
                log_score = np.mean(outcomes * np.log(p_clip) + (1 - outcomes) * np.log(1 - p_clip))
                
                # Store metrics
                metrics[model_name] = {
                    'brier_score': brier,
                    'calibration_error': calibration_error,
                    'coverage': coverage_score,
                    'sharpness': sharpness,
                    'accuracy': accuracy,
                    'log_score': log_score,
                    'sample_count': len(group)
                }
                
                logger.debug(f"Calculated metrics for {model_name}: Brier={brier:.4f}, Calibration Error={calibration_error:.4f}")
                
            except Exception as e:
                logger.error(f"Error calculating metrics for {model_name}: {e}")
        
        return metrics
    
    def calculate_peer_scores(self) -> Dict[str, float]:
        """
        Calculate peer scores for each forecaster.
        Peer score measures how much better/worse a forecaster is compared to the average.
        
        Returns:
            Dictionary of {forecaster_name: peer_score}
        """
        if self.results_df.empty:
            logger.warning("No results available to calculate peer scores")
            return {}
        
        try:
            # Import peer score function
            from metrics import peer_score_df
            
            # Use the peer_score_df function directly
            return peer_score_df(self.results_df)
            
        except Exception as e:
            logger.error(f"Error calculating peer scores: {e}")
            return {}
    
    def get_leaderboard(self, metric: str = 'brier_score', min_predictions: int = 10) -> pd.DataFrame:
        """
        Generate a leaderboard of forecasters sorted by the given metric.
        
        Args:
            metric: Metric to sort by ('brier_score', 'calibration_error', 'peer_score', etc.)
            min_predictions: Minimum number of predictions required to be included
            
        Returns:
            DataFrame with forecaster names and metrics
        """
        # Calculate all metrics
        all_metrics = self.calculate_metrics(min_predictions=min_predictions)
        
        if not all_metrics:
            return pd.DataFrame()
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
        
        # Add peer scores if available and requested
        if metric == 'peer_score' or 'peer_score' not in metrics_df.columns:
            peer_scores = self.calculate_peer_scores()
            if peer_scores:
                metrics_df['peer_score'] = pd.Series(peer_scores)
        
        # Rename the index
        metrics_df.index.name = 'model_name'
        metrics_df.reset_index(inplace=True)
        
        # Sort by the specified metric
        # For some metrics, lower is better
        if metric in ['brier_score', 'calibration_error', 'sharpness']:
            metrics_df = metrics_df.sort_values(by=metric, ascending=True)
        else:
            # For other metrics, higher is better
            metrics_df = metrics_df.sort_values(by=metric, ascending=False)
        
        return metrics_df
    
    def get_calibration_data(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get calibration data for plotting calibration curves.
        
        Args:
            model_name: Name of specific model to get data for (if None, gets all)
            
        Returns:
            Dictionary of calibration data for each forecaster
        """
        if self.results_df.empty:
            logger.warning("No results available to get calibration data")
            return {}
        
        from metrics import calibration_curve
        
        calibration_data = {}
        
        # Filter by model_name if provided
        if model_name:
            models_df = self.results_df[self.results_df['model_name'] == model_name]
            if models_df.empty:
                logger.warning(f"No data found for model {model_name}")
                return {}
            models_to_process = [model_name]
        else:
            models_df = self.results_df
            models_to_process = self.results_df['model_name'].unique()
        
        # Process each model
        for name in models_to_process:
            model_df = models_df[models_df['model_name'] == name]
            
            if len(model_df) < 10:  # Skip if too few predictions
                continue
            
            predictions = model_df['prediction'].values
            outcomes = model_df['outcome'].values
            
            # Calculate calibration curve
            prob_pred, prob_true, bin_total = calibration_curve(predictions, outcomes)
            
            # Store in dictionary
            calibration_data[name] = {
                'prob_pred': prob_pred.tolist(),
                'prob_true': prob_true.tolist(),
                'bin_total': bin_total.tolist()
            }
        
        return calibration_data
    
    def export_results(self, format: str = 'csv', path: Optional[str] = None) -> str:
        """
        Export backtest results to a file.
        
        Args:
            format: Output format ('csv' or 'json')
            path: Output file path (if None, uses default)
            
        Returns:
            Path to the exported file
        """
        if self.results_df.empty:
            logger.warning("No results available to export")
            return ""
        
        if path is None:
            if format == 'csv':
                path = os.path.join(self.data_dir, "backtest_results_export.csv")
            else:
                path = os.path.join(self.data_dir, "backtest_results_export.json")
        
        try:
            if format == 'csv':
                self.results_df.to_csv(path, index=False)
            elif format == 'json':
                self.results_df.to_json(path, orient='records')
            else:
                logger.error(f"Unsupported export format: {format}")
                return ""
            
            logger.info(f"Exported backtest results to {path}")
            return path
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return "" 