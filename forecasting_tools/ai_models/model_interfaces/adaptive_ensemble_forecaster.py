import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import os
import json
import datetime
from sklearn.linear_model import LogisticRegression
from copy import deepcopy

from forecasting_tools.data_models.base_types import ForecastQuestion, Forecast
from forecasting_tools.ai_models.model_interfaces.forecaster import Forecaster
from forecasting_tools.forecast_helpers.metrics import (
    brier_score_df, peer_score_df, calibration_error_df, time_weighted_brier_score
)


class AdaptiveEnsembleForecaster(Forecaster):
    """
    Ensemble forecaster that adapts weights based on historical performance.
    
    Features:
    - Combines multiple forecasters using adaptive weighting
    - Weights update automatically based on recent performance
    - Supports multiple weighting strategies
    - Provides rich explanation of how ensemble was formed
    - Tracks model performance over time for continuous improvement
    """
    
    def __init__(
        self,
        forecasters: List[Forecaster],
        ensemble_method: str = "dynamic_weights",
        window_size: int = 20,
        half_life_days: float = 30.0,
        performance_history_path: Optional[str] = None,
        calibration_correction: bool = True,
        name: str = "Adaptive Ensemble"
    ):
        """
        Initialize the adaptive ensemble forecaster.
        
        Parameters
        ----------
        forecasters : List[Forecaster]
            List of forecaster instances to ensemble
        ensemble_method : str
            Method to combine forecasts:
            - "equal_weights": Simple average of all forecasts
            - "static_weights": User-defined static weights
            - "dynamic_weights": Weights based on recent performance
            - "stacking": Meta-model learns optimal combination
        window_size : int
            Number of most recent questions to use for dynamic weights
        half_life_days : float
            Half-life parameter for time decay in dynamic weighting
        performance_history_path : Optional[str]
            Path to save/load performance history
        calibration_correction : bool
            Whether to apply calibration correction to forecasts
        name : str
            Name of the ensemble forecaster
        """
        self.forecasters = forecasters
        self.ensemble_method = ensemble_method
        self.window_size = window_size
        self.half_life_days = half_life_days
        self.calibration_correction = calibration_correction
        self.name = name
        
        # Default to equal weights
        self.weights = np.ones(len(forecasters)) / len(forecasters)
        
        # Initialize performance history
        self.performance_history_path = performance_history_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "../../../data/ensemble_performance_history.json"
        )
        self.performance_history = self._load_performance_history()
        
        # Initialize stacking model
        self.stacking_model = None
        if ensemble_method == "stacking":
            self.stacking_model = LogisticRegression(random_state=42)
        
        # Calibration correction parameters
        self.calibration_params = {}
        
    def forecast(self, question: ForecastQuestion) -> Forecast:
        """
        Generate a forecast by combining multiple forecasters.
        
        Parameters
        ----------
        question : ForecastQuestion
            The question to forecast
            
        Returns
        -------
        Forecast
            The ensemble forecast
        """
        # Get individual forecasts
        individual_forecasts = []
        for forecaster in self.forecasters:
            try:
                forecast = forecaster.forecast(question)
                individual_forecasts.append(forecast)
            except Exception as e:
                print(f"Error from {forecaster.__class__.__name__}: {e}")
        
        if not individual_forecasts:
            raise ValueError("All forecasters failed to provide forecasts")
        
        # Update weights if using dynamic weights
        if self.ensemble_method == "dynamic_weights":
            self._update_weights()
        
        # Combine forecasts based on ensemble method
        if self.ensemble_method == "stacking" and self.stacking_model is not None:
            # Extract features (individual predictions)
            X = np.array([f.prediction for f in individual_forecasts]).reshape(1, -1)
            
            # If model is trained, use it for prediction
            if hasattr(self.stacking_model, 'coef_'):
                prediction = self.stacking_model.predict_proba(X)[0, 1]
            else:
                # Fall back to equal weights if not trained
                prediction = np.mean([f.prediction for f in individual_forecasts])
        else:
            # For other methods, use weighted average
            prediction = np.sum([
                f.prediction * w for f, w in zip(individual_forecasts, self.weights)
            ])
        
        # Apply calibration correction if enabled
        if self.calibration_correction and self.calibration_params:
            prediction = self._apply_calibration_correction(prediction)
        
        # Calculate confidence interval
        lower, upper = self._calculate_confidence_interval(individual_forecasts)
        
        # Create explanation
        explanation = self._create_ensemble_explanation(individual_forecasts)
        
        # Create ensemble forecast
        ensemble_forecast = Forecast(
            question_id=question.id,
            model=self.name,
            prediction=prediction,
            lower=lower,
            upper=upper,
            explanation=explanation,
            timestamp=datetime.datetime.now(),
            metadata={
                "ensemble_method": self.ensemble_method,
                "forecaster_weights": {
                    forecaster.__class__.__name__: float(weight)
                    for forecaster, weight in zip(self.forecasters, self.weights)
                },
                "individual_forecasts": [
                    {
                        "model": f.model,
                        "prediction": f.prediction,
                        "explanation": f.explanation[:100] + "..." if len(f.explanation) > 100 else f.explanation
                    }
                    for f in individual_forecasts
                ]
            }
        )
        
        return ensemble_forecast
    
    def _update_weights(self) -> None:
        """
        Update weights based on recent performance.
        """
        # Check if we have performance history
        if not self.performance_history:
            return
        
        # Extract performance data for each forecaster
        forecaster_data = {}
        for forecaster in self.forecasters:
            forecaster_name = forecaster.__class__.__name__
            if forecaster_name in self.performance_history:
                # Get recent performance entries
                entries = self.performance_history[forecaster_name][-self.window_size:]
                
                if entries:
                    # Calculate time-weighted Brier score
                    df = pd.DataFrame(entries)
                    if len(df) > 0 and 'outcome' in df and 'prediction' in df and 'timestamp' in df:
                        # Convert timestamp strings to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        # Calculate time-weighted Brier score
                        weighted_score = time_weighted_brier_score(
                            df, 
                            half_life_days=self.half_life_days
                        )
                        
                        forecaster_data[forecaster_name] = weighted_score
        
        # If we have data for at least one forecaster
        if forecaster_data:
            # Convert scores to weights (lower Brier score = higher weight)
            scores = np.array([
                forecaster_data.get(forecaster.__class__.__name__, 0.5)  # Default to 0.5 if no data
                for forecaster in self.forecasters
            ])
            
            # Avoid division by zero
            if np.all(scores == 0):
                self.weights = np.ones(len(self.forecasters)) / len(self.forecasters)
            else:
                # Invert scores (lower is better for Brier score)
                inv_scores = 1.0 / (scores + 0.01)  # Add small constant to avoid division by zero
                
                # Normalize to get weights
                self.weights = inv_scores / np.sum(inv_scores)
    
    def _train_stacking_model(self) -> None:
        """
        Train a meta-model to combine forecasts optimally.
        """
        # Check if we have enough performance history
        if not self.performance_history:
            return
        
        # Collect training data
        X = []
        y = []
        
        # Get all question IDs with outcomes
        question_outcomes = {}
        for forecaster_name, entries in self.performance_history.items():
            for entry in entries:
                if 'question_id' in entry and 'outcome' in entry:
                    question_outcomes[entry['question_id']] = entry['outcome']
        
        # For each question with outcome
        for question_id, outcome in question_outcomes.items():
            # Collect predictions from all forecasters for this question
            question_preds = []
            
            for forecaster in self.forecasters:
                forecaster_name = forecaster.__class__.__name__
                if forecaster_name in self.performance_history:
                    # Find prediction for this question
                    for entry in self.performance_history[forecaster_name]:
                        if entry.get('question_id') == question_id:
                            question_preds.append(entry.get('prediction', 0.5))
                            break
                    else:
                        # No prediction found, use default
                        question_preds.append(0.5)
                else:
                    # No history for this forecaster
                    question_preds.append(0.5)
            
            # Only use questions where we have predictions from all forecasters
            if len(question_preds) == len(self.forecasters):
                X.append(question_preds)
                y.append(outcome)
        
        # Train model if we have enough data
        if len(X) >= 10:  # Require at least 10 samples
            X = np.array(X)
            y = np.array(y)
            
            # Train stacking model
            self.stacking_model.fit(X, y)
    
    def _apply_calibration_correction(self, prediction: float) -> float:
        """
        Apply calibration correction to raw prediction.
        
        Parameters
        ----------
        prediction : float
            Raw prediction probability
            
        Returns
        -------
        float
            Calibrated prediction
        """
        # Simple logistic calibration: p' = 1 / (1 + exp(-a*(p-b)))
        a = self.calibration_params.get('a', 1.0)
        b = self.calibration_params.get('b', 0.5)
        
        # Apply calibration
        calibrated = 1.0 / (1.0 + np.exp(-a * (prediction - b)))
        
        # Ensure prediction is between 0 and 1
        return np.clip(calibrated, 0.001, 0.999)
    
    def _fit_calibration_correction(self) -> None:
        """
        Fit calibration correction parameters from historical data.
        """
        # Collect all ensemble predictions and outcomes
        preds = []
        outcomes = []
        
        # Look for ensemble predictions in performance history
        if self.name in self.performance_history:
            for entry in self.performance_history[self.name]:
                if 'prediction' in entry and 'outcome' in entry:
                    preds.append(entry['prediction'])
                    outcomes.append(entry['outcome'])
        
        # If we have enough data, fit calibration curve
        if len(preds) >= 20:  # Require at least 20 samples
            from sklearn.linear_model import LogisticRegression
            
            # Transform predictions to logit scale
            X = np.array(preds).reshape(-1, 1)
            y = np.array(outcomes)
            
            # Fit logistic regression
            calibrator = LogisticRegression(C=1.0, solver='lbfgs')
            calibrator.fit(X, y)
            
            # Extract parameters
            a = calibrator.coef_[0][0]
            b = -calibrator.intercept_[0] / a if a != 0 else 0.5
            
            # Update calibration parameters
            self.calibration_params = {'a': float(a), 'b': float(b)}
    
    def _calculate_confidence_interval(
        self, individual_forecasts: List[Forecast], confidence_level: float = 0.90
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for ensemble prediction.
        
        Parameters
        ----------
        individual_forecasts : List[Forecast]
            Individual forecasts
        confidence_level : float
            Confidence level (0-1)
            
        Returns
        -------
        Tuple[float, float]
            Lower and upper bounds of confidence interval
        """
        # Extract predictions
        predictions = np.array([f.prediction for f in individual_forecasts])
        
        # If we have interval estimates from individual forecasters, use them
        if all(hasattr(f, 'lower') and hasattr(f, 'upper') for f in individual_forecasts):
            # Extract intervals
            lowers = np.array([f.lower for f in individual_forecasts])
            uppers = np.array([f.upper for f in individual_forecasts])
            
            # Weighted average of bounds
            lower = np.sum(lowers * self.weights)
            upper = np.sum(uppers * self.weights)
        else:
            # Bootstrap confidence interval
            n_bootstrap = 1000
            ensemble_preds = []
            
            for _ in range(n_bootstrap):
                # Resample predictions
                boot_indices = np.random.choice(len(predictions), len(predictions), replace=True)
                boot_preds = predictions[boot_indices]
                boot_weights = self.weights[boot_indices]
                boot_weights = boot_weights / np.sum(boot_weights)  # Renormalize
                
                # Calculate ensemble prediction
                ensemble_pred = np.sum(boot_preds * boot_weights)
                ensemble_preds.append(ensemble_pred)
            
            # Calculate confidence interval from bootstrap samples
            alpha = (1 - confidence_level) / 2.0
            lower = np.percentile(ensemble_preds, 100 * alpha)
            upper = np.percentile(ensemble_preds, 100 * (1 - alpha))
        
        # Ensure bounds are within [0, 1]
        lower = max(0.0, min(lower, 1.0))
        upper = max(0.0, min(upper, 1.0))
        
        return lower, upper
    
    def _create_ensemble_explanation(self, individual_forecasts: List[Forecast]) -> str:
        """
        Create explanation for ensemble forecast.
        
        Parameters
        ----------
        individual_forecasts : List[Forecast]
            Individual forecasts
            
        Returns
        -------
        str
            Explanation text
        """
        explanation = f"## Adaptive Ensemble Forecast ({self.ensemble_method.replace('_', ' ')})\n\n"
        
        # Add ensemble method explanation
        if self.ensemble_method == "equal_weights":
            explanation += "This forecast is a simple average of all forecasters.\n\n"
        elif self.ensemble_method == "static_weights":
            explanation += "This forecast uses predefined weights for each forecaster.\n\n"
        elif self.ensemble_method == "dynamic_weights":
            explanation += "This forecast uses weights based on recent forecaster performance.\n\n"
        elif self.ensemble_method == "stacking":
            explanation += "This forecast uses a meta-model that learns optimal combinations from past data.\n\n"
        
        # Show weights
        explanation += "### Forecaster Weights\n\n"
        for i, (forecaster, weight) in enumerate(zip(self.forecasters, self.weights)):
            explanation += f"- **{forecaster.__class__.__name__}**: {weight:.3f}\n"
        
        # Summary of individual forecasts
        explanation += "\n### Individual Forecasts\n\n"
        for i, forecast in enumerate(individual_forecasts):
            explanation += f"- **{forecast.model}**: {forecast.prediction:.3f}"
            if hasattr(forecast, 'lower') and hasattr(forecast, 'upper'):
                explanation += f" (CI: {forecast.lower:.3f}-{forecast.upper:.3f})\n"
            else:
                explanation += "\n"
        
        return explanation
    
    def _load_performance_history(self) -> Dict[str, List[Dict]]:
        """
        Load performance history from file.
        
        Returns
        -------
        Dict[str, List[Dict]]
            Performance history by forecaster name
        """
        if os.path.exists(self.performance_history_path):
            try:
                with open(self.performance_history_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading performance history: {e}")
        
        return {}
    
    def _save_performance_history(self) -> None:
        """
        Save performance history to file.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.performance_history_path), exist_ok=True)
        
        try:
            with open(self.performance_history_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving performance history: {e}")
    
    def record_forecast_outcome(
        self, forecast: Forecast, outcome: Union[bool, int, float]
    ) -> None:
        """
        Record forecast outcome in performance history.
        
        Parameters
        ----------
        forecast : Forecast
            The forecast
        outcome : Union[bool, int, float]
            The actual outcome
        """
        # Convert outcome to int (0 or 1)
        outcome_int = 1 if outcome else 0
        
        # Record ensemble forecast
        if self.name not in self.performance_history:
            self.performance_history[self.name] = []
        
        self.performance_history[self.name].append({
            'question_id': forecast.question_id,
            'prediction': forecast.prediction,
            'outcome': outcome_int,
            'timestamp': str(forecast.timestamp)
        })
        
        # Record individual forecasts
        if 'metadata' in forecast and 'individual_forecasts' in forecast.metadata:
            for individual in forecast.metadata['individual_forecasts']:
                model_name = individual['model']
                prediction = individual['prediction']
                
                if model_name not in self.performance_history:
                    self.performance_history[model_name] = []
                
                self.performance_history[model_name].append({
                    'question_id': forecast.question_id,
                    'prediction': prediction,
                    'outcome': outcome_int,
                    'timestamp': str(forecast.timestamp)
                })
        
        # Save updated history
        self._save_performance_history()
        
        # Update models if needed
        if self.ensemble_method == "stacking":
            self._train_stacking_model()
        
        if self.calibration_correction:
            self._fit_calibration_correction()
    
    def update_from_results_df(self, df: pd.DataFrame) -> None:
        """
        Update ensemble from a dataframe of forecast results.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns: model, question_id, prediction, outcome, timestamp
        """
        if df.empty:
            return
        
        # Ensure required columns
        required_cols = ['model', 'question_id', 'prediction', 'outcome']
        if not all(col in df.columns for col in required_cols):
            print(f"DataFrame missing required columns. Required: {required_cols}")
            return
        
        # Initialize performance history if needed
        if not self.performance_history:
            self.performance_history = {}
        
        # Process dataframe
        for model_name, group in df.groupby('model'):
            if model_name not in self.performance_history:
                self.performance_history[model_name] = []
            
            for _, row in group.iterrows():
                entry = {
                    'question_id': row['question_id'],
                    'prediction': float(row['prediction']),
                    'outcome': int(row['outcome']),
                    'timestamp': str(row.get('timestamp', datetime.datetime.now()))
                }
                self.performance_history[model_name].append(entry)
        
        # Save updated history
        self._save_performance_history()
        
        # Update models
        if self.ensemble_method == "stacking":
            self._train_stacking_model()
        
        if self.calibration_correction:
            self._fit_calibration_correction()
        
        # Update weights if using dynamic weights
        if self.ensemble_method == "dynamic_weights":
            self._update_weights() 