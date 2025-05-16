import logging
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

from forecasting_tools.ai_models.model_interfaces.forecaster_base import ForecasterBase
from forecasting_tools.ai_models.model_interfaces.forecaster_result import ForecastResult

logger = logging.getLogger(__name__)

class CalibrationSystem:
    """
    A system for tracking forecaster calibration and improving accuracy over time.
    
    This system:
    1. Tracks historical predictions and outcomes
    2. Calculates calibration metrics
    3. Generates calibration curves
    4. Provides recalibration functions to adjust raw forecasts
    """
    
    DEFAULT_CALIBRATION_PATH = "forecasting_tools/data/calibration_data.json"
    
    def __init__(
        self,
        calibration_data_path: Optional[str] = None,
        forecaster_name: str = "default",
        bin_count: int = 10,
        min_samples_for_calibration: int = 20,
        recalibration_method: str = "platt",  # "platt", "isotonic", or "none"
    ):
        """
        Initialize the calibration system.
        
        Args:
            calibration_data_path: Path to save/load calibration data
            forecaster_name: Name of the forecaster being calibrated
            bin_count: Number of bins for calibration curve (higher = more granular)
            min_samples_for_calibration: Minimum samples needed before recalibration is applied
            recalibration_method: Method used for recalibration
        """
        self.calibration_data_path = calibration_data_path or self.DEFAULT_CALIBRATION_PATH
        self.forecaster_name = forecaster_name
        self.bin_count = bin_count
        self.min_samples_for_calibration = min_samples_for_calibration
        self.recalibration_method = recalibration_method
        
        # Load or initialize calibration data
        self.calibration_data = self._load_calibration_data()
        
        # Initialize recalibration parameters
        self.recalibration_params = {}
        self._update_recalibration_params()
        
        logger.info(f"Initialized CalibrationSystem for {forecaster_name}")
    
    def _load_calibration_data(self) -> Dict:
        """Load or initialize calibration data."""
        try:
            if os.path.exists(self.calibration_data_path):
                with open(self.calibration_data_path, 'r') as f:
                    data = json.load(f)
                
                # Ensure our forecaster exists in the data
                if self.forecaster_name not in data:
                    data[self.forecaster_name] = {
                        "predictions": [],
                        "outcomes": [],
                        "metadata": [],
                        "last_updated": datetime.now().isoformat()
                    }
                
                return data
            else:
                # Create new calibration data structure
                data = {
                    self.forecaster_name: {
                        "predictions": [],
                        "outcomes": [],
                        "metadata": [],
                        "last_updated": datetime.now().isoformat()
                    }
                }
                
                # Save the initialized data
                self._save_calibration_data(data)
                return data
                
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
            # Return minimal default data
            return {
                self.forecaster_name: {
                    "predictions": [],
                    "outcomes": [],
                    "metadata": [],
                    "last_updated": datetime.now().isoformat()
                }
            }
    
    def _save_calibration_data(self, data=None):
        """Save calibration data to disk."""
        if data is None:
            data = self.calibration_data
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.calibration_data_path), exist_ok=True)
            
            with open(self.calibration_data_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved calibration data to {self.calibration_data_path}")
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")
    
    def record_prediction(self, question_id: str, prediction: float, metadata: Optional[Dict] = None):
        """
        Record a new prediction (before outcome is known).
        
        Args:
            question_id: Unique identifier for the question
            prediction: Probability forecast (0-1)
            metadata: Optional metadata about the prediction
        """
        if self.forecaster_name not in self.calibration_data:
            self.calibration_data[self.forecaster_name] = {
                "predictions": [],
                "outcomes": [],
                "metadata": [],
                "question_ids": [],
                "last_updated": datetime.now().isoformat()
            }
        
        forecaster_data = self.calibration_data[self.forecaster_name]
        
        # Check if this question_id already exists
        if "question_ids" in forecaster_data and question_id in forecaster_data["question_ids"]:
            # Update existing prediction
            idx = forecaster_data["question_ids"].index(question_id)
            forecaster_data["predictions"][idx] = prediction
            if metadata:
                forecaster_data["metadata"][idx] = metadata
        else:
            # Add new prediction
            forecaster_data["predictions"].append(prediction)
            forecaster_data["outcomes"].append(None)  # Outcome not known yet
            forecaster_data["metadata"].append(metadata or {})
            
            # Ensure question_ids list exists
            if "question_ids" not in forecaster_data:
                forecaster_data["question_ids"] = []
            
            forecaster_data["question_ids"].append(question_id)
        
        forecaster_data["last_updated"] = datetime.now().isoformat()
        self._save_calibration_data()
        
        logger.debug(f"Recorded prediction {prediction} for question {question_id}")
    
    def record_outcome(self, question_id: str, outcome: Union[bool, int]):
        """
        Record the actual outcome for a previously predicted question.
        
        Args:
            question_id: Unique identifier for the question
            outcome: Actual outcome (True/False or 1/0)
        """
        if self.forecaster_name not in self.calibration_data:
            logger.warning(f"No predictions found for forecaster {self.forecaster_name}")
            return
        
        forecaster_data = self.calibration_data[self.forecaster_name]
        
        # Convert outcome to binary (0/1)
        binary_outcome = 1 if outcome else 0
        
        # Find the question in the data
        if "question_ids" in forecaster_data and question_id in forecaster_data["question_ids"]:
            idx = forecaster_data["question_ids"].index(question_id)
            forecaster_data["outcomes"][idx] = binary_outcome
            
            forecaster_data["last_updated"] = datetime.now().isoformat()
            self._save_calibration_data()
            
            # Update recalibration parameters with new data
            self._update_recalibration_params()
            
            logger.debug(f"Recorded outcome {binary_outcome} for question {question_id}")
        else:
            logger.warning(f"No prediction found for question {question_id}")
    
    def get_calibration_curve(self) -> Tuple[List[float], List[float], List[int]]:
        """
        Calculate the calibration curve for this forecaster.
        
        Returns:
            Tuple of (bin_centers, bin_accuracies, bin_counts)
            - bin_centers: Midpoint of each probability bin
            - bin_accuracies: Actual frequency of positive outcomes in each bin
            - bin_counts: Number of samples in each bin
        """
        if self.forecaster_name not in self.calibration_data:
            logger.warning(f"No calibration data found for forecaster {self.forecaster_name}")
            return [], [], []
        
        forecaster_data = self.calibration_data[self.forecaster_name]
        
        # Filter to only include predictions with known outcomes
        predictions = []
        outcomes = []
        for i, outcome in enumerate(forecaster_data["outcomes"]):
            if outcome is not None:
                predictions.append(forecaster_data["predictions"][i])
                outcomes.append(outcome)
        
        if not predictions:
            logger.warning(f"No predictions with outcomes found for forecaster {self.forecaster_name}")
            return [], [], []
        
        # Create bins for the calibration curve
        bin_edges = np.linspace(0, 1, self.bin_count + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate bin indices for each prediction
        bin_indices = np.digitize(predictions, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.bin_count - 1)  # Ensure valid indices
        
        # Calculate accuracy in each bin
        bin_accuracies = []
        bin_counts = []
        
        for bin_idx in range(self.bin_count):
            mask = (bin_indices == bin_idx)
            count = np.sum(mask)
            bin_counts.append(count)
            
            if count > 0:
                accuracy = np.mean([outcomes[i] for i, is_in_bin in enumerate(mask) if is_in_bin])
                bin_accuracies.append(accuracy)
            else:
                bin_accuracies.append(None)  # No samples in this bin
        
        return bin_centers.tolist(), bin_accuracies, bin_counts
    
    def calculate_calibration_metrics(self) -> Dict:
        """
        Calculate various calibration metrics.
        
        Returns:
            Dictionary of calibration metrics
        """
        if self.forecaster_name not in self.calibration_data:
            logger.warning(f"No calibration data found for forecaster {self.forecaster_name}")
            return {
                "reliability": None,
                "resolution": None,
                "sharpness": None,
                "sample_count": 0
            }
        
        forecaster_data = self.calibration_data[self.forecaster_name]
        
        # Filter to only include predictions with known outcomes
        predictions = []
        outcomes = []
        for i, outcome in enumerate(forecaster_data["outcomes"]):
            if outcome is not None:
                predictions.append(forecaster_data["predictions"][i])
                outcomes.append(outcome)
        
        sample_count = len(predictions)
        if sample_count == 0:
            logger.warning(f"No predictions with outcomes found for forecaster {self.forecaster_name}")
            return {
                "reliability": None,
                "resolution": None,
                "sharpness": None,
                "sample_count": 0
            }
        
        # Calculate calibration metrics
        bin_centers, bin_accuracies, bin_counts = self.get_calibration_curve()
        
        # Calculate reliability (mean squared difference between predictions and outcomes)
        reliability = np.mean([(p - o)**2 for p, o in zip(predictions, outcomes)]) if predictions else None
        
        # Calculate sharpness (variance of predictions)
        sharpness = np.var(predictions) if predictions else None
        
        # Calculate resolution (how well predictions separate outcomes)
        # Higher is better, measures how different the conditional means are from the overall mean
        overall_mean = np.mean(outcomes) if outcomes else 0.5
        resolution = np.sum([count * ((acc - overall_mean)**2) for count, acc, center in 
                          zip(bin_counts, bin_accuracies, bin_centers) if acc is not None]) / sample_count
        
        # Calculate Brier skill score (BSS)
        # BSS = 1 - Brier_score / Brier_score_reference
        # Where reference is always predicting the mean outcome
        brier_score = np.mean([(p - o)**2 for p, o in zip(predictions, outcomes)]) if predictions else None
        reference_score = np.mean([(overall_mean - o)**2 for o in outcomes]) if outcomes else None
        
        brier_skill_score = 1 - (brier_score / reference_score) if brier_score is not None and reference_score > 0 else None
        
        return {
            "reliability": reliability,
            "resolution": resolution,
            "sharpness": sharpness,
            "brier_score": brier_score,
            "brier_skill_score": brier_skill_score,
            "sample_count": sample_count,
            "mean_outcome": overall_mean,
            "bin_centers": bin_centers,
            "bin_accuracies": bin_accuracies,
            "bin_counts": bin_counts
        }
    
    def _update_recalibration_params(self):
        """Update the recalibration parameters based on current data."""
        if self.forecaster_name not in self.calibration_data:
            return
        
        forecaster_data = self.calibration_data[self.forecaster_name]
        
        # Filter to only include predictions with known outcomes
        predictions = []
        outcomes = []
        for i, outcome in enumerate(forecaster_data["outcomes"]):
            if outcome is not None:
                predictions.append(forecaster_data["predictions"][i])
                outcomes.append(outcome)
        
        # Only update if we have enough samples for reliable calibration
        if len(predictions) < self.min_samples_for_calibration:
            logger.debug(f"Not enough samples ({len(predictions)}) for recalibration")
            self.recalibration_params = {
                "method": "none",
                "sample_count": len(predictions)
            }
            return
        
        # Calculate parameters for the selected recalibration method
        if self.recalibration_method == "platt":
            # Platt scaling: logistic regression with a single feature
            try:
                from sklearn.linear_model import LogisticRegression
                
                # Convert to the right shape for sklearn
                X = np.array(predictions).reshape(-1, 1)
                y = np.array(outcomes)
                
                # Fit logistic regression
                model = LogisticRegression(solver='lbfgs')
                model.fit(X, y)
                
                self.recalibration_params = {
                    "method": "platt",
                    "coef": model.coef_[0][0],
                    "intercept": model.intercept_[0],
                    "sample_count": len(predictions)
                }
                
                logger.debug(f"Updated Platt scaling parameters: {self.recalibration_params}")
            except Exception as e:
                logger.error(f"Error fitting Platt scaling: {e}")
                self.recalibration_params = {"method": "none"}
        
        elif self.recalibration_method == "isotonic":
            # Isotonic regression: non-parametric monotonic mapping
            try:
                from sklearn.isotonic import IsotonicRegression
                
                # Convert to the right shape for sklearn
                X = np.array(predictions)
                y = np.array(outcomes)
                
                # Fit isotonic regression
                model = IsotonicRegression(out_of_bounds='clip')
                model.fit(X, y)
                
                # Store the model directly since it's not parameterized simply
                self.recalibration_params = {
                    "method": "isotonic",
                    "model": model,
                    "sample_count": len(predictions)
                }
                
                logger.debug(f"Updated isotonic regression parameters")
            except Exception as e:
                logger.error(f"Error fitting isotonic regression: {e}")
                self.recalibration_params = {"method": "none"}
        
        else:
            # No recalibration
            self.recalibration_params = {"method": "none"}
    
    def recalibrate(self, prediction: float) -> float:
        """
        Recalibrate a raw forecast using the current calibration model.
        
        Args:
            prediction: Raw probability forecast (0-1)
            
        Returns:
            Recalibrated probability forecast (0-1)
        """
        # If not enough samples, return the original prediction
        if not self.recalibration_params or self.recalibration_params.get("method") == "none":
            return prediction
        
        # Apply the appropriate recalibration method
        method = self.recalibration_params.get("method")
        
        if method == "platt":
            # Apply logistic regression
            coef = self.recalibration_params.get("coef", 1.0)
            intercept = self.recalibration_params.get("intercept", 0.0)
            
            # Apply the logistic function
            logit = coef * prediction + intercept
            recalibrated = 1.0 / (1.0 + np.exp(-logit))
            
            return float(recalibrated)
        
        elif method == "isotonic":
            # Apply isotonic regression
            model = self.recalibration_params.get("model")
            if model is not None:
                try:
                    recalibrated = model.predict([prediction])[0]
                    return float(recalibrated)
                except Exception as e:
                    logger.error(f"Error applying isotonic recalibration: {e}")
                    return prediction
            else:
                return prediction
        
        # Default - return original
        return prediction
    
    def get_calibration_summary(self) -> Dict:
        """Get a summary of calibration data and metrics."""
        metrics = self.calculate_calibration_metrics()
        
        return {
            "forecaster_name": self.forecaster_name,
            "sample_count": metrics.get("sample_count", 0),
            "metrics": metrics,
            "recalibration_method": self.recalibration_params.get("method", "none"),
            "last_updated": self.calibration_data.get(self.forecaster_name, {}).get("last_updated", "never")
        }


class CalibratedForecaster(ForecasterBase):
    """
    A wrapper forecaster that applies calibration to another forecaster's predictions.
    
    This forecaster applies calibration transformations to improve accuracy over time.
    """
    
    def __init__(
        self,
        base_forecaster: ForecasterBase,
        calibration_data_path: Optional[str] = None,
        recalibration_method: str = "platt",
        min_samples_for_calibration: int = 20
    ):
        """
        Initialize the calibrated forecaster.
        
        Args:
            base_forecaster: The underlying forecaster to calibrate
            calibration_data_path: Path to save/load calibration data
            recalibration_method: Method for recalibration ("platt", "isotonic", or "none")
            min_samples_for_calibration: Minimum samples before applying recalibration
        """
        self.base_forecaster = base_forecaster
        self.model_name = f"Calibrated-{getattr(base_forecaster, 'model_name', 'Forecaster')}"
        
        # Initialize calibration system
        self.calibration = CalibrationSystem(
            calibration_data_path=calibration_data_path,
            forecaster_name=self.model_name,
            recalibration_method=recalibration_method,
            min_samples_for_calibration=min_samples_for_calibration
        )
        
        logger.info(f"Initialized {self.model_name} with {recalibration_method} recalibration")
    
    async def predict(self, question, context=None):
        """
        Return a calibrated probability forecast.
        """
        # Get the base prediction
        base_prediction = await self.base_forecaster.predict(question, context)
        
        # Recalibrate the prediction
        calibrated_prediction = self.calibration.recalibrate(base_prediction)
        
        # Record the prediction (before we know the outcome)
        question_id = str(hash(question.question_text))
        self.calibration.record_prediction(
            question_id=question_id,
            prediction=calibrated_prediction,
            metadata={
                "base_prediction": base_prediction,
                "question_text": question.question_text[:100]  # Store truncated text for reference
            }
        )
        
        logger.debug(f"Calibrated prediction: {base_prediction} -> {calibrated_prediction}")
        return calibrated_prediction
    
    async def explain(self, question, context=None):
        """
        Return an explanation with calibration information.
        """
        # Get the base explanation
        base_explanation = await self.base_forecaster.explain(question, context)
        
        # Get the calibration summary
        summary = self.calibration.get_calibration_summary()
        
        # Add calibration note to the explanation
        calibration_note = (
            f"\n\n## Calibration Information\n\n"
            f"This forecast has been calibrated using the {summary['recalibration_method']} method "
            f"based on {summary['sample_count']} historical predictions."
        )
        
        if summary['sample_count'] > 0:
            metrics = summary['metrics']
            calibration_note += f"\n\nHistorical Brier score: {metrics.get('brier_score', 'N/A'):.4f}"
            if metrics.get('brier_skill_score') is not None:
                calibration_note += f"\nBrier skill score: {metrics.get('brier_skill_score'):.4f}"
        
        explanation = base_explanation + calibration_note
        return explanation
    
    async def confidence_interval(self, question, context=None):
        """
        Return calibrated confidence intervals.
        """
        # Get base confidence interval
        base_interval = await self.base_forecaster.confidence_interval(question, context)
        
        # Calibrate both bounds
        calibrated_lower = self.calibration.recalibrate(base_interval[0])
        calibrated_upper = self.calibration.recalibrate(base_interval[1])
        
        # Ensure lower <= upper after calibration
        if calibrated_lower > calibrated_upper:
            calibrated_lower, calibrated_upper = calibrated_upper, calibrated_lower
        
        return (calibrated_lower, calibrated_upper)
    
    async def get_forecast_result(self, question, context=None):
        """
        Get a complete forecast result with calibration information.
        """
        # Get base prediction and confidence interval
        base_prediction = await self.base_forecaster.predict(question, context)
        base_explanation = await self.base_forecaster.explain(question, context)
        base_interval = await self.base_forecaster.confidence_interval(question, context)
        
        # Calibrate prediction and intervals
        calibrated_prediction = self.calibration.recalibrate(base_prediction)
        calibrated_lower = self.calibration.recalibrate(base_interval[0])
        calibrated_upper = self.calibration.recalibrate(base_interval[1])
        
        # Ensure lower <= upper after calibration
        if calibrated_lower > calibrated_upper:
            calibrated_lower, calibrated_upper = calibrated_upper, calibrated_lower
        
        # Get calibration summary for metadata
        summary = self.calibration.get_calibration_summary()
        
        # Add calibration note to the explanation
        calibration_note = (
            f"\n\n## Calibration Information\n\n"
            f"This forecast has been calibrated using the {summary['recalibration_method']} method "
            f"based on {summary['sample_count']} historical predictions."
        )
        
        if summary['sample_count'] > 0:
            metrics = summary['metrics']
            calibration_note += f"\n\nHistorical Brier score: {metrics.get('brier_score', 'N/A'):.4f}"
            if metrics.get('brier_skill_score') is not None:
                calibration_note += f"\nBrier skill score: {metrics.get('brier_skill_score'):.4f}"
            
            # Add before/after calibration comparison
            calibration_note += (
                f"\n\nUncalibrated prediction: {base_prediction:.4f}"
                f"\nCalibrated prediction: {calibrated_prediction:.4f}"
            )
        
        # Record the prediction
        question_id = str(hash(question.question_text))
        self.calibration.record_prediction(
            question_id=question_id,
            prediction=calibrated_prediction,
            metadata={
                "base_prediction": base_prediction,
                "question_text": question.question_text[:100],
                "calibration_sample_count": summary['sample_count']
            }
        )
        
        # Create the forecast result
        return ForecastResult(
            probability=calibrated_prediction,
            confidence_interval=(calibrated_lower, calibrated_upper),
            rationale=base_explanation + calibration_note,
            model_name=self.model_name,
            metadata={
                "calibration": {
                    "method": summary['recalibration_method'],
                    "sample_count": summary['sample_count'],
                    "base_prediction": base_prediction,
                    "base_interval": base_interval
                },
                "base_forecaster": getattr(self.base_forecaster, "model_name", "UnknownForecaster")
            }
        )
    
    def record_outcome(self, question, outcome: bool):
        """
        Record the actual outcome for a previously predicted question.
        
        Args:
            question: The question that was forecasted
            outcome: Whether the event actually occurred (True/False)
        """
        question_id = str(hash(question.question_text))
        self.calibration.record_outcome(question_id, outcome)
        logger.info(f"Recorded outcome {outcome} for question ID {question_id}")
    
    def get_calibration_stats(self):
        """Get calibration statistics and metrics."""
        return self.calibration.get_calibration_summary() 