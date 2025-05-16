import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

from forecasting_tools.ai_models.model_interfaces.forecaster_base import ForecasterBase
from forecasting_tools.ai_models.model_interfaces.forecaster_result import ForecastResult
from forecasting_tools.ai_models.model_interfaces.active_learning_manager import ActiveLearningManager

logger = logging.getLogger(__name__)

class EnsembleForecaster(ForecasterBase):
    """
    An ensemble forecaster that combines predictions from multiple forecasters.
    
    Supports different ensemble methods:
    - Simple average
    - Weighted average
    - Dynamic weighting (based on past performance)
    """
    
    def __init__(
        self, 
        forecasters: List[ForecasterBase],
        weights: Optional[List[float]] = None,
        ensemble_method: str = "weighted_average",
        confidence_interval_method: str = "bootstrapping",
        use_active_learning: bool = True,
        active_learning_manager: Optional[ActiveLearningManager] = None
    ):
        """
        Initialize the ensemble forecaster.
        
        Args:
            forecasters: List of forecaster instances
            weights: Optional weights for each forecaster (must sum to 1)
            ensemble_method: Method for combining forecasts ("simple_average", "weighted_average")
            confidence_interval_method: Method for calculating confidence intervals 
                                      ("bootstrapping", "variance_propagation")
            use_active_learning: Whether to use active learning for high-uncertainty predictions
            active_learning_manager: Custom ActiveLearningManager instance (optional)
        """
        self.forecasters = forecasters
        
        # Initialize weights (equal if not provided)
        if weights is None:
            self.weights = [1.0 / len(forecasters)] * len(forecasters)
        else:
            if len(weights) != len(forecasters):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of forecasters ({len(forecasters)})")
            
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        self.ensemble_method = ensemble_method
        self.confidence_interval_method = confidence_interval_method
        self.model_name = "EnsembleForecaster"
        
        # Active learning setup
        self.use_active_learning = use_active_learning
        self.active_learning_manager = active_learning_manager or ActiveLearningManager() if use_active_learning else None
        
        logger.info(f"Initialized EnsembleForecaster with {len(forecasters)} forecasters")
        if use_active_learning:
            logger.info(f"Active learning enabled for high-uncertainty predictions")
    
    async def predict(self, question, context=None):
        """
        Return an ensemble probability forecast by combining multiple forecasts.
        """
        predictions = []
        
        # Get predictions from all forecasters
        for i, forecaster in enumerate(self.forecasters):
            try:
                prediction = await forecaster.predict(question, context)
                predictions.append(prediction)
                logger.debug(f"Forecaster {i} predicted: {prediction}")
            except Exception as e:
                logger.error(f"Error getting prediction from forecaster {i}: {e}")
                # Skip this forecaster
        
        if not predictions:
            logger.warning("No valid predictions from any forecaster")
            return 0.5
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == "simple_average":
            ensemble_prediction = sum(predictions) / len(predictions)
        elif self.ensemble_method == "weighted_average":
            # Use only the weights for forecasters that produced predictions
            valid_indices = [i for i, _ in enumerate(self.forecasters) if i < len(predictions)]
            valid_weights = [self.weights[i] for i in valid_indices]
            
            # Normalize weights to sum to 1
            weight_sum = sum(valid_weights)
            normalized_weights = [w / weight_sum for w in valid_weights]
            
            # Compute weighted average
            ensemble_prediction = sum(p * w for p, w in zip(predictions, normalized_weights))
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        logger.info(f"Ensemble prediction: {ensemble_prediction}")
        return ensemble_prediction
    
    async def explain(self, question, context=None):
        """
        Return an explanation that includes contributions from each forecaster.
        """
        explanations = []
        weights = []
        predictions = []
        
        # Get explanations and predictions from all forecasters
        for i, forecaster in enumerate(self.forecasters):
            try:
                explanation = await forecaster.explain(question, context)
                prediction = await forecaster.predict(question, context)
                
                explanations.append(explanation)
                weights.append(self.weights[i])
                predictions.append(prediction)
                
                logger.debug(f"Got explanation from forecaster {i}")
            except Exception as e:
                logger.error(f"Error getting explanation from forecaster {i}: {e}")
                # Skip this forecaster
        
        if not explanations:
            return "No valid explanations from any forecaster"
        
        # Normalize weights for the forecasters that produced explanations
        weight_sum = sum(weights)
        normalized_weights = [w / weight_sum for w in weights]
        
        # Create a combined explanation
        combined = "# Ensemble Forecast Explanation\n\n"
        combined += f"This forecast combines predictions from {len(explanations)} different forecasters.\n\n"
        
        # Add a summary of forecaster contributions
        combined += "## Forecaster Contributions\n\n"
        for i, (pred, weight) in enumerate(zip(predictions, normalized_weights)):
            model_name = getattr(self.forecasters[i], 'model_name', f"Forecaster {i+1}")
            combined += f"- {model_name}: {pred:.2f} (weight: {weight:.2f})\n"
        
        combined += "\n## Individual Explanations\n\n"
        
        # Add individual explanations (abbreviated if very long)
        for i, explanation in enumerate(explanations):
            model_name = getattr(self.forecasters[i], 'model_name', f"Forecaster {i+1}")
            
            # Abbreviate very long explanations
            if len(explanation) > 500:
                explanation_summary = explanation[:500] + "... (truncated)"
            else:
                explanation_summary = explanation
            
            combined += f"### {model_name}\n\n{explanation_summary}\n\n"
        
        return combined
    
    async def confidence_interval(self, question, context=None):
        """
        Return a confidence interval based on the ensemble of forecasters.
        
        Supports multiple methods for interval estimation:
        - bootstrapping: Resamples predictions to account for correlations (more robust)
        - variance_propagation: Simple weighted average of bounds (assumes independence)
        """
        # Get individual forecaster predictions and confidence intervals
        forecaster_predictions = []
        intervals = []
        
        # Get predictions and confidence intervals from all forecasters
        for i, forecaster in enumerate(self.forecasters):
            try:
                prediction = await forecaster.predict(question, context)
                interval = await forecaster.confidence_interval(question, context)
                forecaster_predictions.append(prediction)
                intervals.append(interval)
                logger.debug(f"Forecaster {i} prediction: {prediction}, interval: {interval}")
            except Exception as e:
                logger.error(f"Error getting prediction/interval from forecaster {i}: {e}")
                # Skip this forecaster
        
        if not intervals:
            logger.warning("No valid confidence intervals from any forecaster")
            # Provide a wide default interval
            return (0.3, 0.7)
        
        # Normalize weights for forecasters that produced predictions/intervals
        valid_indices = [i for i, _ in enumerate(self.forecasters) if i < len(forecaster_predictions)]
        valid_weights = [self.weights[i] for i in valid_indices]
        weight_sum = sum(valid_weights)
        normalized_weights = [w / weight_sum for w in valid_weights]
        
        # Calculate ensemble prediction using weights
        ensemble_prediction = sum(p * w for p, w in zip(forecaster_predictions, normalized_weights))
        
        # Method 1: Simple weighted average of bounds (variance propagation)
        lower_bounds = [interval[0] for interval in intervals]
        upper_bounds = [interval[1] for interval in intervals]
        
        simple_lower = sum(lb * w for lb, w in zip(lower_bounds, normalized_weights))
        simple_upper = sum(ub * w for ub, w in zip(upper_bounds, normalized_weights))
        
        # If using variance propagation method or bootstrapping fails, return simple method result
        if self.confidence_interval_method == "variance_propagation":
            logger.info(f"Variance propagation confidence interval: ({simple_lower}, {simple_upper})")
            return (simple_lower, simple_upper)
        
        # Method 2: Bootstrapping approach
        try:
            # Number of bootstrap samples
            n_bootstrap = 1000
            
            # Create bootstrap samples
            bootstrap_samples = []
            
            for _ in range(n_bootstrap):
                # Sample forecasters with replacement, respecting weights
                sampled_indices = np.random.choice(
                    len(forecaster_predictions), 
                    size=len(forecaster_predictions), 
                    replace=True, 
                    p=normalized_weights
                )
                
                # Calculate the ensemble prediction for this bootstrap sample
                sample_preds = [forecaster_predictions[i] for i in sampled_indices]
                bootstrap_samples.append(np.mean(sample_preds))
            
            # Calculate confidence interval from bootstrap distribution (95% CI)
            bootstrap_lower = np.percentile(bootstrap_samples, 2.5)
            bootstrap_upper = np.percentile(bootstrap_samples, 97.5)
            
            # Ensure interval is centered around actual ensemble prediction
            # This addresses any potential bias in the bootstrapping
            bootstrap_mean = np.mean(bootstrap_samples)
            adjustment = ensemble_prediction - bootstrap_mean
            bootstrap_lower += adjustment
            bootstrap_upper += adjustment
            
            # Ensure bounds stay within [0, 1]
            bootstrap_lower = max(0.0, min(1.0, bootstrap_lower))
            bootstrap_upper = max(0.0, min(1.0, bootstrap_upper))
            
            logger.info(f"Bootstrap confidence interval: ({bootstrap_lower}, {bootstrap_upper})")
            logger.debug(f"Simple average interval: ({simple_lower}, {simple_upper})")
            
            return (bootstrap_lower, bootstrap_upper)
        except Exception as e:
            logger.error(f"Error calculating bootstrap confidence interval: {e}")
            # Fallback to simple weighted average
            return (simple_lower, simple_upper)
    
    async def get_forecast_result(self, question, context=None):
        """
        Get a complete ForecastResult object.
        """
        try:
            prediction = await self.predict(question, context)
            explanation = await self.explain(question, context)
            interval = await self.confidence_interval(question, context)
            
            # Check if prediction needs human review
            needs_review = False
            if self.use_active_learning and self.active_learning_manager:
                # Calculate confidence from interval width
                confidence = 1.0 - (interval[1] - interval[0])
                needs_review = self.active_learning_manager.should_review(prediction, confidence)
            
            result = ForecastResult(
                prediction=prediction,
                explanation=explanation,
                confidence_interval=interval,
                needs_human_review=needs_review
            )
            
            return result
        except Exception as e:
            logger.error(f"Error getting forecast result: {e}")
            # Return a default result in case of error
            return ForecastResult(
                prediction=0.5,
                explanation="Error generating forecast",
                confidence_interval=(0.25, 0.75),
                needs_human_review=True
            )
    
    async def get_individual_forecasts(self, question, context=None):
        """
        Get results from individual forecasters for transparency and scenario analysis.
        
        Returns:
            List of ForecastResult objects, one for each forecaster
        """
        return await self._get_individual_forecasts(question, context)
    
    async def _get_individual_forecasts(self, question, context=None):
        """
        Get predictions, explanations, and intervals from all individual forecasters.
        
        Returns:
            List of ForecastResult objects, one for each forecaster
        """
        results = []
        
        for i, forecaster in enumerate(self.forecasters):
            try:
                prediction = await forecaster.predict(question, context)
                explanation = await forecaster.explain(question, context)
                interval = await forecaster.confidence_interval(question, context)
                
                model_name = getattr(forecaster, 'model_name', f"Forecaster {i+1}")
                weight = self.weights[i]
                
                result = ForecastResult(
                    prediction=prediction,
                    explanation=explanation,
                    confidence_interval=interval,
                    metadata={
                        "model_name": model_name,
                        "weight": weight
                    }
                )
                
                results.append(result)
                logger.debug(f"Got individual forecast result from {model_name}")
            except Exception as e:
                logger.error(f"Error getting individual forecast from forecaster {i}: {e}")
                # Skip this forecaster
        
        return results 