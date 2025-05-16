import logging
import random
from forecasting_tools.ai_models.model_interfaces.forecaster_base import ForecasterBase
from forecasting_tools.ai_models.model_interfaces.forecaster_result import ForecastResult

logger = logging.getLogger(__name__)

class SyntheticForecaster(ForecasterBase):
    """
    A synthetic forecaster that returns deterministic or random results.
    Useful for testing, demonstrations, and benchmarking without API dependencies.
    """
    
    def __init__(self, mode="random", fixed_probability=0.7, confidence_width=0.2):
        """
        Initialize the synthetic forecaster.
        
        Args:
            mode: Either "random" or "fixed"
            fixed_probability: The probability to return if mode is "fixed"
            confidence_width: Width of the confidence interval around the probability
        """
        self.mode = mode
        self.fixed_probability = fixed_probability
        self.confidence_width = min(confidence_width, fixed_probability, 1 - fixed_probability)
        logger.info(f"Initialized SyntheticForecaster with mode={mode}, prob={fixed_probability}")
    
    async def predict(self, question, context=None):
        """Return a synthetic probability forecast."""
        if self.mode == "random":
            probability = random.random()
        else:
            probability = self.fixed_probability
            
        logger.debug(f"SyntheticForecaster predicted {probability} for question: {question.question_text}")
        return probability
    
    async def explain(self, question, context=None):
        """Return a synthetic explanation."""
        explanations = [
            f"This is a synthetic explanation for the question: {question.question_text}.",
            "Based on synthetic analysis, this outcome appears likely.",
            "Multiple factors were considered in this synthetic forecast.",
            "Historical patterns suggest this probability is reasonable."
        ]
        
        explanation = "\n\n".join(explanations)
        logger.debug(f"SyntheticForecaster generated explanation for: {question.question_text}")
        return explanation
    
    async def confidence_interval(self, question, context=None):
        """Return a synthetic confidence interval."""
        if self.mode == "random":
            center = random.random()
            width = random.uniform(0.1, 0.4)
            lower = max(0.0, center - width/2)
            upper = min(1.0, center + width/2)
        else:
            center = self.fixed_probability
            width = self.confidence_width
            lower = max(0.0, center - width/2)
            upper = min(1.0, center + width/2)
            
        logger.debug(f"SyntheticForecaster confidence interval: ({lower}, {upper})")
        return (lower, upper)
    
    async def get_forecast_result(self, question, context=None):
        """
        Get a complete forecast result with all components.
        
        This combines predict, explain, and confidence_interval in a single call
        and returns a standardized ForecastResult object.
        """
        probability = await self.predict(question, context)
        rationale = await self.explain(question, context)
        interval = await self.confidence_interval(question, context)
        
        return ForecastResult(
            probability=probability,
            confidence_interval=interval,
            rationale=rationale,
            model_name="SyntheticForecaster",
            metadata={"mode": self.mode}
        ) 