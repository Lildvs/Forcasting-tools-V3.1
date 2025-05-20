from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Dict, Any

from forecasting_tools.data_models.base_types import ForecastQuestion, Forecast


class Forecaster(ABC):
    """
    Abstract base class for forecasters.
    
    All forecaster implementations should inherit from this class
    and implement the required methods.
    """
    
    @abstractmethod
    def forecast(self, question: ForecastQuestion) -> Forecast:
        """
        Generate a forecast for the given question.
        
        Parameters
        ----------
        question : ForecastQuestion
            The question to forecast
            
        Returns
        -------
        Forecast
            The forecast result
        """
        pass
    
    async def predict(self, question: ForecastQuestion) -> float:
        """
        Predict a probability for a binary question.
        
        This is a compatibility method that calls forecast() and returns
        just the prediction value.
        
        Parameters
        ----------
        question : ForecastQuestion
            The question to forecast
            
        Returns
        -------
        float
            Probability between 0 and 1
        """
        forecast = self.forecast(question)
        return forecast.prediction
    
    async def explain(self, question: ForecastQuestion) -> str:
        """
        Generate an explanation for a forecast.
        
        This is a compatibility method that calls forecast() and returns
        just the explanation.
        
        Parameters
        ----------
        question : ForecastQuestion
            The question to forecast
            
        Returns
        -------
        str
            Explanation text
        """
        forecast = self.forecast(question)
        return forecast.explanation
    
    async def confidence_interval(self, question: ForecastQuestion) -> Tuple[float, float]:
        """
        Generate confidence interval for a forecast.
        
        This is a compatibility method that calls forecast() and returns
        just the confidence interval.
        
        Parameters
        ----------
        question : ForecastQuestion
            The question to forecast
            
        Returns
        -------
        Tuple[float, float]
            Lower and upper bounds of confidence interval
        """
        forecast = self.forecast(question)
        # Use default values if not provided
        return (forecast.lower or 0.0, forecast.upper or 1.0) 