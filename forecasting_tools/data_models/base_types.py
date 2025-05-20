from typing import TypeVar, Generic, Optional, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime

T = TypeVar('T')

@dataclass
class ReasonedPrediction(Generic[T]):
    """
    A prediction with accompanying reasoning.
    
    Attributes:
        prediction_value: The actual prediction value (float, int, etc.)
        reasoning: The explanation for the prediction
        confidence_interval: Optional confidence interval as a tuple (lower, upper)
    """
    prediction_value: T
    reasoning: str
    confidence_interval: Optional[tuple[float, float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ForecastQuestion:
    """
    A base class for all forecast questions.
    
    Attributes:
        id: A unique identifier for the question
        text: The text of the question
        resolution_criteria: Criteria for how the question will be resolved
        category: Optional category of the question
        due_date: Optional due date for question resolution
        context: Optional additional context
    """
    id: str
    text: str
    resolution_criteria: str
    category: Optional[str] = None
    due_date: Optional[str] = None
    context: Optional[str] = None


@dataclass
class Forecast:
    """
    A forecast for a question.
    
    Attributes:
        question_id: ID of the question being forecasted
        model: Name of the model/forecaster
        prediction: Probability or value prediction
        lower: Optional lower bound of confidence interval
        upper: Optional upper bound of confidence interval
        explanation: Explanation for the forecast
        timestamp: When the forecast was made
        metadata: Additional metadata about the forecast
    """
    question_id: str
    model: str
    prediction: float
    lower: Optional[float] = None
    upper: Optional[float] = None
    explanation: str = ""
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict[str, Any]] = None 