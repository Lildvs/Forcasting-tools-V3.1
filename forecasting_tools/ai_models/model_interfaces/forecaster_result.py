from dataclasses import dataclass
from typing import Any, Optional, Tuple

@dataclass
class ForecastResult:
    """
    Standardized result format for all forecaster models.
    """
    probability: float  # A number between 0 and 1
    confidence_interval: Tuple[float, float]  # Lower and upper bounds
    rationale: str  # Explanation for the forecast
    model_name: str  # Name of the model that produced this forecast
    raw_output: Optional[Any] = None  # Optional raw output from the model
    metadata: Optional[dict] = None  # Optional additional metadata
    high_uncertainty: bool = False  # Flag for high uncertainty predictions (for active learning)
    confidence_score: float = 1.0  # A score from 0-1 indicating confidence in the prediction 