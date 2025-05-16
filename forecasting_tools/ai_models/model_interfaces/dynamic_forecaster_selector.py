import logging
import re
import json
from typing import Dict, List, Optional, Tuple, Any, Type
from pathlib import Path
import os

from forecasting_tools.ai_models.model_interfaces.forecaster_base import ForecasterBase
from forecasting_tools.ai_models.model_interfaces.forecaster_result import ForecastResult
from forecasting_tools.ai_models.model_interfaces.synthetic_forecaster import SyntheticForecaster
from forecasting_tools.ai_models.model_interfaces.enhanced_llm_forecaster import EnhancedLLMForecaster
from forecasting_tools.ai_models.model_interfaces.expert_forecaster import ExpertForecaster
from forecasting_tools.ai_models.general_llm import GeneralLlm

logger = logging.getLogger(__name__)

class DynamicForecasterSelector:
    """
    Dynamically selects the best forecaster for a given question based on:
    1. Question type/category
    2. Historical performance on similar questions
    3. User-specified preferences
    
    This class maintains a performance registry and learns over time which 
    forecaster works best for which kinds of questions.
    """
    
    DEFAULT_REGISTRY_PATH = "forecasting_tools/data/performance_registry.json"
    
    def __init__(
        self,
        forecaster_registry: Optional[Dict[str, ForecasterBase]] = None,
        performance_registry_path: Optional[str] = None,
        default_forecaster: str = "enhanced_llm",
        allow_learning: bool = True,
        learning_rate: float = 0.1
    ):
        """
        Initialize the dynamic forecaster selector.
        
        Args:
            forecaster_registry: Dictionary mapping forecaster names to instances
            performance_registry_path: Path to save/load performance data
            default_forecaster: Default forecaster if no better option found
            allow_learning: Whether to update performance data based on results
            learning_rate: Rate at which performance scores are updated (0-1)
        """
        # Initialize forecaster registry with default implementations if not provided
        if forecaster_registry is None:
            self.forecaster_registry = {
                "general_llm": GeneralLlm(model="openai/o1"),
                "enhanced_llm": EnhancedLLMForecaster(),
                "synthetic": SyntheticForecaster(),
                "expert": ExpertForecaster()
            }
        else:
            self.forecaster_registry = forecaster_registry
            
        self.default_forecaster = default_forecaster
        self.allow_learning = allow_learning
        self.learning_rate = learning_rate
        
        # Load or initialize performance registry
        self.performance_registry_path = performance_registry_path or self.DEFAULT_REGISTRY_PATH
        self.performance_registry = self._load_performance_registry()
        
        logger.info(f"Initialized DynamicForecasterSelector with {len(self.forecaster_registry)} forecasters")
    
    def _load_performance_registry(self) -> Dict:
        """Load or initialize the performance registry."""
        try:
            if os.path.exists(self.performance_registry_path):
                with open(self.performance_registry_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default registry structure
                registry = {
                    "domain_performance": {},       # Performance by domain
                    "question_type_performance": {},  # Performance by question type
                    "forecaster_metrics": {},       # Overall metrics for each forecaster
                    "question_history": {}          # History of which forecaster was used for which question
                }
                
                # Initialize domains with default values
                for domain in ExpertForecaster.DOMAINS.keys():
                    registry["domain_performance"][domain] = {}
                    for forecaster in self.forecaster_registry.keys():
                        registry["domain_performance"][domain][forecaster] = 0.5  # Neutral starting score
                
                # Initialize question types with default values
                question_types = ["binary_simple", "binary_complex", "numeric", "date"]
                for q_type in question_types:
                    registry["question_type_performance"][q_type] = {}
                    for forecaster in self.forecaster_registry.keys():
                        registry["question_type_performance"][q_type][forecaster] = 0.5
                
                # Initialize forecaster metrics
                for forecaster in self.forecaster_registry.keys():
                    registry["forecaster_metrics"][forecaster] = {
                        "overall_score": 0.5,
                        "questions_answered": 0,
                        "brier_score": 0.0,
                        "calibration": 0.0
                    }
                
                # Save the initialized registry
                self._save_performance_registry(registry)
                return registry
                
        except Exception as e:
            logger.error(f"Error loading performance registry: {e}")
            # Return a minimal default registry
            return {
                "domain_performance": {},
                "question_type_performance": {},
                "forecaster_metrics": {},
                "question_history": {}
            }
    
    def _save_performance_registry(self, registry=None):
        """Save the performance registry to disk."""
        if registry is None:
            registry = self.performance_registry
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.performance_registry_path), exist_ok=True)
            
            with open(self.performance_registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
                
            logger.debug(f"Saved performance registry to {self.performance_registry_path}")
        except Exception as e:
            logger.error(f"Error saving performance registry: {e}")
    
    def _detect_question_type(self, question) -> str:
        """
        Detect the type of question based on its properties.
        
        Returns: "binary_simple", "binary_complex", "numeric", or "date"
        """
        question_text = question.question_text.lower()
        
        # Check for numeric questions (asking for a specific number or range)
        numeric_patterns = [
            r"how many",
            r"what percentage",
            r"what fraction",
            r"what number",
            r"how much",
            r"\d+(\.\d+)?\s*%",
            r"greater than \d+",
            r"less than \d+",
            r"between \d+ and \d+"
        ]
        for pattern in numeric_patterns:
            if re.search(pattern, question_text):
                return "numeric"
        
        # Check for date questions
        date_patterns = [
            r"when will",
            r"what date",
            r"by \d{4}",
            r"by (january|february|march|april|may|june|july|august|september|october|november|december)",
            r"before \d{4}"
        ]
        for pattern in date_patterns:
            if re.search(pattern, question_text):
                return "date"
        
        # Determine if binary question is complex or simple
        complex_indicators = [
            "but only if",
            "unless",
            "except if",
            "if and only if",
            "all of the following",
            "any of the following"
        ]
        
        for indicator in complex_indicators:
            if indicator in question_text:
                return "binary_complex"
        
        # Default to simple binary
        return "binary_simple"
    
    def _detect_domains(self, question) -> List[Tuple[str, float]]:
        """
        Detect applicable domains for the question.
        Uses ExpertForecaster's domain detection.
        """
        if "expert" in self.forecaster_registry:
            expert_forecaster = self.forecaster_registry["expert"]
            if hasattr(expert_forecaster, "_detect_domains"):
                return expert_forecaster._detect_domains(question.question_text)
        
        # Fallback if expert forecaster not available or missing method
        return []
    
    def select_best_forecaster(self, question) -> Tuple[str, ForecasterBase]:
        """
        Select the best forecaster for this question based on:
        1. Question type
        2. Domain(s)
        3. Historical performance
        
        Returns: (forecaster_name, forecaster_instance) tuple
        """
        question_type = self._detect_question_type(question)
        domains = self._detect_domains(question)
        
        logger.info(f"Question type detected: {question_type}")
        if domains:
            logger.info(f"Domains detected: {[d[0] for d in domains]}")
        
        # Score each forecaster
        forecaster_scores = {}
        for forecaster_name in self.forecaster_registry.keys():
            score = 0.0
            count = 0
            
            # Add score from question type performance
            if question_type in self.performance_registry["question_type_performance"]:
                if forecaster_name in self.performance_registry["question_type_performance"][question_type]:
                    type_score = self.performance_registry["question_type_performance"][question_type][forecaster_name]
                    score += type_score
                    count += 1
            
            # Add scores from domain performance (weighted by domain relevance)
            domain_score = 0.0
            domain_count = 0
            for domain, relevance in domains:
                if domain in self.performance_registry["domain_performance"]:
                    if forecaster_name in self.performance_registry["domain_performance"][domain]:
                        domain_score += self.performance_registry["domain_performance"][domain][forecaster_name] * relevance
                        domain_count += relevance
            
            if domain_count > 0:
                score += domain_score / domain_count
                count += 1
            
            # Add overall score
            if forecaster_name in self.performance_registry["forecaster_metrics"]:
                overall_score = self.performance_registry["forecaster_metrics"][forecaster_name]["overall_score"]
                score += overall_score
                count += 1
            
            # Calculate average (if any scores were added)
            if count > 0:
                forecaster_scores[forecaster_name] = score / count
            else:
                # Default score if no data available
                forecaster_scores[forecaster_name] = 0.5
        
        # Select the highest scoring forecaster
        best_forecaster = max(forecaster_scores.items(), key=lambda x: x[1])[0]
        
        # Fallback to default if no clear winner
        if best_forecaster not in self.forecaster_registry:
            best_forecaster = self.default_forecaster
            
        logger.info(f"Selected forecaster: {best_forecaster} (score: {forecaster_scores.get(best_forecaster, 0.5):.3f})")
        
        # Record this selection in the history
        question_id = hash(question.question_text)
        self.performance_registry["question_history"][str(question_id)] = {
            "forecaster": best_forecaster,
            "question_type": question_type,
            "domains": [d[0] for d in domains[:3]] if domains else ["general"],
            "resolved": False
        }
        
        # Save the updated registry
        if self.allow_learning:
            self._save_performance_registry()
        
        return best_forecaster, self.forecaster_registry[best_forecaster]
    
    def update_performance(self, question, forecaster_name: str, brier_score: float, calibration: float = None):
        """
        Update the performance registry with a new result.
        
        Args:
            question: The question that was forecasted
            forecaster_name: Name of the forecaster used
            brier_score: Brier score for the forecast (lower is better)
            calibration: Optional calibration score (higher is better)
        """
        if not self.allow_learning:
            return
            
        # Only update for forecasters in the registry
        if forecaster_name not in self.forecaster_registry:
            logger.warning(f"Cannot update performance for unknown forecaster: {forecaster_name}")
            return
        
        # Convert Brier score to a performance score (0-1, higher is better)
        # Brier score ranges from 0 (perfect) to 1 (worst)
        performance_score = 1.0 - brier_score
        
        # Apply learning rate to limit how quickly scores change
        lr = self.learning_rate
        
        # Update overall forecaster metrics
        if forecaster_name in self.performance_registry["forecaster_metrics"]:
            metrics = self.performance_registry["forecaster_metrics"][forecaster_name]
            
            # Update overall score with learning rate
            old_score = metrics["overall_score"]
            metrics["overall_score"] = old_score * (1 - lr) + performance_score * lr
            
            # Update Brier score as a running average
            n = metrics["questions_answered"]
            if n > 0:
                old_brier = metrics["brier_score"]
                metrics["brier_score"] = (old_brier * n + brier_score) / (n + 1)
            else:
                metrics["brier_score"] = brier_score
                
            # Update calibration if provided
            if calibration is not None:
                if n > 0:
                    old_calibration = metrics["calibration"]
                    metrics["calibration"] = (old_calibration * n + calibration) / (n + 1)
                else:
                    metrics["calibration"] = calibration
            
            # Increment questions answered
            metrics["questions_answered"] += 1
        
        # Update question type performance
        question_type = self._detect_question_type(question)
        if question_type in self.performance_registry["question_type_performance"]:
            if forecaster_name in self.performance_registry["question_type_performance"][question_type]:
                old_score = self.performance_registry["question_type_performance"][question_type][forecaster_name]
                new_score = old_score * (1 - lr) + performance_score * lr
                self.performance_registry["question_type_performance"][question_type][forecaster_name] = new_score
        
        # Update domain performance
        domains = self._detect_domains(question)
        for domain, _ in domains:
            if domain in self.performance_registry["domain_performance"]:
                if forecaster_name in self.performance_registry["domain_performance"][domain]:
                    old_score = self.performance_registry["domain_performance"][domain][forecaster_name]
                    new_score = old_score * (1 - lr) + performance_score * lr
                    self.performance_registry["domain_performance"][domain][forecaster_name] = new_score
        
        # Mark question as resolved in history
        question_id = hash(question.question_text)
        if str(question_id) in self.performance_registry["question_history"]:
            self.performance_registry["question_history"][str(question_id)]["resolved"] = True
            self.performance_registry["question_history"][str(question_id)]["brier_score"] = brier_score
            if calibration is not None:
                self.performance_registry["question_history"][str(question_id)]["calibration"] = calibration
        
        # Save updated registry
        self._save_performance_registry()
        logger.info(f"Updated performance for {forecaster_name}: Brier score = {brier_score:.4f}")
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of forecaster performance."""
        return {
            "forecaster_metrics": self.performance_registry["forecaster_metrics"],
            "domain_leaders": self._get_domain_leaders(),
            "question_type_leaders": self._get_question_type_leaders()
        }
    
    def _get_domain_leaders(self) -> Dict[str, str]:
        """Get the best forecaster for each domain."""
        domain_leaders = {}
        for domain, forecasters in self.performance_registry["domain_performance"].items():
            if forecasters:
                # Find forecaster with highest score for this domain
                best_forecaster = max(forecasters.items(), key=lambda x: x[1])[0]
                domain_leaders[domain] = best_forecaster
        return domain_leaders
    
    def _get_question_type_leaders(self) -> Dict[str, str]:
        """Get the best forecaster for each question type."""
        type_leaders = {}
        for q_type, forecasters in self.performance_registry["question_type_performance"].items():
            if forecasters:
                # Find forecaster with highest score for this question type
                best_forecaster = max(forecasters.items(), key=lambda x: x[1])[0]
                type_leaders[q_type] = best_forecaster
        return type_leaders


class DynamicForecaster(ForecasterBase):
    """
    A forecaster that dynamically selects the best underlying forecaster
    for each question.
    
    This is a wrapper around DynamicForecasterSelector that implements
    the ForecasterBase interface.
    """
    
    def __init__(
        self,
        forecaster_registry: Optional[Dict[str, ForecasterBase]] = None,
        performance_registry_path: Optional[str] = None,
        default_forecaster: str = "enhanced_llm",
        allow_learning: bool = True,
        learning_rate: float = 0.1
    ):
        """
        Initialize the dynamic forecaster.
        
        Args:
            forecaster_registry: Dictionary mapping forecaster names to instances
            performance_registry_path: Path to save/load performance data
            default_forecaster: Default forecaster if no better option found
            allow_learning: Whether to update performance data based on results
            learning_rate: Rate at which performance scores are updated (0-1)
        """
        self.selector = DynamicForecasterSelector(
            forecaster_registry=forecaster_registry,
            performance_registry_path=performance_registry_path,
            default_forecaster=default_forecaster,
            allow_learning=allow_learning,
            learning_rate=learning_rate
        )
        
        self.model_name = "DynamicForecaster"
        logger.info(f"Initialized DynamicForecaster")
    
    async def predict(self, question, context=None):
        """Predict using the best forecaster for this question."""
        forecaster_name, forecaster = self.selector.select_best_forecaster(question)
        logger.info(f"Using {forecaster_name} for prediction")
        return await forecaster.predict(question, context)
    
    async def explain(self, question, context=None):
        """Get explanation from the best forecaster for this question."""
        forecaster_name, forecaster = self.selector.select_best_forecaster(question)
        logger.info(f"Using {forecaster_name} for explanation")
        
        explanation = await forecaster.explain(question, context)
        
        # Add a note about which forecaster was selected
        return f"*Selected forecaster: {forecaster_name}*\n\n{explanation}"
    
    async def confidence_interval(self, question, context=None):
        """Get confidence interval from the best forecaster for this question."""
        forecaster_name, forecaster = self.selector.select_best_forecaster(question)
        logger.info(f"Using {forecaster_name} for confidence interval")
        return await forecaster.confidence_interval(question, context)
    
    async def get_forecast_result(self, question, context=None):
        """Get complete forecast result using the best forecaster."""
        forecaster_name, forecaster = self.selector.select_best_forecaster(question)
        logger.info(f"Using {forecaster_name} for full forecast")
        
        # Get result from selected forecaster
        result = await forecaster.get_forecast_result(question, context)
        
        # Add information about the selection process
        question_type = self.selector._detect_question_type(question)
        domains = self.selector._detect_domains(question)
        
        # Create a new result with additional metadata
        return ForecastResult(
            probability=result.probability,
            confidence_interval=result.confidence_interval,
            rationale=f"*Selected forecaster: {forecaster_name} (question type: {question_type})*\n\n{result.rationale}",
            model_name=f"Dynamic-{forecaster_name}",
            metadata={
                "selected_forecaster": forecaster_name,
                "question_type": question_type,
                "detected_domains": [d[0] for d in domains[:3]] if domains else [],
                "original_result": result.metadata
            }
        )
        
    def update_performance(self, question, brier_score: float, calibration: float = None):
        """Update performance metrics for the last used forecaster."""
        # Get the forecaster that was used for this question
        question_id = hash(question.question_text)
        history = self.selector.performance_registry["question_history"]
        
        if str(question_id) in history:
            forecaster_name = history[str(question_id)]["forecaster"]
            self.selector.update_performance(
                question=question,
                forecaster_name=forecaster_name,
                brier_score=brier_score,
                calibration=calibration
            )
        else:
            logger.warning(f"No record of which forecaster was used for this question")
    
    def get_performance_summary(self):
        """Get performance summary from the selector."""
        return self.selector.get_performance_summary() 