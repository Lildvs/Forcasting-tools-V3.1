import logging
import os
import json
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime
import numpy as np

from forecasting_tools.ai_models.model_interfaces.forecaster_result import ForecastResult
from forecasting_tools.data_models.questions import BinaryQuestion, NumericQuestion, DateQuestion

logger = logging.getLogger(__name__)

class ActiveLearningManager:
    """
    Manages the active learning process for forecasting models.
    
    This class is responsible for:
    1. Tracking uncertainty in forecasts
    2. Flagging low-confidence predictions for human review
    3. Storing human feedback for retraining models
    4. Prioritizing questions for review based on uncertainty and importance
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        uncertainty_threshold: float = 0.2,  # Confidence interval width threshold for flagging
        confidence_threshold: float = 0.6,   # Minimum confidence score required
        middle_prob_threshold: float = 0.15, # Flag if prob is within +/- this value from 0.5
        max_review_queue: int = 100          # Maximum number of questions to queue for review
    ):
        """
        Initialize the active learning manager.
        
        Args:
            data_dir: Directory for storing active learning data
            uncertainty_threshold: Width of confidence interval to trigger review flag
            confidence_threshold: Minimum confidence score to avoid flagging
            middle_prob_threshold: Flag predictions close to 0.5
            max_review_queue: Maximum number of questions to keep in review queue
        """
        self.data_dir = data_dir or "forecasting_tools/data/active_learning"
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_threshold = confidence_threshold
        self.middle_prob_threshold = middle_prob_threshold
        self.max_review_queue = max_review_queue
        self.data_file = os.path.join(self.data_dir, "active_learning_data.json")
        
        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load or initialize active learning data
        self.active_learning_data = self._load_data()
        logger.info(f"Initialized ActiveLearningManager with {len(self.active_learning_data.get('flagged_questions', []))} flagged questions")
    
    def _load_data(self) -> Dict:
        """Load or initialize active learning data."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Ensure required fields exist
                if "flagged_questions" not in data:
                    data["flagged_questions"] = []
                if "reviewed_questions" not in data:
                    data["reviewed_questions"] = []
                if "metadata" not in data:
                    data["metadata"] = {}
                
                return data
            else:
                # Create new data structure
                data = {
                    "flagged_questions": [],
                    "reviewed_questions": [],
                    "metadata": {
                        "last_updated": datetime.now().isoformat(),
                        "version": "1.0"
                    }
                }
                
                # Save the initialized data
                self._save_data(data)
                return data
                
        except Exception as e:
            logger.error(f"Error loading active learning data: {e}")
            # Return minimal default data
            return {
                "flagged_questions": [],
                "reviewed_questions": [],
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
    
    def _save_data(self, data=None):
        """Save active learning data to disk."""
        if data is None:
            data = self.active_learning_data
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.data_dir, exist_ok=True)
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved active learning data to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving active learning data: {e}")
    
    def evaluate_forecast(self, forecast_result: ForecastResult, question: Any) -> bool:
        """
        Evaluate a forecast result to determine if it should be flagged for review.
        
        Args:
            forecast_result: The forecast result to evaluate
            question: The question being forecast
            
        Returns:
            bool: True if the forecast is flagged for human review, False otherwise
        """
        # Extract the probability and confidence interval
        probability = forecast_result.probability
        interval = forecast_result.confidence_interval
        
        # Check if probability is close to 0.5 (high uncertainty)
        is_close_to_half = abs(probability - 0.5) < self.middle_prob_threshold
        
        # Check if confidence interval is too wide
        interval_width = interval[1] - interval[0]
        is_interval_wide = interval_width > self.uncertainty_threshold
        
        # Get confidence score from metadata if available
        confidence_score = 1.0
        if forecast_result.metadata and "confidence_score" in forecast_result.metadata:
            confidence_score = forecast_result.metadata["confidence_score"]
        
        is_low_confidence = confidence_score < self.confidence_threshold
        
        # Determine if forecast should be flagged
        should_flag = (is_close_to_half or is_interval_wide or is_low_confidence)
        
        if should_flag:
            logger.info(f"Flagging forecast: prob={probability}, interval_width={interval_width}, confidence={confidence_score}")
            self._flag_question(question, forecast_result)
        
        return should_flag
    
    def _flag_question(self, question: Any, forecast_result: ForecastResult):
        """
        Flag a question for human review.
        
        Args:
            question: The question to flag
            forecast_result: The associated forecast result
        """
        # Create a unique ID for the question
        question_id = self._get_question_id(question)
        
        # Check if this question is already flagged
        for flagged in self.active_learning_data["flagged_questions"]:
            if flagged.get("question_id") == question_id:
                logger.debug(f"Question already flagged: {question_id}")
                return
        
        # Add to the flagged questions list
        flagged_entry = {
            "question_id": question_id,
            "question_text": question.question_text,
            "probability": forecast_result.probability,
            "confidence_interval": forecast_result.confidence_interval,
            "model_name": forecast_result.model_name,
            "timestamp": datetime.now().isoformat(),
            "reviewed": False,
            "importance": self._calculate_importance(question, forecast_result)
        }
        
        # Add any metadata if available
        if forecast_result.metadata:
            flagged_entry["metadata"] = forecast_result.metadata
        
        # Store additional question info if available
        if hasattr(question, "background_info") and question.background_info:
            flagged_entry["background_info"] = question.background_info
        
        # Add to flagged questions
        self.active_learning_data["flagged_questions"].append(flagged_entry)
        
        # Limit the size of the queue if needed
        if len(self.active_learning_data["flagged_questions"]) > self.max_review_queue:
            # Sort by importance (higher first)
            sorted_flags = sorted(
                self.active_learning_data["flagged_questions"], 
                key=lambda x: x.get("importance", 0), 
                reverse=True
            )
            # Keep only the most important ones
            self.active_learning_data["flagged_questions"] = sorted_flags[:self.max_review_queue]
        
        # Save the updated data
        self._save_data()
        logger.info(f"Added question to review queue: {question_id}")
    
    def _calculate_importance(self, question: Any, forecast_result: ForecastResult) -> float:
        """
        Calculate importance score for prioritizing questions.
        Higher scores indicate questions that should be prioritized for review.
        
        Args:
            question: The question being evaluated
            forecast_result: The forecast result
            
        Returns:
            float: Importance score (0-1)
        """
        # Start with base importance
        importance = 0.5
        
        # Factor 1: Uncertainty (higher uncertainty -> higher importance)
        interval_width = forecast_result.confidence_interval[1] - forecast_result.confidence_interval[0]
        uncertainty_factor = min(1.0, interval_width / 0.5)  # Normalize to 0-1
        
        # Factor 2: Probability near 0.5 (closer to 0.5 -> higher importance)
        prob_distance = abs(forecast_result.probability - 0.5)
        prob_factor = 1.0 - (prob_distance * 2)  # Convert to 0-1 (1 at 0.5, 0 at 0 or 1)
        
        # Factor 3: Question difficulty/importance if available in metadata
        metadata_importance = 0.5
        if forecast_result.metadata and "question_importance" in forecast_result.metadata:
            metadata_importance = forecast_result.metadata["question_importance"]
        
        # Combine factors (can adjust weights as needed)
        importance = 0.4 * uncertainty_factor + 0.4 * prob_factor + 0.2 * metadata_importance
        
        return importance
    
    def _get_question_id(self, question: Any) -> str:
        """
        Generate a unique identifier for the question.
        
        Args:
            question: The question to generate an ID for
            
        Returns:
            str: A unique ID for the question
        """
        if hasattr(question, "id_of_question") and question.id_of_question:
            return f"q_{question.id_of_question}"
        return f"q_{hash(question.question_text)}"
    
    def get_flagged_questions(self, limit: int = 10, sort_by_importance: bool = True) -> List[Dict]:
        """
        Get a list of flagged questions for review.
        
        Args:
            limit: Maximum number of questions to return
            sort_by_importance: Whether to sort by importance score
            
        Returns:
            List[Dict]: List of flagged questions with their forecast details
        """
        flagged = self.active_learning_data.get("flagged_questions", [])
        
        # Filter to only include unreviewed questions
        unreviewed = [q for q in flagged if not q.get("reviewed", False)]
        
        # Sort by importance if requested
        if sort_by_importance:
            unreviewed = sorted(unreviewed, key=lambda x: x.get("importance", 0), reverse=True)
        
        return unreviewed[:limit]
    
    def submit_review(self, question_id: str, human_probability: float, feedback: str = "", update_model: bool = True):
        """
        Submit a human review for a flagged question.
        
        Args:
            question_id: The ID of the question being reviewed
            human_probability: The human's probability estimate
            feedback: Optional feedback or explanation
            update_model: Whether to use this feedback to update models
        """
        # Find the flagged question
        for i, question in enumerate(self.active_learning_data["flagged_questions"]):
            if question.get("question_id") == question_id:
                # Update the question with review data
                self.active_learning_data["flagged_questions"][i]["reviewed"] = True
                self.active_learning_data["flagged_questions"][i]["human_probability"] = human_probability
                self.active_learning_data["flagged_questions"][i]["review_feedback"] = feedback
                self.active_learning_data["flagged_questions"][i]["review_timestamp"] = datetime.now().isoformat()
                
                # Add to the reviewed questions list
                reviewed_entry = self.active_learning_data["flagged_questions"][i].copy()
                self.active_learning_data["reviewed_questions"].append(reviewed_entry)
                
                # Save updated data
                self._save_data()
                logger.info(f"Recorded human review for question: {question_id}")
                return True
        
        logger.warning(f"Question not found for review: {question_id}")
        return False
    
    def get_training_data(self, limit: int = 100) -> List[Dict]:
        """
        Get training data from reviewed questions for model retraining.
        
        Args:
            limit: Maximum number of questions to return
            
        Returns:
            List[Dict]: List of reviewed questions with human probabilities
        """
        reviewed = self.active_learning_data.get("reviewed_questions", [])
        
        # Sort by review timestamp (newest first)
        sorted_reviews = sorted(
            reviewed, 
            key=lambda x: x.get("review_timestamp", ""), 
            reverse=True
        )
        
        return sorted_reviews[:limit]
    
    def purge_old_data(self, days_threshold: int = 30):
        """
        Remove old data from the active learning system.
        
        Args:
            days_threshold: Number of days after which data is considered old
        """
        now = datetime.now()
        threshold = days_threshold * 24 * 60 * 60  # Convert to seconds
        
        # Filter flagged questions
        self.active_learning_data["flagged_questions"] = [
            q for q in self.active_learning_data["flagged_questions"]
            if self._is_recent(q.get("timestamp", ""), now, threshold)
        ]
        
        # Filter reviewed questions, but keep more of these as they're valuable training data
        self.active_learning_data["reviewed_questions"] = [
            q for q in self.active_learning_data["reviewed_questions"]
            if self._is_recent(q.get("review_timestamp", ""), now, threshold * 3)
        ]
        
        # Update metadata
        self.active_learning_data["metadata"]["last_updated"] = now.isoformat()
        
        # Save updated data
        self._save_data()
        logger.info("Purged old active learning data")
    
    def _is_recent(self, timestamp_str: str, now: datetime, threshold_seconds: float) -> bool:
        """Check if a timestamp is recent enough to keep."""
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            diff = (now - timestamp).total_seconds()
            return diff < threshold_seconds
        except:
            # If we can't parse the timestamp, assume it's old
            return False 