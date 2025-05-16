import logging
import re
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from collections import defaultdict
import pandas as pd

from forecasting_tools.ai_models.model_interfaces.forecaster_base import ForecasterBase
from forecasting_tools.ai_models.model_interfaces.forecaster_result import ForecastResult
from forecasting_tools.ai_models.general_llm import GeneralLlm

logger = logging.getLogger(__name__)

class HistoricalForecaster(ForecasterBase):
    """
    A forecaster that leverages historical data from similar questions.
    
    This forecaster:
    1. Maintains a database of past questions and outcomes
    2. Identifies similar questions for a new query
    3. Uses historical outcomes to inform predictions
    4. Combines historical data with other models
    """
    
    DEFAULT_HISTORY_PATH = "forecasting_tools/data/historical_questions.json"
    
    def __init__(
        self,
        history_data_path: Optional[str] = None,
        embedding_model: str = "openai/o1",
        similarity_threshold: float = 0.75,
        max_history_size: int = 1000,
        time_decay_factor: float = 0.9,  # How much to discount older questions (per year)
        base_forecaster: Optional[ForecasterBase] = None
    ):
        """
        Initialize the historical forecaster.
        
        Args:
            history_data_path: Path to save/load historical question data
            embedding_model: Model to use for semantic similarity
            similarity_threshold: Minimum similarity score to consider a question relevant
            max_history_size: Maximum number of historical questions to store
            time_decay_factor: Factor to discount older questions (per year)
            base_forecaster: Optional forecaster to use when no relevant history exists
        """
        self.history_data_path = history_data_path or self.DEFAULT_HISTORY_PATH
        self.similarity_threshold = similarity_threshold
        self.max_history_size = max_history_size
        self.time_decay_factor = time_decay_factor
        self.base_forecaster = base_forecaster
        self.model_name = "HistoricalForecaster"
        
        # Initialize LLM for embeddings and similarity
        self.llm = GeneralLlm(model=embedding_model)
        
        # Load or initialize historical data
        self.history_data = self._load_history_data()
        
        logger.info(f"Initialized HistoricalForecaster with {len(self.history_data.get('questions', []))} historical questions")
    
    def _load_history_data(self) -> Dict:
        """Load or initialize historical question data."""
        try:
            if os.path.exists(self.history_data_path):
                with open(self.history_data_path, 'r') as f:
                    data = json.load(f)
                
                # Ensure required fields exist
                if "questions" not in data:
                    data["questions"] = []
                if "embeddings" not in data:
                    data["embeddings"] = []
                if "metadata" not in data:
                    data["metadata"] = {}
                
                return data
            else:
                # Create new historical data structure
                data = {
                    "questions": [],
                    "embeddings": [],
                    "outcomes": [],
                    "categories": [],
                    "timestamps": [],
                    "metadata": {
                        "last_updated": datetime.now().isoformat(),
                        "version": "1.0"
                    }
                }
                
                # Save the initialized data
                self._save_history_data(data)
                return data
                
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            # Return minimal default data
            return {
                "questions": [],
                "embeddings": [],
                "outcomes": [],
                "categories": [],
                "timestamps": [],
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
    
    def _save_history_data(self, data=None):
        """Save historical data to disk."""
        if data is None:
            data = self.history_data
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.history_data_path), exist_ok=True)
            
            with open(self.history_data_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved historical data to {self.history_data_path}")
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for a text using the LLM."""
        try:
            # For models supporting embeddings directly
            if hasattr(self.llm, "embed") and callable(self.llm.embed):
                embedding = await self.llm.embed(text)
                return embedding
            
            # Fallback: Ask the model to create an embedding via generation
            prompt = f"Create a numerical embedding vector representing this text: {text}"
            response = await self.llm.invoke(prompt)
            
            # Try to extract a vector from the response
            # This is a simplistic approach and might need refinement
            vector_pattern = r"\[([0-9., -]+)\]"
            match = re.search(vector_pattern, response)
            if match:
                vector_str = match.group(1)
                try:
                    # Parse the vector string into a list of floats
                    vector = [float(x.strip()) for x in vector_str.split(",")]
                    return vector
                except ValueError:
                    pass
            
            # If extraction failed, use a hashed representation as fallback
            logger.warning("Could not extract embedding vector from model response, using fallback method")
            import hashlib
            hash_vals = []
            for i in range(20):  # Create a 20-dimensional vector
                h = hashlib.md5(f"{text}_{i}".encode()).hexdigest()
                hash_vals.append(int(h[:8], 16) / (2**32))  # Convert to float between 0-1
            return hash_vals
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return a default vector
            return [0.0] * 20
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
            
        try:
            # Convert to numpy arrays for calculation
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Handle different dimensionality
            min_dim = min(len(vec1), len(vec2))
            vec1 = vec1[:min_dim]
            vec2 = vec2[:min_dim]
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def add_historical_question(
        self, 
        question_text: str, 
        outcome: Optional[bool] = None, 
        category: Optional[str] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add a historical question to the database.
        
        Args:
            question_text: The text of the question
            outcome: True/False outcome (if known)
            category: Optional category for the question
            timestamp: Optional timestamp (ISO format)
            metadata: Additional metadata about the question
        """
        if "questions" not in self.history_data:
            self.history_data["questions"] = []
        if "embeddings" not in self.history_data:
            self.history_data["embeddings"] = []
        if "outcomes" not in self.history_data:
            self.history_data["outcomes"] = []
        if "categories" not in self.history_data:
            self.history_data["categories"] = []
        if "timestamps" not in self.history_data:
            self.history_data["timestamps"] = []
        
        # Generate embedding
        embedding = await self._get_embedding(question_text)
        
        # Use current timestamp if none provided
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Add to history data
        self.history_data["questions"].append(question_text)
        self.history_data["embeddings"].append(embedding)
        self.history_data["outcomes"].append(1 if outcome else 0 if outcome is not None else None)
        self.history_data["categories"].append(category or "general")
        self.history_data["timestamps"].append(timestamp)
        
        # Update metadata
        self.history_data["metadata"]["last_updated"] = datetime.now().isoformat()
        self.history_data["metadata"]["count"] = len(self.history_data["questions"])
        
        # Limit size if needed
        if len(self.history_data["questions"]) > self.max_history_size:
            # Remove oldest entries
            self.history_data["questions"] = self.history_data["questions"][-self.max_history_size:]
            self.history_data["embeddings"] = self.history_data["embeddings"][-self.max_history_size:]
            self.history_data["outcomes"] = self.history_data["outcomes"][-self.max_history_size:]
            self.history_data["categories"] = self.history_data["categories"][-self.max_history_size:]
            self.history_data["timestamps"] = self.history_data["timestamps"][-self.max_history_size:]
        
        # Save updated data
        self._save_history_data()
        
        logger.info(f"Added historical question: {question_text[:50]}... (outcome: {outcome})")
    
    async def find_similar_questions(self, question_text: str, min_similarity: float = None) -> List[Dict]:
        """
        Find questions in the history that are similar to the given question.
        
        Args:
            question_text: The question to find similar questions for
            min_similarity: Optional override for similarity threshold
            
        Returns:
            List of similar questions with similarity scores and outcomes
        """
        if not self.history_data.get("questions"):
            logger.warning("No historical questions available for comparison")
            return []
        
        # Use provided threshold or default
        threshold = min_similarity if min_similarity is not None else self.similarity_threshold
        
        # Get embedding for the query question
        query_embedding = await self._get_embedding(question_text)
        
        # Calculate similarity with all historical questions
        similar_questions = []
        
        for i, (hist_question, hist_embedding, hist_outcome, hist_category, hist_timestamp) in enumerate(zip(
            self.history_data["questions"],
            self.history_data["embeddings"],
            self.history_data["outcomes"],
            self.history_data["categories"],
            self.history_data["timestamps"]
        )):
            # Skip questions without outcomes
            if hist_outcome is None:
                continue
                
            # Calculate similarity
            similarity = self._calculate_similarity(query_embedding, hist_embedding)
            
            # Apply time decay if timestamp available
            time_weight = 1.0
            if hist_timestamp:
                try:
                    hist_date = datetime.fromisoformat(hist_timestamp.split('+')[0].split('Z')[0])
                    current_date = datetime.now()
                    years_diff = (current_date - hist_date).days / 365.0
                    time_weight = self.time_decay_factor ** years_diff
                except Exception as e:
                    logger.warning(f"Error calculating time weight: {e}")
            
            # Apply time decay to similarity
            adjusted_similarity = similarity * time_weight
            
            # Add to results if above threshold
            if adjusted_similarity >= threshold:
                similar_questions.append({
                    "question": hist_question,
                    "similarity": similarity,
                    "adjusted_similarity": adjusted_similarity,
                    "outcome": bool(hist_outcome),
                    "category": hist_category,
                    "timestamp": hist_timestamp,
                    "time_weight": time_weight
                })
        
        # Sort by adjusted similarity (descending)
        similar_questions.sort(key=lambda x: x["adjusted_similarity"], reverse=True)
        
        logger.debug(f"Found {len(similar_questions)} similar questions above threshold {threshold}")
        return similar_questions
    
    def _calculate_historical_probability(self, similar_questions: List[Dict]) -> Tuple[float, float]:
        """
        Calculate probability based on similar historical questions.
        
        Args:
            similar_questions: List of similar questions with similarity and outcome
            
        Returns:
            Tuple of (probability, confidence)
        """
        if not similar_questions:
            return (0.5, 0.0)  # Default with zero confidence
        
        # Calculate similarity-weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        
        for q in similar_questions:
            # Use adjusted similarity as weight
            weight = q["adjusted_similarity"]
            outcome = 1.0 if q["outcome"] else 0.0
            
            weighted_sum += weight * outcome
            total_weight += weight
        
        if total_weight > 0:
            probability = weighted_sum / total_weight
        else:
            probability = 0.5
        
        # Calculate confidence based on total weight and number of similar questions
        confidence = min(1.0, total_weight / 2.0)  # Cap at 1.0
        
        return (probability, confidence)
    
    async def predict(self, question, context=None):
        """
        Return a probability forecast based on historical data.
        """
        # Find similar questions
        similar_questions = await self.find_similar_questions(question.question_text)
        
        # If no similar questions found and base forecaster is available, use it
        if not similar_questions and self.base_forecaster:
            logger.info("No similar historical questions found, using base forecaster")
            return await self.base_forecaster.predict(question, context)
        
        # Calculate probability from historical data
        probability, confidence = self._calculate_historical_probability(similar_questions)
        
        # If confidence is low and base forecaster is available, blend with base forecast
        if confidence < 0.5 and self.base_forecaster:
            base_probability = await self.base_forecaster.predict(question, context)
            
            # Blend based on confidence
            blended_probability = (probability * confidence) + (base_probability * (1 - confidence))
            logger.debug(f"Blended historical ({probability:.3f}) and base ({base_probability:.3f}) forecasts with confidence {confidence:.3f}")
            return blended_probability
        
        logger.debug(f"Historical forecast: {probability:.3f} (confidence: {confidence:.3f})")
        return probability
    
    async def explain(self, question, context=None):
        """
        Return an explanation based on historical similar questions.
        """
        # Find similar questions
        similar_questions = await self.find_similar_questions(question.question_text)
        
        # If no similar questions found and base forecaster is available, use it
        if not similar_questions and self.base_forecaster:
            logger.info("No similar historical questions found, using base forecaster for explanation")
            base_explanation = await self.base_forecaster.explain(question, context)
            return f"No similar historical questions found.\n\n{base_explanation}"
        
        # Calculate probability from historical data
        probability, confidence = self._calculate_historical_probability(similar_questions)
        
        # Create explanation from historical data
        explanation = f"# Historical Analysis\n\n"
        explanation += f"This forecast is based on {len(similar_questions)} similar historical questions "
        explanation += f"with a confidence level of {confidence:.2f}.\n\n"
        
        # Add table of similar questions
        explanation += "## Most Similar Historical Questions\n\n"
        explanation += "| Question | Similarity | Outcome | Date |\n"
        explanation += "|----------|------------|---------|------|\n"
        
        # Add up to 5 most similar questions
        for q in similar_questions[:5]:
            # Truncate question if too long
            q_text = q["question"]
            if len(q_text) > 50:
                q_text = q_text[:47] + "..."
            
            # Format date if available
            date_str = "N/A"
            if q["timestamp"]:
                try:
                    date_obj = datetime.fromisoformat(q["timestamp"].split('+')[0].split('Z')[0])
                    date_str = date_obj.strftime("%Y-%m-%d")
                except:
                    date_str = q["timestamp"][:10]
            
            explanation += f"| {q_text} | {q['similarity']:.2f} | {'Yes' if q['outcome'] else 'No'} | {date_str} |\n"
        
        explanation += "\n"
        
        # Add category breakdown if we have categorized questions
        categories = {}
        for q in similar_questions:
            cat = q["category"]
            if cat not in categories:
                categories[cat] = {"count": 0, "yes": 0, "weight": 0}
            categories[cat]["count"] += 1
            if q["outcome"]:
                categories[cat]["yes"] += 1
            categories[cat]["weight"] += q["adjusted_similarity"]
        
        if len(categories) > 1:
            explanation += "## Category Analysis\n\n"
            explanation += "| Category | Count | Yes % | Weight |\n"
            explanation += "|----------|-------|-------|--------|\n"
            
            for cat, data in sorted(categories.items(), key=lambda x: x[1]["weight"], reverse=True):
                yes_pct = data["yes"] / data["count"] * 100 if data["count"] > 0 else 0
                explanation += f"| {cat} | {data['count']} | {yes_pct:.1f}% | {data['weight']:.2f} |\n"
                
            explanation += "\n"
        
        # Get base forecaster explanation if available and confidence is low
        if confidence < 0.5 and self.base_forecaster:
            base_explanation = await self.base_forecaster.explain(question, context)
            explanation += f"## Additional Analysis\n\n"
            explanation += f"Due to limited historical precedent (confidence: {confidence:.2f}), "
            explanation += f"this forecast is supplemented with additional analysis:\n\n"
            explanation += base_explanation
        
        return explanation
    
    async def confidence_interval(self, question, context=None):
        """
        Return a confidence interval based on historical data.
        """
        # Find similar questions
        similar_questions = await self.find_similar_questions(question.question_text)
        
        # If no similar questions found and base forecaster is available, use it
        if not similar_questions and self.base_forecaster:
            logger.info("No similar historical questions found, using base forecaster for confidence interval")
            return await self.base_forecaster.confidence_interval(question, context)
        
        # Calculate probability from historical data
        probability, confidence = self._calculate_historical_probability(similar_questions)
        
        # Calculate interval width based on confidence
        # Lower confidence = wider interval
        base_width = 0.3  # Default width
        width = base_width * (1 + (1 - confidence))  # Adjust width by confidence
        
        # Ensure interval stays within [0, 1]
        lower = max(0.0, probability - width/2)
        upper = min(1.0, probability + width/2)
        
        # If we have very few similar questions, make the interval wider
        if len(similar_questions) < 3:
            lower = max(0.0, lower - 0.1)
            upper = min(1.0, upper + 0.1)
        
        # If confidence is low and base forecaster is available, blend with base interval
        if confidence < 0.5 and self.base_forecaster:
            base_interval = await self.base_forecaster.confidence_interval(question, context)
            
            # Blend intervals based on confidence
            blended_lower = (lower * confidence) + (base_interval[0] * (1 - confidence))
            blended_upper = (upper * confidence) + (base_interval[1] * (1 - confidence))
            
            logger.debug(f"Blended historical and base confidence intervals with confidence {confidence:.3f}")
            return (blended_lower, blended_upper)
        
        logger.debug(f"Historical confidence interval: ({lower:.3f}, {upper:.3f})")
        return (lower, upper)
    
    async def get_forecast_result(self, question, context=None):
        """
        Get a complete forecast result with historical data.
        """
        # Find similar questions
        similar_questions = await self.find_similar_questions(question.question_text)
        
        # If no similar questions found and base forecaster is available, use it
        if not similar_questions and self.base_forecaster:
            logger.info("No similar historical questions found, using base forecaster")
            base_result = await self.base_forecaster.get_forecast_result(question, context)
            return ForecastResult(
                probability=base_result.probability,
                confidence_interval=base_result.confidence_interval,
                rationale="No similar historical questions found.\n\n" + base_result.rationale,
                model_name=f"Historical-{base_result.model_name}",
                metadata={
                    "similar_questions_count": 0,
                    "base_result": base_result.metadata
                }
            )
        
        # Calculate probability from historical data
        probability, confidence = self._calculate_historical_probability(similar_questions)
        
        # Get base result if confidence is low
        base_result = None
        if confidence < 0.5 and self.base_forecaster:
            base_result = await self.base_forecaster.get_forecast_result(question, context)
            
            # Blend probability based on confidence
            blended_probability = (probability * confidence) + (base_result.probability * (1 - confidence))
            probability = blended_probability
        
        # Generate explanation
        explanation = await self.explain(question, context)
        
        # Calculate confidence interval
        interval = await self.confidence_interval(question, context)
        
        # Create metadata
        metadata = {
            "similar_questions_count": len(similar_questions),
            "similar_questions": [
                {"question": q["question"], "similarity": q["similarity"], "outcome": q["outcome"]}
                for q in similar_questions[:5]  # Include top 5 for reference
            ],
            "historical_confidence": confidence
        }
        
        if base_result:
            metadata["base_result"] = base_result.metadata
        
        return ForecastResult(
            probability=probability,
            confidence_interval=interval,
            rationale=explanation,
            model_name="HistoricalForecaster",
            metadata=metadata
        )
    
    async def bulk_import_historical_data(self, data_file_path: str):
        """
        Import historical questions and outcomes from a CSV or JSON file.
        
        Args:
            data_file_path: Path to data file (CSV or JSON)
        """
        try:
            # Determine file type
            if data_file_path.endswith('.csv'):
                # Import from CSV
                df = pd.read_csv(data_file_path)
                
                # Extract data from dataframe
                imported_count = 0
                for _, row in df.iterrows():
                    # Get required fields
                    if 'question' not in row or pd.isna(row['question']):
                        continue
                        
                    question_text = str(row['question'])
                    
                    # Get outcome if available
                    outcome = None
                    if 'outcome' in row and not pd.isna(row['outcome']):
                        outcome_val = row['outcome']
                        if isinstance(outcome_val, bool):
                            outcome = outcome_val
                        elif isinstance(outcome_val, (int, float)):
                            outcome = bool(outcome_val)
                        elif isinstance(outcome_val, str):
                            outcome = outcome_val.lower() in ['true', 'yes', '1', 't', 'y']
                    
                    # Get optional fields
                    category = str(row['category']) if 'category' in row and not pd.isna(row['category']) else None
                    timestamp = str(row['timestamp']) if 'timestamp' in row and not pd.isna(row['timestamp']) else None
                    
                    # Add to history
                    await self.add_historical_question(
                        question_text=question_text,
                        outcome=outcome,
                        category=category,
                        timestamp=timestamp
                    )
                    imported_count += 1
                    
                logger.info(f"Imported {imported_count} questions from CSV: {data_file_path}")
                
            elif data_file_path.endswith('.json'):
                # Import from JSON
                with open(data_file_path, 'r') as f:
                    data = json.load(f)
                
                # Check if it's a list of questions
                if isinstance(data, list):
                    imported_count = 0
                    for item in data:
                        if not isinstance(item, dict) or 'question' not in item:
                            continue
                            
                        question_text = item['question']
                        outcome = item.get('outcome')
                        category = item.get('category')
                        timestamp = item.get('timestamp')
                        
                        await self.add_historical_question(
                            question_text=question_text,
                            outcome=outcome,
                            category=category,
                            timestamp=timestamp
                        )
                        imported_count += 1
                        
                    logger.info(f"Imported {imported_count} questions from JSON list: {data_file_path}")
                
                # Check if it's a dict with questions array
                elif isinstance(data, dict) and 'questions' in data and isinstance(data['questions'], list):
                    imported_count = 0
                    for i, question in enumerate(data['questions']):
                        # Skip if not a string
                        if not isinstance(question, str):
                            continue
                            
                        # Get corresponding data
                        outcome = data.get('outcomes', [])[i] if i < len(data.get('outcomes', [])) else None
                        category = data.get('categories', [])[i] if i < len(data.get('categories', [])) else None
                        timestamp = data.get('timestamps', [])[i] if i < len(data.get('timestamps', [])) else None
                        
                        await self.add_historical_question(
                            question_text=question,
                            outcome=outcome,
                            category=category,
                            timestamp=timestamp
                        )
                        imported_count += 1
                        
                    logger.info(f"Imported {imported_count} questions from JSON dict: {data_file_path}")
                
            else:
                logger.error(f"Unsupported file format: {data_file_path}")
                
        except Exception as e:
            logger.error(f"Error importing historical data: {e}")
    
    def get_statistics(self):
        """Get statistics about the historical question database."""
        if not self.history_data.get("questions"):
            return {
                "count": 0,
                "resolved_count": 0,
                "categories": {}
            }
        
        # Count questions
        total_count = len(self.history_data["questions"])
        
        # Count resolved questions
        resolved_count = sum(1 for outcome in self.history_data["outcomes"] if outcome is not None)
        
        # Count by category
        categories = defaultdict(int)
        for category in self.history_data.get("categories", []):
            categories[category or "uncategorized"] += 1
        
        # Count positive outcomes by category
        positive_by_category = defaultdict(int)
        for i, outcome in enumerate(self.history_data["outcomes"]):
            if outcome == 1:  # Positive outcome
                category = self.history_data["categories"][i] if i < len(self.history_data["categories"]) else "uncategorized"
                positive_by_category[category] += 1
        
        # Calculate positive ratios
        positive_ratios = {}
        for category, count in categories.items():
            positive_count = positive_by_category.get(category, 0)
            positive_ratios[category] = positive_count / count if count > 0 else 0
        
        return {
            "count": total_count,
            "resolved_count": resolved_count,
            "categories": dict(categories),
            "positive_ratios": positive_ratios,
            "last_updated": self.history_data.get("metadata", {}).get("last_updated", "unknown")
        } 