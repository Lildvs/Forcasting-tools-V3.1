import logging
import re
from typing import Optional, Tuple, Any, Dict

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.model_interfaces.forecaster_base import ForecasterBase
from forecasting_tools.ai_models.model_interfaces.forecaster_result import ForecastResult

logger = logging.getLogger(__name__)

class EnhancedLLMForecaster(ForecasterBase):
    """
    An LLM forecaster with enhanced prompt engineering and output parsing.
    
    Uses chain-of-thought reasoning, explicit uncertainty quantification,
    and other techniques to improve forecast quality.
    """
    
    def __init__(
        self, 
        model_name: str = "openai/o1", 
        temperature: float = 0.2,
        include_recent_evidence: bool = True,
        use_chain_of_thought: bool = True,
        calibration_prompt: bool = True,
    ):
        """
        Initialize the enhanced LLM forecaster.
        
        Args:
            model_name: LLM model identifier
            temperature: Model temperature (0-1)
            include_recent_evidence: Whether to ask for recent evidence
            use_chain_of_thought: Whether to use chain-of-thought prompting
            calibration_prompt: Whether to include calibration instructions
        """
        self.llm = GeneralLlm(model=model_name, temperature=temperature)
        self.include_recent_evidence = include_recent_evidence
        self.use_chain_of_thought = use_chain_of_thought
        self.calibration_prompt = calibration_prompt
        self.model_name = model_name
        logger.info(f"Initialized EnhancedLLMForecaster with model={model_name}")
    
    async def predict(self, question, context=None):
        """
        Return a probability forecast using enhanced prompting.
        """
        prompt = self._build_forecast_prompt(question, context)
        response = await self.llm.invoke(prompt)
        
        try:
            probability = self._extract_probability(response)
            logger.debug(f"EnhancedLLMForecaster predicted {probability} for question: {question.question_text}")
            return probability
        except Exception as e:
            logger.error(f"Error extracting probability: {e}. Response: {response[:100]}...")
            # Fallback to simpler approach if extraction fails
            return await self._fallback_prediction(question, context)
    
    async def explain(self, question, context=None):
        """
        Return a detailed explanation using chain-of-thought.
        """
        prompt = self._build_explanation_prompt(question, context)
        response = await self.llm.invoke(prompt)
        
        # Clean up the response to remove any artifacts
        explanation = self._clean_explanation(response)
        logger.debug(f"EnhancedLLMForecaster generated explanation for: {question.question_text}")
        return explanation
    
    async def confidence_interval(self, question, context=None):
        """
        Return a confidence interval with explicit calibration instructions.
        """
        prompt = self._build_interval_prompt(question, context)
        response = await self.llm.invoke(prompt)
        
        try:
            interval = self._extract_interval(response)
            logger.debug(f"EnhancedLLMForecaster confidence interval: {interval}")
            return interval
        except Exception as e:
            logger.error(f"Error extracting interval: {e}. Response: {response[:100]}...")
            # Fallback to simpler approach
            return await self._fallback_interval(question, context)
    
    async def get_forecast_result(self, question, context=None):
        """
        Get a complete forecast result with all components.
        """
        probability = await self.predict(question, context)
        rationale = await self.explain(question, context)
        interval = await self.confidence_interval(question, context)
        
        return ForecastResult(
            probability=probability,
            confidence_interval=interval,
            rationale=rationale,
            model_name=f"EnhancedLLM-{self.model_name}",
            metadata={
                "chain_of_thought": self.use_chain_of_thought,
                "calibration_prompt": self.calibration_prompt
            }
        )
    
    def _build_forecast_prompt(self, question, context=None):
        """Build an enhanced prompt for forecasting."""
        base_prompt = f"Forecast the probability of this question: {question.question_text}"
        
        if hasattr(question, 'background_info') and question.background_info:
            base_prompt += f"\n\nBackground: {question.background_info}"
            
        if hasattr(question, 'resolution_criteria') and question.resolution_criteria:
            base_prompt += f"\n\nResolution criteria: {question.resolution_criteria}"
            
        if context:
            base_prompt += f"\n\nAdditional context: {context}"
            
        if self.include_recent_evidence:
            base_prompt += "\n\nBefore making your prediction, list recent evidence relevant to this question."
            
        if self.use_chain_of_thought:
            base_prompt += "\n\nThink step by step before giving your final answer:"
            base_prompt += "\n1. What are the key factors that affect this outcome?"
            base_prompt += "\n2. What evidence supports a YES outcome?"
            base_prompt += "\n3. What evidence supports a NO outcome?"
            base_prompt += "\n4. Are there any relevant historical patterns?"
            base_prompt += "\n5. What are the biggest uncertainties?"
            
        if self.calibration_prompt:
            base_prompt += "\n\nBe careful to avoid overconfidence. If you're very uncertain, your probability should be closer to 50%."
            
        base_prompt += "\n\nAfter your reasoning, provide your probability forecast as a number between 0 and 1."
        base_prompt += "\nYour final answer should be in the format: 'Final probability: [0-1]'"
        
        return base_prompt
    
    def _build_explanation_prompt(self, question, context=None):
        """Build an enhanced prompt for explanation."""
        base_prompt = f"Explain your forecast for this question: {question.question_text}"
        
        if hasattr(question, 'background_info') and question.background_info:
            base_prompt += f"\n\nBackground: {question.background_info}"
            
        if context:
            base_prompt += f"\n\nAdditional context: {context}"
            
        base_prompt += "\n\nIn your explanation, include:"
        base_prompt += "\n- Your interpretation of the question"
        base_prompt += "\n- Key evidence you considered"
        base_prompt += "\n- Major uncertainties"
        base_prompt += "\n- How your reasoning led to your probability forecast"
        
        return base_prompt
    
    def _build_interval_prompt(self, question, context=None):
        """Build an enhanced prompt for confidence intervals."""
        base_prompt = f"Provide an 80% confidence interval for the probability of: {question.question_text}"
        
        if hasattr(question, 'background_info') and question.background_info:
            base_prompt += f"\n\nBackground: {question.background_info}"
            
        if context:
            base_prompt += f"\n\nAdditional context: {context}"
            
        base_prompt += "\n\nAn 80% confidence interval means you believe there's an 80% chance the true probability falls within this range."
        base_prompt += "\nIf you provided 100 such intervals, about 80 should contain the true value."
        base_prompt += "\nBe careful not to make your interval too narrow (overconfident) or too wide (uninformative)."
        
        base_prompt += "\n\nAfter your reasoning, provide your 80% confidence interval as two numbers between 0 and 1."
        base_prompt += "\nYour final answer should be in the format: 'Confidence interval: [lower bound, upper bound]'"
        
        return base_prompt
    
    def _extract_probability(self, response: str) -> float:
        """Extract the probability from the LLM response."""
        # Look for patterns like "Final probability: 0.75" or just a number at the end
        patterns = [
            r"[Ff]inal probability:?\s*([0-9]*\.?[0-9]+)",
            r"[Pp]robability:?\s*([0-9]*\.?[0-9]+)",
            r"([0-9]*\.?[0-9]+)\s*$",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    probability = float(match.group(1))
                    # Ensure it's in [0, 1]
                    return max(0.0, min(1.0, probability))
                except ValueError:
                    continue
        
        # If no pattern matches, try to find any decimal between 0 and 1
        matches = re.findall(r"([0-9]*\.?[0-9]+)", response)
        for match in matches:
            try:
                value = float(match)
                if 0 <= value <= 1:
                    return value
            except ValueError:
                continue
        
        # Fallback
        logger.warning(f"Couldn't extract probability from response: {response[:100]}...")
        return 0.5
    
    def _clean_explanation(self, response: str) -> str:
        """Clean up the explanation."""
        # Remove lines that contain probability estimates or confidence intervals
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if re.search(r"[Ff]inal probability:?\s*[0-9]*\.?[0-9]+", line):
                continue
            if re.search(r"[Cc]onfidence interval:?\s*\[\s*[0-9]*\.?[0-9]+\s*,\s*[0-9]*\.?[0-9]+\s*\]", line):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_interval(self, response: str) -> Tuple[float, float]:
        """Extract the confidence interval from the LLM response."""
        # Look for patterns like "Confidence interval: [0.65, 0.85]"
        interval_pattern = r"[Cc]onfidence interval:?\s*\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]"
        match = re.search(interval_pattern, response)
        
        if match:
            try:
                lower = float(match.group(1))
                upper = float(match.group(2))
                # Ensure bounds are in [0, 1] and lower <= upper
                lower = max(0.0, min(1.0, lower))
                upper = max(0.0, min(1.0, upper))
                if lower > upper:
                    lower, upper = upper, lower
                return (lower, upper)
            except ValueError:
                pass
        
        # Alternative patterns
        alt_pattern = r"([0-9]*\.?[0-9]+)\s*(?:to|-)\s*([0-9]*\.?[0-9]+)"
        match = re.search(alt_pattern, response)
        
        if match:
            try:
                lower = float(match.group(1))
                upper = float(match.group(2))
                # Ensure bounds are in [0, 1] and lower <= upper
                lower = max(0.0, min(1.0, lower))
                upper = max(0.0, min(1.0, upper))
                if lower > upper:
                    lower, upper = upper, lower
                return (lower, upper)
            except ValueError:
                pass
        
        # Fallback: look for any two numbers in [0, 1]
        numbers = []
        matches = re.findall(r"([0-9]*\.?[0-9]+)", response)
        for match in matches:
            try:
                value = float(match)
                if 0 <= value <= 1:
                    numbers.append(value)
                    if len(numbers) == 2:
                        break
            except ValueError:
                continue
        
        if len(numbers) >= 2:
            lower, upper = sorted(numbers[:2])
            return (lower, upper)
        
        # Last resort fallback
        logger.warning(f"Couldn't extract interval from response: {response[:100]}...")
        return (0.4, 0.6)
    
    async def _fallback_prediction(self, question, context=None):
        """Fallback method for prediction if main method fails."""
        simple_prompt = f"What is the probability of this: {question.question_text}? Respond with just a number between 0 and 1."
        response = await self.llm.invoke(simple_prompt)
        
        try:
            # Try to extract any float
            matches = re.findall(r"([0-9]*\.?[0-9]+)", response)
            for match in matches:
                try:
                    value = float(match)
                    if 0 <= value <= 1:
                        return value
                except ValueError:
                    continue
        except Exception:
            pass
        
        return 0.5
    
    async def _fallback_interval(self, question, context=None):
        """Fallback method for confidence interval if main method fails."""
        # Default to a wide interval around the prediction
        prediction = await self.predict(question, context)
        width = 0.2
        lower = max(0.0, prediction - width)
        upper = min(1.0, prediction + width)
        return (lower, upper) 