import logging
import re
from typing import Dict, List, Optional, Tuple, Any

from forecasting_tools.ai_models.model_interfaces.forecaster_base import ForecasterBase
from forecasting_tools.ai_models.model_interfaces.forecaster_result import ForecastResult
from forecasting_tools.ai_models.general_llm import GeneralLlm

logger = logging.getLogger(__name__)

class ExpertForecaster(ForecasterBase):
    """
    A specialized forecaster that leverages domain expertise for specific question categories.
    
    This forecaster maintains a registry of domain experts and routes questions to the
    appropriate expert based on category detection. It uses specialized prompts and
    knowledge for each domain to improve forecast quality.
    """
    
    # Domain categories and their related keywords
    DOMAINS = {
        "politics": ["election", "government", "president", "vote", "political", "congress", "democracy"],
        "economics": ["inflation", "gdp", "economy", "recession", "market", "financial", "unemployment", "interest rate"],
        "technology": ["ai", "artificial intelligence", "tech", "software", "hardware", "algorithm", "computer"],
        "science": ["research", "scientific", "experiment", "physics", "chemistry", "biology", "discovery"],
        "health": ["disease", "pandemic", "virus", "medical", "health", "vaccine", "treatment", "infection"],
        "climate": ["climate", "weather", "warming", "environment", "carbon", "emission", "temperature", "renewable"],
        "geopolitics": ["war", "conflict", "military", "international", "country", "nation", "treaty", "sanction"]
    }
    
    def __init__(
        self, 
        model_name: str = "openai/o1", 
        temperature: float = 0.1,
        default_domain: str = None,
        domain_weights: Dict[str, float] = None
    ):
        """
        Initialize the expert forecaster.
        
        Args:
            model_name: LLM model identifier
            temperature: Model temperature (lower for more deterministic outputs)
            default_domain: Default domain if no domain is detected
            domain_weights: Optional custom weights for different domains
        """
        self.llm = GeneralLlm(model=model_name, temperature=temperature)
        self.model_name = model_name
        self.default_domain = default_domain
        
        # Initialize domain weights (equal if not provided)
        if domain_weights is None:
            self.domain_weights = {domain: 1.0 for domain in self.DOMAINS.keys()}
        else:
            self.domain_weights = domain_weights
            
        logger.info(f"Initialized ExpertForecaster with model={model_name}")
    
    def _detect_domains(self, question_text: str) -> List[Tuple[str, float]]:
        """
        Detect relevant domains for a given question.
        
        Returns a list of (domain, score) tuples sorted by relevance.
        """
        domains_scores = []
        question_lower = question_text.lower()
        
        for domain, keywords in self.DOMAINS.items():
            # Calculate a score based on keyword frequency
            score = 0
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    score += 1
            
            # Apply domain weight
            weighted_score = score * self.domain_weights.get(domain, 1.0)
            
            if weighted_score > 0:
                domains_scores.append((domain, weighted_score))
        
        # Sort by score in descending order
        return sorted(domains_scores, key=lambda x: x[1], reverse=True)
    
    def _build_expert_prompt(self, question, domain: str, context=None):
        """Build a domain-specific expert prompt."""
        base_prompt = f"You are a world-class expert in {domain} with a track record of accurate forecasting.\n\n"
        base_prompt += f"Forecast the probability of this question: {question.question_text}\n\n"
        
        # Add domain-specific prompting based on the detected domain
        if domain == "politics":
            base_prompt += "Consider electoral dynamics, polling accuracy, historical precedents, and institutional constraints.\n"
            base_prompt += "Political events are often influenced by public opinion, strategic behavior, and systematic forces.\n"
        elif domain == "economics":
            base_prompt += "Consider market signals, economic indicators, central bank policies, and historical market behavior.\n"
            base_prompt += "Pay special attention to leading indicators and lagging effects in your analysis.\n"
        elif domain == "technology":
            base_prompt += "Consider technology adoption curves, technical feasibility, competitive dynamics, and scaling challenges.\n"
            base_prompt += "Technological progress often follows predictable patterns but can be disrupted by breakthrough innovations.\n"
        elif domain == "science":
            base_prompt += "Consider the quality of existing evidence, replication status, methodological rigor, and theoretical plausibility.\n"
            base_prompt += "Scientific progress depends on both experimental results and theoretical frameworks.\n"
        elif domain == "health":
            base_prompt += "Consider epidemiological models, transmission dynamics, intervention efficacy, and biological constraints.\n"
            base_prompt += "Health outcomes depend on both biological factors and human behavior.\n"
        elif domain == "climate":
            base_prompt += "Consider climate models, historical trends, feedback loops, and policy implementation timelines.\n"
            base_prompt += "Climate predictions should account for both physical systems and socioeconomic factors.\n"
        elif domain == "geopolitics":
            base_prompt += "Consider strategic interests, balance of power, historical relationships, and domestic political constraints.\n"
            base_prompt += "Geopolitical outcomes depend on both capabilities and intentions of relevant actors.\n"
        
        if hasattr(question, 'background_info') and question.background_info:
            base_prompt += f"\nBackground: {question.background_info}"
            
        if hasattr(question, 'resolution_criteria') and question.resolution_criteria:
            base_prompt += f"\nResolution criteria: {question.resolution_criteria}"
            
        if context:
            base_prompt += f"\nAdditional context: {context}"
        
        base_prompt += "\n\nThink step by step, considering multiple perspectives and scenarios. Identify key uncertainties and how they affect your forecast."
        base_prompt += "\nAfter careful analysis, provide your probability forecast as a number between 0 and 1."
        base_prompt += "\nYour final answer should be in the format: 'Final probability: [0-1]'"
        
        return base_prompt
    
    async def predict(self, question, context=None):
        """
        Return a probability forecast using domain expertise.
        """
        # Detect relevant domains
        domains = self._detect_domains(question.question_text)
        
        if not domains and self.default_domain:
            # Use default domain if specified and no domains detected
            domains = [(self.default_domain, 1.0)]
        elif not domains:
            # Use general forecasting if no domains detected and no default
            domains = [("general", 1.0)]
        
        # Use the highest-scoring domain for the prompt
        top_domain = domains[0][0]
        prompt = self._build_expert_prompt(question, top_domain, context)
        
        response = await self.llm.invoke(prompt)
        
        try:
            probability = self._extract_probability(response)
            logger.debug(f"ExpertForecaster ({top_domain}) predicted {probability} for: {question.question_text}")
            return probability
        except Exception as e:
            logger.error(f"Error extracting probability: {e}. Response: {response[:100]}...")
            # Fallback
            return 0.5
    
    async def explain(self, question, context=None):
        """
        Return a detailed expert explanation.
        """
        # Detect relevant domains
        domains = self._detect_domains(question.question_text)
        
        if not domains and self.default_domain:
            domains = [(self.default_domain, 1.0)]
        elif not domains:
            domains = [("general", 1.0)]
        
        top_domain = domains[0][0]
        
        prompt = f"You are a world-class expert in {top_domain}. Explain your reasoning about this question:\n\n"
        prompt += f"{question.question_text}\n\n"
        
        if hasattr(question, 'background_info') and question.background_info:
            prompt += f"Background: {question.background_info}\n\n"
            
        prompt += "In your explanation:\n"
        prompt += f"1. Highlight key factors specific to {top_domain} that influence this outcome\n"
        prompt += "2. Discuss major uncertainties and how they affect your assessment\n"
        prompt += "3. Consider potential objections to your analysis\n"
        prompt += "4. Provide concrete historical examples or relevant data points\n"
        
        response = await self.llm.invoke(prompt)
        explanation = f"## Expert Analysis ({top_domain.capitalize()})\n\n{response}"
        
        # Add expert credentials
        explanation += f"\n\n*This forecast was generated using domain expertise in {top_domain}*"
        
        logger.debug(f"ExpertForecaster generated explanation for: {question.question_text}")
        return explanation
    
    async def confidence_interval(self, question, context=None):
        """
        Return a domain-appropriate confidence interval.
        """
        # Detect relevant domains
        domains = self._detect_domains(question.question_text)
        
        if not domains and self.default_domain:
            domains = [(self.default_domain, 1.0)]
        elif not domains:
            domains = [("general", 1.0)]
        
        top_domain = domains[0][0]
        
        prompt = f"You are a world-class expert in {top_domain} with expertise in uncertainty quantification.\n\n"
        prompt += f"Provide an 80% confidence interval for the probability of this question: {question.question_text}\n\n"
        
        if hasattr(question, 'background_info') and question.background_info:
            prompt += f"Background: {question.background_info}\n\n"
            
        prompt += "An 80% confidence interval means you believe there's an 80% chance the true probability falls within this range.\n"
        prompt += "Be aware of domain-specific uncertainty patterns and avoid overconfidence.\n"
        prompt += "Your interval should be in the format: 'Confidence interval: [lower bound, upper bound]'"
        
        response = await self.llm.invoke(prompt)
        
        try:
            interval = self._extract_interval(response)
            logger.debug(f"ExpertForecaster confidence interval: {interval}")
            return interval
        except Exception as e:
            logger.error(f"Error extracting interval: {e}. Response: {response[:100]}...")
            # Fallback with domain-specific width
            width = 0.3  # Default width
            
            # Adjust width based on domain (some domains have higher inherent uncertainty)
            if top_domain in ["politics", "geopolitics"]:
                width = 0.4  # Higher uncertainty
            elif top_domain in ["technology", "science"]:
                width = 0.35
            elif top_domain in ["economics", "climate"]:
                width = 0.3
            elif top_domain in ["health"]:
                width = 0.25  # Lower uncertainty
                
            prediction = await self.predict(question, context)
            lower = max(0.0, prediction - width/2)
            upper = min(1.0, prediction + width/2)
            return (lower, upper)
    
    async def get_forecast_result(self, question, context=None):
        """
        Get a complete forecast result with all components.
        """
        probability = await self.predict(question, context)
        rationale = await self.explain(question, context)
        interval = await self.confidence_interval(question, context)
        
        # Detect relevant domains for metadata
        domains = self._detect_domains(question.question_text)
        if not domains and self.default_domain:
            domains = [(self.default_domain, 1.0)]
        elif not domains:
            domains = [("general", 1.0)]
        
        return ForecastResult(
            probability=probability,
            confidence_interval=interval,
            rationale=rationale,
            model_name=f"ExpertForecaster-{self.model_name}",
            metadata={
                "domains": domains,
                "primary_domain": domains[0][0]
            }
        )
    
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
        
        # Fallback
        return (0.4, 0.6) 