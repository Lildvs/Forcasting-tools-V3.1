import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Coroutine

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.base_types import ForecastQuestion
from forecasting_tools.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot2025,
)
from forecasting_tools.forecast_bots.official_bots.forecaster_assumptions import (
    FORECASTER_THOUGHT_PROCESS,
    FORCASTER_DATA_COLLECTION_AND_ANALYSIS
)
from forecasting_tools.forecast_helpers.crawl4ai_searcher import Crawl4AISearcher
from forecasting_tools.forecast_helpers.browser_searcher import BrowserSearcher

# Create a BinaryQuestion alias for compatibility
BinaryQuestion = ForecastQuestion

logger = logging.getLogger(__name__)


class ResearchSource:
    def __init__(self, title: str, url: str, content: str, source_type: str, published_date: str = None):
        self.title = title
        self.url = url
        self.content = content
        self.source_type = source_type
        self.published_date = published_date


class MainBot(Q1TemplateBot2025):
    """
    The verified highest accuracy bot available.
    """

    def __init__(
        self,
        *,
        research_reports_per_question: int = 3,
        predictions_per_research_report: int = 5,
        use_research_summary_to_forecast: bool = False,
        use_browser_automation: bool = False,
        progress_callback: Optional[Callable[[str], Coroutine]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            **kwargs,
        )
        self.research_sources: List[ResearchSource] = []
        self.use_browser_automation = use_browser_automation
        # Instead of sharing a single forecaster instance which can cause event loop issues,
        # we'll create a new forecaster for each request
        self.forecaster_config = {
            "model": "openai/o1", 
            "temperature": 0.2
        }
        # Progress callback for real-time updates
        self.progress_callback = progress_callback
        # Check if browser automation is available
        if self.use_browser_automation:
            is_available = BrowserSearcher.is_available()
            if not is_available:
                logger.warning("Browser automation requested but Playwright is not available.")
                logger.warning("Install with: pip install playwright && python -m playwright install")
                self.use_browser_automation = False

    def _get_forecaster(self):
        """Create a fresh forecaster instance to avoid event loop issues"""
        return GeneralLlm(**self.forecaster_config)
        
    async def send_progress_update(self, message: str) -> None:
        """Send a progress update via the callback if available"""
        if self.progress_callback:
            await self.progress_callback(message)
        logger.info(f"Progress update: {message}")

    def _format_sources(self) -> str:
        """Format the collected sources into a readable string."""
        if not self.research_sources:
            return "No sources available."
        
        formatted_sources = "\n\nSources Used:\n"
        for i, source in enumerate(self.research_sources, 1):
            formatted_sources += f"\n{i}. {source.title}\n"
            formatted_sources += f"   URL: {source.url}\n"
            if source.published_date:
                formatted_sources += f"   Published: {source.published_date}\n"
            formatted_sources += f"   Source Type: {source.source_type}\n"
        
        return formatted_sources

    async def run_research(self, question: MetaculusQuestion) -> str:
        # Avoid using the concurrency limiter directly which can cause event loop issues
        try:
            await self.send_progress_update("Starting research process...")
            research = ""
            self.research_sources = []  # Reset sources for new research
            
            if os.getenv("PERPLEXITY_API_KEY"):
                await self.send_progress_update("Using Perplexity for advanced web search...")
                # Configure Perplexity Sonar with high search context
                model = GeneralLlm(
                    model="perplexity/sonar-pro",
                    temperature=0.1,
                    web_search_options={"search_context_size": "high"},
                    reasoning_effort="high"
                )
                
                # Format the appropriate analysis framework based on question type
                framework_type = "binary"  # default
                if hasattr(question, 'question_type'):
                    if question.question_type == "multiple_choice":
                        framework_type = "multiple_choice"
                    elif question.question_type == "numeric":
                        framework_type = "numeric"
                
                # Format the scenario analysis as a list for compatibility
                scenario_analysis = []
                for name, scenario in FORECASTER_THOUGHT_PROCESS["scenario_analysis"][framework_type]["scenarios"].items():
                    scenario_analysis.append(f"{name.replace('_', ' ').title()}: {scenario['description']}")
                formatted_scenario_analysis = "\n".join(scenario_analysis)
                
                # Format the epistemological principles as bullet points
                epistemological_principles = "\n".join([
                    f"- {principle}" for principle in FORECASTER_THOUGHT_PROCESS["forecasting_principles"]["epistemological"]
                ])
                
                # Format the prompt
                prompt = clean_indents(
                    f"""
                    You are an assistant to a superforecaster.
                    The superforecaster will give you a question they intend to forecast on.
                    To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                    You do not produce forecasts yourself.

                    Question:
                    {question.question_text}

                    Background:
                    {question.background_info if question.background_info else "No background information provided."}

                    Resolution criteria:
                    {question.resolution_criteria if question.resolution_criteria else "No resolution criteria provided."}

                    Fine print:
                    {question.fine_print if question.fine_print else "No fine print provided."}

                    Using the following data collection and analysis framework:
                    Data Collection: {FORCASTER_DATA_COLLECTION_AND_ANALYSIS["Data Collection"]}
                    World Model: {FORCASTER_DATA_COLLECTION_AND_ANALYSIS["World Model"]}
                    Data Prioritization: {FORCASTER_DATA_COLLECTION_AND_ANALYSIS["Data Prioritization"]}
                    Idea Generation: {FORCASTER_DATA_COLLECTION_AND_ANALYSIS["Idea Generation"]}

                    Using the following analysis framework:
                    {formatted_scenario_analysis}

                    And considering these base assumptions:
                    {epistemological_principles}
                    """
                )
                
                await self.send_progress_update("Performing intelligent web search and analysis...")
                # Get the research response
                research = await model.invoke(prompt)
                await self.send_progress_update("Web search completed, processing results...")
                
                # Add Perplexity as a source
                self.research_sources.append(
                    ResearchSource(
                        title="Perplexity Sonar Research",
                        url="https://www.perplexity.ai",
                        content=research,
                        source_type="AI Research",
                        published_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
                
                # If we have additional research sources, combine them
                if os.getenv("EXA_API_KEY"):
                    await self.send_progress_update("Enhancing research with Exa.ai focused search...")
                    exa_research = await self._call_exa_smart_searcher(question.question_text)
                    research = f"{research}\n\nAdditional Research:\n{exa_research}"
                
                if os.getenv("CRAWL4AI_API_KEY"):
                    try:
                        await self.send_progress_update("Performing deep web research with Crawl4AI...")
                        crawl4ai_searcher = Crawl4AISearcher()
                        crawl4ai_research = await crawl4ai_searcher.get_formatted_search_results(
                            query=question.question_text,
                            depth=2  # Default depth
                        )
                        research = f"{research}\n\nDeep Web Research:\n{crawl4ai_research}"
                        
                        # Add Crawl4AI as a source
                        self.research_sources.append(
                            ResearchSource(
                                title="Crawl4AI Deep Search Results",
                                url="https://crawl4ai.com",
                                content=crawl4ai_research,
                                source_type="Deep Web Search",
                                published_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error using Crawl4AI: {e}")
                        await self.send_progress_update(f"Error with deep web research: {e}")
                    finally:
                        # Always clean up resources, even if an exception occurred
                        if 'crawl4ai_searcher' in locals():
                            await crawl4ai_searcher.close()
                
                # Add browser-based research if enabled
                if self.use_browser_automation:
                    try:
                        await self.send_progress_update("Starting browser-based research...")
                        browser_searcher = BrowserSearcher()
                        # Use the question text to search on metaforecast.org
                        browser_research = await browser_searcher.get_formatted_search_results(
                            query=question.question_text
                        )
                        research = f"{research}\n\nBrowser-Based Research:\n{browser_research}"
                        
                        # Add Browser research as a source
                        self.research_sources.append(
                            ResearchSource(
                                title="Browser Automated Research",
                                url="https://metaforecast.org",
                                content=browser_research,
                                source_type="Browser Automation",
                                published_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error using Browser automation: {e}")
                        await self.send_progress_update(f"Error with browser research: {e}")
                    finally:
                        # Always clean up resources
                        if 'browser_searcher' in locals():
                            await browser_searcher.close()
                
            elif os.getenv("OPENROUTER_API_KEY"):
                # Fallback to OpenRouter with similar configuration
                await self.send_progress_update("Using OpenRouter for research...")
                model = GeneralLlm(
                    model="openrouter/perplexity/sonar-reasoning",
                    temperature=0.1,
                    web_search_options={"search_context_size": "high"},
                    reasoning_effort="high"
                )
                research = await self._call_perplexity(
                    question.question_text, use_open_router=True
                )
                
                # Add OpenRouter as a source
                self.research_sources.append(
                    ResearchSource(
                        title="OpenRouter Research",
                        url="https://openrouter.ai",
                        content=research,
                        source_type="AI Research",
                        published_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
            else:
                # Fallback to basic research if no Perplexity available
                await self.send_progress_update("No primary research API available, using basic research methods...")
                research = await self._get_basic_research(question)
            
            # Add the formatted sources to the research
            research += self._format_sources()
            
            await self.send_progress_update("Research phase completed.")
            logger.info(
                f"Found Research for URL {question.page_url}:\n{research}"
            )
            return research
        except Exception as e:
            error_msg = f"Error in run_research: {e}"
            logger.error(error_msg)
            await self.send_progress_update(f"Error in research: {e}")
            # Return a minimal research report to avoid breaking the application
            return f"Error performing research: {str(e)}\n\nPlease try again or check API credentials."

    async def _get_basic_research(self, question: MetaculusQuestion) -> str:
        """Fallback research method when Perplexity is not available"""
        try:
            research_parts = []
            
            if os.getenv("EXA_API_KEY"):
                await self.send_progress_update("Using Exa.ai for research...")
                exa_research = await self._call_exa_smart_searcher(question.question_text)
                research_parts.append(exa_research)
                
                # Add Exa as a source
                self.research_sources.append(
                    ResearchSource(
                        title="Exa Smart Search Results",
                        url="https://exa.ai",
                        content=exa_research,
                        source_type="Web Search",
                        published_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
            
            if os.getenv("CRAWL4AI_API_KEY"):
                try:
                    await self.send_progress_update("Performing deep web research with Crawl4AI...")
                    crawl4ai_searcher = Crawl4AISearcher()
                    crawl4ai_research = await crawl4ai_searcher.get_formatted_search_results(
                        query=question.question_text,
                        depth=2  # Default depth
                    )
                    research_parts.append(crawl4ai_research)
                    
                    # Add Crawl4AI as a source
                    self.research_sources.append(
                        ResearchSource(
                            title="Crawl4AI Deep Search Results",
                            url="https://crawl4ai.com",
                            content=crawl4ai_research,
                            source_type="Deep Web Search",
                            published_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                    )
                    
                except Exception as e:
                    logger.error(f"Error using Crawl4AI: {e}")
                    await self.send_progress_update(f"Error with deep web research: {e}")
                finally:
                    # Always clean up resources, even if an exception occurred
                    if 'crawl4ai_searcher' in locals():
                        await crawl4ai_searcher.close()
            
            # Add browser-based research if enabled
            if self.use_browser_automation:
                try:
                    await self.send_progress_update("Starting browser-based research...")
                    browser_searcher = BrowserSearcher()
                    # Use the question text to search on metaforecast.org
                    browser_research = await browser_searcher.get_formatted_search_results(
                        query=question.question_text
                    )
                    research_parts.append(browser_research)
                    
                    # Add Browser research as a source
                    self.research_sources.append(
                        ResearchSource(
                            title="Browser Automated Research",
                            url="https://metaforecast.org",
                            content=browser_research,
                            source_type="Browser Automation",
                            published_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                    )
                except Exception as e:
                    logger.error(f"Error using Browser automation: {e}")
                    await self.send_progress_update(f"Error with browser research: {e}")
                finally:
                    # Always clean up resources
                    if 'browser_searcher' in locals():
                        await browser_searcher.close()
            
            return "\n\n".join(research_parts) if research_parts else "No research data available."
        except Exception as e:
            error_msg = f"Error in _get_basic_research: {e}"
            logger.error(error_msg)
            await self.send_progress_update(f"Error in basic research: {e}")
            return f"Error performing basic research: {str(e)}"

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        return {
            "default": GeneralLlm(model="openai/o1", temperature=1),
            "summarizer": GeneralLlm(
                model="openai/gpt-4o-mini", temperature=0
            ),
        }

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        # Use the forecaster's predict and explain methods
        try:
            # Create a new forecaster for this request
            forecaster = self._get_forecaster()
            
            await self.send_progress_update("Analyzing research and generating probability estimate...")
            # Get the probability from the forecaster
            prediction = await forecaster.predict(question)
            
            await self.send_progress_update("Developing detailed reasoning for forecast...")
            # Get the explanation
            reasoning = await forecaster.explain(question)
            
            # Get confidence interval (optional, for future use)
            confidence = await forecaster.confidence_interval(question)
            
            await self.send_progress_update(f"Forecast complete: {prediction:.2f} probability")
            
            logger.info(
                f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
            )
            
            # Create a ReasonedPrediction that works with either the model or dictionary format
            # This makes it compatible with both V3.1 and V4
            reasoned_prediction = ReasonedPrediction(
                prediction_value=prediction, 
                reasoning=reasoning
            )
            
            # Convert to dictionary to ensure compatibility with Pydantic V2 validation
            # When this is used in ResearchWithPredictions
            return reasoned_prediction.model_dump() if hasattr(reasoned_prediction, 'model_dump') else reasoned_prediction
        except Exception as e:
            error_msg = f"Error using forecaster: {e}. Falling back to standard method."
            logger.error(error_msg)
            await self.send_progress_update(f"Error in forecasting: {e}, using backup method...")
            # Fall back to the original implementation if the forecaster fails
            result = await super()._run_forecast_on_binary(question, research)
            # Also convert the result to dictionary if needed
            return result.model_dump() if hasattr(result, 'model_dump') else result
