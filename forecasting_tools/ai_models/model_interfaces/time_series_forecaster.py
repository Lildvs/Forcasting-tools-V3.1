import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
import os
from datetime import datetime, timedelta
import json

# Time series libraries
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

try:
    # Prophet is optional - not all users will have it installed
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    # Tensorflow/Keras is optional - not all users will have it installed
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from forecasting_tools.ai_models.model_interfaces.forecaster_base import ForecasterBase
from forecasting_tools.ai_models.model_interfaces.forecaster_result import ForecastResult
from forecasting_tools.data_models.questions import BinaryQuestion, NumericQuestion, DateQuestion

logger = logging.getLogger(__name__)

class TimeSeriesForecaster(ForecasterBase):
    """
    A forecaster that uses time series analysis to make predictions.
    
    This forecaster can automatically identify and use relevant time series data
    for questions that involve forecasting trends or events over time. It supports
    multiple time series models:
    
    - ARIMA: For simple trends with seasonal components
    - Prophet: For more complex seasonality and trend detection
    - LSTM: For complex patterns and dependencies
    """
    
    def __init__(
        self,
        model_type: str = "auto",
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        auto_fetch: bool = True,
        forecast_horizon: int = 30,  # Default forecast horizon in days
        min_history_points: int = 10,
        fallback_forecaster: Optional[ForecasterBase] = None
    ):
        """
        Initialize the time series forecaster.
        
        Args:
            model_type: Type of time series model to use ("arima", "prophet", "lstm", or "auto")
            data_dir: Directory for storing time series data
            cache_dir: Directory for caching model results
            auto_fetch: Whether to automatically fetch data when needed
            forecast_horizon: Default forecast horizon in days
            min_history_points: Minimum number of historical data points required
            fallback_forecaster: Forecaster to use when time series analysis is not applicable
        """
        self.model_type = model_type
        self.data_dir = data_dir or "forecasting_tools/data/time_series"
        self.cache_dir = cache_dir or "forecasting_tools/data/cache/time_series"
        self.auto_fetch = auto_fetch
        self.forecast_horizon = forecast_horizon
        self.min_history_points = min_history_points
        self.fallback_forecaster = fallback_forecaster
        self.model_name = f"TimeSeriesForecaster_{model_type}"
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check if optional dependencies are available
        if model_type == "prophet" and not PROPHET_AVAILABLE:
            logger.warning("Prophet not available. Install with 'pip install prophet'")
            self.model_type = "arima"
            
        if model_type == "lstm" and not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Install with 'pip install tensorflow'")
            self.model_type = "arima"
        
        logger.info(f"Initialized TimeSeriesForecaster with model_type={self.model_type}")
    
    async def predict(self, question, context=None):
        """
        Return a probability forecast based on time series analysis.
        
        Args:
            question: The question to forecast
            context: Optional context information
            
        Returns:
            float: Probability forecast (0-1)
        """
        # Check if time series analysis is applicable
        if not self._is_time_series_applicable(question):
            logger.info("Time series analysis not applicable to this question")
            if self.fallback_forecaster:
                logger.info("Using fallback forecaster")
                return await self.fallback_forecaster.predict(question, context)
            return 0.5
        
        try:
            # Get time series data
            data = await self._fetch_time_series_data(question, context)
            if data is None or len(data) < self.min_history_points:
                logger.warning(f"Insufficient data: found {len(data) if data is not None else 0} points, need {self.min_history_points}")
                if self.fallback_forecaster:
                    return await self.fallback_forecaster.predict(question, context)
                return 0.5
            
            # Select the best model for this data
            model_type = self._select_best_model(data, question)
            logger.info(f"Selected {model_type} model for time series forecasting")
            
            # Get forecast horizon from question if possible
            horizon = self._extract_forecast_horizon(question) or self.forecast_horizon
            
            # Run the appropriate forecasting model
            if model_type == "arima":
                forecast_result = await self._forecast_with_arima(data, horizon)
            elif model_type == "prophet":
                forecast_result = await self._forecast_with_prophet(data, horizon)
            elif model_type == "lstm":
                forecast_result = await self._forecast_with_lstm(data, horizon)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Extract the forecast data
            forecast_df = forecast_result['forecast']
            
            # Convert forecast to probability based on question type
            probability = self._forecast_to_probability(forecast_df, question)
            
            # Cache the forecast result for explanations
            self._last_forecast = {
                'question_key': self._get_data_key(question),
                'forecast_result': forecast_result,
                'probability': probability,
                'model_type': model_type,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to cache
            cache_path = os.path.join(self.cache_dir, f"{self._get_data_key(question)}_forecast.json")
            try:
                with open(cache_path, 'w') as f:
                    # Convert non-serializable parts to strings
                    serializable_forecast = {
                        'question_key': self._last_forecast['question_key'],
                        'probability': self._last_forecast['probability'],
                        'model_type': self._last_forecast['model_type'],
                        'timestamp': self._last_forecast['timestamp'],
                        'forecast_summary': forecast_result['summary']
                    }
                    json.dump(serializable_forecast, f)
            except Exception as e:
                logger.error(f"Error saving forecast cache: {e}")
            
            return probability
            
        except Exception as e:
            logger.error(f"Error in time series prediction: {e}")
            if self.fallback_forecaster:
                logger.info("Using fallback forecaster due to error")
                return await self.fallback_forecaster.predict(question, context)
            return 0.5
    
    def _extract_forecast_horizon(self, question) -> Optional[int]:
        """
        Extract the forecast horizon from the question.
        
        Args:
            question: The question to analyze
            
        Returns:
            int or None: Forecast horizon in days
        """
        question_text = question.question_text.lower()
        
        # Look for time periods in the question
        time_patterns = [
            (r"(\d+)\s*day", lambda x: int(x)),
            (r"(\d+)\s*week", lambda x: int(x) * 7),
            (r"(\d+)\s*month", lambda x: int(x) * 30),
            (r"(\d+)\s*year", lambda x: int(x) * 365),
            (r"by\s+(\d{4})", lambda x: (datetime(int(x), 1, 1) - datetime.now()).days),
            (r"end\s+of\s+(\d{4})", lambda x: (datetime(int(x), 12, 31) - datetime.now()).days)
        ]
        
        import re
        for pattern, converter in time_patterns:
            matches = re.findall(pattern, question_text)
            if matches:
                try:
                    return converter(matches[0])
                except:
                    pass
        
        # If we have a close time, use that
        if hasattr(question, 'close_time') and question.close_time:
            days_until_close = (question.close_time - datetime.now()).days
            if days_until_close > 0:
                return days_until_close
        
        return None
    
    def _forecast_to_probability(self, forecast_df, question) -> float:
        """
        Convert a time series forecast to a probability.
        
        Args:
            forecast_df: DataFrame with forecast results
            question: The question being forecast
            
        Returns:
            float: Probability forecast (0-1)
        """
        try:
            # Extract the forecasted values
            forecast_values = forecast_df['forecast'].values
            
            # For binary questions, we need to extract the threshold and direction
            if isinstance(question, BinaryQuestion):
                threshold, direction = self._extract_threshold(question)
                
                if threshold is not None:
                    # Calculate probability based on how many forecast points exceed the threshold
                    if direction == "above":
                        probability = np.mean(forecast_values > threshold)
                    else:  # below
                        probability = np.mean(forecast_values < threshold)
                else:
                    # If no threshold found, use trend direction
                    trend = forecast_values[-1] - forecast_values[0]
                    probability = 1 / (1 + np.exp(-trend * 0.1))  # Sigmoid to bound between 0 and 1
            
            # For numeric questions, we can check if the forecast is within the bounds
            elif isinstance(question, NumericQuestion):
                # Find the central forecast value
                center_forecast = forecast_values[-1]
                
                # Check where it falls within the question bounds
                lower_bound = question.lower_bound
                upper_bound = question.upper_bound
                
                # Scale to 0-1 range
                if upper_bound > lower_bound:
                    scaled_value = (center_forecast - lower_bound) / (upper_bound - lower_bound)
                    probability = max(0, min(1, scaled_value))
                else:
                    probability = 0.5
            
            # For date questions, calculate probability based on when the event is likely to occur
            elif isinstance(question, DateQuestion):
                # This would be more complex in practice
                probability = 0.5
            
            else:
                probability = 0.5
                
            return probability
            
        except Exception as e:
            logger.error(f"Error converting forecast to probability: {e}")
            return 0.5
    
    def _extract_threshold(self, question) -> Tuple[Optional[float], str]:
        """
        Extract a numeric threshold from a binary question.
        
        Args:
            question: The binary question to analyze
            
        Returns:
            Tuple of (threshold, direction): The threshold value and whether values should be above/below it
        """
        question_text = question.question_text.lower()
        resolution_criteria = (question.resolution_criteria or "").lower()
        combined_text = question_text + " " + resolution_criteria
        
        # Common patterns for thresholds
        import re
        
        # Look for patterns like "exceed X", "above X", "more than X"
        above_patterns = [
            r"exceed\s+(\d+(?:\.\d+)?)",
            r"above\s+(\d+(?:\.\d+)?)",
            r"more than\s+(\d+(?:\.\d+)?)",
            r"greater than\s+(\d+(?:\.\d+)?)",
            r"higher than\s+(\d+(?:\.\d+)?)",
            r"at least\s+(\d+(?:\.\d+)?)",
            r"reach\s+(\d+(?:\.\d+)?)",
            r"surpass\s+(\d+(?:\.\d+)?)"
        ]
        
        for pattern in above_patterns:
            matches = re.findall(pattern, combined_text)
            if matches:
                try:
                    return float(matches[0]), "above"
                except:
                    pass
        
        # Look for patterns like "below X", "less than X"
        below_patterns = [
            r"below\s+(\d+(?:\.\d+)?)",
            r"less than\s+(\d+(?:\.\d+)?)",
            r"under\s+(\d+(?:\.\d+)?)",
            r"lower than\s+(\d+(?:\.\d+)?)",
            r"at most\s+(\d+(?:\.\d+)?)",
            r"fall below\s+(\d+(?:\.\d+)?)"
        ]
        
        for pattern in below_patterns:
            matches = re.findall(pattern, combined_text)
            if matches:
                try:
                    return float(matches[0]), "below"
                except:
                    pass
        
        # No threshold found
        return None, "above"
    
    async def explain(self, question, context=None):
        """
        Return an explanation of the time series forecast.
        
        Args:
            question: The question to explain
            context: Optional context information
            
        Returns:
            str: Explanation of the forecast
        """
        # If time series isn't applicable or we don't have a last forecast, use fallback
        if not self._is_time_series_applicable(question) or not hasattr(self, '_last_forecast'):
            if self.fallback_forecaster:
                return await self.fallback_forecaster.explain(question, context)
            return "Time series analysis is not applicable to this question."
        
        try:
            # Ensure we have forecast data for this question
            question_key = self._get_data_key(question)
            if not hasattr(self, '_last_forecast') or self._last_forecast.get('question_key') != question_key:
                # Try to load from cache
                cache_path = os.path.join(self.cache_dir, f"{question_key}_forecast.json")
                if os.path.exists(cache_path):
                    with open(cache_path, 'r') as f:
                        self._last_forecast = json.load(f)
                else:
                    # No cached forecast, run prediction again
                    await self.predict(question, context)
            
            if not hasattr(self, '_last_forecast'):
                return "No forecast data available for explanation."
            
            # Create a detailed explanation
            forecast_result = self._last_forecast.get('forecast_result')
            model_type = self._last_forecast.get('model_type')
            probability = self._last_forecast.get('probability')
            
            explanation = f"# Time Series Forecast Analysis\n\n"
            
            # Explain the model used
            explanation += f"## Model: {model_type.upper()}\n\n"
            
            if model_type == "arima":
                explanation += "ARIMA (AutoRegressive Integrated Moving Average) models are used for forecasting time series data by capturing trends and seasonality.\n\n"
            elif model_type == "prophet":
                explanation += "Prophet is a forecasting model developed by Facebook that handles seasonality and trend shifts effectively.\n\n"
            elif model_type == "lstm":
                explanation += "LSTM (Long Short-Term Memory) neural networks are powerful for capturing complex patterns and long-term dependencies in time series data.\n\n"
            
            # Include summary information
            if forecast_result and 'summary' in forecast_result:
                explanation += "## Model Details\n\n"
                for key, value in forecast_result['summary'].items():
                    explanation += f"- **{key}**: {value}\n"
                explanation += "\n"
            
            # Explain the forecast trend
            if forecast_result and 'forecast' in forecast_result:
                forecast_df = forecast_result['forecast']
                first_value = forecast_df['forecast'].values[0]
                last_value = forecast_df['forecast'].values[-1]
                
                if last_value > first_value:
                    trend_direction = "upward"
                    percent_change = (last_value - first_value) / first_value * 100
                elif last_value < first_value:
                    trend_direction = "downward"
                    percent_change = (first_value - last_value) / first_value * 100
                else:
                    trend_direction = "flat"
                    percent_change = 0
                
                explanation += f"## Forecast Trend\n\n"
                explanation += f"The time series analysis shows a {trend_direction} trend "
                
                if trend_direction != "flat":
                    explanation += f"with a {percent_change:.1f}% change over the forecast period.\n\n"
                else:
                    explanation += "with no significant change over the forecast period.\n\n"
                
                # Add confidence information
                explanation += f"## Confidence Analysis\n\n"
                explanation += f"The forecast suggests a {probability:.1%} probability for the event, "
                
                # Calculate confidence band width
                avg_width = np.mean(forecast_df['upper_bound'] - forecast_df['lower_bound'])
                relative_width = avg_width / np.mean(forecast_df['forecast']) * 100
                
                if relative_width < 10:
                    confidence_level = "high"
                elif relative_width < 30:
                    confidence_level = "moderate"
                else:
                    confidence_level = "low"
                
                explanation += f"with a {confidence_level} level of confidence "
                explanation += f"(relative confidence interval width: {relative_width:.1f}%).\n\n"
            
            # Add specific insights for the question type
            if isinstance(question, BinaryQuestion):
                threshold, direction = self._extract_threshold(question)
                if threshold is not None:
                    explanation += f"## Threshold Analysis\n\n"
                    explanation += f"The question asks whether the value will be {direction} {threshold}.\n"
                    if forecast_result and 'forecast' in forecast_result:
                        forecast_values = forecast_result['forecast']['forecast'].values
                        if direction == "above":
                            percent_above = np.mean(forecast_values > threshold) * 100
                            explanation += f"Based on the forecast, values are projected to be above the threshold {percent_above:.1f}% of the time.\n\n"
                        else:
                            percent_below = np.mean(forecast_values < threshold) * 100
                            explanation += f"Based on the forecast, values are projected to be below the threshold {percent_below:.1f}% of the time.\n\n"
            
            # End with a conclusion
            explanation += "## Conclusion\n\n"
            explanation += f"Based on the time series analysis, the forecast suggests a **{probability:.1%}** probability for the outcome.\n"
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating time series explanation: {e}")
            if self.fallback_forecaster:
                return await self.fallback_forecaster.explain(question, context)
            return "Unable to generate a time series explanation."
    
    async def confidence_interval(self, question, context=None):
        """
        Return a confidence interval for the time series forecast.
        
        Args:
            question: The question to forecast
            context: Optional context information
            
        Returns:
            Tuple[float, float]: (lower_bound, upper_bound) probability interval
        """
        # If time series isn't applicable, use fallback
        if not self._is_time_series_applicable(question):
            if self.fallback_forecaster:
                return await self.fallback_forecaster.confidence_interval(question, context)
            return (0.3, 0.7)
        
        try:
            # Ensure we have forecast data for this question
            question_key = self._get_data_key(question)
            if not hasattr(self, '_last_forecast') or self._last_forecast.get('question_key') != question_key:
                # Try to load from cache
                cache_path = os.path.join(self.cache_dir, f"{question_key}_forecast.json")
                if os.path.exists(cache_path):
                    with open(cache_path, 'r') as f:
                        self._last_forecast = json.load(f)
                else:
                    # No cached forecast, run prediction again
                    await self.predict(question, context)
            
            if not hasattr(self, '_last_forecast'):
                return (0.3, 0.7)
            
            # Get the probability forecast
            probability = self._last_forecast.get('probability', 0.5)
            
            # Extract forecast uncertainty
            forecast_result = self._last_forecast.get('forecast_result')
            if forecast_result and 'forecast' in forecast_result:
                forecast_df = forecast_result['forecast']
                
                # Calculate relative uncertainty
                avg_forecast = np.mean(forecast_df['forecast'].values)
                avg_lower = np.mean(forecast_df['lower_bound'].values)
                avg_upper = np.mean(forecast_df['upper_bound'].values)
                
                # Convert time series uncertainty to probability uncertainty
                if avg_forecast > 0:
                    lower_ratio = avg_lower / avg_forecast
                    upper_ratio = avg_upper / avg_forecast
                    
                    # Apply these ratios to the probability
                    probability_lower = max(0, probability * lower_ratio)
                    probability_upper = min(1, probability * upper_ratio)
                    
                    # Ensure interval is not too narrow
                    min_width = 0.05  # Minimum width of 5 percentage points
                    if probability_upper - probability_lower < min_width:
                        mid = (probability_upper + probability_lower) / 2
                        probability_lower = max(0, mid - min_width/2)
                        probability_upper = min(1, mid + min_width/2)
                    
                    return (probability_lower, probability_upper)
            
            # Fallback: use a wider interval for time series (more uncertainty)
            interval_width = 0.3
            lower = max(0, probability - interval_width/2)
            upper = min(1, probability + interval_width/2)
            
            return (lower, upper)
            
        except Exception as e:
            logger.error(f"Error calculating time series confidence interval: {e}")
            if self.fallback_forecaster:
                return await self.fallback_forecaster.confidence_interval(question, context)
            return (0.3, 0.7)
    
    async def get_forecast_result(self, question, context=None):
        """
        Get a complete forecast result with all components.
        
        Args:
            question: The question to forecast
            context: Optional context information
            
        Returns:
            ForecastResult: Complete forecast result
        """
        probability = await self.predict(question, context)
        rationale = await self.explain(question, context)
        interval = await self.confidence_interval(question, context)
        
        # Create metadata about the time series forecast
        metadata = {}
        
        if hasattr(self, '_last_forecast'):
            forecast_result = self._last_forecast.get('forecast_result')
            if forecast_result and 'summary' in forecast_result:
                metadata['model_summary'] = forecast_result['summary']
            
            metadata['model_type'] = self._last_forecast.get('model_type')
            
            # Add visualization data (limited amount)
            if forecast_result:
                if 'forecast' in forecast_result:
                    # Convert DataFrame to list of dicts for JSON serialization
                    forecast_df = forecast_result['forecast']
                    metadata['forecast_data'] = {
                        'dates': [d.strftime('%Y-%m-%d') for d in forecast_df.index],
                        'values': forecast_df['forecast'].tolist(),
                        'lower_bound': forecast_df['lower_bound'].tolist(),
                        'upper_bound': forecast_df['upper_bound'].tolist()
                    }
                
                if 'historical' in forecast_result:
                    # Include a sample of historical data (last 30 points)
                    historical_df = forecast_result['historical']
                    sample_size = min(30, len(historical_df))
                    metadata['historical_data'] = {
                        'dates': [d.strftime('%Y-%m-%d') for d in historical_df.index[-sample_size:]],
                        'values': historical_df.iloc[-sample_size:, 0].tolist()
                    }
        
        return ForecastResult(
            probability=probability,
            confidence_interval=interval,
            rationale=rationale,
            model_name=f"TimeSeriesForecaster_{self.model_type}",
            metadata=metadata
        )
    
    def _is_time_series_applicable(self, question) -> bool:
        """
        Determine if time series analysis is applicable to this question.
        
        Args:
            question: The question to analyze
            
        Returns:
            bool: True if time series analysis is applicable
        """
        # Check if the question has any time-related keywords
        time_keywords = [
            "trend", "increase", "decrease", "growth", "decline", "rate", 
            "by the end of", "by year", "monthly", "weekly", "quarterly",
            "annual", "per year", "percent change", "percentage", "GDP",
            "stock", "price", "market", "index", "rate", "unemployment",
            "inflation", "population", "sales", "revenue", "temperature"
        ]
        
        question_text = question.question_text.lower()
        has_time_keywords = any(keyword in question_text for keyword in time_keywords)
        
        # For numeric questions, time series is often applicable
        is_numeric = isinstance(question, NumericQuestion)
        
        # For date questions, time series can help with event prediction
        is_date = isinstance(question, DateQuestion)
        
        # For binary questions, check if it's about a threshold being crossed
        is_threshold_binary = False
        if isinstance(question, BinaryQuestion):
            threshold_keywords = [
                "exceed", "above", "below", "more than", "less than",
                "greater than", "higher than", "lower than", "at least",
                "at most", "reach", "surpass", "fall below"
            ]
            is_threshold_binary = any(keyword in question_text for keyword in threshold_keywords)
        
        return (is_numeric or is_date or is_threshold_binary) and has_time_keywords
    
    def _get_data_key(self, question) -> str:
        """
        Generate a unique key for the question to use with data storage.
        
        Args:
            question: The question to generate a key for
            
        Returns:
            str: A unique key for the question
        """
        if hasattr(question, 'id_of_question') and question.id_of_question:
            return f"q_{question.id_of_question}"
        return f"q_{hash(question.question_text)}"
    
    async def _fetch_time_series_data(self, question, context=None) -> Optional[pd.DataFrame]:
        """
        Fetch time series data relevant to the question.
        
        Args:
            question: The question to fetch data for
            context: Optional context information
            
        Returns:
            pd.DataFrame or None: DataFrame with time series data (datetime index, value column)
        """
        # Check if we have cached data
        data_key = self._get_data_key(question)
        data_path = os.path.join(self.data_dir, f"{data_key}.csv")
        
        if os.path.exists(data_path):
            logger.info(f"Loading cached time series data from {data_path}")
            try:
                return pd.read_csv(data_path, parse_dates=['date'], index_col='date')
            except Exception as e:
                logger.error(f"Error loading cached data: {e}")
        
        if not self.auto_fetch:
            logger.warning("Auto fetch disabled and no cached data available")
            return None
        
        # Identify the data series needed based on the question
        # This could be extended to use an API or structured data source
        try:
            # Extract entities from the question that we need data for
            entities = self._extract_entities(question)
            if not entities:
                logger.warning("No entities identified in the question")
                return None
            
            # For each entity, try to fetch relevant data
            dfs = []
            for entity, entity_type in entities:
                entity_data = await self._fetch_entity_data(entity, entity_type, question)
                if entity_data is not None:
                    dfs.append(entity_data)
            
            if not dfs:
                logger.warning("No data could be fetched for the identified entities")
                return None
            
            # Combine the data if we have multiple entities
            if len(dfs) == 1:
                combined_df = dfs[0]
            else:
                # Merge all dataframes on date
                combined_df = dfs[0]
                for df in dfs[1:]:
                    combined_df = pd.merge(combined_df, df, left_index=True, right_index=True, how='outer')
                
                # Fill missing values with forward fill then backward fill
                combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
            
            # Save the data for future use
            combined_df.to_csv(data_path)
            logger.info(f"Saved time series data to {data_path}")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error fetching time series data: {e}")
            return None
    
    def _extract_entities(self, question) -> List[Tuple[str, str]]:
        """
        Extract entities from the question that might have associated time series data.
        
        Args:
            question: The question to extract entities from
            
        Returns:
            List of (entity, entity_type) tuples
        """
        # This could be enhanced with NLP/entity recognition
        # For now, using a simple keyword approach
        
        question_text = question.question_text.lower()
        background = (question.background_info or "").lower()
        combined_text = question_text + " " + background
        
        entities = []
        
        # Check for economic indicators
        economic_indicators = {
            "gdp": "economic", 
            "inflation": "economic",
            "unemployment": "economic",
            "interest rate": "economic",
            "consumer price index": "economic",
            "cpi": "economic",
            "stock market": "financial",
            "s&p 500": "financial",
            "s&p500": "financial",
            "dow jones": "financial",
            "nasdaq": "financial",
            "bitcoin": "crypto",
            "btc": "crypto",
            "ethereum": "crypto",
            "eth": "crypto"
        }
        
        for indicator, indicator_type in economic_indicators.items():
            if indicator in combined_text:
                entities.append((indicator, indicator_type))
        
        # Check for countries/regions
        countries = [
            "us", "usa", "united states", "china", "europe", "eu", "uk", 
            "united kingdom", "japan", "india", "global", "world"
        ]
        
        for country in countries:
            country_pattern = f" {country} "  # Spaces to avoid partial matches
            if country_pattern in f" {combined_text} ":
                # Pair country with any economic indicators found
                for entity, entity_type in entities:
                    if entity_type in ["economic", "financial"]:
                        entities.append((f"{country}_{entity}", entity_type))
        
        return entities
    
    async def _fetch_entity_data(self, entity: str, entity_type: str, question) -> Optional[pd.DataFrame]:
        """
        Fetch time series data for a specific entity.
        
        Args:
            entity: The entity to fetch data for
            entity_type: The type of entity
            question: The original question (for context)
            
        Returns:
            pd.DataFrame or None: DataFrame with time series data
        """
        # This method would connect to data sources or APIs
        # For now, we'll simulate with some synthetic data
        
        # In a real implementation, this would use proper data sources
        # such as FRED API for economic data, Yahoo Finance for stocks, etc.
        
        try:
            # Generate synthetic data based on entity type
            today = datetime.now()
            date_range = pd.date_range(end=today, periods=365, freq='D')
            
            if entity_type == "economic":
                # Economic data tends to change slowly with some seasonality
                base = 100
                trend = np.linspace(0, 30, len(date_range)) 
                seasonality = 10 * np.sin(np.linspace(0, 12*np.pi, len(date_range)))
                noise = np.random.normal(0, 5, len(date_range))
                values = base + trend + seasonality + noise
                
            elif entity_type == "financial":
                # Financial data often follows random walk with drift
                base = 100
                drift = np.linspace(0, 50, len(date_range))
                random_walk = np.cumsum(np.random.normal(0, 10, len(date_range)))
                values = base + drift + random_walk
                
            elif entity_type == "crypto":
                # Crypto data can be more volatile
                base = 1000
                drift = np.linspace(0, 500, len(date_range))
                random_walk = np.cumsum(np.random.normal(0, 50, len(date_range)))
                values = base + drift + random_walk
                
            else:
                # Default case
                base = 100
                trend = np.linspace(0, 20, len(date_range))
                noise = np.random.normal(0, 10, len(date_range))
                values = base + trend + noise
            
            # Create DataFrame
            df = pd.DataFrame({
                entity: values
            }, index=date_range)
            
            # Ensure the index is named 'date'
            df.index.name = 'date'
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for entity {entity}: {e}")
            return None
    
    def _select_best_model(self, data: pd.DataFrame, question) -> str:
        """
        Select the best time series model based on the data and question.
        
        Args:
            data: DataFrame with time series data
            question: The question to forecast
            
        Returns:
            str: Model type to use ('arima', 'prophet', or 'lstm')
        """
        if self.model_type != "auto":
            return self.model_type
            
        # Check data characteristics to select appropriate model
        n_points = len(data)
        
        # For very short series, use simple ARIMA
        if n_points < 30:
            return "arima"
            
        # For medium length series with potential seasonality, use Prophet
        if n_points < 100 and PROPHET_AVAILABLE:
            return "prophet"
            
        # For longer series with complex patterns, use LSTM
        if n_points >= 100 and TENSORFLOW_AVAILABLE:
            return "lstm"
            
        # Default fallback
        return "arima"
    
    async def _forecast_with_arima(self, data: pd.DataFrame, horizon: int = 30) -> dict:
        """
        Generate forecasts using ARIMA model.
        
        Args:
            data: DataFrame with time series data
            horizon: Forecast horizon in days
            
        Returns:
            dict: Dictionary with forecast results
        """
        try:
            # Use the first column if multiple columns exist
            if len(data.columns) > 1:
                logger.info(f"Multiple columns found in data, using the first one: {data.columns[0]}")
                series = data[data.columns[0]]
            else:
                series = data[data.columns[0]]
                
            # Ensure data is sampled daily
            if data.index.freq is None or data.index.freq.name != 'D':
                # Resample to daily frequency if needed
                series = series.resample('D').mean()
                # Fill missing values
                series = series.fillna(method='ffill').fillna(method='bfill')
            
            # Simple auto-ARIMA approach
            # In a full implementation, could use pmdarima for auto-ARIMA
            try:
                # First try with seasonal component (common in economic data)
                model = sm.tsa.statespace.SARIMAX(
                    series,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)
            except:
                # Fallback to non-seasonal ARIMA
                model = ARIMA(series, order=(1, 1, 1))
                results = model.fit()
            
            # Generate forecast
            forecast = results.forecast(steps=horizon)
            
            # Calculate confidence intervals
            # In ARIMA these are usually based on the standard error
            pred_ci = results.get_forecast(steps=horizon).conf_int(alpha=0.05)
            lower_bound = pred_ci.iloc[:, 0]
            upper_bound = pred_ci.iloc[:, 1]
            
            # Format the results
            forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=horizon)
            
            forecast_df = pd.DataFrame({
                'forecast': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }, index=forecast_dates)
            
            # For visualization, include some historical data
            historical = series.iloc[-60:] if len(series) > 60 else series
            historical_df = pd.DataFrame({
                'historical': historical
            })
            
            # Create a summary of the model
            summary = {
                'model_type': 'ARIMA',
                'parameters': str(results.specification),
                'AIC': results.aic,
                'BIC': results.bic
            }
            
            return {
                'forecast': forecast_df,
                'historical': historical_df,
                'summary': summary
            }
        
        except Exception as e:
            logger.error(f"Error in ARIMA forecasting: {e}")
            raise 
    
    async def _forecast_with_prophet(self, data: pd.DataFrame, horizon: int = 30) -> dict:
        """
        Generate forecasts using Facebook Prophet model.
        
        Args:
            data: DataFrame with time series data
            horizon: Forecast horizon in days
            
        Returns:
            dict: Dictionary with forecast results
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available. Install with 'pip install prophet'")
            
        try:
            # Use the first column if multiple columns exist
            if len(data.columns) > 1:
                series = data[data.columns[0]]
            else:
                series = data[data.columns[0]]
                
            # Prophet requires 'ds' and 'y' columns
            prophet_df = pd.DataFrame({
                'ds': data.index,
                'y': series.values
            })
            
            # Initialize and fit the Prophet model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                seasonality_mode='multiplicative',
                interval_width=0.95
            )
            
            # Add holidays if available (future enhancement)
            # model.add_country_holidays(country_name='US')
            
            # Fit the model
            model.fit(prophet_df)
            
            # Make future dataframe for prediction
            future = model.make_future_dataframe(periods=horizon)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Format the results
            forecast_start_idx = len(prophet_df)
            forecast_part = forecast.iloc[forecast_start_idx:].copy()
            historical_part = forecast.iloc[:forecast_start_idx].copy()
            
            # Convert to same format as ARIMA output
            forecast_df = pd.DataFrame({
                'forecast': forecast_part['yhat'],
                'lower_bound': forecast_part['yhat_lower'],
                'upper_bound': forecast_part['yhat_upper'],
                'trend': forecast_part['trend']
            }, index=pd.DatetimeIndex(forecast_part['ds']))
            
            historical_df = pd.DataFrame({
                'historical': prophet_df['y'],
                'fitted': historical_part['yhat'],
                'trend': historical_part['trend']
            }, index=pd.DatetimeIndex(prophet_df['ds']))
            
            # Create a summary of the model components
            model_components = model.component_modes()
            summary = {
                'model_type': 'Prophet',
                'components': model_components,
                'seasonality_mode': model.seasonality_mode,
                'changepoints': list(model.changepoints.astype(str))
            }
            
            return {
                'forecast': forecast_df,
                'historical': historical_df,
                'summary': summary,
                'prophet_forecast': forecast  # Keep full Prophet output for visualization
            }
        
        except Exception as e:
            logger.error(f"Error in Prophet forecasting: {e}")
            raise
            
    async def _forecast_with_lstm(self, data: pd.DataFrame, horizon: int = 30) -> dict:
        """
        Generate forecasts using LSTM neural network.
        
        Args:
            data: DataFrame with time series data
            horizon: Forecast horizon in days
            
        Returns:
            dict: Dictionary with forecast results
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with 'pip install tensorflow'")
            
        try:
            # Use the first column if multiple columns exist
            if len(data.columns) > 1:
                series = data[data.columns[0]]
            else:
                series = data[data.columns[0]]
                
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
            
            # Prepare data for LSTM (create sequences)
            def create_sequences(data, seq_length):
                xs, ys = [], []
                for i in range(len(data) - seq_length):
                    x = data[i:i+seq_length]
                    y = data[i+seq_length]
                    xs.append(x)
                    ys.append(y)
                return np.array(xs), np.array(ys)
            
            # Choose sequence length based on data characteristics
            # For daily data, using a month of lookback is common
            seq_length = 30 if len(scaled_data) > 60 else len(scaled_data) // 3
            
            # Create training sequences
            X, y = create_sequences(scaled_data, seq_length)
            
            # Reshape for LSTM [samples, time steps, features]
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Build a simple LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            
            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Early stopping to prevent overfitting
            early_stopping = EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train the model
            model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                verbose=0,
                callbacks=[early_stopping]
            )
            
            # Generate predictions one step at a time for the forecast horizon
            # First, we need the latest sequence of data
            last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
            
            # Now forecast step by step
            forecast_scaled = []
            current_sequence = last_sequence.copy()
            
            for _ in range(horizon):
                # Predict next value
                next_pred = model.predict(current_sequence, verbose=0)[0][0]
                forecast_scaled.append(next_pred)
                
                # Update sequence for next prediction
                current_sequence = np.append(
                    current_sequence[:, 1:, :],
                    [[next_pred]],
                    axis=1
                )
            
            # Invert scaling to get actual values
            forecast_values = scaler.inverse_transform(
                np.array(forecast_scaled).reshape(-1, 1)
            ).flatten()
            
            # Since LSTM doesn't natively provide confidence intervals,
            # we'll estimate them based on model error on training data
            y_pred = model.predict(X, verbose=0).flatten()
            y_true = y.flatten()
            
            # Calculate mean absolute error
            mae = np.mean(np.abs(y_pred - y_true))
            
            # Estimate confidence intervals as prediction ± 2*MAE (approximately 95% CI)
            lower_bound = forecast_values - 2 * mae
            upper_bound = forecast_values + 2 * mae
            
            # Format results
            forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=horizon)
            
            forecast_df = pd.DataFrame({
                'forecast': forecast_values,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }, index=forecast_dates)
            
            # Include some historical data
            historical = series.iloc[-60:] if len(series) > 60 else series
            historical_df = pd.DataFrame({
                'historical': historical
            })
            
            # Create a summary
            summary = {
                'model_type': 'LSTM',
                'sequence_length': seq_length,
                'training_loss': model.history.history['loss'][-1],
                'mean_absolute_error': mae
            }
            
            return {
                'forecast': forecast_df,
                'historical': historical_df,
                'summary': summary
            }
        
        except Exception as e:
            logger.error(f"Error in LSTM forecasting: {e}")
            raise 