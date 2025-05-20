import logging
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import asyncio

import streamlit as st
from pydantic import BaseModel

from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.util.jsonable import Jsonable
from forecasting_tools.forecast_helpers.backtest_manager import BacktestManager
from forecasting_tools.ai_models.model_interfaces.forecaster_base import ForecasterBase
from forecasting_tools.ai_models.model_interfaces.ensemble_forecaster import EnsembleForecaster
from metrics import brier_score, calibration_curve, coverage, peer_score

logger = logging.getLogger(__name__)

class ModelPerformanceInput(Jsonable, BaseModel):
    action: str  # "view", "backtest", "export"
    models: list = []
    num_questions: int = 50
    categories: list = []
    metric: str = "brier_score"

class ModelPerformanceOutput(Jsonable, BaseModel):
    success: bool
    message: str
    data: dict = {}

class ModelPerformancePage(ToolPage):
    PAGE_DISPLAY_NAME: str = "📊 Model Performance"
    URL_PATH: str = "/model-performance"
    INPUT_TYPE = ModelPerformanceInput
    OUTPUT_TYPE = ModelPerformanceOutput

    # Form input keys
    ACTION_SELECT = "action_select_performance"
    MODEL_SELECT = "model_select_performance"
    NUM_QUESTIONS = "num_questions_performance"
    CATEGORY_SELECT = "category_select_performance"
    METRIC_SELECT = "metric_select_performance"
    TAB_SELECT = "tab_select_performance"
    
    # Constants
    DEFAULT_BACKTEST_PATH = "forecasting_tools/data/backtest"
    
    # Model registry (to be populated with available forecaster models)
    MODEL_REGISTRY = {}

    @classmethod
    async def _display_intro_text(cls) -> None:
        st.write(
            "This dashboard allows you to evaluate forecasting model performance, run backtests, "
            "view calibration plots, and compare models on a leaderboard."
        )
        st.info(
            "Backtesting runs forecasting models on historical resolved questions to measure "
            "their accuracy and compare performance. This helps identify which models perform "
            "best on different types of questions."
        )

    @classmethod
    def _get_available_models(cls) -> Dict[str, ForecasterBase]:
        """Get dictionary of available forecaster models."""
        # This is a placeholder - in a real implementation, you would
        # discover and load available forecaster models dynamically
        
        # For demo purposes, we're returning a static set of models
        # These would be initialized with actual forecaster instances in practice
        
        if not cls.MODEL_REGISTRY:
            # Import some forecasters for demonstration
            try:
                from forecasting_tools.ai_models.model_interfaces.ensemble_forecaster import EnsembleForecaster
                
                # Create some placeholder models (would be actual instances in a real implementation)
                cls.MODEL_REGISTRY = {
                    "GeneralLLM": None,  # Placeholder
                    "ExpertForecaster": None,  # Placeholder
                    "EnsembleForecaster": None,  # Placeholder
                    "CalibratedForecaster": None,  # Placeholder
                    "TimeSeriesForecaster": None,  # Placeholder
                }
            except ImportError:
                logger.warning("Could not import forecaster models")
                
        return cls.MODEL_REGISTRY

    @classmethod
    async def _get_input(cls) -> ModelPerformanceInput | None:
        # Get available models
        available_models = cls._get_available_models()
        
        # Create tabs for different dashboard sections
        tabs = st.tabs(["Model Leaderboard", "Calibration Plots", "Run Backtest", "Settings"])
        
        with tabs[0]:  # Model Leaderboard
            st.subheader("Model Performance Leaderboard")
            
            # Create a manager to access backtest results
            backtest_manager = BacktestManager(data_dir=cls.DEFAULT_BACKTEST_PATH)
            
            # Get the leaderboard data
            col1, col2 = st.columns([3, 1])
            
            with col2:
                metric = st.selectbox(
                    "Sort by metric",
                    options=[
                        "brier_score", "calibration_error", "coverage", 
                        "accuracy", "log_score", "peer_score"
                    ],
                    format_func=lambda x: x.replace("_", " ").title(),
                    key=cls.METRIC_SELECT
                )
                
                min_predictions = st.slider(
                    "Minimum predictions",
                    min_value=5,
                    max_value=100,
                    value=10,
                    step=5,
                    help="Minimum number of predictions required for a model to be included"
                )
            
            with col1:
                leaderboard_df = backtest_manager.get_leaderboard(metric=metric, min_predictions=min_predictions)
                
                if leaderboard_df.empty:
                    st.warning("No backtest data available. Run a backtest first.")
                else:
                    # Format the DataFrame for display
                    display_df = leaderboard_df.copy()
                    
                    # Rename columns for display
                    rename_map = {
                        'model_name': 'Model',
                        'brier_score': 'Brier Score',
                        'calibration_error': 'Calibration Error',
                        'coverage': 'Coverage',
                        'sharpness': 'Sharpness',
                        'accuracy': 'Accuracy',
                        'log_score': 'Log Score',
                        'peer_score': 'Peer Score',
                        'sample_count': 'Predictions'
                    }
                    display_df.rename(columns=rename_map, inplace=True)
                    
                    # Format scores to be more readable
                    for col in ['Brier Score', 'Calibration Error', 'Log Score', 'Peer Score']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
                    
                    for col in ['Coverage', 'Accuracy']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].map(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
                    
                    # Show the leaderboard with highlighting
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Model": st.column_config.TextColumn("Model"),
                            "Predictions": st.column_config.NumberColumn("Predictions", format="%d")
                        }
                    )
            
            if st.button("Export Leaderboard", key="export_leaderboard"):
                if not leaderboard_df.empty:
                    csv = leaderboard_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="model_leaderboard.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data to export")
        
        with tabs[1]:  # Calibration Plots
            st.subheader("Calibration Plots")
            
            # Create a manager to access backtest results
            backtest_manager = BacktestManager(data_dir=cls.DEFAULT_BACKTEST_PATH)
            
            # Get available models from backtest results
            if not backtest_manager.results_df.empty:
                available_backtest_models = backtest_manager.results_df['model_name'].unique().tolist()
            else:
                available_backtest_models = []
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                selected_models = st.multiselect(
                    "Select models to display",
                    options=available_backtest_models,
                    default=available_backtest_models[:min(3, len(available_backtest_models))],
                    key="calib_models"
                )
                
                plot_type = st.selectbox(
                    "Plot type",
                    options=["Plotly Interactive", "Matplotlib Static"],
                    index=0,
                    key="plot_type"
                )
                
                n_bins = st.slider(
                    "Number of bins",
                    min_value=5,
                    max_value=20,
                    value=10,
                    step=1,
                    key="n_bins"
                )
            
            with col1:
                if not available_backtest_models:
                    st.warning("No backtest data available. Run a backtest first.")
                elif not selected_models:
                    st.info("Select one or more models to display calibration plots.")
                else:
                    # Get calibration data for selected models
                    calibration_data = {}
                    for model in selected_models:
                        model_data = backtest_manager.get_calibration_data(model)
                        if model in model_data:
                            calibration_data[model] = model_data[model]
                    
                    if not calibration_data:
                        st.warning("No calibration data available for selected models.")
                    else:
                        if plot_type == "Plotly Interactive":
                            # Create interactive Plotly calibration plot
                            fig = go.Figure()
                            
                            # Add perfect calibration line
                            fig.add_trace(go.Scatter(
                                x=[0, 1],
                                y=[0, 1],
                                mode='lines',
                                name='Perfect Calibration',
                                line=dict(dash='dash', color='black')
                            ))
                            
                            # Add each model's calibration curve
                            for model, data in calibration_data.items():
                                prob_pred = data['prob_pred']
                                prob_true = data['prob_true']
                                bin_total = data['bin_total']
                                
                                # Calculate marker size based on bin count
                                size = np.array(bin_total) / max(max(bin_total), 1) * 15 + 5
                                
                                fig.add_trace(go.Scatter(
                                    x=prob_pred,
                                    y=prob_true,
                                    mode='markers',
                                    name=model,
                                    marker=dict(size=size),
                                    hovertemplate=
                                    'Predicted: %{x:.2f}<br>'+
                                    'Observed: %{y:.2f}<br>'+
                                    'Samples: %{text}<extra></extra>',
                                    text=bin_total
                                ))
                            
                            fig.update_layout(
                                title='Calibration Curve',
                                xaxis_title='Predicted Probability',
                                yaxis_title='Observed Frequency',
                                xaxis=dict(range=[0, 1]),
                                yaxis=dict(range=[0, 1]),
                                height=500,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        else:
                            # Create static Matplotlib calibration plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Add perfect calibration line
                            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                            
                            # Add each model's calibration curve
                            for model, data in calibration_data.items():
                                prob_pred = data['prob_pred']
                                prob_true = data['prob_true']
                                bin_total = data['bin_total']
                                
                                ax.scatter(prob_pred, prob_true, s=np.array(bin_total) / max(max(bin_total), 1) * 100 + 20, 
                                        label=model, alpha=0.7)
                            
                            ax.set_xlim(0, 1)
                            ax.set_ylim(0, 1)
                            ax.set_xlabel('Predicted Probability')
                            ax.set_ylabel('Observed Frequency')
                            ax.set_title('Calibration Curve')
                            ax.grid(alpha=0.3)
                            ax.legend()
                            
                            st.pyplot(fig)
        
        with tabs[2]:  # Run Backtest
            st.subheader("Run Backtest on Historical Questions")
            
            with st.form("backtest_form"):
                available_model_names = list(available_models.keys())
                
                selected_models = st.multiselect(
                    "Select models to test",
                    options=available_model_names,
                    default=available_model_names[:min(2, len(available_model_names))],
                    key=cls.MODEL_SELECT
                )
                
                num_questions = st.slider(
                    "Number of questions",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    key=cls.NUM_QUESTIONS
                )
                
                # Placeholder for actual categories in a real implementation
                categories = st.multiselect(
                    "Question categories (optional)",
                    options=["Economics", "Politics", "Science", "Technology", "Health", "Other"],
                    default=[],
                    key=cls.CATEGORY_SELECT
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    run_button = st.form_submit_button("Run Backtest")
                
                with col2:
                    append_results = st.checkbox("Append to existing results", value=True,
                                              help="If checked, new results will be added to existing data")
                
                if run_button:
                    if not selected_models:
                        st.error("Please select at least one model to test.")
                    else:
                        return ModelPerformanceInput(
                            action="backtest",
                            models=selected_models,
                            num_questions=num_questions,
                            categories=categories if categories else []
                        )
        
        with tabs[3]:  # Settings
            st.subheader("Settings")
            
            backtest_manager = BacktestManager(data_dir=cls.DEFAULT_BACKTEST_PATH)
            
            st.write("Backtest Data Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export Backtest Data"):
                    if not backtest_manager.results_df.empty:
                        # Get file format
                        format = st.selectbox(
                            "Export format",
                            options=["csv", "json"],
                            index=0
                        )
                        
                        # Export the data
                        export_path = backtest_manager.export_results(format=format)
                        
                        if export_path:
                            try:
                                with open(export_path, 'r') as f:
                                    file_data = f.read()
                                
                                st.download_button(
                                    label=f"Download {format.upper()}",
                                    data=file_data,
                                    file_name=f"backtest_results.{format}",
                                    mime=f"text/{'csv' if format == 'csv' else 'json'}"
                                )
                            except Exception as e:
                                st.error(f"Error preparing download: {e}")
                        else:
                            st.error("Error exporting results")
                    else:
                        st.warning("No backtest data to export")
            
            with col2:
                if st.button("Clear Backtest Data", type="secondary"):
                    if st.checkbox("Confirm data deletion"):
                        # Create a new empty DataFrame with correct columns
                        empty_df = pd.DataFrame({
                            'question_id': [],
                            'question_text': [],
                            'model_name': [],
                            'prediction': [],
                            'confidence_interval_lower': [],
                            'confidence_interval_upper': [],
                            'outcome': [],
                            'prediction_time': [],
                            'resolution_time': [],
                            'category': [],
                            'difficulty': [],
                            'tags': []
                        })
                        
                        # Replace current data with empty DataFrame
                        backtest_manager.results_df = empty_df
                        backtest_manager._save_results()
                        st.success("Backtest data cleared successfully")
                        
                        # Refresh the page to show updated state
                        st.rerun()
            
            # Display data summary
            if not backtest_manager.results_df.empty:
                st.write("Backtest Data Summary")
                
                # Summary stats
                total_questions = backtest_manager.results_df['question_id'].nunique()
                total_forecasts = len(backtest_manager.results_df)
                total_models = backtest_manager.results_df['model_name'].nunique()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Questions", total_questions)
                col2.metric("Models", total_models)
                col3.metric("Total Forecasts", total_forecasts)
                
                # Show sample of data
                with st.expander("Preview Data"):
                    st.dataframe(backtest_manager.results_df.head(10))
        
        return None  # For now, only return if running a backtest

    @classmethod
    async def _run_tool(cls, input: ModelPerformanceInput) -> ModelPerformanceOutput:
        if input.action == "backtest":
            with st.spinner("Running backtest on historical questions..."):
                try:
                    # Create a BacktestManager
                    backtest_manager = BacktestManager(data_dir=cls.DEFAULT_BACKTEST_PATH)
                    
                    # Get forecaster instances for selected models
                    # NOTE: In a real implementation, you would load actual forecaster instances
                    # This is just a placeholder for demonstration
                    forecasters = {}
                    for model_name in input.models:
                        # Create mock forecaster instance (would be real in actual implementation)
                        forecaster = await cls._create_mock_forecaster(model_name)
                        forecasters[model_name] = forecaster
                    
                    # Run the backtest
                    results_df = await backtest_manager.run_backtest(
                        forecasters=forecasters,
                        num_questions=input.num_questions,
                        categories=input.categories if input.categories else None
                    )
                    
                    # Calculate metrics
                    metrics = backtest_manager.calculate_metrics()
                    
                    return ModelPerformanceOutput(
                        success=True,
                        message=f"Successfully ran backtest with {len(forecasters)} forecasters on {input.num_questions} questions.",
                        data={
                            "results_shape": results_df.shape,
                            "metrics": metrics
                        }
                    )
                    
                except Exception as e:
                    logger.error(f"Error running backtest: {e}")
                    return ModelPerformanceOutput(
                        success=False,
                        message=f"Error running backtest: {str(e)}"
                    )
        
        return ModelPerformanceOutput(
            success=False,
            message="Invalid action or not implemented"
        )

    @classmethod
    async def _display_outputs(cls, outputs: list[ModelPerformanceOutput]) -> None:
        if not outputs:
            return
        
        output = outputs[0]
        
        if output.success:
            st.success(output.message)
            
            # Display results if available
            if "results_shape" in output.data:
                rows, cols = output.data["results_shape"]
                st.write(f"Added {rows} predictions to the backtest database.")
                
                # Display metrics
                if "metrics" in output.data and output.data["metrics"]:
                    st.subheader("Backtest Results")
                    
                    # Display metrics for each model in a tabular format
                    metrics_data = []
                    for model, metrics in output.data["metrics"].items():
                        metrics_row = {"Model": model}
                        metrics_row.update(metrics)
                        metrics_data.append(metrics_row)
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    # Format for display
                    display_cols = ["Model", "brier_score", "calibration_error", "accuracy", "sample_count"]
                    display_df = metrics_df[display_cols].copy() if set(display_cols).issubset(metrics_df.columns) else metrics_df
                    
                    # Rename columns
                    rename_map = {
                        "brier_score": "Brier Score",
                        "calibration_error": "Calibration Error",
                        "accuracy": "Accuracy",
                        "sample_count": "Samples"
                    }
                    display_df.rename(columns=rename_map, inplace=True)
                    
                    # Format scores
                    for col in ["Brier Score", "Calibration Error"]:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
                    
                    if "Accuracy" in display_df.columns:
                        display_df["Accuracy"] = display_df["Accuracy"].map(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
                    
                    st.table(display_df)
                    
                # Show a link to view the full leaderboard
                st.write("View the Model Leaderboard tab to see complete results and compare all models.")
        else:
            st.error(output.message)

    @classmethod
    async def _save_run_to_coda(
        cls,
        input_to_tool: Jsonable,
        output: Jsonable,
        is_premade_example: bool,
    ) -> None:
        # Not saving to Coda for this page
        pass
    
    @classmethod
    async def _create_mock_forecaster(cls, model_name: str) -> ForecasterBase:
        """
        Create a mock forecaster for demonstration purposes.
        
        In a real implementation, you would load or create actual forecaster instances.
        This mock implementation simulates different forecasting behaviors.
        """
        # Create a mock forecaster that generates different quality predictions
        # based on the model name
        from forecasting_tools.ai_models.model_interfaces.forecaster_base import ForecasterBase
        
        class MockForecaster(ForecasterBase):
            def __init__(self, name, quality=0.7):
                self.model_name = name
                self.quality = quality  # 0-1 quality factor (higher = better)
            
            async def predict(self, question, context=None):
                # Simulate prediction based on quality and question
                # In a real implementation, this would use actual prediction logic
                
                # Generate a baseline prediction
                if hasattr(question, 'resolve_to_true'):
                    # For demonstration, we'll peek at the answer but add noise
                    # In a real situation, the model wouldn't know the true outcome
                    true_outcome = 1.0 if question.resolve_to_true else 0.0
                    
                    # Add noise based on quality (higher quality = less noise)
                    noise = np.random.normal(0, 1.0 - self.quality)
                    
                    # Mix the true outcome with noise
                    raw_prediction = self.quality * true_outcome + (1.0 - self.quality) * 0.5 + noise * 0.2
                    
                    # Clip to valid probability range
                    prediction = max(0.01, min(0.99, raw_prediction))
                    return prediction
                else:
                    # If we don't know the truth, return a value based on model name hash
                    import hashlib
                    name_hash = int(hashlib.md5(question.question_text.encode()).hexdigest(), 16)
                    return 0.2 + (name_hash % 600) / 1000.0  # Range 0.2-0.8
            
            async def explain(self, question, context=None):
                return f"Mock explanation from {self.model_name}"
            
            async def confidence_interval(self, question, context=None):
                # Generate confidence interval based on quality
                # Higher quality models have narrower intervals
                prediction = await self.predict(question, context)
                width = 0.4 * (1.0 - self.quality)
                lower = max(0.01, prediction - width/2)
                upper = min(0.99, prediction + width/2)
                return (lower, upper)
        
        # Assign different quality factors to different models
        quality_map = {
            "GeneralLLM": 0.65,
            "ExpertForecaster": 0.78,
            "EnsembleForecaster": 0.82,
            "CalibratedForecaster": 0.80,
            "TimeSeriesForecaster": 0.75
        }
        
        quality = quality_map.get(model_name, 0.7)
        return MockForecaster(model_name, quality) 