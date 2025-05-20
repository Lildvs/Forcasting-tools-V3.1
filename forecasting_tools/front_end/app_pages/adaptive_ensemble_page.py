import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
from typing import List, Dict, Optional
import datetime
from PIL import Image
from io import BytesIO

# Add the root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from forecasting_tools.front_end.helpers.app_page import AppPage
from forecasting_tools.ai_models.model_interfaces.adaptive_ensemble_forecaster import AdaptiveEnsembleForecaster
from forecasting_tools.ai_models.model_interfaces.forecaster import Forecaster
from forecasting_tools.data_models.base_types import ForecastQuestion, Forecast
from forecasting_tools.forecast_helpers.metrics import (
    brier_score_df, calibration_curve_df, coverage_df, peer_score_df,
    calibration_error_df, sharpness_df, time_weighted_brier_score,
    model_performance_over_time, aggregate_metrics
)

# Import available forecasters
from forecasting_tools.ai_models.model_registry import get_available_forecasters


class AdaptiveEnsemblePage(AppPage):
    PAGE_DISPLAY_NAME: str = "🔄 Adaptive Ensemble"
    URL_PATH: str = "/adaptive-ensemble"
    IS_DEFAULT_PAGE: bool = False

    @classmethod
    def main(cls):
        st.title("Adaptive Ensemble Forecaster")
        
        # Add explanatory text
        st.markdown("""
        This page demonstrates an advanced ensemble forecaster that combines multiple models and 
        automatically adapts weights based on historical performance. The ensemble leverages metrics like 
        Brier Score, Calibration, Peer Score, and Coverage to continuously improve accuracy over time.
        """)
        
        # Sidebar configuration
        st.sidebar.header("Ensemble Configuration")
        
        # Get available forecasters
        available_forecasters = get_available_forecasters()
        
        # Default selected forecasters (choose 3-5 diverse ones if available)
        default_selected = cls._get_default_selection(available_forecasters)
        
        # Let user select forecasters
        selected_forecaster_names = st.sidebar.multiselect(
            "Select Forecasters to Ensemble",
            list(available_forecasters.keys()),
            default=default_selected
        )
        
        # Error if no forecasters selected
        if not selected_forecaster_names:
            st.warning("Please select at least one forecaster to continue.")
            return
        
        # Get instances of selected forecasters
        selected_forecasters = [
            available_forecasters[name] for name in selected_forecaster_names
        ]
        
        # Ensemble method selection
        ensemble_method = st.sidebar.selectbox(
            "Ensemble Method",
            ["equal_weights", "dynamic_weights", "stacking"],
            index=1,  # Default to dynamic weights
            format_func=lambda x: {
                "equal_weights": "Equal Weights",
                "dynamic_weights": "Dynamic Weights (auto-updated)",
                "stacking": "Stacking (meta-learner)"
            }.get(x, x)
        )
        
        # Advanced options (collapsible)
        with st.sidebar.expander("Advanced Options"):
            window_size = st.number_input(
                "Window Size (# questions)",
                min_value=5,
                max_value=100,
                value=20,
                help="Number of most recent questions to use for dynamic weights"
            )
            
            half_life = st.number_input(
                "Half-Life (days)",
                min_value=1,
                max_value=180,
                value=30,
                help="Time-weighting half-life. Lower values focus more on recent performance."
            )
            
            calibration_correction = st.checkbox(
                "Apply Calibration Correction",
                value=True,
                help="Automatically correct for calibration issues based on historical data"
            )
        
        # Create ensemble forecaster
        ensemble_name = "Adaptive Ensemble"
        ensemble = AdaptiveEnsembleForecaster(
            forecasters=selected_forecasters,
            ensemble_method=ensemble_method,
            window_size=int(window_size),
            half_life_days=float(half_life),
            calibration_correction=calibration_correction,
            name=ensemble_name
        )
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs([
            "Ensemble Dashboard",
            "Make Forecast",
            "Performance History"
        ])
        
        # Tab 1: Ensemble Dashboard
        with tab1:
            cls._render_ensemble_dashboard(ensemble)
        
        # Tab 2: Make Forecast
        with tab2:
            cls._render_make_forecast(ensemble)
        
        # Tab 3: Performance History
        with tab3:
            cls._render_performance_history(ensemble)
    
    @staticmethod
    def _get_default_selection(available_forecasters: Dict[str, Forecaster]) -> List[str]:
        """Get default selection of diverse forecasters."""
        # Priority categories to ensure diversity
        categories = [
            ["LLM", "BasicLLM", "GPT", "Claude"],  # LLM-based
            ["Historical", "TimeSeries"],  # Data-driven
            ["Expert", "Domain"],  # Expert/domain-specific
            ["Community", "Human"],  # Human/community
            ["Calibrated"]  # Calibrated models
        ]
        
        selected = []
        # Try to get one from each category
        for category_terms in categories:
            for name in available_forecasters:
                if any(term.lower() in name.lower() for term in category_terms) and name not in selected:
                    selected.append(name)
                    break
        
        # If we haven't selected enough, add more
        all_names = list(available_forecasters.keys())
        while len(selected) < min(5, len(all_names)):
            for name in all_names:
                if name not in selected:
                    selected.append(name)
                    break
        
        return selected[:5]  # Limit to 5
    
    @classmethod
    def _render_ensemble_dashboard(cls, ensemble: AdaptiveEnsembleForecaster):
        """Render the ensemble dashboard with visualizations."""
        st.subheader("Ensemble Dashboard")
        
        # Show current weights
        st.markdown("### Current Ensemble Weights")
        
        # Create weight visualization
        fig = px.bar(
            x=[f.__class__.__name__ for f in ensemble.forecasters],
            y=ensemble.weights,
            labels={
                'x': 'Forecaster',
                'y': 'Weight'
            },
            title="Current Model Weights",
            color=ensemble.weights,
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show performance metrics if available
        if ensemble.performance_history:
            st.markdown("### Historical Performance")
            
            # Build a dataframe from performance history
            history_data = []
            for model_name, entries in ensemble.performance_history.items():
                for entry in entries:
                    if all(k in entry for k in ['question_id', 'prediction', 'outcome', 'timestamp']):
                        history_data.append({
                            'model': model_name,
                            'question_id': entry['question_id'],
                            'prediction': entry['prediction'],
                            'outcome': entry['outcome'],
                            'timestamp': pd.to_datetime(entry['timestamp'])
                        })
            
            if history_data:
                history_df = pd.DataFrame(history_data)
                
                # Calculate Brier scores by model
                models = history_df['model'].unique()
                brier_scores = []
                
                for model in models:
                    model_df = history_df[history_df['model'] == model]
                    if len(model_df) >= 5:  # Need enough data points
                        brier = brier_score_df(model_df)
                        brier_scores.append({
                            'Model': model,
                            'Brier Score': brier,
                            'Sample Size': len(model_df)
                        })
                
                if brier_scores:
                    brier_df = pd.DataFrame(brier_scores).sort_values('Brier Score')
                    
                    # Show metrics table
                    st.dataframe(brier_df)
                    
                    # Highlight ensemble performance
                    ensemble_row = brier_df[brier_df['Model'] == ensemble.name]
                    if not ensemble_row.empty:
                        ensemble_brier = ensemble_row['Brier Score'].iloc[0]
                        best_brier = brier_df['Brier Score'].min()
                        
                        cols = st.columns(2)
                        cols[0].metric(
                            "Ensemble Brier Score", 
                            f"{ensemble_brier:.4f}",
                            f"{best_brier - ensemble_brier:.4f}" if ensemble_brier > best_brier else None
                        )
                        
                        # Calculate improvement
                        avg_forecaster_brier = brier_df[brier_df['Model'] != ensemble.name]['Brier Score'].mean()
                        improvement = avg_forecaster_brier - ensemble_brier
                        improvement_pct = (improvement / avg_forecaster_brier) * 100
                        
                        cols[1].metric(
                            "Improvement vs Avg",
                            f"{improvement_pct:.1f}%",
                            help="Percentage improvement in Brier score compared to average of individual forecasters"
                        )
                
                # Show calibration plot for ensemble
                ensemble_df = history_df[history_df['model'] == ensemble.name]
                if len(ensemble_df) >= 10:
                    st.markdown("### Calibration Analysis")
                    
                    prob_pred, prob_true, bin_total = calibration_curve_df(ensemble_df)
                    
                    fig = go.Figure()
                    
                    # Add calibration curve
                    fig.add_trace(
                        go.Scatter(
                            x=prob_pred, 
                            y=prob_true,
                            mode='lines+markers',
                            name='Calibration Curve',
                            marker=dict(size=10, color=bin_total, colorscale='Viridis',
                                       showscale=True, colorbar=dict(title='Count')),
                            line=dict(width=2)
                        )
                    )
                    
                    # Add perfect calibration line
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1], 
                            y=[0, 1],
                            mode='lines',
                            name='Perfect Calibration',
                            line=dict(dash='dash', color='gray')
                        )
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title="Ensemble Calibration Curve",
                        xaxis_title="Predicted Probability",
                        yaxis_title="Observed Frequency",
                        legend=dict(x=0, y=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance history available yet. Make some forecasts to build history.")
    
    @classmethod
    def _render_make_forecast(cls, ensemble: AdaptiveEnsembleForecaster):
        """Render interface to make a new forecast."""
        st.subheader("Make a Forecast")
        
        # Question input
        question_text = st.text_area(
            "Forecast Question",
            value="",
            height=100,
            placeholder="Enter a binary question to forecast (e.g., 'Will the Federal Reserve raise interest rates in the next meeting?')"
        )
        
        # Resolution criteria
        resolution_criteria = st.text_area(
            "Resolution Criteria",
            value="",
            height=50,
            placeholder="Specify how the question will be resolved (e.g., 'This resolves YES if the Federal Reserve announces a rate increase.')"
        )
        
        # Optional fields
        with st.expander("Additional Information (Optional)"):
            # Due date
            due_date = st.date_input(
                "Resolution Date",
                value=None,
                help="When is this question expected to resolve?"
            )
            
            # Category
            category = st.selectbox(
                "Category",
                ["", "Economics", "Politics", "Technology", "Science", "Sports", "Other"],
                index=0
            )
            
            # Additional context
            context = st.text_area(
                "Additional Context",
                value="",
                height=100,
                placeholder="Provide any additional context, background information, or relevant links"
            )
        
        # Generate forecast button
        if st.button("Generate Forecast"):
            if not question_text:
                st.error("Please enter a question to forecast.")
            else:
                # Show spinner during forecast generation
                with st.spinner("Generating ensemble forecast..."):
                    # Create question object
                    question = ForecastQuestion(
                        id=f"q_{hash(question_text) % 10000}",
                        text=question_text,
                        resolution_criteria=resolution_criteria,
                        category=category if category else None,
                        due_date=due_date.isoformat() if due_date else None,
                        context=context if context else None
                    )
                    
                    try:
                        # Generate forecast
                        forecast = ensemble.forecast(question)
                        
                        # Display result
                        st.success("Forecast generated successfully!")
                        
                        # Show forecast
                        st.markdown("## Forecast Result")
                        
                        # Main prediction
                        cols = st.columns(3)
                        cols[0].metric(
                            "Prediction", 
                            f"{forecast.prediction:.1%}",
                            help="Probability estimate for YES outcome"
                        )
                        
                        # Confidence interval
                        interval_text = f"{forecast.lower:.1%} - {forecast.upper:.1%}"
                        cols[1].metric(
                            "90% Confidence Interval",
                            interval_text,
                            help="90% confidence interval for the prediction"
                        )
                        
                        # Visualize prediction with gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=forecast.prediction * 100,
                            title={"text": "Probability"},
                            gauge={
                                "axis": {"range": [0, 100], "ticksuffix": "%"},
                                "bar": {"color": "darkblue"},
                                "steps": [
                                    {"range": [0, 25], "color": "lightgray"},
                                    {"range": [25, 50], "color": "gray"},
                                    {"range": [50, 75], "color": "lightblue"},
                                    {"range": [75, 100], "color": "royalblue"}
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.75,
                                    "value": 50
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show explanation
                        st.markdown("### Explanation")
                        st.markdown(forecast.explanation)
                        
                        # Option to save result
                        st.markdown("### Save Result")
                        outcome = st.radio(
                            "If you know the outcome, you can record it to improve future forecasts:",
                            ["Unknown (Save Forecast Only)", "YES", "NO"],
                            index=0
                        )
                        
                        if st.button("Save Forecast"):
                            # Get outcome
                            outcome_val = None
                            if outcome == "YES":
                                outcome_val = 1
                            elif outcome == "NO":
                                outcome_val = 0
                            
                            # Save forecast
                            if outcome_val is not None:
                                ensemble.record_forecast_outcome(forecast, outcome_val)
                                st.success("Forecast and outcome saved! Ensemble will use this to improve.")
                            else:
                                # Just save the forecast for now
                                if "saved_forecasts" not in st.session_state:
                                    st.session_state.saved_forecasts = {}
                                
                                st.session_state.saved_forecasts[forecast.question_id] = {
                                    "forecast": forecast,
                                    "question": question
                                }
                                st.success("Forecast saved! You can record the outcome later when known.")
                    
                    except Exception as e:
                        st.error(f"Error generating forecast: {str(e)}")
    
    @classmethod
    def _render_performance_history(cls, ensemble: AdaptiveEnsembleForecaster):
        """Render performance history and allow uploading additional data."""
        st.subheader("Performance History")
        
        # Add explanation
        st.markdown("""
        This section shows the performance history of the ensemble and its component forecasters.
        You can also upload additional forecast results to enhance the adaptive capabilities.
        """)
        
        # Show history stats
        history = ensemble.performance_history
        if history:
            # Count questions and forecasts
            total_questions = set()
            total_forecasts = 0
            forecasters = set()
            
            for model, entries in history.items():
                forecasters.add(model)
                for entry in entries:
                    if 'question_id' in entry:
                        total_questions.add(entry['question_id'])
                    total_forecasts += 1
            
            # Display stats
            cols = st.columns(3)
            cols[0].metric("Total Questions", len(total_questions))
            cols[1].metric("Total Forecasts", total_forecasts)
            cols[2].metric("Forecasters", len(forecasters))
            
            # Performance over time
            st.markdown("### Performance Trends")
            
            # Build dataframe from history
            history_data = []
            for model, entries in history.items():
                for entry in entries:
                    if all(k in entry for k in ['question_id', 'prediction', 'outcome', 'timestamp']):
                        history_data.append({
                            'model': model,
                            'question_id': entry['question_id'],
                            'prediction': entry['prediction'],
                            'outcome': entry['outcome'],
                            'timestamp': pd.to_datetime(entry['timestamp'])
                        })
            
            if history_data:
                history_df = pd.DataFrame(history_data)
                
                # Window size selection
                window_options = {
                    "Last 10 Questions": 10,
                    "Last 20 Questions": 20,
                    "Last 50 Questions": 50,
                    "All Questions": len(total_questions)
                }
                
                selected_window = st.selectbox(
                    "Analysis Window",
                    list(window_options.keys()),
                    index=1  # Default to 20 questions
                )
                
                window_size = window_options[selected_window]
                
                # Calculate performance over time
                # Group by model and sort by timestamp
                history_df = history_df.sort_values('timestamp')
                
                # Get unique models
                models = sorted(history_df['model'].unique())
                
                # Calculate rolling Brier score
                rolling_brier_scores = []
                
                for model in models:
                    model_df = history_df[history_df['model'] == model]
                    
                    # Need enough data points
                    if len(model_df) >= 5:
                        # Calculate rolling Brier scores
                        for i in range(5, len(model_df) + 1):
                            window = model_df.iloc[:i].tail(min(window_size, i))
                            
                            if len(window) >= 5:  # Need at least 5 data points
                                brier = brier_score_df(window)
                                
                                rolling_brier_scores.append({
                                    'Model': model,
                                    'Question Count': i,
                                    'Timestamp': window['timestamp'].max(),
                                    'Brier Score': brier,
                                    'Is Ensemble': model == ensemble.name
                                })
                
                if rolling_brier_scores:
                    rolling_df = pd.DataFrame(rolling_brier_scores)
                    
                    # Plot trend
                    fig = px.line(
                        rolling_df,
                        x='Question Count',
                        y='Brier Score',
                        color='Model',
                        title=f"Brier Score Trend (Rolling Window: {window_size} questions)",
                        labels={
                            'Question Count': 'Cumulative Questions',
                            'Brier Score': 'Brier Score (lower is better)'
                        }
                    )
                    
                    # Highlight ensemble if present
                    ensemble_df = rolling_df[rolling_df['Model'] == ensemble.name]
                    if not ensemble_df.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=ensemble_df['Question Count'],
                                y=ensemble_df['Brier Score'],
                                mode='lines+markers',
                                name=ensemble.name,
                                line=dict(width=4, dash='solid', color='red'),
                                marker=dict(size=8)
                            )
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show final scores
                    last_scores = []
                    for model in models:
                        model_rows = rolling_df[rolling_df['Model'] == model]
                        if not model_rows.empty:
                            last_row = model_rows.iloc[-1]
                            last_scores.append({
                                'Model': model,
                                'Brier Score': last_row['Brier Score'],
                                'Questions': last_row['Question Count']
                            })
                    
                    if last_scores:
                        last_df = pd.DataFrame(last_scores).sort_values('Brier Score')
                        st.dataframe(last_df)
                    
                else:
                    st.info("Not enough data to calculate performance trends.")
        else:
            st.info("No performance history available yet.")
        
        # Upload additional data
        st.markdown("### Upload Additional Data")
        
        uploaded_file = st.file_uploader(
            "Upload forecast results CSV",
            type="csv",
            help="CSV file with columns: model, question_id, prediction, outcome, timestamp (optional)"
        )
        
        if uploaded_file is not None:
            # Process uploaded file
            try:
                df = pd.read_csv(uploaded_file)
                
                # Check required columns
                required_cols = ['model', 'question_id', 'prediction', 'outcome']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                else:
                    # Preview data
                    st.markdown("#### Data Preview")
                    st.dataframe(df.head())
                    
                    # Import button
                    if st.button("Import Data"):
                        with st.spinner("Importing data and updating ensemble..."):
                            # Update ensemble with data
                            ensemble.update_from_results_df(df)
                            st.success(f"Successfully imported {len(df)} forecast records!")
                            
                            # Force refresh
                            st.rerun()
            
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}") 