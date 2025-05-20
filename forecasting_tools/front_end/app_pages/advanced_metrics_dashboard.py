import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
import os
import sys

# Add the root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from forecasting_tools.front_end.helpers.app_page import AppPage
from forecasting_tools.forecast_helpers.metrics import (
    brier_score_df, calibration_curve_df, coverage_df, peer_score_df,
    calibration_error_df, sharpness_df, time_weighted_brier_score,
    model_performance_over_time, aggregate_metrics
)


class AdvancedMetricsDashboardPage(AppPage):
    PAGE_DISPLAY_NAME: str = "📊 Advanced Metrics"
    URL_PATH: str = "/advanced-metrics"
    IS_DEFAULT_PAGE: bool = False

    @classmethod
    def main(cls):
        st.title("Advanced Forecasting Metrics Dashboard")

        # Sidebar for configuration
        st.sidebar.header("Dashboard Configuration")
        data_source = st.sidebar.selectbox(
            "Select Data Source",
            ["Benchmark Results", "Uploaded Data", "Sample Data"],
            index=0
        )

        # Load data based on selection
        if data_source == "Benchmark Results":
            df = cls._load_benchmark_data()
        elif data_source == "Uploaded Data":
            df = cls._load_uploaded_data()
        else:
            df = cls._load_sample_data()

        if df is None or df.empty:
            st.warning("No data available. Please run benchmarks or upload data.")
            return

        # Show data summary
        st.subheader("Data Summary")
        cols = st.columns(4)
        cols[0].metric("Total Questions", len(df['question_id'].unique()))
        cols[1].metric("Total Forecasts", len(df))
        cols[2].metric("Models", len(df['model'].unique()) if 'model' in df.columns else "N/A")
        cols[3].metric("Time Range", f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
                      if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']) 
                      else "N/A")

        # Tabs for different analysis views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overall Performance", 
            "Calibration Analysis", 
            "Time Series Analysis",
            "Confidence Intervals",
            "Model Comparison"
        ])

        with tab1:
            cls._render_overall_performance(df)

        with tab2:
            cls._render_calibration_analysis(df)

        with tab3:
            cls._render_time_series_analysis(df)

        with tab4:
            cls._render_confidence_intervals(df)

        with tab5:
            cls._render_model_comparison(df)

    @staticmethod
    def _load_benchmark_data() -> pd.DataFrame:
        """Load benchmark results data."""
        DATA_PATH = os.path.join(os.path.dirname(__file__), '../../../data/benchmark_results.csv')
        
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            # Filter to only resolved questions
            df = df.dropna(subset=['outcome'])
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            return df
        
        return pd.DataFrame()

    @staticmethod
    def _load_uploaded_data() -> pd.DataFrame:
        """Load data from user upload."""
        uploaded_file = st.sidebar.file_uploader("Upload forecast results CSV", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Basic validation
            required_cols = ['prediction', 'outcome']
            if not all(col in df.columns for col in required_cols):
                st.sidebar.error(f"Uploaded file must contain columns: {', '.join(required_cols)}")
                return pd.DataFrame()
            
            # Ensure timestamp is datetime if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            return df
        
        return pd.DataFrame()

    @staticmethod
    def _load_sample_data() -> pd.DataFrame:
        """Load sample data for demonstration."""
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        n_models = 5
        n_questions = 50
        
        # Generate timestamps over the last year
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=365)
        timestamps = [start_date + pd.Timedelta(days=i) for i in 
                     np.random.randint(0, 365, n_samples)]
        
        # Model names
        model_names = [f"Model_{i}" for i in range(1, n_models+1)]
        
        # Generate data
        data = {
            'model': np.random.choice(model_names, n_samples),
            'question_id': np.random.randint(1, n_questions+1, n_samples),
            'timestamp': timestamps,
            'prediction': np.random.beta(2, 2, n_samples),  # Beta distribution for predictions
            'outcome': np.random.binomial(1, 0.5, n_samples),  # Binary outcomes
            'lower': [],
            'upper': []
        }
        
        # Generate confidence intervals
        for pred in data['prediction']:
            width = np.random.uniform(0.1, 0.3)  # Random interval width
            lower = max(0, pred - width/2)
            upper = min(1, pred + width/2)
            data['lower'].append(lower)
            data['upper'].append(upper)
            
        return pd.DataFrame(data)

    @classmethod
    def _render_overall_performance(cls, df: pd.DataFrame):
        """Render overall performance metrics."""
        st.subheader("Overall Model Performance")
        
        # Calculate aggregate metrics
        metrics_df = aggregate_metrics(df)
        
        if not metrics_df.empty:
            # Metrics table
            st.dataframe(metrics_df.style.highlight_min(['Brier Score', 'Calibration Error']).highlight_max(['Log Score', 'Sharpness']))
            
            # Visualization
            fig = px.bar(
                metrics_df, 
                x='Model', 
                y=['Brier Score', 'Calibration Error', 'Peer Score'],
                barmode='group',
                title="Model Performance Comparison",
                labels={
                    'value': 'Score (lower is better)',
                    'variable': 'Metric'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Unable to calculate aggregate metrics with the available data.")

    @classmethod
    def _render_calibration_analysis(cls, df: pd.DataFrame):
        """Render calibration analysis."""
        st.subheader("Calibration Analysis")
        
        # Model selection for calibration
        if 'model' in df.columns:
            models = ['All Models'] + sorted(df['model'].unique().tolist())
            selected_model = st.selectbox("Select Model for Calibration Analysis", models)
            
            if selected_model != 'All Models':
                model_df = df[df['model'] == selected_model]
            else:
                model_df = df
        else:
            model_df = df
        
        # Calculate calibration curve
        n_bins = st.slider("Number of Calibration Bins", min_value=5, max_value=20, value=10)
        prob_pred, prob_true, bin_total = calibration_curve_df(model_df, n_bins=n_bins)
        
        if len(prob_pred) > 0:
            # Create calibration plot
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
            
            # Add calibration error
            cal_error = calibration_error_df(model_df, n_bins=n_bins)
            
            # Update layout
            fig.update_layout(
                title=f"Calibration Curve (Error: {cal_error:.4f})",
                xaxis_title="Predicted Probability",
                yaxis_title="Observed Frequency",
                legend=dict(x=0, y=1),
                width=800,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sharpness
            sharp = sharpness_df(model_df)
            st.metric("Prediction Sharpness", f"{sharp:.4f}", 
                     help="Variance of predictions. Higher values indicate more decisive predictions.")
            
            # Distribution of predictions
            fig2 = px.histogram(
                model_df, 
                x="prediction", 
                nbins=20,
                title="Distribution of Predictions",
                labels={"prediction": "Predicted Probability"}
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Unable to calculate calibration curve with the available data.")

    @classmethod
    def _render_time_series_analysis(cls, df: pd.DataFrame):
        """Render time series analysis of model performance."""
        st.subheader("Performance Over Time")
        
        # Check if timestamp is available
        if 'timestamp' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            st.info("Timestamp data not available for time series analysis.")
            return
        
        # Time window selection
        window_options = {
            "Weekly": "7D",
            "Biweekly": "14D",
            "Monthly": "30D",
            "Quarterly": "90D"
        }
        
        selected_window = st.selectbox(
            "Time Window", 
            list(window_options.keys()),
            index=2  # Default to monthly
        )
        
        window_size = window_options[selected_window]
        
        # Calculate performance over time
        time_df = model_performance_over_time(df, window_size=window_size)
        
        if not time_df.empty:
            # Line chart of metrics over time
            fig = px.line(
                time_df, 
                x="Time", 
                y=["Brier Score", "Calibration Error"],
                color="Model",
                title=f"Model Performance Metrics Over Time ({selected_window} Windows)",
                labels={
                    "value": "Score (lower is better)",
                    "Time": "Time Period",
                    "variable": "Metric"
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate time-weighted metrics
            if 'model' in df.columns:
                models = sorted(df['model'].unique().tolist())
                results = []
                
                half_life = st.slider(
                    "Time Weighting Half-Life (days)", 
                    min_value=7, 
                    max_value=180, 
                    value=30,
                    help="Recent forecasts are weighted more heavily. This controls how quickly older forecasts decay in importance."
                )
                
                for model in models:
                    model_df = df[df['model'] == model]
                    tw_brier = time_weighted_brier_score(model_df, half_life_days=half_life)
                    results.append({
                        "Model": model,
                        "Time-Weighted Brier Score": tw_brier
                    })
                
                results_df = pd.DataFrame(results).sort_values("Time-Weighted Brier Score")
                
                st.subheader(f"Time-Weighted Performance (Half-Life: {half_life} days)")
                st.dataframe(results_df.style.highlight_min(['Time-Weighted Brier Score']))
            else:
                tw_brier = time_weighted_brier_score(df, half_life_days=30)
                st.metric("Time-Weighted Brier Score", f"{tw_brier:.4f}")
        else:
            st.info("Not enough time series data to analyze performance over time.")

    @classmethod
    def _render_confidence_intervals(cls, df: pd.DataFrame):
        """Render confidence interval analysis."""
        st.subheader("Confidence Interval Analysis")
        
        # Check if confidence intervals are available
        if not {'lower', 'upper'}.issubset(df.columns):
            st.info("Confidence interval data not available.")
            return
        
        # Calculate coverage
        coverage_val = coverage_df(df)
        
        # Calculate average interval width
        df['interval_width'] = df['upper'] - df['lower']
        avg_width = df['interval_width'].mean()
        
        cols = st.columns(2)
        cols[0].metric("Coverage", f"{coverage_val*100:.1f}%", 
                     help="Percentage of actual outcomes that fall within the confidence intervals")
        cols[1].metric("Average Interval Width", f"{avg_width:.3f}", 
                     help="Average width of confidence intervals (smaller is better if coverage is adequate)")
        
        # Plot interval width vs correctness
        df['within_interval'] = (df['outcome'] >= df['lower']) & (df['outcome'] <= df['upper'])
        
        fig = px.scatter(
            df, 
            x="interval_width",
            y="within_interval",
            color="model" if "model" in df.columns else None,
            title="Confidence Interval Analysis",
            labels={
                "interval_width": "Confidence Interval Width",
                "within_interval": "Outcome Within Interval",
                "model": "Model"
            },
            opacity=0.6
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution of interval widths
        if 'model' in df.columns:
            fig2 = px.box(
                df, 
                x="model", 
                y="interval_width",
                title="Distribution of Confidence Interval Widths by Model",
                labels={
                    "model": "Model",
                    "interval_width": "Interval Width"
                }
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        else:
            fig2 = px.histogram(
                df, 
                x="interval_width",
                nbins=30,
                title="Distribution of Confidence Interval Widths",
                labels={
                    "interval_width": "Interval Width"
                }
            )
            
            st.plotly_chart(fig2, use_container_width=True)

    @classmethod
    def _render_model_comparison(cls, df: pd.DataFrame):
        """Render model comparison analysis."""
        st.subheader("Model Comparison")
        
        if 'model' not in df.columns:
            st.info("Model comparison not available without model identifiers.")
            return
        
        # Calculate peer scores
        peer_df = peer_score_df(df)
        
        if not peer_df.empty:
            # Sort by peer score
            peer_df = peer_df.sort_values("Peer Score")
            
            # Plot peer scores
            fig = px.bar(
                peer_df, 
                x="Model", 
                y="Peer Score",
                title="Peer Score Comparison (Difference from Average)",
                labels={
                    "Peer Score": "Peer Score (negative is better)",
                    "Model": "Model"
                },
                color="Peer Score",
                color_continuous_scale=px.colors.diverging.RdBu_r
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Advanced analysis - comparing with aggregate decision
            st.subheader("Comparing with Aggregate Decision")
            
            # Group by question and create ensemble prediction
            question_groups = df.groupby('question_id')
            ensemble_results = []
            
            for question, group in question_groups:
                if len(group) > 1:  # Only if we have multiple models for this question
                    # Simple average ensemble
                    avg_pred = group['prediction'].mean()
                    outcome = group['outcome'].iloc[0]  # All should be same
                    
                    # For each model, compare with ensemble
                    for _, row in group.iterrows():
                        ensemble_results.append({
                            'Model': row['model'],
                            'Question': question,
                            'Model Prediction': row['prediction'],
                            'Ensemble Prediction': avg_pred,
                            'Outcome': outcome,
                            'Model-Ensemble Diff': row['prediction'] - avg_pred,
                            'Correct Direction': (row['prediction'] > avg_pred and outcome == 1) or 
                                               (row['prediction'] < avg_pred and outcome == 0)
                        })
            
            if ensemble_results:
                ensemble_df = pd.DataFrame(ensemble_results)
                
                # Calculate "beats ensemble" percentage
                model_edge = ensemble_df.groupby('Model')['Correct Direction'].mean().reset_index()
                model_edge.columns = ['Model', 'Beats Ensemble %']
                model_edge['Beats Ensemble %'] = model_edge['Beats Ensemble %'] * 100
                
                # Sort by percentage
                model_edge = model_edge.sort_values('Beats Ensemble %', ascending=False)
                
                st.dataframe(model_edge)
                
                # Plot
                fig2 = px.bar(
                    model_edge,
                    x="Model",
                    y="Beats Ensemble %",
                    title="Model vs. Ensemble (% of times model corrects ensemble direction)",
                    labels={
                        "Beats Ensemble %": "% Correct Direction vs Ensemble",
                        "Model": "Model"
                    },
                    color="Beats Ensemble %",
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Not enough data to compare with ensemble decisions.") 