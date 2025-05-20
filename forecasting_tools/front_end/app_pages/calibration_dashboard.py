import logging
import re
import asyncio
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import dotenv
import streamlit as st
from pydantic import BaseModel

from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.util.jsonable import Jsonable
from forecasting_tools.ai_models.model_interfaces.calibration_system import CalibrationSystem

logger = logging.getLogger(__name__)

class CalibrationDashboardPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "📊 Calibration Dashboard"
    URL_PATH: str = "/calibration-dashboard"
    INPUT_TYPE = Jsonable
    OUTPUT_TYPE = Jsonable

    # Form input keys
    FORECASTER_SELECT = "forecaster_select_calibration"
    DATE_RANGE = "date_range_calibration"
    REFRESH_BUTTON = "refresh_button_calibration"
    
    # Default calibration data path
    DEFAULT_CALIBRATION_PATH = "forecasting_tools/data/calibration_data.json"

    @classmethod
    async def _display_intro_text(cls) -> None:
        st.write(
            "This dashboard displays calibration metrics for the forecasting models. "
            "Use it to evaluate forecaster accuracy and improve future predictions."
        )

    @classmethod
    async def _get_input(cls) -> Jsonable | None:
        # Load the available forecasters from the calibration data
        forecasters = cls._get_available_forecasters()
        
        # Initialize session state for calibration selection
        if "calibration_forecaster" not in st.session_state:
            st.session_state["calibration_forecaster"] = forecasters[0] if forecasters else ""
        if "calibration_refresh" not in st.session_state:
            st.session_state["calibration_refresh"] = False
            
        # Define callbacks
        def update_selected_forecaster():
            st.session_state["calibration_forecaster"] = st.session_state[cls.FORECASTER_SELECT]
            
        def on_refresh_click():
            st.session_state["calibration_refresh"] = True
        
        # Create selection interface
        col1, col2 = st.columns(2)
        
        with col1:
            selected_forecaster = st.selectbox(
                "Select Forecaster",
                options=forecasters,
                key=cls.FORECASTER_SELECT,
                on_change=update_selected_forecaster
            )
        
        with col2:
            st.button("Refresh Data", on_click=on_refresh_click)
        
        # Check if refresh was clicked or no forecasters available
        if st.session_state["calibration_refresh"] or not forecasters:
            # Reset refresh flag
            st.session_state["calibration_refresh"] = False
            
            return Jsonable(value={
                "forecaster": st.session_state["calibration_forecaster"], 
                "refresh": True
            })
        
        # Allow viewing without explicitly submitting
        if forecasters:
            return Jsonable(value={
                "forecaster": st.session_state["calibration_forecaster"], 
                "refresh": False
            })
            
        return None

    @classmethod
    async def _run_tool(cls, input: Jsonable) -> Jsonable:
        selected_forecaster = input.value.get("forecaster")
        
        if not selected_forecaster:
            st.warning("No forecaster data available. Please make some forecasts first.")
            return Jsonable(value={"success": False})
        
        # Load calibration data for the selected forecaster
        calibration_system = CalibrationSystem(
            forecaster_name=selected_forecaster,
            calibration_data_path=cls.DEFAULT_CALIBRATION_PATH
        )
        
        # Get calibration metrics
        calibration_summary = calibration_system.get_calibration_summary()
        metrics = calibration_summary.get("metrics", {})
        
        return Jsonable(value={
            "forecaster": selected_forecaster,
            "summary": calibration_summary,
            "metrics": metrics,
            "success": True
        })

    @classmethod
    async def _display_outputs(cls, outputs: list[Jsonable]) -> None:
        if not outputs or not outputs[0].value.get("success", False):
            st.warning("No data to display.")
            return
        
        output = outputs[0].value
        forecaster = output.get("forecaster")
        summary = output.get("summary", {})
        metrics = output.get("metrics", {})
        
        # Display summary information
        st.subheader(f"Calibration Summary: {forecaster}")
        
        # Create three columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Sample Count",
                summary.get("sample_count", 0)
            )
        
        with col2:
            brier_score = metrics.get("brier_score")
            if brier_score is not None:
                st.metric(
                    "Brier Score",
                    f"{brier_score:.4f}"
                )
            else:
                st.metric("Brier Score", "N/A")
        
        with col3:
            bss = metrics.get("brier_skill_score")
            if bss is not None:
                st.metric(
                    "Brier Skill Score",
                    f"{bss:.4f}"
                )
            else:
                st.metric("Brier Skill Score", "N/A")
        
        # Display calibration method
        st.info(f"Recalibration method: {summary.get('recalibration_method', 'none')}")
        
        # Show last updated time
        last_updated = summary.get("last_updated", "never")
        if last_updated != "never":
            try:
                last_updated_dt = datetime.fromisoformat(last_updated.split('+')[0].split('Z')[0])
                last_updated_str = last_updated_dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                last_updated_str = last_updated
            st.text(f"Last updated: {last_updated_str}")
        
        # Display calibration curve
        cls._display_calibration_curve(metrics)
        
        # Display prediction vs outcome scatter plot
        cls._display_prediction_vs_outcome(metrics)
        
        # Display reliability diagram
        cls._display_reliability_diagram(metrics)
        
        # Display raw data table if available
        cls._display_data_table(summary, forecaster)

    @classmethod
    def _get_available_forecasters(cls) -> list[str]:
        """Get list of forecasters with calibration data."""
        try:
            if os.path.exists(cls.DEFAULT_CALIBRATION_PATH):
                with open(cls.DEFAULT_CALIBRATION_PATH, 'r') as f:
                    data = json.load(f)
                return list(data.keys())
            else:
                return []
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
            return []

    @classmethod
    def _display_calibration_curve(cls, metrics: dict) -> None:
        """Display the calibration curve."""
        bin_centers = metrics.get("bin_centers", [])
        bin_accuracies = metrics.get("bin_accuracies", [])
        bin_counts = metrics.get("bin_counts", [])
        
        if not bin_centers or not bin_accuracies:
            return
        
        st.subheader("Calibration Curve")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter out None values
        valid_points = [(x, y, c) for x, y, c in zip(bin_centers, bin_accuracies, bin_counts) if y is not None]
        if not valid_points:
            st.write("Not enough data to display calibration curve.")
            return
            
        x, y, counts = zip(*valid_points)
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Plot actual calibration curve, with point size proportional to count
        sizes = [max(20, min(500, 20 + c * 5)) for c in counts]
        scatter = ax.scatter(x, y, s=sizes, alpha=0.6, label='Observed frequency')
        
        # Add bin counts as annotations
        for i, (xi, yi, count) in enumerate(zip(x, y, counts)):
            if count > 0:
                ax.annotate(f"{count}", (xi, yi), textcoords="offset points", 
                           xytext=(0,10), ha='center')
        
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Observed frequency')
        ax.set_title('Calibration Curve')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        st.pyplot(fig)
        
        st.info(
            "The calibration curve shows how well the predicted probabilities match the observed frequencies. "
            "Points on the diagonal line represent perfect calibration. Points above the line indicate "
            "underconfidence (predicted probabilities are too low), while points below indicate "
            "overconfidence (predicted probabilities are too high)."
        )

    @classmethod
    def _display_prediction_vs_outcome(cls, metrics: dict) -> None:
        """Display scatter plot of predictions vs outcomes."""
        # This visualization would need data that's not directly available in the metrics
        # We would need to extract raw prediction-outcome pairs from the calibration data
        # For now, we'll skip this visualization
        pass

    @classmethod
    def _display_reliability_diagram(cls, metrics: dict) -> None:
        """Display reliability diagram (similar to calibration curve but with histogram)."""
        bin_centers = metrics.get("bin_centers", [])
        bin_accuracies = metrics.get("bin_accuracies", [])
        bin_counts = metrics.get("bin_counts", [])
        
        if not bin_centers or not bin_accuracies or not bin_counts:
            return
        
        st.subheader("Reliability Diagram")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Filter out None values from bin_accuracies
        valid_indices = [i for i, acc in enumerate(bin_accuracies) if acc is not None]
        if not valid_indices:
            st.write("Not enough data to display reliability diagram.")
            return
            
        valid_centers = [bin_centers[i] for i in valid_indices]
        valid_accuracies = [bin_accuracies[i] for i in valid_indices]
        valid_counts = [bin_counts[i] for i in valid_indices]
        
        # Top subplot: Reliability curve
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.plot(valid_centers, valid_accuracies, 'b-o', label='Model')
        
        # Fill area between curves
        ax1.fill_between(valid_centers, valid_centers, valid_accuracies, alpha=0.2, color='blue')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Observed frequency')
        ax1.set_title('Reliability Diagram')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # Bottom subplot: Histogram of predictions
        ax2.bar(bin_centers, bin_counts, width=1.0/len(bin_centers), alpha=0.6, color='blue')
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Predicted probability')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info(
            "The reliability diagram shows both calibration (top) and the distribution of predictions (bottom). "
            "The blue shaded area represents calibration error. A well-calibrated model will have points close "
            "to the diagonal and a small shaded area."
        )

    @classmethod
    def _display_data_table(cls, summary: dict, forecaster: str) -> None:
        """Display a table of raw prediction-outcome data if available."""
        # We would need to extract this data from the full calibration data file
        # For now, we'll just provide a button to download the full data
        
        st.subheader("Raw Calibration Data")
        
        try:
            if os.path.exists(cls.DEFAULT_CALIBRATION_PATH):
                with open(cls.DEFAULT_CALIBRATION_PATH, 'r') as f:
                    data = json.load(f)
                
                if forecaster in data:
                    forecaster_data = data[forecaster]
                    
                    # Create a DataFrame for display
                    if "predictions" in forecaster_data and "outcomes" in forecaster_data:
                        predictions = forecaster_data["predictions"]
                        outcomes = forecaster_data["outcomes"]
                        question_ids = forecaster_data.get("question_ids", [""] * len(predictions))
                        metadata = forecaster_data.get("metadata", [{}] * len(predictions))
                        
                        # Extract question text from metadata if available
                        questions = []
                        for m in metadata:
                            if isinstance(m, dict) and "question_text" in m:
                                questions.append(m["question_text"])
                            else:
                                questions.append("")
                        
                        # Create DataFrame with available data
                        df_data = {
                            "Prediction": predictions,
                            "Outcome": outcomes,
                        }
                        
                        # Add questions if available
                        if any(questions):
                            df_data["Question"] = questions
                        
                        df = pd.DataFrame(df_data)
                        
                        # Filter to only show resolved predictions
                        resolved_df = df[df["Outcome"].notnull()].copy()
                        
                        if not resolved_df.empty:
                            # Calculate additional metrics
                            resolved_df["Error"] = (resolved_df["Prediction"] - resolved_df["Outcome"]).abs()
                            resolved_df["Squared Error"] = (resolved_df["Prediction"] - resolved_df["Outcome"])**2
                            
                            # Display table
                            st.dataframe(resolved_df)
                            
                            # Provide download link
                            csv = resolved_df.to_csv(index=False)
                            st.download_button(
                                label="Download data as CSV",
                                data=csv,
                                file_name=f"{forecaster}_calibration_data.csv",
                                mime="text/csv"
                            )
                        else:
                            st.write("No resolved predictions available yet.")
                    else:
                        st.write("Calibration data format not as expected.")
                else:
                    st.write(f"No data found for forecaster: {forecaster}")
            else:
                st.write("Calibration data file not found.")
        except Exception as e:
            st.error(f"Error loading raw calibration data: {e}")


if __name__ == "__main__":
    dotenv.load_dotenv()
    CalibrationDashboardPage.main()