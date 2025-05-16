import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from metrics import brier_score_df, calibration_curve_df, coverage_df
from forecasting_tools.front_end.helpers.app_page import AppPage

class MetricsDashboardPage(AppPage):
    PAGE_DISPLAY_NAME: str = "📊 Metrics Dashboard"
    URL_PATH: str = "/metrics-dashboard"
    IS_DEFAULT_PAGE: bool = False

    @classmethod
    def main(cls):
        st.title("Forecasting Metrics Dashboard")

        # Load real benchmark results
        DATA_PATH = os.path.join(os.path.dirname(__file__), '../../../data/benchmark_results.csv')
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            df = df.dropna(subset=['outcome'])  # Only use resolved questions
            if df.empty:
                st.warning("No resolved benchmark results yet.")
        else:
            st.warning("No benchmark results found. Please run the benchmark first.")
            df = pd.DataFrame()

        if not df.empty:
            # Brier Score
            brier = brier_score_df(df)
            st.metric("Brier Score", f"{brier:.3f}")

            # Calibration Plot
            prob_pred, prob_true, bin_total = calibration_curve_df(df)
            fig, ax = plt.subplots()
            ax.plot(prob_pred, prob_true, marker='o', label='Model')
            ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Observed Frequency")
            ax.set_title("Calibration (Reliability Diagram)")
            ax.legend()
            st.pyplot(fig)

            # Coverage
            if {'lower', 'upper'}.issubset(df.columns):
                coverage_val = coverage_df(df)
                st.metric("Coverage", f"{coverage_val*100:.1f}%")
            else:
                st.info("No confidence interval data available for coverage metric.")

            # Peer Score (if multiple models)
            if 'model' in df.columns:
                models = df['model'].unique()
                all_scores = []
                for m in models:
                    m_df = df[df['model'] == m]
                    all_scores.append((m, brier_score_df(m_df)))
                peer_df = pd.DataFrame(all_scores, columns=['Model', 'Brier Score'])
                peer_df['Peer Score'] = peer_df['Brier Score'] - peer_df['Brier Score'].mean()
                st.subheader("Peer Score Leaderboard")
                st.table(peer_df)
            else:
                st.info("No model column found for peer score calculation.")

        # Optionally, add trend lines, filters, or export buttons 