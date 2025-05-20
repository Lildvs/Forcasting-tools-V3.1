import logging
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import asyncio

import streamlit as st
from pydantic import BaseModel

from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.util.jsonable import Jsonable
from forecasting_tools.ai_models.model_interfaces.active_learning_manager import ActiveLearningManager
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.front_end.helpers.report_displayer import ReportDisplayer

logger = logging.getLogger(__name__)

class ActiveLearningInput(Jsonable, BaseModel):
    question_id: str
    human_probability: float
    feedback: str
    update_model: bool = True

class ActiveLearningOutput(Jsonable, BaseModel):
    success: bool
    message: str

class ActiveLearningPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "🔄 Active Learning"
    URL_PATH: str = "/active-learning"
    INPUT_TYPE = ActiveLearningInput
    OUTPUT_TYPE = ActiveLearningOutput

    # Form input keys
    QUESTION_SELECT = "question_select_active_learning"
    HUMAN_PROBABILITY = "human_probability_active_learning"
    FEEDBACK_TEXT = "feedback_text_active_learning"
    UPDATE_MODEL = "update_model_active_learning"
    REFRESH_BUTTON = "refresh_button_active_learning"
    BATCH_SIZE = "batch_size_active_learning"
    SORT_BY = "sort_by_active_learning"
    
    # Constants
    DEFAULT_ACTIVE_LEARNING_PATH = "forecasting_tools/data/active_learning"

    @classmethod
    async def _display_intro_text(cls) -> None:
        st.write(
            "This page shows forecasts with high uncertainty or low confidence that need human review. "
            "By providing your feedback, you help improve the forecasting models through active learning."
        )
        st.info(
            "Active learning is a machine learning approach where the system identifies uncertain predictions "
            "and requests human input to improve future performance. This feedback loop allows models to learn "
            "from difficult cases and gradually improve their accuracy."
        )

    @classmethod
    async def _get_input(cls) -> ActiveLearningInput | None:
        # Initialize the active learning manager
        active_learning_manager = ActiveLearningManager(data_dir=cls.DEFAULT_ACTIVE_LEARNING_PATH)
        
        # Create controls for filtering and sorting
        col1, col2, col3 = st.columns(3)
        
        with col1:
            batch_size = st.selectbox(
                "Number of questions to display",
                options=[5, 10, 20, 50],
                index=1,  # Default to 10
                key=cls.BATCH_SIZE
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                options=["Importance", "Date (newest first)", "Probability (closest to 0.5)"],
                index=0,
                key=cls.SORT_BY
            )
        
        with col3:
            refresh = st.button("Refresh Queue")
        
        # Get flagged questions
        flagged_questions = active_learning_manager.get_flagged_questions(limit=batch_size)
        
        # Sort according to user preference
        if sort_by == "Date (newest first)":
            flagged_questions = sorted(
                flagged_questions,
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )
        elif sort_by == "Probability (closest to 0.5)":
            flagged_questions = sorted(
                flagged_questions,
                key=lambda x: abs(x.get("probability", 0.5) - 0.5)
            )
        # Default is already sorted by importance
        
        if not flagged_questions:
            st.info("No forecasts currently need review. Check back later!")
            return None
        
        # Display the flagged questions in an interactive list
        st.subheader(f"Questions Requiring Review ({len(flagged_questions)})")
        
        selected_question = None
        
        # Display each question card and allow selection
        for i, question in enumerate(flagged_questions):
            question_id = question.get("question_id", "")
            question_text = question.get("question_text", "No question text")
            probability = question.get("probability", 0.5)
            interval = question.get("confidence_interval", [0.3, 0.7])
            model_name = question.get("model_name", "Unknown model")
            importance = question.get("importance", 0.5)
            
            # Create an expandable card for each question
            with st.expander(f"{i+1}. {question_text[:100]}{'...' if len(question_text) > 100 else ''}", expanded=(i==0)):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Question**: {question_text}")
                    
                    # Show background info if available
                    if "background_info" in question and question["background_info"]:
                        st.markdown("**Background**:")
                        st.markdown(question["background_info"])
                
                with col2:
                    st.metric("Forecast", f"{probability:.2f}")
                    st.write(f"Confidence Interval: [{interval[0]:.2f}, {interval[1]:.2f}]")
                    st.progress(importance, text=f"Importance: {importance:.2f}")
                
                # Add a button to select this question for review
                if st.button(f"Review this question"):
                    selected_question = question
                    st.session_state["selected_question"] = question
        
        # If a question was selected, show the review form
        if "selected_question" in st.session_state:
            selected_question = st.session_state["selected_question"]
            
            st.subheader("Provide Your Feedback")
            
            # Show the selected question details
            question_id = selected_question.get("question_id", "")
            question_text = selected_question.get("question_text", "")
            model_probability = selected_question.get("probability", 0.5)
            
            st.markdown(f"**Selected Question**: {question_text}")
            st.markdown(f"**Model Forecast**: {model_probability:.2f}")
            
            # Create a form for the human review
            with st.form("active_learning_form"):
                human_probability = st.slider(
                    "Your probability estimate",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(model_probability),  # Start with model's prediction
                    step=0.01,
                    format="%.2f",
                    key=cls.HUMAN_PROBABILITY
                )
                
                feedback = st.text_area(
                    "Feedback or explanation (optional)",
                    placeholder="Explain your reasoning or provide information that might be useful for the model",
                    key=cls.FEEDBACK_TEXT
                )
                
                update_model = st.checkbox(
                    "Use for model retraining",
                    value=True,
                    help="If checked, this feedback will be used to improve the model in the next training cycle",
                    key=cls.UPDATE_MODEL
                )
                
                submit_button = st.form_submit_button("Submit Review")
                
                if submit_button:
                    return ActiveLearningInput(
                        question_id=question_id,
                        human_probability=human_probability,
                        feedback=feedback,
                        update_model=update_model
                    )
        
        return None

    @classmethod
    async def _run_tool(cls, input: ActiveLearningInput) -> ActiveLearningOutput:
        # Initialize the active learning manager
        active_learning_manager = ActiveLearningManager(data_dir=cls.DEFAULT_ACTIVE_LEARNING_PATH)
        
        with st.spinner("Processing your feedback..."):
            # Submit the human review
            success = active_learning_manager.submit_review(
                question_id=input.question_id,
                human_probability=input.human_probability,
                feedback=input.feedback,
                update_model=input.update_model
            )
            
            if success:
                # Clear the selected question from session state
                if "selected_question" in st.session_state:
                    del st.session_state["selected_question"]
                
                return ActiveLearningOutput(
                    success=True,
                    message="Thank you for your feedback! Your input will help improve the forecasting models."
                )
            else:
                return ActiveLearningOutput(
                    success=False,
                    message="Error submitting feedback. The question may have already been reviewed or was removed from the queue."
                )

    @classmethod
    async def _display_outputs(cls, outputs: list[ActiveLearningOutput]) -> None:
        if not outputs:
            return
        
        output = outputs[0]
        
        if output.success:
            st.success(output.message)
        else:
            st.error(output.message)
        
        # Display statistics about the active learning system
        active_learning_manager = ActiveLearningManager(data_dir=cls.DEFAULT_ACTIVE_LEARNING_PATH)
        
        # Get summary statistics
        flagged_count = len(active_learning_manager.active_learning_data.get("flagged_questions", []))
        reviewed_count = len(active_learning_manager.active_learning_data.get("reviewed_questions", []))
        unreviewed_count = len([q for q in active_learning_manager.active_learning_data.get("flagged_questions", []) 
                               if not q.get("reviewed", False)])
        
        # Show the statistics
        st.subheader("Active Learning Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Questions in Queue", unreviewed_count)
        
        with col2:
            st.metric("Total Flagged", flagged_count)
        
        with col3:
            st.metric("Total Reviewed", reviewed_count)
        
        # Show a visualization of the feedback history if we have reviewed questions
        if reviewed_count > 0:
            st.subheader("Feedback Analysis")
            
            reviewed_questions = active_learning_manager.active_learning_data.get("reviewed_questions", [])
            
            # Create a dataframe for analysis
            data = []
            for q in reviewed_questions:
                data.append({
                    'question_id': q.get('question_id', ''),
                    'model_probability': q.get('probability', 0.5),
                    'human_probability': q.get('human_probability', 0.5),
                    'difference': abs(q.get('probability', 0.5) - q.get('human_probability', 0.5)),
                    'timestamp': q.get('review_timestamp', '')
                })
            
            if data:
                df = pd.DataFrame(data)
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Set up a 2-column layout for visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Plot model vs human predictions
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(df['model_probability'], df['human_probability'], alpha=0.7)
                    ax.set_xlabel('Model Probability')
                    ax.set_ylabel('Human Probability')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line for perfect alignment
                    ax.set_title('Model vs Human Probabilities')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    # Plot the distribution of differences
                    fig, ax = plt.subplots(figsize=(6, 6))
                    df['difference'].hist(ax=ax, bins=20, alpha=0.7)
                    ax.set_xlabel('Absolute Difference')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Model-Human Differences')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Show tips for feedback based on the data
                avg_diff = df['difference'].mean()
                if avg_diff > 0.2:
                    st.warning(
                        f"The average difference between human and model predictions is {avg_diff:.2f}. "
                        "This suggests there may be systematic issues with the model that need addressing."
                    )
                else:
                    st.success(
                        f"The average difference between human and model predictions is {avg_diff:.2f}. "
                        "The model appears to be reasonably aligned with human judgments."
                    )

    @classmethod
    async def _save_run_to_coda(
        cls,
        input_to_tool: Jsonable,
        output: Jsonable,
        is_premade_example: bool,
    ) -> None:
        # Not saving to Coda for this page
        pass

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    ActiveLearningPage.main() 