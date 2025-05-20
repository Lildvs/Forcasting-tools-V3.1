import logging
import re
import asyncio
import json
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union

from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.front_end.helpers.report_displayer import ReportDisplayer
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.util.jsonable import Jsonable
from forecasting_tools.ai_models.model_interfaces.ensemble_forecaster import EnsembleForecaster
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.model_interfaces.enhanced_llm_forecaster import EnhancedLLMForecaster
from forecasting_tools.ai_models.model_interfaces.expert_forecaster import ExpertForecaster
from forecasting_tools.ai_models.model_interfaces.forecaster_result import ForecastResult

logger = logging.getLogger(__name__)

class ScenarioKey(Jsonable, BaseModel):
    name: str
    value: float
    initial_value: float
    min_value: float = 0.0 
    max_value: float = 1.0
    step: float = 0.1
    description: str = ""
    
class ScenarioInput(Jsonable, BaseModel):
    question: BinaryQuestion
    key_factors: List[ScenarioKey] = Field(default_factory=list)
    
class ScenarioOutput(Jsonable, BaseModel):
    baseline_report: BinaryReport
    scenario_reports: List[BinaryReport] = Field(default_factory=list)
    forecaster_breakdown: Dict[str, Any] = Field(default_factory=dict)

class ScenarioAnalysisPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "🔍 Transparency & Scenario Analysis"
    URL_PATH: str = "/scenario-analysis"
    INPUT_TYPE = ScenarioInput
    OUTPUT_TYPE = ScenarioOutput
    
    # Form input keys
    QUESTION_TEXT_BOX = "question_text_box_scenario"
    RESOLUTION_CRITERIA_BOX = "resolution_criteria_box_scenario"
    FINE_PRINT_BOX = "fine_print_box_scenario"
    BACKGROUND_INFO_BOX = "background_info_box_scenario"
    METACULUS_URL_INPUT = "metaculus_url_input_scenario"
    FETCH_BUTTON = "fetch_button_scenario"
    
    # Forecaster keys
    USE_LLM = "use_llm_scenario"
    USE_ENHANCED_LLM = "use_enhanced_llm_scenario"
    USE_EXPERT = "use_expert_scenario"
    
    LLM_WEIGHT = "llm_weight_scenario"
    ENHANCED_LLM_WEIGHT = "enhanced_llm_weight_scenario"
    EXPERT_WEIGHT = "expert_weight_scenario"
    
    # Scenario keys
    KEY_FACTOR_NAME = "key_factor_name_scenario"
    KEY_FACTOR_VALUE = "key_factor_value_scenario"
    
    @classmethod
    async def _display_intro_text(cls) -> None:
        st.write(
            "This page provides transparency into ensemble forecasts and allows you to "
            "analyze how changes to key factors affect predictions."
        )
        st.info(
            "You can view the breakdown of individual forecaster contributions "
            "and explore 'what-if' scenarios by adjusting key assumptions."
        )

    @classmethod
    async def _get_input(cls) -> ScenarioInput | None:
        st.write("### Question Details")
        
        question_text = st.text_area(
            "Yes/No Binary Question", 
            height=100,
            key=cls.QUESTION_TEXT_BOX
        )
        
        resolution_criteria = st.text_area(
            "Resolution Criteria (optional)",
            height=100,
            key=cls.RESOLUTION_CRITERIA_BOX,
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fine_print = st.text_area(
                "Fine Print (optional)",
                height=100, 
                key=cls.FINE_PRINT_BOX
            )
        
        with col2:
            background_info = st.text_area(
                "Background Info (optional)", 
                height=100,
                key=cls.BACKGROUND_INFO_BOX
            )
        
        st.write("### Forecaster Configuration")
        
        # Three columns for forecaster selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            use_llm = st.checkbox("Standard LLM", 
                                value=True,
                                key=cls.USE_LLM)
            if use_llm:
                llm_weight = st.slider("Weight", 
                                    min_value=0.0, 
                                    max_value=1.0, 
                                    value=0.3,
                                    key=cls.LLM_WEIGHT)
            else:
                llm_weight = 0.0
        
        with col2:
            use_enhanced_llm = st.checkbox("Enhanced LLM",
                                        value=True,
                                        key=cls.USE_ENHANCED_LLM)
            if use_enhanced_llm:
                enhanced_llm_weight = st.slider("Weight",
                                            min_value=0.0,
                                            max_value=1.0,
                                            value=0.4,
                                            key=cls.ENHANCED_LLM_WEIGHT)
            else:
                enhanced_llm_weight = 0.0
        
        with col3:
            use_expert = st.checkbox("Expert Forecaster",
                                   value=True,
                                   key=cls.USE_EXPERT)
            if use_expert:
                expert_weight = st.slider("Weight",
                                       min_value=0.0,
                                       max_value=1.0,
                                       value=0.3,
                                       key=cls.EXPERT_WEIGHT)
            else:
                expert_weight = 0.0
        
        # Key factors section
        st.write("### Key Factors / Assumptions")
        st.info("Define key factors that can be adjusted to explore 'what-if' scenarios")
        
        key_factors = []
        
        # Get existing factors from session state
        if 'key_factors' not in st.session_state:
            st.session_state.key_factors = []
        
        # Add new factor form
        with st.form("add_factor_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                factor_name = st.text_input("Factor Name", placeholder="e.g., GDP Growth Rate")
                factor_description = st.text_area("Description", placeholder="How this factor affects the forecast", height=100)
            
            with col2:
                factor_value = st.slider("Initial Value", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
                col1, col2 = st.columns(2)
                with col1:
                    min_value = st.number_input("Min Value", value=0.0, step=0.1)
                with col2:    
                    max_value = st.number_input("Max Value", value=1.0, step=0.1)
                step = st.number_input("Step Size", value=0.1, min_value=0.01, max_value=0.5, step=0.01)
            
            add_factor_button = st.button("Add Factor")
            
            if add_factor_button and factor_name:
                # Create new factor and add to list
                new_factor = ScenarioKey(
                    name=factor_name,
                    value=factor_value,
                    initial_value=factor_value,
                    min_value=min_value,
                    max_value=max_value,
                    step=step,
                    description=factor_description
                )
                st.session_state.key_factors.append(new_factor.model_dump())
                st.rerun()  # Refresh to show the updated list
        
        # Display existing factors
        if st.session_state.key_factors:
            st.write("#### Defined Factors")
            
            for i, factor in enumerate(st.session_state.key_factors):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{factor['name']}**: {factor['description']}")
                
                with col2:
                    st.write(f"Value: {factor['value']:.2f}")
                
                with col3:
                    if st.button("Remove"):
                        st.session_state.key_factors.pop(i)
                        st.rerun()
        
        # Run forecast button
        if st.button("Run Analysis"):
            if not question_text:
                st.error("Question Text is required.")
                return None
                
            # Ensure at least one forecaster is selected
            if not any([use_llm, use_enhanced_llm, use_expert]):
                st.error("Please select at least one forecaster.")
                return None
            
            question = BinaryQuestion(
                question_text=question_text,
                background_info=background_info,
                resolution_criteria=resolution_criteria,
                fine_print=fine_print,
                page_url="",
                api_json={},
            )
            
            # Store forecaster configurations in session state
            st.session_state['forecaster_config'] = {
                'use_llm': use_llm,
                'use_enhanced_llm': use_enhanced_llm,
                'use_expert': use_expert,
                
                'llm_weight': llm_weight,
                'enhanced_llm_weight': enhanced_llm_weight,
                'expert_weight': expert_weight,
            }
            
            # Convert key factors to model objects
            key_factors = [ScenarioKey(**factor) for factor in st.session_state.key_factors]
            
            return ScenarioInput(
                question=question, 
                key_factors=key_factors
            )
        
        return None

    @classmethod
    async def _run_tool(cls, input: ScenarioInput) -> ScenarioOutput:
        config = st.session_state.get('forecaster_config', {})
        
        with st.spinner("Analyzing baseline scenario..."):
            # Create forecasters based on configuration
            forecasters = []
            weights = []
            
            # Create standard LLM
            if config.get('use_llm', False):
                forecasters.append(GeneralLlm(model="openai/o1", temperature=0.2))
                weights.append(config.get('llm_weight', 0.3))
            
            # Create enhanced LLM
            if config.get('use_enhanced_llm', False):
                forecasters.append(EnhancedLLMForecaster(model_name="openai/o1", temperature=0.2))
                weights.append(config.get('enhanced_llm_weight', 0.4))
            
            # Create expert forecaster
            if config.get('use_expert', False):
                forecasters.append(ExpertForecaster(model_name="openai/o1", temperature=0.1))
                weights.append(config.get('expert_weight', 0.3))
            
            # Normalize weights
            if sum(weights) > 0:
                weights = [w/sum(weights) for w in weights]
            else:
                weights = [1.0/len(forecasters) for _ in forecasters]
            
            # Create the ensemble
            ensemble = EnsembleForecaster(
                forecasters=forecasters,
                weights=weights,
                ensemble_method="weighted_average"
            )
            
            # Run baseline forecast
            forecast_results = await ensemble._get_individual_forecasts(input.question)
            
            # Get overall prediction
            baseline_prediction = await ensemble.predict(input.question)
            baseline_explanation = await ensemble.explain(input.question)
            baseline_interval = await ensemble.confidence_interval(input.question)
            
            # Create baseline report
            baseline_report = BinaryReport(
                question=input.question,
                prediction=baseline_prediction,
                explanation=baseline_explanation,
                other_notes=f"Confidence Interval: [{baseline_interval[0]:.2f}, {baseline_interval[1]:.2f}]",
            )
            
            # Create forecaster breakdown for visualization
            forecaster_breakdown = {
                "forecasters": [],
                "predictions": [],
                "weights": [],
                "explanations": []
            }
            
            for i, result in enumerate(forecast_results):
                model_name = getattr(forecasters[i], 'model_name', f"Forecaster {i+1}")
                forecaster_breakdown["forecasters"].append(model_name)
                forecaster_breakdown["predictions"].append(result.prediction)
                forecaster_breakdown["weights"].append(weights[i])
                forecaster_breakdown["explanations"].append(result.explanation)
        
        # Run scenario analyses if key factors are provided
        scenario_reports = []
        
        if input.key_factors:
            with st.spinner(f"Running scenario analysis for {len(input.key_factors)} key factors..."):
                for factor in input.key_factors:
                    # Only run if the factor value is different from initial
                    if factor.value != factor.initial_value:
                        # Create modified question with factor adjustment in background info
                        modified_bg_info = input.question.background_info or ""
                        modified_bg_info += f"\n\nASSUMPTION CHANGE: {factor.name} adjusted to {factor.value} (from {factor.initial_value}). {factor.description}"
                        
                        modified_question = BinaryQuestion(
                            question_text=input.question.question_text,
                            background_info=modified_bg_info,
                            resolution_criteria=input.question.resolution_criteria,
                            fine_print=input.question.fine_print,
                            page_url="",
                            api_json={},
                        )
                        
                        # Run forecast with modified question
                        scenario_prediction = await ensemble.predict(modified_question)
                        scenario_explanation = await ensemble.explain(modified_question)
                        scenario_interval = await ensemble.confidence_interval(modified_question)
                        
                        # Create scenario report
                        scenario_report = BinaryReport(
                            question=modified_question,
                            prediction=scenario_prediction,
                            explanation=scenario_explanation,
                            other_notes=f"Scenario: {factor.name} = {factor.value} (baseline: {factor.initial_value})\nConfidence Interval: [{scenario_interval[0]:.2f}, {scenario_interval[1]:.2f}]",
                        )
                        
                        scenario_reports.append(scenario_report)
        
        return ScenarioOutput(
            baseline_report=baseline_report,
            scenario_reports=scenario_reports,
            forecaster_breakdown=forecaster_breakdown
        )

    @classmethod
    async def _display_outputs(cls, outputs: list[ScenarioOutput]) -> None:
        if not outputs:
            return
        
        output = outputs[0]
        
        # 1. Display baseline forecast
        st.header("Baseline Forecast")
        ReportDisplayer.display_report(output.baseline_report)
        
        # 2. Display forecaster breakdown
        st.header("Ensemble Breakdown")
        
        if output.forecaster_breakdown:
            # Create forecaster breakdown visualization
            forecasters = output.forecaster_breakdown["forecasters"]
            predictions = output.forecaster_breakdown["predictions"]
            weights = output.forecaster_breakdown["weights"]
            explanations = output.forecaster_breakdown["explanations"]
            
            # Create a DataFrame for the table
            df = pd.DataFrame({
                "Forecaster": forecasters,
                "Prediction": predictions,
                "Weight": weights
            })
            
            # Display as table
            st.dataframe(
                df,
                column_config={
                    "Forecaster": st.column_config.TextColumn("Forecaster"),
                    "Prediction": st.column_config.NumberColumn("Prediction", format="%.3f"),
                    "Weight": st.column_config.NumberColumn("Weight", format="%.2f"),
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Create bar chart for weights
            fig_weights = px.bar(
                df, 
                x="Forecaster", 
                y="Weight", 
                title="Ensemble Weights",
                color="Forecaster",
                text_auto=".2f"
            )
            fig_weights.update_layout(yaxis_range=[0, max(weights) * 1.2])
            st.plotly_chart(fig_weights, use_container_width=True)
            
            # Create bar chart for predictions
            fig_preds = px.bar(
                df, 
                x="Forecaster", 
                y="Prediction", 
                title="Individual Predictions",
                color="Forecaster",
                text_auto=".3f"
            )
            
            # Add horizontal line for ensemble prediction
            fig_preds.add_hline(
                y=output.baseline_report.prediction, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Ensemble: {output.baseline_report.prediction:.3f}"
            )
            
            fig_preds.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig_preds, use_container_width=True)
            
            # Show individual rationales
            with st.expander("Individual Forecaster Rationales"):
                for i, (forecaster, explanation) in enumerate(zip(forecasters, explanations)):
                    st.subheader(forecaster)
                    st.write(explanation)
                    if i < len(forecasters) - 1:
                        st.divider()
        
        # 3. Display scenario analyses
        if output.scenario_reports:
            st.header("Scenario Analysis")
            
            # Create tabs for each scenario
            scenario_tabs = st.tabs([f"Scenario {i+1}" for i in range(len(output.scenario_reports))])
            
            for i, (tab, report) in enumerate(zip(scenario_tabs, output.scenario_reports)):
                with tab:
                    # Extract scenario info from other_notes
                    scenario_name = ""
                    if report.other_notes:
                        scenario_match = re.search(r"Scenario: (.+?) = (.+?) \(baseline: (.+?)\)", report.other_notes)
                        if scenario_match:
                            factor_name = scenario_match.group(1)
                            factor_value = float(scenario_match.group(2))
                            baseline_value = float(scenario_match.group(3))
                            scenario_name = f"{factor_name}: {baseline_value} → {factor_value}"
                    
                    st.subheader(scenario_name)
                    
                    # Calculate change from baseline
                    change = report.prediction - output.baseline_report.prediction
                    change_pct = (change / output.baseline_report.prediction) * 100 if output.baseline_report.prediction != 0 else float('inf')
                    
                    # Display change metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Baseline Prediction", 
                            f"{output.baseline_report.prediction:.3f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Scenario Prediction", 
                            f"{report.prediction:.3f}",
                            f"{change:+.3f} ({change_pct:+.1f}%)"
                        )
                    
                    with col3:
                        # Extract confidence intervals
                        baseline_ci = [0.0, 0.0]
                        scenario_ci = [0.0, 0.0]
                        
                        if output.baseline_report.other_notes:
                            ci_match = re.search(r"Confidence Interval: \[(.+?), (.+?)\]", output.baseline_report.other_notes)
                            if ci_match:
                                baseline_ci = [float(ci_match.group(1)), float(ci_match.group(2))]
                        
                        if report.other_notes:
                            ci_match = re.search(r"Confidence Interval: \[(.+?), (.+?)\]", report.other_notes)
                            if ci_match:
                                scenario_ci = [float(ci_match.group(1)), float(ci_match.group(2))]
                        
                        st.metric(
                            "Confidence Interval Width",
                            f"{scenario_ci[1] - scenario_ci[0]:.3f}",
                            f"{(scenario_ci[1] - scenario_ci[0]) - (baseline_ci[1] - baseline_ci[0]):+.3f}"
                        )
                    
                    # Display confidence intervals as a chart
                    fig = go.Figure()
                    
                    # Add baseline prediction with confidence interval
                    fig.add_trace(go.Scatter(
                        x=["Baseline"],
                        y=[output.baseline_report.prediction],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=[baseline_ci[1] - output.baseline_report.prediction],
                            arrayminus=[output.baseline_report.prediction - baseline_ci[0]]
                        ),
                        marker=dict(color='blue', size=12),
                        name="Baseline"
                    ))
                    
                    # Add scenario prediction with confidence interval
                    fig.add_trace(go.Scatter(
                        x=["Scenario"],
                        y=[report.prediction],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=[scenario_ci[1] - report.prediction],
                            arrayminus=[report.prediction - scenario_ci[0]]
                        ),
                        marker=dict(color='red', size=12),
                        name="Scenario"
                    ))
                    
                    fig.update_layout(
                        title="Prediction Comparison with Confidence Intervals",
                        yaxis=dict(
                            title="Probability",
                            range=[0, 1]
                        ),
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display the scenario report details
                    with st.expander("Full Scenario Report"):
                        ReportDisplayer.display_report(report)
        
        # 4. Interactive scenario builder
        if 'key_factors' in st.session_state and st.session_state.key_factors:
            st.header("Interactive Scenario Builder")
            st.info("Adjust key factors and re-run the analysis to see how changes affect the forecast")
            
            key_factors = [ScenarioKey(**factor) for factor in st.session_state.key_factors]
            
            # Create sliders for each factor
            factors_changed = False
            for i, factor in enumerate(key_factors):
                new_value = st.slider(
                    f"{factor.name}",
                    min_value=factor.min_value,
                    max_value=factor.max_value,
                    value=factor.value,
                    step=factor.step,
                    help=factor.description,
                    key=f"factor_slider_{i}"
                )
                
                # Update factor value if changed
                if new_value != factor.value:
                    st.session_state.key_factors[i]['value'] = new_value
                    factors_changed = True
            
            # Re-run button
            if factors_changed and st.button("Re-run Scenario Analysis"):
                st.rerun()
            
            # Reset button
            if st.button("Reset to Baseline"):
                # Reset all factors to initial values
                for i, factor in enumerate(st.session_state.key_factors):
                    st.session_state.key_factors[i]['value'] = factor['initial_value']
                st.rerun()

    @classmethod
    async def _save_run_to_coda(
        cls,
        input_to_tool: Jsonable,
        output: Jsonable,
        is_premade_example: bool,
    ) -> None:
        # Not implementing Coda integration for this page
        pass 