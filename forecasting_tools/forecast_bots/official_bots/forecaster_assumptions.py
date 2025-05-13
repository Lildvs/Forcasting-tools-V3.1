"""
This module contains the core assumptions and thought processes used by the forecasting bots.
These serve as the single source of truth for all forecasting logic and assumptions.
"""

FORCASTER_DATA_COLLECTION_AND_ANALYSIS = {
    "Data Collection": [
        "Provided with resources R, gather as much data as you can, from this list of desired data, about X"
    ],
    "World Model": [
        '"World model" here likely means, "A set of stored symbolic and probabilistic assumptions about the world / people / organizations."',
        "Given a forecasting LLM L, and forecasting question Q, contribute a world model that will result in L being highly accurate."
    ],
    "Human Elicitation": {
        "enabled": False,
        "prompts": [
            "You are a professional forecaster interviewing for a job.",
            "Given that you want to get as much useful new data about topic X, using Perplexity Sonar's deep research capabilities with high search context and reasoning effort, use resources R to send off elicitions and gather data from humans."
        ]
    },
    "Data Prioritization": [
        "Given the potential to fetch from data source D, and resources R, fetch the most useful data for providing marginal insight on unknown variables V.",
        "Write 5 forecasting questions on topic X. After this, a separate epistemic system will be run with a fixed number of resources R on each question. Choose questions such that the final results are maximally useful to end-users."
    ],
    "Idea Generation": [
        "Given a person with details D, generate a plan for that person to optimize their question."
    ],
    "Enhanced Forecasting": {
        "enabled": False,
        "prompts": [
            "Using information I, and N LLM calls, make your best forecast on variable V.",
            "Using an existing LEP L, and resources R, write a set of forecasting questions. Expect that forecasting resources F will be used on these questions, and you are trying to optimize utility function U."
        ]
    }
}

FORECASTER_THOUGHT_PROCESS = {
    # Combined priority steps and chain-of-thought into an integrated reasoning process
    "priority_chain_of_thought": {
        "enabled": True,
        "steps": [
            # From original priority_steps
            "Prioritize obvious logical steps in your analysis",
            "Focus on verifiable factual events rather than speculation",
            "Consider historical precedents and base rates",
            "Evaluate current trends and their momentum",
            "Assess expert opinions while considering potential biases",
            
            # From chain_of_thought steps with enhancements
            "Identify the key variables that will affect this outcome and their relationships",
            "Analyze related historical events to establish base rates and patterns",
            "Project current trends forward with appropriate uncertainty",
            "Identify potential inflection points or discontinuities that could change trajectories",
            "Quantify your uncertainty for each factor and how they compound",
            "Synthesize all factors into a coherent probabilistic forecast"
        ]
    },
    
    # Consolidated forecasting principles from base_assumptions, constitutional_principles, and core_principles
    "forecasting_principles": {
        "core": [
            "Weight the status quo outcome more heavily (the world changes slowly most of the time)",
            "Express appropriate epistemic humility with wide confidence intervals for unknown unknowns",
            "Allocate some probability to unexpected outcomes and edge cases"
        ],
        "epistemological": [
            "The world changes slowly most of the time",
            "Status quo bias is a valid consideration",
            "Black swan events are rare but possible"
        ],
        "methodological": [
            "Maintain appropriate uncertainty (avoid overconfidence)",
            "Consider base rates and outside views first",
            "Recognize the limits of your knowledge",
            "Distinguish between facts, interpretations, and speculations",
            "Weight the status quo outcome more heavily in your probability distribution"
        ],
        "analytical": [
            "Expert consensus often provides valuable insight",
            "Historical patterns can inform future outcomes",
            "Update incrementally on new evidence",
            "Be aware of cognitive biases that might affect your judgment",
            "Allocate probability mass to edge cases that aren't immediately apparent"
        ],
        "communication": [
            "Quantify your uncertainty explicitly",
            "Set wide 90/10 confidence intervals to account for unknown unknowns",
            "Leave some moderate probability on most options to account for unexpected outcomes"
        ]
    },
    
    "considerations": [
        "Most societies aim to do good as a whole. Sometimes those societies do bad things for what their societies think is good and moral. Yet every once in a while an evil yet powerful person will rise to the highest ranks of their respective societies. This leader then tends to have that society bully other societies or groups of humans which then disrupts the balance of power in that region."
    ],

    # Merged analysis framework and tree of thoughts into scenario-based analysis
    "scenario_analysis": {
        "binary": {
            "timeline": "The time left until the outcome to the question is known",
            "scenarios": {
                "status_quo": {
                    "description": "What happens if current trends continue?",
                    "outcome": "The status quo outcome if nothing changed",
                    "probability": "Likelihood of this scenario",
                    "indicators": "Early signals this scenario is unfolding"
                },
                "yes_outcome": {
                    "description": "A scenario that results in a Yes outcome",
                    "key_assumptions": "Critical assumptions for this scenario",
                    "probability": "Likelihood of this scenario",
                    "indicators": "Early signals this scenario is unfolding"
                },
                "no_outcome": {
                    "description": "A scenario that results in a No outcome",
                    "key_assumptions": "Critical assumptions for this scenario",
                    "probability": "Likelihood of this scenario", 
                    "indicators": "Early signals this scenario is unfolding"
                }
            }
        },
        "multiple_choice": {
            "timeline": "The time left until the outcome to the question is known",
            "scenarios": {
                "status_quo": {
                    "description": "What happens if current trends continue?",
                    "outcome": "The status quo outcome if nothing changed",
                    "probability": "Likelihood of this scenario",
                    "indicators": "Early signals this scenario is unfolding"
                },
                "unexpected": {
                    "description": "A scenario that results in an unexpected outcome",
                    "key_assumptions": "Critical assumptions for this scenario",
                    "probability": "Likelihood of this scenario",
                    "indicators": "Early signals this scenario is unfolding"
                }
            }
        },
        "numeric": {
            "timeline": "The time left until the outcome to the question is known",
            "scenarios": {
                "status_quo": {
                    "description": "The outcome if nothing changed",
                    "value": "Numeric prediction for this scenario",
                    "probability": "Likelihood of this scenario"
                },
                "trend_continuation": {
                    "description": "The outcome if current trends continued",
                    "value": "Numeric prediction for this scenario",
                    "probability": "Likelihood of this scenario"
                },
                "expert_consensus": {
                    "description": "The expectations of experts and markets",
                    "value": "Numeric prediction based on expert views",
                    "probability": "Likelihood of this scenario" 
                },
                "low_surprise": {
                    "description": "An unexpected scenario resulting in a low outcome",
                    "value": "Numeric prediction for this scenario",
                    "probability": "Likelihood of this scenario"
                },
                "high_surprise": {
                    "description": "An unexpected scenario resulting in a high outcome",
                    "value": "Numeric prediction for this scenario",
                    "probability": "Likelihood of this scenario"
                }
            }
        }
    },
    
    # Unified approach to multiple perspective forecasting
    "perspective_analysis": {
        "enabled": True,
        "perspectives": {
            "historical": {
                "description": "Based primarily on historical data and precedents",
                "key_reference_points": "Identify 2-3 key historical reference points",
                "base_rate": "Establish the base rate from historical data",
                "adjustment": "Adjustments needed for current circumstances"
            },
            "current_trends": {
                "description": "Based on recent developments and momentum",
                "key_trends": "Identify 2-3 most important current trends",
                "trajectory": "Project these trends forward",
                "inflection_points": "Possible points where trends might change"
            },
            "expert_consensus": {
                "description": "Based on what domain experts are predicting",
                "key_experts": "Identify relevant expert opinions",
                "consensus_view": "Summarize the consensus position",
                "dissenting_views": "Note significant disagreements"
            },
            "outside_view": {
                "description": "Based on similar situations without specific details",
                "reference_class": "Define the appropriate reference class",
                "base_rate": "Establish the base rate in this reference class",
                "adjustment": "Minimal adjustments for specifics of this case"
            }
        },
        "reconciliation": {
            "weighting": "Assign relative weights to each perspective",
            "synthesis": "Synthesize insights across perspectives",
            "justification": "Explain which aspects of each you're incorporating and why"
        }
    },
    
    "world_model_building": {
        "enabled": True,
        "steps": [
            "Identify the 3-5 most important causal factors affecting this outcome",
            "Map the relationships between these factors (what influences what)",
            "Estimate the strength and direction of each relationship",
            "Identify feedback loops or non-linear dynamics",
            "Determine which factors are most uncertain vs. most predictable"
        ]
    },
    
    "output_formats": {
        "binary": "Probability: ZZ%",
        "multiple_choice": "Option_X: Probability_X",
        "numeric": "Percentile XX: XX"
    },
    
    # DEPRECATED: For backward compatibility only - these are now fully integrated into forecasting_principles
    # This will be removed in a future version - use forecasting_principles instead
    "core_principles": [
        "Good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time",
        "Good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns",
        "Good forecasters leave some moderate probability on most options to account for unexpected outcomes"
    ],
}

# More modular template system
def build_forecast_prompt(question_type, question_data, research, include_sections=None):
    """
    Build a comprehensive forecast prompt with selective section inclusion.
    
    Args:
        question_type: Type of forecast ('binary', 'multiple_choice', 'numeric')
        question_data: Dict containing question properties
        research: Research information to include
        include_sections: List of sections to include (defaults to all)
    
    Returns:
        A formatted prompt string
    """
    if include_sections is None:
        include_sections = ['principles', 'priority_chain', 'scenarios', 'perspectives', 'world_model']
    
    # Format the question section
    question_section = f"""
    You are a professional forecaster interviewing for a job.

    Your interview question is:
    {question_data.get('question_text', '')}
    """
    
    if question_type == 'multiple_choice':
        question_section += f"\nThe options are: {question_data.get('options', [])}"

    question_section += f"""
    
    Question background:
    {question_data.get('background_info', '')}
    
    This question's outcome will be determined by the specific criteria below:
    {question_data.get('resolution_criteria', '')}
    
    {question_data.get('fine_print', '')}
    """
    
    if question_type == 'numeric' and question_data.get('unit_of_measure'):
        question_section += f"\nUnits for answer: {question_data.get('unit_of_measure')}"
    
    # Format the research section
    research_section = f"""
    Your research assistant says:
    {research}
    
    Today is {question_data.get('current_date', '')}
    """
    
    # Format each optional section
    sections = {}
    
    # Priority chain of thought section
    if 'priority_chain' in include_sections:
        sections['priority_chain'] = format_priority_chain_section()
    
    # Principles section
    if 'principles' in include_sections:
        sections['principles'] = format_principles_section()
    
    # Scenario analysis section
    if 'scenarios' in include_sections:
        sections['scenarios'] = format_scenarios_section(question_type)
    
    # Perspectives section
    if 'perspectives' in include_sections:
        sections['perspectives'] = format_perspectives_section()
    
    # World model section
    if 'world_model' in include_sections:
        sections['world_model'] = format_world_model_section()
    
    # Format the output section
    output_section = format_output_section(question_type)
    
    # Combine all sections
    template_parts = [question_section, research_section]
    for section_name in include_sections:
        if section_name in sections:
            template_parts.append(sections[section_name])
    template_parts.append(output_section)
    
    # Clean up any empty sections and return
    formatted_template = "\n\n".join(template_parts)
    formatted_template = formatted_template.replace("\n\n\n", "\n\n")
    
    return formatted_template.strip()

def format_priority_chain_section():
    """Format the priority chain of thought section"""
    steps = "\n".join([f"{i+1}. {step}" for i, step in enumerate(FORECASTER_THOUGHT_PROCESS["priority_chain_of_thought"]["steps"])])
    return f"""
    First, reason through your forecast methodically:
    {steps}
    """

def format_principles_section():
    """Format the forecasting principles section with emphasis on core principles"""
    principles = []
    
    # Highlight the core principles first
    principles.append("--- Core Principles ---")
    for item in FORECASTER_THOUGHT_PROCESS["forecasting_principles"]["core"]:
        principles.append(f"• {item}")
    
    # Then include the remaining categorized principles
    for category, items in FORECASTER_THOUGHT_PROCESS["forecasting_principles"].items():
        if category != "core":  # Skip core since we already included it
            principles.append(f"\n--- {category.title()} Principles ---")
            for item in items:
                principles.append(f"• {item}")
    
    return f"""
    As you generate this forecast, adhere to these forecasting principles:
    {' '.join(principles)}
    """

def format_scenarios_section(question_type):
    """Format the scenario analysis section for the given question type"""
    scenario_data = FORECASTER_THOUGHT_PROCESS["scenario_analysis"][question_type]
    
    scenarios_text = f"Consider the following analysis framework:\n\n"
    scenarios_text += f"Timeline: {scenario_data['timeline']}\n\n"
    scenarios_text += "Analyze these scenarios:\n"
    
    for name, scenario in scenario_data["scenarios"].items():
        scenarios_text += f"\n- {name.replace('_', ' ').title()} Scenario: {scenario['description']}\n"
        for key, value in scenario.items():
            if key != "description":
                scenarios_text += f"  • {key.replace('_', ' ').title()}: {value}\n"
    
    return scenarios_text

def format_perspectives_section():
    """Format the perspectives analysis section"""
    perspectives = FORECASTER_THOUGHT_PROCESS["perspective_analysis"]["perspectives"]
    
    perspectives_text = "Generate forecasts from these different perspectives:\n"
    
    for name, perspective in perspectives.items():
        perspectives_text += f"\n- {name.replace('_', ' ').title()} Perspective: {perspective['description']}\n"
        for key, value in perspective.items():
            if key != "description":
                perspectives_text += f"  • {key.replace('_', ' ').title()}: {value}\n"
    
    reconciliation = FORECASTER_THOUGHT_PROCESS["perspective_analysis"]["reconciliation"]
    perspectives_text += "\nThen reconcile these perspectives:\n"
    for key, value in reconciliation.items():
        perspectives_text += f"• {key.replace('_', ' ').title()}: {value}\n"
    
    return perspectives_text

def format_world_model_section():
    """Format the world model building section"""
    steps = "\n".join([f"{i+1}. {step}" for i, step in enumerate(FORECASTER_THOUGHT_PROCESS["world_model_building"]["steps"])])
    return f"""
    Before finalizing your forecast, construct a world model by:
    {steps}
    """

def format_output_section(question_type):
    """Format the output section for the given question type"""
    output_format = FORECASTER_THOUGHT_PROCESS["output_formats"][question_type]
    
    if question_type == "binary":
        return f"""
        The last thing you write is your final answer as: "{output_format}", 0-100
        """
    elif question_type == "multiple_choice":
        return f"""
        The last thing you write is your final probabilities for each option as:
        {output_format}
        """
    elif question_type == "numeric":
        return f"""
        The last thing you write is your final percentiles for the outcome. 
        You always give the 1st, 10th, 25th, 50th, 75th, 90th, and 99th percentiles. 
        Each percentile is given in the format:
        {output_format}
        """
    return ""

# For backward compatibility with existing code
def format_advanced_prompting_template(template_type, include_all=True, **kwargs):
    """
    Format the advanced prompting template with appropriate values.
    
    Args:
        template_type: Type of template to format ('binary', 'multiple_choice', 'numeric')
        include_all: Whether to include all advanced prompting techniques
        kwargs: Optional overrides for specific template sections
    
    Returns:
        Formatted template string
    """
    include_sections = []
    
    if include_all or kwargs.get("include_priority_cot", False):
        include_sections.append('priority_chain')
    
    if include_all or kwargs.get("include_principles", False):
        include_sections.append('principles')
    
    if include_all or kwargs.get("include_scenarios", False) or kwargs.get("include_tot", False):
        include_sections.append('scenarios')
    
    if include_all or kwargs.get("include_perspectives", False) or kwargs.get("include_self_consistency", False):
        include_sections.append('perspectives')
    
    if include_all or kwargs.get("include_world_model", False):
        include_sections.append('world_model')
    
    # Create mock question data with minimal information
    question_data = {
        'current_date': kwargs.get('current_date', '')
    }
    
    # Use the new build_forecast_prompt function to generate the template
    template = build_forecast_prompt(
        question_type=template_type,
        question_data=question_data,
        research="",
        include_sections=include_sections
    )
    
    return template

# Helper function for retrieving principles by reference - useful for backward compatibility
def get_core_principle(index):
    """
    Get a core principle by index. Uses the new forecasting_principles structure
    but maintains backward compatibility with old code that references core_principles by index.
    
    Args:
        index: Index of the core principle (0-2)
        
    Returns:
        The core principle as a string
    """
    if 0 <= index < len(FORECASTER_THOUGHT_PROCESS["forecasting_principles"]["core"]):
        return FORECASTER_THOUGHT_PROCESS["forecasting_principles"]["core"][index]
    else:
        return FORECASTER_THOUGHT_PROCESS["core_principles"][index] 