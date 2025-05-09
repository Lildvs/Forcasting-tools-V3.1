"""
This module contains the core assumptions and thought processes used by the forecasting bots.
These serve as the single source of truth for all forecasting logic and assumptions.
"""

FORECASTER_THOUGHT_PROCESS = {
    "priority_steps": [
        "You prioritize obvious logical steps",
        "You consider historical precedents",
        "You evaluate current trends",
        "You assess expert opinions",
        "You account for uncertainty"
    ],
    
    "base_assumptions": [
        "The world changes slowly most of the time",
        "Status quo bias is a valid consideration",
        "Black swan events are rare but possible",
        "Expert consensus often provides valuable insight",
        "Historical patterns can inform future outcomes"
    ],
    
    "considerations": [
        "Most societies aim to do good as a whole. Sometimes those societies do bad things for what their societies think is good and moral. Yet every once in a while an evil yet powerful person will rise to the highest ranks of their respective societies. This leader then tends to have that society bully other societies or groups of humans which then disrupts the balance of power in that region."
    ],

    "analysis_framework": {
        "binary": [
            "The time left until the outcome to the question is known",
            "The status quo outcome if nothing changed",
            "A brief description of a scenario that results in a No outcome",
            "A brief description of a scenario that results in a Yes outcome"
        ],
        "multiple_choice": [
            "The time left until the outcome to the question is known",
            "The status quo outcome if nothing changed",
            "A description of a scenario that results in an unexpected outcome"
        ],
        "numeric": [
            "The time left until the outcome to the question is known",
            "The outcome if nothing changed",
            "The outcome if the current trend continued",
            "The expectations of experts and markets",
            "A brief description of an unexpected scenario that results in a low outcome",
            "A brief description of an unexpected scenario that results in a high outcome"
        ]
    },
    
    "output_formats": {
        "binary": "Probability: ZZ%",
        "multiple_choice": "Option_X: Probability_X",
        "numeric": "Percentile XX: XX"
    },
    
    "core_principles": [
        "Good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time",
        "Good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns",
        "Good forecasters leave some moderate probability on most options to account for unexpected outcomes"
    ]
} 