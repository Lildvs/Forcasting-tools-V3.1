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