"""
Compatibility module to handle imports for modules that were moved.
"""

# Import the time_series_forecaster from its new location 
# to maintain backward compatibility with code that imports from the old path
from forecasting_tools.ai_models.model_interfaces.time_series_forecaster import *

# Re-export any other moved modules here if needed 