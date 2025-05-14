import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now try to import from the forecasting_tools package
try:
    from forecasting_tools.front_end.Home import run_forecasting_streamlit_app
    print("Successfully imported run_forecasting_streamlit_app!")
except ImportError as e:
    print(f"Import failed: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    
    # List all the directories to help debug
    print("\nChecking directories:")
    if os.path.exists("forecasting_tools"):
        print("  - forecasting_tools directory exists")
        subdirs = [d for d in os.listdir("forecasting_tools") if os.path.isdir(os.path.join("forecasting_tools", d))]
        print(f"  - Subdirectories: {subdirs}")
    else:
        print("  - forecasting_tools directory NOT found") 