from forecasting_tools.ai_models.model_interfaces.forecaster_base import ForecasterBase

class CommunityForecaster(ForecasterBase):
    def __init__(self, community_data):
        self.community_data = community_data

    def predict(self, question, context=None):
        return self.community_data.get('median', 0.5)

    def explain(self, question, context=None):
        return "Forecast is the current Metaculus community median."

    def confidence_interval(self, question, context=None):
        return (
            self.community_data.get('lower', 0.4),
            self.community_data.get('upper', 0.6)
        ) 