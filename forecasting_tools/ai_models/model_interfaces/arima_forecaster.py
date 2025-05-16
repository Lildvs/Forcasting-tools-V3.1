from forecasting_tools.ai_models.model_interfaces.forecaster_base import ForecasterBase

class ARIMAForecaster(ForecasterBase):
    def __init__(self, model):
        self.model = model

    def predict(self, question, context=None):
        # Use ARIMA to forecast
        forecast = self.model.forecast(steps=1)
        return float(forecast[0])

    def explain(self, question, context=None):
        return "Forecast based on ARIMA time series model."

    def confidence_interval(self, question, context=None):
        # Use ARIMA's conf_int method
        conf = self.model.get_forecast(steps=1).conf_int()
        return tuple(conf.iloc[0]) 