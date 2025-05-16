from abc import ABC, abstractmethod

class ForecasterBase(ABC):
    @abstractmethod
    def predict(self, question, context=None):
        """Return a probability or forecast result for the given question."""
        pass

    @abstractmethod
    def explain(self, question, context=None):
        """Return a rationale or explanation for the forecast."""
        pass

    @abstractmethod
    def confidence_interval(self, question, context=None):
        """Return a (lower, upper) confidence interval for the forecast."""
        pass 