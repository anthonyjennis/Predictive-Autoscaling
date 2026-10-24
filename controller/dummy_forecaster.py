"""Stand-in with the same interface as model/forecaster.py: 'next hour = this hour' (naive forecast)."""


class DummyForecaster:
    def __init__(self, artifact_dir=None):
        pass

    def predict(self, history, quantile=0.9):
        return float(history["rps"].iloc[-1])