from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression, LinearRegression
from src.feature_extractor import TfidfFeatureExtractor
from src.transcript_preprocessor import TranscriptPreprocessor

class BaseModel(ABC):
    @abstractmethod
    def fit(self, train_data):
        pass

    @abstractmethod
    def predict(self, test_data):
        pass

    def temporal_split(self, X, y, test_size=0.2):
        """ Temporal split : train = first 80%, test = last 20%"""
        split_idx = int(len(X) * (1 - test_size))

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, X_test, y_train, y_test


class LogisticRegressionModel(BaseModel):
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    '''def predict_proba(self, test_data):
        return self.model.predict_proba(test_data)'''

    def predict(self, X_test):
        return self.model.predict(X_test)
