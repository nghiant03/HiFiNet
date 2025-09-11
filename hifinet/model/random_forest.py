from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from hifinet.model.feature_extractor import TimeSeriesFeatureExtractor


class RandomForestModel:
    def __init__(self, sequence_columns, n_estimators=100, **kwargs):
        self.sequence_columns = sequence_columns
        self.n_estimators = n_estimators
        self.pipeline = self._build_pipeline(**kwargs)
        
    def _build_pipeline(self, **kwargs):
        return Pipeline([
            ("feature_extractor", TimeSeriesFeatureExtractor(self.sequence_columns)),
            ("classifier", RandomForestClassifier(
                n_estimators=self.n_estimators,
                n_jobs=-1,
                **kwargs
            ))
        ])
    
    def fit(self, x, y):
        self.pipeline.fit(x, y)
        return self
        
    def predict(self, x):
        return self.pipeline.predict(x)
    
    def predict_proba(self, x):
        return self.pipeline.predict_proba(x)
