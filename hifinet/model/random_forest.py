import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


class TimeSeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, sequence_columns: list[str], fc_parameters: list[str] | None= None):
        self.sequence_columns = sequence_columns
        self.fc_parameters = fc_parameters
        
    def transform(self, x, y=None):
        # Create a melted DataFrame for tsfresh
        ts_data = []
        for idx, row in x.iterrows():
            for col in self.sequence_columns:
                sequence = row[col]
                for t, value in enumerate(sequence):
                    ts_data.append({
                        "id": idx,
                        "time": t,
                        "value": value,
                        "variable": col
                    })
        
        ts_df = pd.DataFrame(ts_data)
        
        # Extract features using tsfresh
        features = extract_features(
            ts_df,
            column_id="id",
            column_sort="time",
            column_value="value",
            column_kind="variable",
            default_fc_parameters=self.fc_parameters,
        )
        features = impute(features)
        return features

    def fit(self, x, y=None):
        return self


class RandomForestModel:
    def __init__(self, sequence_columns=None, n_estimators=100, **kwargs):
        self.sequence_columns = sequence_columns or ["feature_1", "flow"]
        self.n_estimators = n_estimators
        self.pipeline = self._build_pipeline(**kwargs)
        
    def _build_pipeline(self, **kwargs):
        return Pipeline([
            ("feature_extractor", TimeSeriesFeatureExtractor(self.sequence_columns)),
            ("classifier", RandomForestClassifier(
                n_estimators=self.n_estimators,
                n_jobs=settings.N_JOBS,
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
