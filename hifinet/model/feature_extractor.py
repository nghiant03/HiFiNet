import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh import extract_relevant_features


class TimeSeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        sequence_columns: list[str],
        fc_parameters: dict[str, dict] | None = None,
    ):
        self.sequence_columns = sequence_columns
        self.fc_parameters = fc_parameters

    def transform(self, x, y):
        long_data = x.explode(self.sequence_columns).reset_index(drop=True)
        for c in self.sequence_columns:
            long_data[c] = pd.to_numeric(long_data[c], errors="coerce")

        features = extract_relevant_features(
            long_data,
            y,
            column_id="seq_id",
            default_fc_parameters=self.fc_parameters,
            n_jobs=2
        )
        return features

    def fit(self, x, y=None):
        return self
