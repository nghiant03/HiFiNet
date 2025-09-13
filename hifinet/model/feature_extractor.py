import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


class TimeSeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        sequence_columns: list[str],
        fc_parameters: dict[str, dict] | None = None,
    ):
        self.sequence_columns = sequence_columns
        self.fc_parameters = fc_parameters

    def transform(self, x, y=None):
        long_data = x.explode(self.sequence_columns).reset_index(drop=True)
        for c in self.sequence_columns:
            long_data[c] = pd.to_numeric(long_data[c], errors='coerce')

        features = extract_features(
            long_data,
            column_id="id",
            default_fc_parameters=self.fc_parameters,
            n_jobs=2
        )
        features = impute(features)
        return features

    def fit(self, x, y=None):
        return self
