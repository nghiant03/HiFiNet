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
        ts_data = []
        sample_ids = []
        for _, row in x.iterrows():
            series_id = str(row["id"])
            if "day" in x.columns:
                series_id = f"{series_id}_{row['day']}"
            sample_ids.append(series_id)
            for col in self.sequence_columns:
                sequence = row[col]
                for t, value in enumerate(sequence):
                    ts_data.append(
                        {
                            "id": series_id,
                            "time": t,
                            "value": value,
                            "variable": col,
                        }
                    )

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
        features = features.reindex(sample_ids)
        features.index = x.index
        return features

    def fit(self, x, y=None):
        return self
