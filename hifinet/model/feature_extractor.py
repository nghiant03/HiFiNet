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
        if "day" in x.columns:
            series_id = x["id"].astype(str) + "_" + x["day"].astype(str)
        else:
            series_id = x["id"].astype(str)

        sample_ids = series_id.tolist()

        ts_df = (
            x.assign(id=series_id)[["id", *self.sequence_columns]]
            .melt(id_vars="id", var_name="variable", value_name="value")
            .explode("value")
        )
        ts_df["time"] = ts_df.groupby(["id", "variable"]).cumcount()
        ts_df = ts_df[["id", "time", "value", "variable"]]

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
