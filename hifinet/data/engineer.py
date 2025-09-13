import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from tsfresh import extract_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute


class TimeSeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        sequence_columns: list[str],
        fc_parameters: dict[str, dict] | None = None,
    ):
        self.sequence_columns = sequence_columns
        self.fc_parameters = fc_parameters

    def _to_long(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x[["seq_id"] + self.sequence_columns].copy()

        df = df.melt(
            id_vars="seq_id",
            value_vars=self.sequence_columns,
            var_name="kind",
            value_name="value",
        )

        df = df.explode("value", ignore_index=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df

    def fit(self, x: pd.DataFrame, y: pd.Series):
        long_data = self._to_long(x)

        y_series = pd.Series(
            np.asarray(y),
            index=pd.Index(x["seq_id"].to_numpy(), name="seq_id"),
        )
        feats = extract_relevant_features(
            long_data,
            y_series,
            column_id="seq_id",
            column_kind="kind",
            column_value="value",
            default_fc_parameters=self.fc_parameters,
            n_jobs=2,
        )

        self.selected_columns_ = feats.columns.tolist()
        self.is_fitted_ = True
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "is_fitted_")
        x_long = self._to_long(x)

        feats_all = extract_features(
            x_long,
            column_id="seq_id",
            column_kind="kind",
            column_value="value",
            default_fc_parameters=self.fc_parameters,
            n_jobs=2,
        )
        impute(feats_all)

        assert isinstance(feats_all, pd.DataFrame), f"Extractor return {type(feats_all)}, expected DataFrame"

        feats_all = feats_all.reindex(columns=self.selected_columns_, fill_value=0)

        return feats_all.reindex(index=x["seq_id"].values)
