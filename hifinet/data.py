import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from tsfresh import extract_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        fc_parameters: dict[str, dict] | None = None,
    ):
        self.fc_parameters = fc_parameters

    def _to_long(self, x: pd.DataFrame) -> pd.DataFrame:
        sequence_columns = [column for column in x.columns if re.match(r"feature_*", column)]
        sequence_columns.append("target")
        df = x[["seq_id"] + sequence_columns].copy()

        df = df.melt(
            id_vars="seq_id",
            value_vars=sequence_columns,
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
            n_jobs=16,
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
            n_jobs=16,
        )
        impute(feats_all)

        assert isinstance(feats_all, pd.DataFrame), (
            f"Extractor return {type(feats_all)}, expected DataFrame"
        )

        feats_all = feats_all.reindex(columns=self.selected_columns_, fill_value=0)

        return feats_all.reindex(index=x["seq_id"].values)

def split(data, temp):
    match temp:
        case 0:
            return (
                data[data["datetime"] < "2023-08-01"],
                data[
                    (data["datetime"] >= "2023-08-01")
                    | (data["datetime"] < "2023-10-01")
                ],
                data[
                    (data["datetime"] >= "2023-10-01")
                    | (data["datetime"] < "2023-11-01")
                ],
            )
        case 1:
            return (
                data[
                    (data["datetime"] >= "2023-02-01")
                    | (data["datetime"] < "2023-09-01")
                ],
                data[
                    (data["datetime"] >= "2023-09-01")
                    | (data["datetime"] < "2023-11-01")
                ],
                data[
                    (data["datetime"] >= "2023-11-01")
                    | (data["datetime"] < "2023-12-01")
                ],
            )
        case 2:
            return (
                data[
                    (data["datetime"] >= "2023-03-01")
                    | (data["datetime"] < "2023-10-01")
                ],
                data[
                    (data["datetime"] >= "2023-10-01")
                    | (data["datetime"] < "2023-12-01")
                ],
                data[data["datetime"] >= "2023-12-03"],
            )
        case _:
            raise NotImplementedError

def format_data(data):
    data["seq_idx"] = data.groupby("id").cumcount()
    data["seq_id"] = data["id"].astype(str) + "_" + data["seq_idx"].astype(str)
    return data
