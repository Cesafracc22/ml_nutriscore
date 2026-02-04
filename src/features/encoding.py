import joblib
import pandas as pd
import numpy as np
from collections import Counter
from typing import Optional, Dict, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, MultiLabelBinarizer
from tqdm import tqdm

from data.countries_mappings import normalize_country


CATEGORICAL_FEATURES = ["countries", "pnns_groups_1", "pnns_groups_2"]

"""
Helper function to parse the countries column.
"""
def parse_countries(country_string: str) -> List[str]:
    countries = []
    for c in str(country_string).split(','):
        c = c.strip()
        if c and c.lower() != 'unknown':
            countries.append(normalize_country(c))
    return countries


"""
This class encodes the categorical features of the dataset.
It inherit from BaseEstimator and TransformerMixin to be used in a scikit-learn pipeline.
It contains 3 different encoders for the 3 different categorical features.

Note: we choose the following encoding methods based on the characteristics of the features:
- Countries: Multi-Label Binarization of top N countries
- pnns_groups_1: One-Hot Encoding (low cardinality)
- pnns_groups_2: Target Encoding (medium cardinality)
"""

class FeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, top_n_countries: int = 15, target_col: str = 'nutriscore_grade'):
        self.top_n_countries = top_n_countries
        self.target_col = target_col
        self.encoders_: Dict[str, BaseEstimator] = {}
        self.top_countries_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEncoder':
        self.encoders_ = {}
        self.top_countries_ = None

        
        if y is None and self.target_col in X.columns:
            y = X[self.target_col]

        for feature in CATEGORICAL_FEATURES:
            if feature not in X.columns:
                continue

            if feature == "countries":
                self._fit_countries_encoder(X[feature])
            elif feature == "pnns_groups_1":
                self._fit_onehot_encoder(X[feature])
            elif feature == "pnns_groups_2":
                self._fit_target_encoder(X[feature], y)

        return self

    def _fit_countries_encoder(self, X_countries: pd.Series) -> None:
        country_lists = X_countries.apply(parse_countries).tolist()

        country_counts = Counter()
        for country_list in country_lists:
            # Set to avoid counting duplicates within same product
            country_counts.update(set(country_list))

        self.top_countries_ = [
            country for country, _ in country_counts.most_common(self.top_n_countries)
        ]

        filtered_data = [
            list(set(c for c in country_list if c in self.top_countries_))
            for country_list in country_lists
        ]

        encoder = MultiLabelBinarizer(classes=self.top_countries_)
        encoder.fit(filtered_data)
        self.encoders_["countries"] = encoder

    def _fit_onehot_encoder(self, X_feature: pd.Series) -> None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(X_feature.values.reshape(-1, 1))
        self.encoders_["pnns_groups_1"] = encoder

    def _fit_target_encoder(self, X_feature: pd.Series, y: Optional[pd.Series]) -> None:
        if y is None:
            raise ValueError("TargetEncoder for pnns_groups_2 requires y parameter")

        encoder = TargetEncoder(smooth="auto")
        encoder.fit(X_feature.values.reshape(-1, 1), y)
        self.encoders_["pnns_groups_2"] = encoder

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = X.copy()

        features_to_encode = [f for f in CATEGORICAL_FEATURES
                             if f in X_encoded.columns and f in self.encoders_]

        initial_cols = len(X_encoded.columns)

        with tqdm(total=len(features_to_encode), desc="           Step 2.3: Encoding categorical features",
                  unit="feature", leave=False, mininterval=0.05, miniters=1) as pbar:
            for feature in features_to_encode:
                encoder = self.encoders_[feature]

                if isinstance(encoder, MultiLabelBinarizer):
                    X_encoded = self._transform_multilabel(X_encoded, feature, encoder)
                elif isinstance(encoder, OneHotEncoder):
                    X_encoded = self._transform_onehot(X_encoded, feature, encoder)
                else:
                    X_encoded = self._transform_target(X_encoded, feature, encoder)

                pbar.update(1)

        return X_encoded

    def _transform_multilabel(
        self, X: pd.DataFrame, feature: str, encoder: MultiLabelBinarizer
    ) -> pd.DataFrame:
        country_lists = X[feature].apply(parse_countries).tolist()


        filtered_data = []
        for country_list in country_lists:
            filtered = list(set(c for c in country_list if c in self.top_countries_))
            filtered_data.append(filtered)

        transformed = encoder.transform(filtered_data)
        feature_names = [f"{feature}_{country}" for country in encoder.classes_]

        X_encoded = X.drop(columns=[feature])
        X_encoded = pd.concat([
            X_encoded,
            pd.DataFrame(transformed, columns=feature_names, index=X.index)
        ], axis=1)

        return X_encoded

    def _transform_onehot(
        self, X: pd.DataFrame, feature: str, encoder: OneHotEncoder
    ) -> pd.DataFrame:
        transformed = encoder.transform(X[feature].values.reshape(-1, 1))
        feature_names = encoder.get_feature_names_out([feature])

        X_encoded = X.drop(columns=[feature])
        X_encoded = pd.concat([
            X_encoded,
            pd.DataFrame(transformed, columns=feature_names, index=X.index)
        ], axis=1)

        return X_encoded

    def _transform_target(
        self, X: pd.DataFrame, feature: str, encoder: TargetEncoder
    ) -> pd.DataFrame:
        transformed = encoder.transform(X[feature].values.reshape(-1, 1))
        transformed = np.asarray(transformed).squeeze()
        if transformed.ndim > 1:
            transformed = transformed[:, 0]

        X[feature] = transformed
        return X

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'FeatureEncoder':
        return joblib.load(path)
