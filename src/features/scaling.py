import joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import skew
from tqdm import tqdm


SCALERS = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
    'robust': RobustScaler
}

METADATA_COLS = ['nutriscore_grade', 'split_group', 'product_name', 'brands', 'code']


LOG1P_BEFORE_STANDARD = [
    'energy_100g',
    'fat_to_protein_ratio',
    'sugar_to_carb_ratio',
    'saturated_to_total_fat_ratio',
]

"""
This class scales the numerical features of the dataset.
It inherit from BaseEstimator and TransformerMixin to be used in a scikit-learn pipeline.
We implemented different scalers: StandardScaler, MinMaxScaler and RobustScaler.

Note: We set as default the StandardScaler, since it is the one that gives the best results.
Some highly skewed features are transformed with the log1p function before scaling to better 
usage with distance based models like KNN.
"""
class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'standard', skew_threshold: float = 1.0):
        self.method = method
        self.skew_threshold = skew_threshold
        self.scalers_: Dict[str, BaseEstimator] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureScaler':
        self.scalers_ = {}

        numerical_cols = X.select_dtypes(include=[np.number]).columns
        features_to_scale = []
        for col in numerical_cols:
            if col not in METADATA_COLS:
                features_to_scale.append(col)

        if len(features_to_scale) == 0:
            raise ValueError(
                "No numerical features found to scale. "
                "Ensure input contains numerical columns (excluding metadata)."
            )

        for feature in features_to_scale:
            if self.method == 'auto':
                scaler = self._select_scaler_by_skewness(X[feature])
            else:
                scaler = SCALERS[self.method]()

            values = X[feature].values.astype(float)
            if self.method == 'standard' and feature in LOG1P_BEFORE_STANDARD:
                values = np.log1p(np.clip(values, a_min=0.0, a_max=None))

            scaler.fit(values.reshape(-1, 1))
            self.scalers_[feature] = scaler

        return self

    def _select_scaler_by_skewness(self, X_feature: pd.Series) -> BaseEstimator:
        feature_skewness = abs(skew(X_feature.dropna()))
        return MinMaxScaler() if feature_skewness > self.skew_threshold else StandardScaler()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = X.copy()

        scalers_list = [(feature, scaler) for feature, scaler in self.scalers_.items()
                        if feature in X_scaled.columns]

        scaler_types = {}
        for _, scaler in scalers_list:
            scaler_name = type(scaler).__name__
            scaler_types[scaler_name] = scaler_types.get(scaler_name, 0) + 1

        with tqdm(total=len(scalers_list), desc="           Step 2.5: Scaling numerical features",
                  unit="feature", leave=False, mininterval=0.05, miniters=1) as pbar:
            for feature, scaler in scalers_list:
                values = X_scaled[feature].values.astype(float)
                if self.method == 'standard' and feature in LOG1P_BEFORE_STANDARD:
                    values = np.log1p(np.clip(values, a_min=0.0, a_max=None))

                X_scaled[feature] = scaler.transform(
                    values.reshape(-1, 1)
                ).flatten()
                pbar.update(1)

        return X_scaled

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'FeatureScaler':
        return joblib.load(path)
