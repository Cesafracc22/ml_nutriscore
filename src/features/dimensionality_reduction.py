"""PCA-based dimensionality reduction."""
import joblib
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


METADATA_COLS = ['nutriscore_grade', 'split_group', 'product_name', 'brands', 'code']

"""
This class reduces the dimensionality of the dataset using PCA.
It inherit from BaseEstimator and TransformerMixin to be used in a scikit-learn pipeline.
"""

class FeatureReducer(BaseEstimator, TransformerMixin):
    """Reduces dimensions using PCA with variance threshold."""

    def __init__(
        self, variance_threshold: float = 0.95, n_components: Optional[int] = None
    ):
        self.variance_threshold = variance_threshold
        self.n_components = n_components
        self.pca_: Optional[PCA] = None
        self.feature_columns_: Optional[list] = None
        self.n_components_selected_: Optional[int] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    """
    This method fits the PCA model to the dataset.
    It first checks if there are any numerical columns in the dataset.
    Then, it selects the feature columns for PCA.
    It fits the PCA model to the dataset and returns the fitted model.
    """
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureReducer':
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) == 0:
            raise ValueError("No numerical columns found for PCA")

        self.feature_columns_ = [
            col for col in numerical_cols if col not in METADATA_COLS
        ]

        if len(self.feature_columns_) == 0:
            raise ValueError("No numerical feature columns found for PCA (all excluded)")

        X_array = X[self.feature_columns_].values

        if self.n_components is not None:
            n_comp = min(self.n_components, X_array.shape[1])
        else:
            pca_full = PCA()
            pca_full.fit(X_array)
            cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
            n_comp = np.argmax(cumsum_variance >= self.variance_threshold) + 1
            n_comp = max(1, n_comp)

        self.n_components_selected_ = n_comp
        self.pca_ = PCA(n_components=n_comp)
        self.pca_.fit(X_array)
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_

        cumulative_variance = np.cumsum(self.explained_variance_ratio_)[-1]
        print(f"PCA: {len(self.feature_columns_)} features → {n_comp} components ({cumulative_variance*100:.1f}% variance)")

        return self

    """
    This method transforms the dataset using the fitted PCA model.
    First checks if the required feature columns are present in the dataset.
    Then, it transforms the dataset using the fitted PCA model.
    """
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        
        if self.pca_ is None:
            raise ValueError("PCA has not been fitted. Call fit() first.")

        missing_cols = [col for col in self.feature_columns_ if col not in X.columns]
        if missing_cols:
            if len(self.feature_columns_) > 10:
                expected_cols = self.feature_columns_[:10]
                print("Expected columns (first 10):", expected_cols, "...")
            else:
                print("Expected columns:", self.feature_columns_)
            raise ValueError("Missing feature columns required for PCA transformation:", missing_cols)


        X_array = X[self.feature_columns_].values
        X_transformed = self.pca_.transform(X_array)

        pc_columns = [f'PC{i+1}' for i in range(self.n_components_selected_)]
        X_pca = pd.DataFrame(X_transformed, columns=pc_columns, index=X.index)

        for col in METADATA_COLS:
            if col in X.columns:
                X_pca[col] = X[col].values

        return X_pca

    def get_explained_variance_ratio(self) -> np.ndarray:
        
        if self.pca_ is None:
            raise ValueError("PCA has not been fitted. Call fit() first.")
        return self.explained_variance_ratio_

    def get_cumulative_variance(self) -> np.ndarray:
        
        if self.pca_ is None:
            raise ValueError("PCA has not been fitted. Call fit() first.")
        return np.cumsum(self.explained_variance_ratio_)

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'FeatureReducer':
        
        return joblib.load(path)
