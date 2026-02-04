import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.pipeline import Pipeline

from features.outlier_removal import MissingValueTransformer, OutlierRemovalTransformer
from features.encoding import FeatureEncoder
from features.feature_engineering import FeatureEngineer
from features.scaling import FeatureScaler
from features.dimensionality_reduction import FeatureReducer


"""
This class creates the preprocessing pipeline.
It contains the different steps of the preprocessing pipeline and the different parameters.

Note: every step is optional and can be disabled by setting the corresponding flag to False.
We decided this in order to better solve and understand some issues we encountered.
"""

def create_preprocessing_pipeline(
    missing_threshold: float = 0.95,
    top_n_countries: int = 15,
    scaling_method: str = 'auto',
    scaling_skew_threshold: float = 1.0,
    pca_variance_threshold: float = 0.95,
    target_col: str = 'nutriscore_grade',
    include_feature_engineering: bool = True,
    feature_engineering_kwargs: Optional[dict] = None,
    remove_statistical_outliers: bool = False,
    include_pca: bool = True,
    include_encoding: bool = True,
    include_scaling: bool = True
) -> Pipeline:
    steps = [
        ('missing_values', MissingValueTransformer(
            threshold_drop_feature=missing_threshold,
            target_col=target_col
        )),
        ('outlier_removal', OutlierRemovalTransformer(
            target_col=target_col,
            remove_statistical_outliers=remove_statistical_outliers
        )),
    ]

    if include_encoding:
        steps.append(('encoding', FeatureEncoder(top_n_countries=top_n_countries, target_col=target_col)))

    if include_feature_engineering:
        fe_kwargs = feature_engineering_kwargs or {}
        steps.append(('feature_engineering', FeatureEngineer(**fe_kwargs)))

    if include_scaling:
        steps.append(('scaling', FeatureScaler(method=scaling_method, skew_threshold=scaling_skew_threshold)))

    if include_pca:
        steps.append(('pca', FeatureReducer(variance_threshold=pca_variance_threshold)))

    return Pipeline(steps)


class PreprocessingPipeline:
    def __init__(
        self,
        missing_threshold: float = 0.95,
        top_n_countries: int = 15,
        scaling_method: str = 'auto',
        scaling_skew_threshold: float = 1.0,
        pca_variance_threshold: float = 0.95,
        target_col: str = 'nutriscore_grade',
        split_group_col: str = 'split_group',
        preserve_cols: Optional[List[str]] = None,
        include_feature_engineering: bool = True,
        feature_engineering_kwargs: Optional[dict] = None,
        remove_statistical_outliers: bool = False,
        include_pca: bool = True,
        include_encoding: bool = True,
        include_scaling: bool = True
    ):
        self.target_col = target_col
        self.split_group_col = split_group_col
        self.preserve_cols = preserve_cols or []

        self.pipeline = create_preprocessing_pipeline(
            missing_threshold=missing_threshold,
            top_n_countries=top_n_countries,
            scaling_method=scaling_method,
            scaling_skew_threshold=scaling_skew_threshold,
            pca_variance_threshold=pca_variance_threshold,
            target_col=target_col,
            include_feature_engineering=include_feature_engineering,
            feature_engineering_kwargs=feature_engineering_kwargs,
            remove_statistical_outliers=remove_statistical_outliers,
            include_pca=include_pca,
            include_encoding=include_encoding,
            include_scaling=include_scaling
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PreprocessingPipeline':
        if self.split_group_col in X.columns:
            train_mask = X[self.split_group_col] == 'train'
            X_train = X[train_mask]
            print(f"Fitting on train split only: {len(X_train):,} samples")
            self.pipeline.fit(X_train, y)
        else:
            self.pipeline.fit(X, y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        if self.split_group_col in X.columns:
            train_mask = X[self.split_group_col] == 'train'
            X_train = X[train_mask]
            print(f"Fitting on train split only: {len(X_train):,} samples")
            self.pipeline.fit(X_train, y)
            return self.pipeline.transform(X)
        else:
            return self.pipeline.fit_transform(X, y)
