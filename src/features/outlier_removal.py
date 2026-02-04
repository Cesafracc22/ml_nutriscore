import pandas as pd
from typing import Optional, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from data.preprocessing import MissingValueHandler


"""
This class handles the missing values in the dataset.
It inherit from BaseEstimator and TransformerMixin to be used in a scikit-learn pipeline.

Note: we choose the following methods to handle the missing values:
- Dropping features with more than 95% missing values
- Imputing the missing values with the median value
- Labeling the missing values as 'unknown'
- Removing the samples without Nutri-Score labels
"""
class MissingValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, threshold_drop_feature: float = 0.95, target_col: str = 'nutriscore_grade'
    ):
        self.threshold_drop_feature = threshold_drop_feature
        self.target_col = target_col
        self.handler = MissingValueHandler(threshold_drop_feature=threshold_drop_feature)
        self.dropped_features_: Optional[list] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MissingValueTransformer':
        self.handler.analyze_missing_values(X)
        self.dropped_features_ = self.handler.dropped_features.copy()
        return self

    def transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        X_work = X.copy()
        if self.target_col not in X_work.columns and y is not None:
            X_work[self.target_col] = y.values

        result = self.handler.handle_missing_values(X_work, target_col=self.target_col)
        return result

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X, y)


class OutlierRemovalTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        target_col: str = 'nutriscore_grade',
        remove_statistical_outliers: bool = False
    ):
        self.target_col = target_col
        self.remove_statistical_outliers = remove_statistical_outliers

        self.valid_ranges_ = {
            'fat_100g': (0, 100),
            'saturated-fat_100g': (0, 100),
            'carbohydrates_100g': (0, 100),
            'sugars_100g': (0, 100),
            'fiber_100g': (0, 100),
            'proteins_100g': (0, 100),
            'salt_100g': (0, 100),
            'energy_100g': (0, 5000),
            'energy-kcal_100g': (0, 1000),
        }

        self.rows_removed_ = 0
        self.outlier_report_: Optional[Dict] = None

    """
    This method detects the outliers in the dataset using the interquartile range (IQR) method.

    Note: this is disabled by default, since it is too aggressive for nutritional data.
    Nutritional data from openfoodfacts contains very skewed data with some extreme values that are
    correct.
    """
    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        outlier_info = {}

        for col in self.valid_ranges_.keys():
            if col not in df.columns:
                continue

            col_info = {}

            negative_mask = df[col] < 0
            col_info['negative_count'] = int(negative_mask.sum())

            min_val, max_val = self.valid_ranges_[col]
            below_min = (df[col] < min_val).sum()
            above_max = (df[col] > max_val).sum()
            col_info['below_min'] = int(below_min)
            col_info['above_max'] = int(above_max)
            col_info['valid_range'] = self.valid_ranges_[col]

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            statistical_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            col_info['statistical_outliers'] = int(statistical_outliers)
            col_info['IQR_bounds'] = (float(lower_bound), float(upper_bound))

            col_info['min'] = float(df[col].min())
            col_info['max'] = float(df[col].max())
            col_info['median'] = float(df[col].median())

            outlier_info[col] = col_info

        return outlier_info

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OutlierRemovalTransformer':
        self.outlier_report_ = {}
        self.outlier_report_['before_cleaning'] = self._detect_outliers(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_clean = X.copy()
        initial_rows = len(df_clean)

        if self.outlier_report_ is None:
            self.outlier_report_ = {}
        if 'before_cleaning' not in self.outlier_report_:
            self.outlier_report_['before_cleaning'] = self._detect_outliers(df_clean)

        total_issues = 0
        for col, info in self.outlier_report_['before_cleaning'].items():
            issues = info['negative_count'] + info['above_max']
            total_issues += issues

        rows_removed = 0
        removal_reasons = {}

        cols_to_check = [col for col in self.valid_ranges_.keys() if col in df_clean.columns]

        with tqdm(total=len(cols_to_check), desc="           Step 2.1: Detecting and removing outliers",
                  unit="feature", leave=False, mininterval=0.05, miniters=1) as pbar:
            for col in cols_to_check:
                min_val, max_val = self.valid_ranges_[col]

                invalid_mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
                invalid_count = invalid_mask.sum()

                if invalid_count > 0:
                    removal_reasons[col] = {
                        'count': int(invalid_count),
                        'reason': f'Outside valid range ({min_val}, {max_val})'
                    }

                    df_clean = df_clean[~invalid_mask]
                    rows_removed += invalid_count

                pbar.update(1)

        if self.remove_statistical_outliers:
            nutritional_cols = list(self.valid_ranges_.keys())
            existing_cols = [col for col in nutritional_cols if col in df_clean.columns]

            combined_extreme_mask = pd.Series(False, index=df_clean.index)

            with tqdm(total=len(existing_cols), desc="           Step 2.2: Statistical outlier detection",
                      unit="feature", leave=False) as pbar:
                for col in existing_cols:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    statistical_lower = Q1 - 3 * IQR
                    statistical_upper = Q3 + 3 * IQR

                    min_val, max_val = self.valid_ranges_[col]

                    lower_bound = max(0, statistical_lower)
                    upper_bound = min(statistical_upper, max_val)

                    extreme_mask = (df_clean[col] > upper_bound)
                    extreme_count = extreme_mask.sum()

                    if extreme_count > 0:
                        combined_extreme_mask = combined_extreme_mask | extreme_mask

                        if col not in removal_reasons:
                            removal_reasons[col] = {
                                'count': int(extreme_count),
                                'reason': f'Extreme statistical outlier (3*IQR, upper bound: {upper_bound:.2f})'
                            }

                    pbar.update(1)

            if combined_extreme_mask.any():
                statistical_outliers_count = combined_extreme_mask.sum()
                df_clean = df_clean[~combined_extreme_mask]
                rows_removed += statistical_outliers_count

        final_rows = len(df_clean)
        total_removed = initial_rows - final_rows
        removal_pct = (total_removed / initial_rows) * 100

        self.outlier_report_['after_cleaning'] = self._detect_outliers(df_clean)
        self.outlier_report_['removal_summary'] = removal_reasons
        self.outlier_report_['rows_removed'] = int(total_removed)
        self.outlier_report_['removal_percentage'] = float(removal_pct)

        self.rows_removed_ = total_removed
        return df_clean

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
