import joblib
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

"""
This class creates derived features from the dataset.
It inherit from BaseEstimator and TransformerMixin to be used in a scikit-learn pipeline.

Note: we choose 4 different types of derived features:
- Nutrient ratios
- Energy density
- Caloric contributions
- Boolean flags
"""

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        add_ratios: bool = True,
        add_energy_density: bool = True,
        add_caloric_contributions: bool = True,
        add_boolean_flags: bool = True
    ):
        self.add_ratios = add_ratios
        self.add_energy_density = add_energy_density
        self.add_caloric_contributions = add_caloric_contributions
        self.add_boolean_flags = add_boolean_flags

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_engineered = X.copy()

        steps = [
            (self.add_ratios, self._add_ratios, "nutrient ratios"),
            (self.add_energy_density, self._add_energy_density, "energy density"),
            (self.add_caloric_contributions, self._add_caloric_contributions, "caloric contributions"),
            (self.add_boolean_flags, self._add_boolean_flags, "threshold flags")
        ]

        active_steps = [(func, name) for enabled, func, name in steps if enabled]

        with tqdm(total=len(active_steps), desc="           Step 2.4: Engineering features",
                  unit="operation", leave=False, mininterval=0.05, miniters=1) as pbar:
            for func, name in active_steps:
                X_engineered = func(X_engineered)
                pbar.update(1)

        derived_cols = [col for col in X_engineered.columns if col not in X.columns]
        if derived_cols:
            for col in derived_cols:
                if X_engineered[col].isna().any():
                    X_engineered[col] = X_engineered[col].fillna(0.0)

        return X_engineered

    def _add_ratios(self, X: pd.DataFrame) -> pd.DataFrame:
        if 'fat_100g' in X.columns and 'proteins_100g' in X.columns:
            X['fat_to_protein_ratio'] = np.divide(
                X['fat_100g'], X['proteins_100g'],
                out=np.full_like(X['fat_100g'], np.nan, dtype=float),
                where=X['proteins_100g'] != 0
            )

        if 'sugars_100g' in X.columns and 'carbohydrates_100g' in X.columns:
            X['sugar_to_carb_ratio'] = np.divide(
                X['sugars_100g'], X['carbohydrates_100g'],
                out=np.full_like(X['sugars_100g'], np.nan, dtype=float),
                where=X['carbohydrates_100g'] != 0
            )

        if 'saturated-fat_100g' in X.columns and 'fat_100g' in X.columns:
            X['saturated_to_total_fat_ratio'] = np.divide(
                X['saturated-fat_100g'], X['fat_100g'],
                out=np.full_like(X['saturated-fat_100g'], np.nan, dtype=float),
                where=X['fat_100g'] != 0
            )

        return X

    def _add_energy_density(self, X: pd.DataFrame) -> pd.DataFrame:
        if 'energy-kcal_100g' in X.columns:
            X['energy_density'] = X['energy-kcal_100g'] / 100.0
        return X

    def _add_caloric_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        if 'fat_100g' in X.columns:
            X['calories_from_fat'] = X['fat_100g'] * 9.0

        if 'carbohydrates_100g' in X.columns:
            X['calories_from_carbs'] = X['carbohydrates_100g'] * 4.0

        if 'proteins_100g' in X.columns:
            X['calories_from_protein'] = X['proteins_100g'] * 4.0

        return X

    def _add_boolean_flags(self, X: pd.DataFrame) -> pd.DataFrame:
        if 'fat_100g' in X.columns:
            X['high_fat'] = (X['fat_100g'] > 20.0).astype(int)

        if 'sugars_100g' in X.columns:
            X['high_sugar'] = (X['sugars_100g'] > 15.0).astype(int)

        if 'salt_100g' in X.columns:
            X['high_salt'] = (X['salt_100g'] > 1.5).astype(int)

        return X

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        return joblib.load(path)
