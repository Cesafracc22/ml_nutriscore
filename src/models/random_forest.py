from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .base_model import BaseModel
from .model_registry import register_model


@register_model(
    name='random_forest',
    description='Random Forest ensemble classifier',
    category='ensemble',
    requires_probability=True
)
class RandomForestModel(BaseModel):

    def __init__(
        self,
        n_estimators: int = 300,           # Number of trees in the forest
        max_depth: Optional[int] = 15,     # Maximum depth of the trees
        min_samples_split: int = 2,        # Minimum number of samples required to split an internal node
        min_samples_leaf: int = 2,         # Minimum number of samples required to be at a leaf node
        max_features: str = 'sqrt',        # Number of features to consider for splitting
        class_weight: str = 'balanced',    # Class weight
        n_jobs: int = -1,                  # Number of jobs to run in parallel
        random_state: int = 42,            # Random state
        **kwargs
    ):
        hyperparameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'class_weight': class_weight,
            'n_jobs': n_jobs,
            'random_state': random_state,
            **kwargs
        }
        super().__init__(model_name='RandomForest', **hyperparameters)

    def _build_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(**self.hyperparameters)

    def _fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        self.model.fit(X_train, y_train)

        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        metrics = {
            'accuracy': train_accuracy,
        }

        return metrics
