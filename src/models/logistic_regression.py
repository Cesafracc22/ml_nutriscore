from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .base_model import BaseModel
from .model_registry import register_model

"""
This class implements the Logistic Regression classifier.
It inherits from the BaseModel class and implements the _build_model and _fit methods
that are called by the train method of the BaseModel class.
"""
@register_model(
    name='logistic_regression',
    description='Logistic Regression baseline model',
    category='linear',
    requires_probability=True
)
class LogisticRegressionModel(BaseModel):

    def __init__(
        self,
        C: float = 1.0,
        solver: str = 'lbfgs',
        max_iter: int = 1000,
        class_weight: str = 'balanced',
        random_state: int = 42,
        **kwargs
    ):
        hyperparameters = {
            'C': C,
            'solver': solver,
            'max_iter': max_iter,
            'class_weight': class_weight,
            'random_state': random_state,
            **kwargs
        }
        super().__init__(model_name='LogisticRegression', **hyperparameters)

    def _build_model(self) -> LogisticRegression:
        return LogisticRegression(**self.hyperparameters)

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
