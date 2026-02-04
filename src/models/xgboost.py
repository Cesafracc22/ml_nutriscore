from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from .base_model import BaseModel
from .model_registry import register_model



class LabelEncodedXGBClassifier(XGBClassifier):
    """
    XGBClassifier wrapper that handles string labels internally.
    
    Note: this is a workaround to handle the string labels of the target variable,
    because XGBClassifier does not support string labels.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._le = LabelEncoder()

    def fit(self, X, y, **kwargs):
        y_enc = self._le.fit_transform(np.asarray(y).ravel())
        return super().fit(X, y_enc, **kwargs)

    def predict(self, X, **kwargs):
        y_enc = super().predict(X, **kwargs)
        return self._le.inverse_transform(y_enc.astype(int))

    def score(self, X, y, **kwargs):
        return accuracy_score(y, self.predict(X))


@register_model(
    name='xgboost',
    description='XGBoost gradient boosting classifier',
    category='ensemble',
    requires_probability=True
)
class XGBoostModel(BaseModel):

    """
    The hyperparemeters were chosen based on the execution of the tune_model.py script.
    that performed a grid search over the hyperparameters.
    """

    def __init__(
        self,
        n_estimators: int = 300,           # Number of boosting rounds
        max_depth: int = 6,                # Maximum depth of the trees
        learning_rate: float = 0.1,        # Learning rate
        subsample: float = 1.0,            # Subsample ratio of the training instances
        colsample_bytree: float = 0.8,     # Subsample ratio of the features
        reg_alpha: float = 0.1,            # L1 regularization term on weights
        reg_lambda: float = 1.0,           # L2 regularization term on weights
        min_child_weight: int = 1,
        objective: str = 'multi:softprob', # Objective function
        eval_metric: str = 'mlogloss',     # Evaluation metric
        random_state: int = 42,
        n_jobs: int = -1,                  # Number of jobs to run in parallel (-1 means use all available cores)
        **kwargs
    ):
        hyperparameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'min_child_weight': min_child_weight,
            'objective': objective,
            'eval_metric': eval_metric,
            'random_state': random_state,
            'n_jobs': n_jobs,
            **kwargs
        }
        super().__init__(model_name='XGBoost', **hyperparameters)

    def _build_model(self) -> LabelEncodedXGBClassifier:
        return LabelEncodedXGBClassifier(**self.hyperparameters)

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
