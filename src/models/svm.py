from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from .base_model import BaseModel
from .model_registry import register_model


@register_model(
    name='svm',
    description='Support Vector Machine classifier',
    category='kernel-based',
    requires_probability=True
)
class SVMModel(BaseModel):

    """
    The hyperparemeters were chosen based on the execution of the tune_model.py script.
    that performed a grid search over the hyperparameters.
    """

    def __init__(
        self,
        C: float = 10.0,                 # regularization parameter
        kernel: str = 'rbf',             # kernel type
        gamma: str = 'scale',            # kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        degree: int = 3,                 # degree for 'poly'
        class_weight: str = 'balanced',  # class weight
        probability: bool = True,        # whether to enable probability estimates
        cache_size: int = 2000,          # cache size for the solver
        max_iter: int = -1,              # maximum number of iterations (-1 means no limit)
        random_state: int = 42,          # random state
        **kwargs
    ):
        hyperparameters = {
            'C': C,
            'kernel': kernel,
            'gamma': gamma,
            'degree': degree,
            'class_weight': class_weight,
            'probability': probability,
            'cache_size': cache_size,
            'max_iter': max_iter,
            'random_state': random_state,
            **kwargs
        }
        super().__init__(model_name='SVM', **hyperparameters)

    def _build_model(self) -> SVC:
        return SVC(**self.hyperparameters)

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
