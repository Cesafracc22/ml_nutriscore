from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from .base_model import BaseModel
from .model_registry import register_model


@register_model(
    name='knn',
    description='K-Nearest Neighbors classifier',
    category='instance-based',
    requires_probability=True
)
class KNNModel(BaseModel):

    """
    The hyperparemeters were chosen based on the execution of the tune_model.py script.
    that performed a grid search over the hyperparameters.
    """

    def __init__(
        self,
        n_neighbors: int = 15,           # number of neighbors to consider
        weights: str = 'distance',       # weight function used in prediction
        algorithm: str = 'auto',         # algorithm used to compute the nearest neighbors
        metric: str = 'minkowski',       # metric used to measure the distance between instances
        p: int = 2,                      # power parameter for the Minkowski metric
        n_jobs: int = -1,                # number of jobs to run in parallel (-1 means use all available cores)
        **kwargs
    ):
        hyperparameters = {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'algorithm': algorithm,
            'metric': metric,
            'p': p,
            'n_jobs': n_jobs,
            **kwargs
        }
        super().__init__(model_name='KNN', **hyperparameters)

    def _build_model(self) -> KNeighborsClassifier:
        return KNeighborsClassifier(**self.hyperparameters)

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
