import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

sys.path.append(str(Path(__file__).parent.parent))

from src.models import ModelRegistry


"""
This script tunes the hyperparameters of a machine learning model for Nutri-Score prediction.
It loads the preprocessed data, initializes the model, tunes the hyperparameters and saves the model.
It also displays the best hyperparameters and the best F1-Macro score.

Note: The different choices for hyperparameters were made based on the best practices for the different models
and the best results we achieved in the first experiments.
"""

PARAM_GRIDS = {
    'logistic_regression': {
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['lbfgs'],
        'class_weight': ['balanced'],
    },
    'knn': {
        'n_neighbors': [5, 10, 15, 20, 30],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'euclidean', 'manhattan'],
    },
    'svm': {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto'],
    },
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
    },
}


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with GridSearchCV')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='data/splits')
    parser.add_argument('--output-dir', type=str, default='models/tuning')
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=-1)
    args = parser.parse_args()

    if args.model not in PARAM_GRIDS:
        print("Model not supported. Available:", list(PARAM_GRIDS.keys()))
        sys.exit(1)

    data_dir = Path(args.data_dir)
    X_train = pd.read_csv(data_dir / 'X_train.csv').select_dtypes(include=['number'])
    y_train = pd.read_csv(data_dir / 'y_train.csv').values.ravel()

    print("Model:", args.model)
    print("Samples:", X_train.shape[0], "Features:", X_train.shape[1])

    estimator = ModelRegistry.create_model(args.model)._build_model()
    param_grid = PARAM_GRIDS[args.model]

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=args.cv,
        n_jobs=args.n_jobs,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print("\nBest F1-Macro:", round(grid_search.best_score_, 4))
    print("Best params:", grid_search.best_params_)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'model': args.model,
        'best_score': float(grid_search.best_score_),
        'best_params': grid_search.best_params_,
        'cv_folds': args.cv,
        'timestamp': datetime.now().isoformat(),
    }

    output_path = output_dir / f"{args.model}_tuning.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved to", output_path)


if __name__ == '__main__':
    main()
