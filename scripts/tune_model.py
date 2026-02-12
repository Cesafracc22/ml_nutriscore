import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix

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
        'n_neighbors': [3, 5, 7, 15, 31, 63, 127],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean'],
    },
    'svm': {
        'C': [2 ** i for i in range(-5, 16, 5)],
        'kernel': ['rbf'],
        'gamma': [2 ** i for i in range(-15, 6, 5)],
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
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
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

    # stratified CV for class imbalance
    cv_stratified = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=cv_stratified,
        n_jobs=args.n_jobs,
        verbose=1,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    print("\nBest F1-Macro:", round(grid_search.best_score_, 4))
    print("Best params:", grid_search.best_params_)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a detailed report for every tested hyperparameter combination
    cv_results = grid_search.cv_results_
    all_combinations = []
    n_candidates = len(cv_results['params'])
    labels = np.unique(y_train)
    # Confusion matrix cv
    cv_for_cm = cv_stratified

    for i in range(n_candidates):
        mean_train = float(cv_results['mean_train_score'][i])
        std_train = float(cv_results['std_train_score'][i])
        mean_test = float(cv_results['mean_test_score'][i])
        std_test = float(cv_results['std_test_score'][i])
        gap = mean_train - mean_test

        # Confusion matrix for this hyperparameter combination
        params = cv_results['params'][i]
        estimator.set_params(**params)
        y_pred = cross_val_predict(
            estimator,
            X_train,
            y_train,
            cv=cv_for_cm,
            n_jobs=args.n_jobs
        )
        cm = confusion_matrix(y_train, y_pred, labels=labels).tolist()

        combination_result = {
            'params': cv_results['params'][i],
            'mean_train_score': mean_train,
            'std_train_score': std_train,
            'mean_test_score': mean_test,
            'std_test_score': std_test,
            'overfit_gap': gap,
            'is_overfitting': bool(gap > 0.05),
            'rank_test_score': int(cv_results['rank_test_score'][i]),
            'confusion_matrix': cm,
        }
        all_combinations.append(combination_result)

    # Overall overfitting assessment for the best-ranked combination
    best_index = int(np.argmin(cv_results['rank_test_score']))
    best_gap = float(
        cv_results['mean_train_score'][best_index] - cv_results['mean_test_score'][best_index]
    )
    overfit_threshold = 0.05

    results = {
        'model': args.model,
        'best_score': float(grid_search.best_score_),
        'best_params': grid_search.best_params_,
        'cv_folds': args.cv,
        'timestamp': datetime.now().isoformat(),
        'overfitting_threshold': overfit_threshold,
        'best_overfit_gap': best_gap,
        'best_is_overfitting': bool(best_gap > overfit_threshold),
        'confusion_matrix_labels': labels.tolist(),
        'all_combinations': all_combinations,
    }

    output_path = output_dir / f"{args.model}_tuning.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved to", output_path)


if __name__ == '__main__':
    main()
