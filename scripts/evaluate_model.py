import argparse
import sys
from pathlib import Path
import pandas as pd
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.models import ModelRegistry

"""
This script evaluates a trained model on a test set.
It loads the test set and the trained model, and evaluates the model on the test set.
It displays the evaluation metrics and the classification report.
"""


BLUE = '\033[0;34m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
NC = '\033[0m'
BOLD = '\033[1m'

BOX_H = "━"
BOX_V = "┃"
BOX_TL = "┏"
BOX_TR = "┓"
BOX_BL = "┗"
BOX_BR = "┛"


def print_header(text):
    width = 70
    print(f"{CYAN}{BOX_TL}{BOX_H * (width-2)}{BOX_TR}{NC}")
    print(f"{CYAN}{BOX_V}{NC} {BOLD}{text:^{width-4}}{NC} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_BL}{BOX_H * (width-2)}{BOX_BR}{NC}\n")


def print_step(step, desc):
    print(f"{BLUE}━━━ {BOLD}[{step}]{NC} {BLUE}{desc}{NC}")


def print_success(msg):
    print(f"      {GREEN}{msg}{NC}")


def print_error(msg):
    print(f"      {YELLOW}{msg}{NC}")


def print_info(msg):
    print(f"      {msg}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the trained model file (.joblib)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        help='Type of model (e.g., logistic_regression, knn, svm, random_forest, xgboost, naive_bayes)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/splits',
        help='Directory containing test data (default: data/splits)'
    )
    parser.add_argument(
        '--show-report',
        action='store_true',
        help='Show detailed classification report'
    )
    parser.add_argument(
        '--show-confusion-matrix',
        action='store_true',
        help='Show confusion matrix'
    )

    args = parser.parse_args()

    print_header("Evaluation")

    model_path = Path(args.model_path)
    if not model_path.exists():
        print_error(f"Model file not found: {model_path}")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print_error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    try:
        X_test = pd.read_csv(data_dir / 'X_test.csv')
        y_test = pd.read_csv(data_dir / 'y_test.csv').values.ravel()

        non_numeric_cols = X_test.select_dtypes(include=['object']).columns.tolist()
        if non_numeric_cols:
            X_test = X_test.select_dtypes(include=['number'])

        print_info(f"Test set: {X_test.shape[0]:,} samples")
    except FileNotFoundError as e:
        print_error(f"Test data not found")
        sys.exit(1)

    try:
        model = ModelRegistry.create_model(args.model_type)
        model.load(model_path)
    except Exception as e:
        print_error(f"Error loading model: {e}")
        sys.exit(1)

    # Load metadata and show train/val performance
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        history = metadata.get('training_history', {})
        train_metrics = history.get('train_metrics', {})
        val_metrics = history.get('val_metrics', {})

        if train_metrics and val_metrics:
            train_acc = train_metrics.get('accuracy', 0)
            val_acc = val_metrics.get('accuracy', 0)
            gap = train_acc - val_acc

            print_info(f"Train accuracy: {train_acc:.4f} | Val accuracy: {val_acc:.4f}")

            if gap > 0.05:
                print_info(f"Train-val gap: {YELLOW}{gap:.4f}{NC} (possible overfitting)")
            else:
                print_info(f"Train-val gap: {GREEN}{gap:.4f}{NC} (good)")
        print()

    test_metrics = model.evaluate(X_test, y_test, dataset_name='test', verbose=True)

    if args.show_report:
        print()
        print(f"{BOLD}Classification Report:{NC}")
        print(model.get_classification_report(X_test, y_test))

    if args.show_confusion_matrix:
        print()
        print(f"{BOLD}Confusion Matrix:{NC}")
        cm = model.get_confusion_matrix(X_test, y_test)
        print(cm)

    print()
    print_success(f"Test Accuracy: {BOLD}{test_metrics['accuracy']:.4f}{NC}")
    print_success(f"Test F1 (weighted): {BOLD}{test_metrics['f1_weighted']:.4f}{NC}")


if __name__ == '__main__':
    main()
