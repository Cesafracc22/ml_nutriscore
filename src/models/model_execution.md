# Model execution

## QuickStart

Run the interactive script:

```bash
./run-training-and-evaluation.sh
```

This trains and evaluates any model. Preprocessing must be ran first with `./run-preprocessing.sh`.

## Preprocessing

Modify change preprocessing with these flags:

```bash
# Skip PCA
python scripts/run_preprocessing.py --no-pca

# Change scaling method (standard, minmax, or auto)
python scripts/run_preprocessing.py --scale-method minmax

# Remove outliers
python scripts/run_preprocessing.py --remove-outliers
```

## Training

Train a model manually:

```bash
# Basic
python scripts/train_model.py --model logistic_regression

# With custom version
python scripts/train_model.py --model logistic_regression --version v2

# List available models
python scripts/train_model.py --list-models
```

Training will save two files:
- `{model_name}_v1.joblib` - the trained model
- `{model_name}_v1_metadata.json` - hyperparameters and metrics

## Evaluation

Evaluate a trained model on test set:

```bash
python scripts/evaluate_model.py \
    --model-path models/trained/logistic_regression/logistic_regression_v1.joblib \
    --model-type logistic_regression \
    --show-report \
    --show-confusion-matrix
```

## Metrics

- **Accuracy** - % correct predictions
- **Precision** - how many positive predictions were right
- **Recall** - how many actual positives were found
- **F1-Score** - balanced metric for imbalanced classes