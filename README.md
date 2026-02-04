# Nutri-Score ML Prediction - OpenFoodFacts API

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

## Overview

This project implements a multi-class classification system for predicting Nutri-Score grades of food products based on their nutritional composistion and score. The Nutri-Score is a nutrition label (goes from a through e). Grade 'a' represents the best nutritional profile, while grade 'e' indicates products with worst characteristics.

The classification task uses nutritional data from the Open Food Facts database to train and evaluate five different machine learning algorithms. The project includes a complete data pipeline from raw data acquisition through preprocessing, model training, and evaluation. All models are trained with class balancing techniques and evaluated using stratified cross-validation to ensure reliable performance estimates across all Nutri-Score categories.

## Quick Start

The project provides bash scripts for each step of the pipeline. (For windows, need to convert to .bat files)

### 1. Data Download and Preprocessing

```bash
./run-preprocessing.sh
```

This script performs the following operations:
- Downloads the Open Food Facts dataset (approximately 3GB)
- Filters products with complete nutritional information
- Samples 250,000 products for training
- Executes the 7-stage preprocessing pipeline
- Creates stratified train/validation/test splits (70%/15%/15%)
- Saves processed data to `data/splits/`

### 2. Model Training and Evaluation

```bash
./run-training-and-evaluation.sh
```

The script shows a menu to select from the five available models:
1. Logistic Regression
2. K-Nearest Neighbors
3. Support Vector Machine
4. Random Forest
5. XGBoost

After model selection, the script:
- Trains the model using 5-fold stratified cross-validation
- Evaluates performance on the validation set
- Returns classification metrics and confusion matrix
- Saves the trained model to `models/trained/<model_name>/`

### 3. Manual Training (Alternative)

Can also run the training python file directly:

```bash
python scripts/train_model.py --model svm
```

Available model ids: `logistic_regression`, `knn`, `svm`, `random_forest`, `xgboost`

### 4. Model Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate_model.py \
    --model-path models/trained/svm/svm_v1.joblib \
    --model-type svm \
    --show-report \
    --show-confusion-matrix
```

## Installation

### Prerequisites
- Python 3.8+

### Setup

Clone the repository:
```bash
git clone https://github.com/djacoo/ml-nutriscore-prediction.git
cd ml-nutriscore-prediction
```

Create and activate a virtual environment:
```bash
python -m venv ml-predictor
source ml-predictor/bin/activate  # On Windows: ml-predictor\Scripts\activate
```

Install dependencies using one of the following methods:

**1: using pip with pyproject.toml**
```bash
pip install -e .
```

**2: using requirements.txt**
```bash
pip install -r requirements.txt
```

## Project Structure

```
ml-nutriscore-prediction/
├── data/
│   ├── raw/                    # Original downloaded dataset
│   ├── processed/              # Filtered dataset (250k products)
│   └── splits/                 # Train/validation/test sets
│       ├── X_train.csv
│       ├── y_train.csv
│       ├── X_val.csv
│       ├── y_val.csv
│       ├── X_test.csv
│       └── y_test.csv
├── src/
│   ├── data/                   # Data loading utils
│   ├── features/               # Preprocessing pipeline
│   └── models/                 # Models
│       ├── base_model.py       # Abstract base class for all models
│       ├── model_registry.py   # Model registration/registry
│       ├── logistic_regression_model.py
│       ├── knn_model.py
│       ├── svm_model.py
│       ├── random_forest_model.py
│       └── xgboost_model.py
├── scripts/
│   ├── download_data.py        # Dataset downloader
│   ├── train_model.py          # Model trainer script
│   ├── evaluate_model.py       # Model evaluator script
│   └── tune_model.py           # Hyperparameter tuner script
├── models/trained/             # Saved model files/metadata
│   ├── logistic_regression/
│   ├── knn/
│   ├── svm/
│   ├── random_forest/
│   └── xgboost/
├── docs/
│   ├── notebooks/              # Jupyter notebooks
│   │   ├── eda.ipynb          # Exploratory data analysis (EDA)
│   │   ├── model_evaluation.ipynb
│   │   └── preprocessing.ipynb
│   └── preprocessing_pipeline.md
├── run-preprocessing.sh        # scripts for auto
└── run-training-and-evaluation.sh
```

## Dataset

The project uses data from Open Food Facts, an open database of food products with ingredients, nutritional information, and other metadata.

**Dataset Characteristics:**
- Source: Open Food Facts API
- Total products in database: >3 million
- Filtered dataset: 250,000 products with complete nutritional data
- Features: 15 nutritional attributes
  - Energy (kcal and kJ)
  - Macronutrients (fat, saturated fat, carbohydrates, sugars, proteins)
  - Micronutrients (salt, fiber, sodium)
  - Additional metrics (fruits/vegetables percentage, etc.)
- Target variable: Nutri-Score grade (a, b, c, d, e)
- Class distribution: Imbalanced (addressed via class weighting)

**Data Splitting:**
- Training set: 70% (175,000 products)
- Validation set: 15% (37,500 products)
- Test set: 15% (37,500 products)
- Stratification: Applied to maintain class proportions across splits

## Preprocessing Pipeline

The preprocessing pipeline goes with 7 sequential stages used to prepare raw nutritional data for the machine learning models. Each transformation is fitted on the training set only to prevent data leakage.

**Pipeline Stages:**
1. **Missing Value Imputation** - Median imputation for nutritional features
2. **Duplicate Removal** - Elimination of identical product entries
3. **Outlier Detection** - Domain-specific outlier handling for nutritional ranges
4. **Feature Engineering** - Creation of nutritional ratios and thresholds
5. **Adaptive Scaling** - StandardScaler
6. **Dimensionality Reduction** - PCA with 95% variance applied
7. **Validation** - Data integrity checks and range validation

The pipeline is implemented as a system where each stage can be configured independently. The documentation of each stage is to be found in [docs/preprocessing_pipeline.md](docs/preprocessing_pipeline.md).

## Models

All models inherit from a common `BaseModel` abstract class that defines a consistent interface for training, prediction, and evaluation. This system allows for easy addition of new models.

### 1. Logistic Regression

**Type:** Linear classifier with L2 regularization
**Use case:** Baseline model

**Performance:**
- Validation accuracy: 70.2%
- F1-macro: 68.1%
- Training time: short
- Cross-validation (5-fold): 70.6% ± 0.1%

### 2. K-Nearest Neighbors

**Type:** Instance-based learning with distance weighting
**Use case:** Non-parametric baseline

**Performance:**
- Validation accuracy: 77.1%
- F1-macro: 75.1%
- Training time: short
- Cross-validation (5-fold): 76.5% ± 0.1%

### 3. Support Vector Machine

**Type:** Kernel-based classifier with RBF kernel
**Use case:** Best performing model

**Performance:**
- Validation accuracy: 83.5%
- F1-macro: 81.6%
- Training time: long
- Cross-validation (5-fold): 83.2% ± 0.1%

**Note:** SVM achieves the best performance but has the longest training time.

### 4. Random Forest

**Type:** Ensemble of decision trees
**Use case:** Good balance of performance and speed

**Performance:**
- Validation accuracy: 79.1%
- F1-macro: 77.2%
- Training time: medium
- Cross-validation (5-fold): 79.1% ± 0.2%

**Characteristics:** Shows some overfitting (train accuracy: 89.1%)

### 5. XGBoost

**Type:** Gradient boosting with regularization
**Use case:** Strong performance with efficient training

**Performance:**
- Validation accuracy: 81.2%
- F1-macro: 78.9%
- Training time: short
- Cross-validation (5-fold): 81.1% ± 0.3%

## Model Comparison

| Model | Val Accuracy | F1-Macro | Training Time | Notes |
|-------|-------------|----------|---------------|-------|
| SVM | 83.5% | 81.6% | long | Best performance |
| XGBoost | 81.2% | 78.9% | short | Strong performance, fast training |
| Random Forest | 79.1% | 77.2% | medium | Good balance of speed/accuracy |
| KNN | 77.1% | 75.1% | short | Non-parametric baseline |
| Logistic Regression | 70.2% | 68.1% | short | Linear baseline |

---

**Authors:** Jacopo Parretti VR536104, Cesare Fraccaroli VR533061
