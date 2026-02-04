# Preprocessing Pipeline

This document describes the complete data preprocessing workflow for transforming raw Open-Food-Facts data into machine learning-ready features for Nutri-Score prediction.

## How to Run

### Automated Execution

Execute the following command from the repository root:

```bash
./run-preprocessing.sh
```

This script orchestrates the entire preprocessing workflow:

1. Creates or activates the `ml-predictor` virtual environment
2. Installs all dependencies from [pyproject.toml](../pyproject.toml)
3. Downloads and filters 250,000 product samples from Open Food Facts
4. Applies all data preprocessing transformations
5. Generates stratified train/validation/test splits

The script preserves existing virtual environments, allowing repeated execution without reinstalling dependencies. All processed data is written to [data/splits/](../data/splits/).

### Manual Execution

For step-by-step execution:

```bash
# create and activate venv
python3 -m venv ml-predictor
source ml-predictor/bin/activate

# install dependencies
pip install uv  # fast package installer
uv pip install -e .

# download raw data
python scripts/download_data.py

# run preprocessing
python scripts/run_preprocessing.py
```

## Pipeline Architecture

The preprocessing pipeline consists of seven sequential stages, each implemented as a scikit-learn transformer. All transformers are fitted exclusively on training data to prevent information leakage.

### Stage 1: Data pull from API

**Implementation**: [scripts/download_data.py](../scripts/download_data.py)

The pipeline downloads the complete Open Food Facts dataset (approximately 7GB compressed) and filters for products containing Nutri-Score labels (grades a-e). A random sample of 250,000 products is extracted and saved to `data/processed/openfoodfacts_filtered.csv`.

### Stage 2: Missing Value Handling

**Implementation**: `MissingValueTransformer` ([src/features/outlier_removal.py:9](../src/features/outlier_removal.py#L9))

Missing value treatment follows a two-phase strategy:

**Feature-level filtering**: Columns with more than 95% missing values are dropped, as they provide insufficient information for prediction.

**Imputation strategy** (applied to retained features):
- **Numerical features**: Imputed using the median value from the training set, which provides robustness against outliers
- **Additives count**: Missing values default to 0, assuming unreported additives are absent
- **Categorical features**: Missing entries are labeled as 'unknown'
- **Target variable**: Samples without Nutri-Score labels are removed entirely

### Stage 3: Outlier Detection and Removal

**Implementation**: `OutlierRemovalTransformer` ([src/features/outlier_removal.py:43](../src/features/outlier_removal.py#L43))

Outlier detection applies domain-based validation rules derived from nutritional constraints:

- **Macronutrients** (fat, carbohydrates, sugars, fiber, proteins): Valid range 0-100 g/100g
- **Saturated fat**: Valid range 0-100 g/100g
- **Salt**: Valid range 0-50 g/100g

Values outside these physiologically plausible ranges indicate data entry errors or measurement artifacts. This approach removes approximately 1,500 samples (1.5% of the dataset).

The pipeline includes an optional statistical outlier removal method using the interquartile range (IQR), but this is disabled by default (`remove_statistical_outliers=False`). Statistical methods are too aggressive for nutritional data, incorrectly flagging legitimate products such as high-fiber cereals and protein concentrates as outliers.

### Stage 4: Categorical feature encoding

**Implementation**: `FeatureEncoder` ([src/features/encoding.py:14](../src/features/encoding.py#L14))

Three categorical features are encoded using methods suited to their specific characteristics:

**Countries**: Products may be sold across multiple countries, requiring multi-label representation. The encoder retains the top 15 countries by frequency and applies `MultiLabelBinarizer` to create binary indicator columns.

**Food Groups** (`pnns_groups_1`): Broad taxonomic categories (e.g., beverages, dairy, cereals) are encoded using one-hot encoding, as the number of unique categories remains manageable.

**Food Subgroups** (`pnns_groups_2`): This granular categorization contains numerous unique values. Target encoding replaces each category with its mean target value from the training set, capturing the category-target relationship while controlling dimensionality. Smoothing prevents overfitting on rare categories.

This stage transforms 3 categorical features into approximately 30-35 numerical columns.

### Stage 5: Feature Engineering

**Implementation**: `FeatureEngineer` ([src/features/feature_engineering.py:9](../src/features/feature_engineering.py#L9))

Feature engineering creates 10 derived features from existing nutritional measurements:

**Nutritional Ratios** (4 features):
- `sugar_to_carb_ratio`: Proportion of carbohydrates consisting of simple sugars
- `saturated_to_total_fat_ratio`: Saturated fat fraction of total fat content
- `fat_to_protein_ratio`: Macronutrient balance indicator
- Additional nutrient ratios

**Energy Decomposition** (4 features):
- `energy_density`: Energy content per 100g (kcal/100g)
- `calories_from_fat`: Energy from fat (9 kcal/g conversion factor)
- `calories_from_carbs`: Energy from carbohydrates (4 kcal/g conversion factor)
- `calories_from_protein`: Energy from protein (4 kcal/g conversion factor)

**Health Threshold Flags** (2-3 features):
- `high_sugar`: Binary indicator for sugar content >15g/100g (WHO threshold)
- `high_salt`: Binary indicator for salt content >1.5g/100g (WHO threshold)
- `high_fat`: Binary indicator for fat content >20g/100g

Division operations use safe division with NaN handling. Resulting NaN values are imputed as 0.0.

### Stage 6: Feature Scaling

**Implementation**: `FeatureScaler` ([src/features/scaling.py:20](../src/features/scaling.py#L20))

Feature scaling normalizes numerical features to comparable ranges using automatic scaler selection based on distribution skewness:

- **MinMaxScaler**: Applied when absolute skewness >1.0, compressing values to [0,1] range
- **StandardScaler**: Applied when |skewness| ≤1.0, standardizing to zero mean and unit variance

This adaptive approach accounts for the varying distributions observed in nutritional features. Metadata columns (identifiers, product names, target variable) are excluded from scaling.

### Stage 7: Dimensionality Reduction

**Implementation**: `FeatureReducer` ([src/features/dimensionality_reduction.py:13](../src/features/dimensionality_reduction.py#L13))

Principal Component Analysis (PCA) reduces the feature space while retaining 95% of total variance. The dataset contains approximately 48-50 features after earlier transformations. PCA projection typically produces 19-20 principal components that collectively preserve 95% of training variance.

Benefits of this compression:
- Reduced computational cost for model training
- Mitigation of multicollinearity through orthogonal components
- Noise reduction by excluding low-variance components

## Data Partitioning

Following preprocessing, the dataset is split into training, validation, and test sets using a 70/15/15 ratio:

- **Training set**: 172,395 samples (70%) — used for fitting models and estimating transformation parameters
- **Validation set**: 36,942 samples (15%) — used for hyperparameter tuning and model selection
- **Test set**: 36,942 samples (15%) — reserved for final performance evaluation

The split is stratified by Nutri-Score grade, ensuring each partition maintains the original class distribution. This prevents evaluation bias and achieves class balance across splits.

Of the initial 250,000 samples, 246,279 remain after outlier removal, representing a 1.5% attrition rate.

## Data Leakage Prevention

All transformation parameters—including imputation statistics, encoding mappings, scaling parameters, PCA components, and target encoding values—are computed exclusively from the training partition. Validation and test data are transformed using these training-derived parameters, ensuring no information leakage affects model evaluation.

The implementation identifies training samples using the `split_group` column when present, or fits on the complete dataset if split information is unavailable.

## Configuration Parameters

Pipeline behavior can be customized in [scripts/run_preprocessing.py](../scripts/run_preprocessing.py):

```python
PreprocessingPipeline(
    missing_threshold=0.95,              # Drop columns with >95% missing values
    top_n_countries=15,                  # Number of countries to retain in encoding
    scaling_method='auto',               # Options: 'auto', 'standard', 'minmax', 'robust'
    scaling_skew_threshold=1.0,          # Skewness cutoff for automatic scaler selection
    pca_variance_threshold=0.95,         # Proportion of variance to retain in PCA
    target_col='nutriscore_grade',       # Target variable column name
    include_feature_engineering=True,    # Enable/disable feature engineering stage
    remove_statistical_outliers=False,   # Enable/disable IQR-based outlier removal
    include_pca=True                     # Enable/disable dimensionality reduction
)
```

Disabling feature engineering or PCA allows evaluation of their contribution to model performance. Statistical outlier removal should remain disabled for nutritional data.

## Output Files

### Raw Data
- `data/raw/openfoodfacts_raw.csv.gz` — Original compressed download from Open Food Facts

### Intermediate Files
- `data/processed/openfoodfacts_filtered.csv` — 250,000 filtered products with Nutri-Score labels
- `data/processed/openfoodfacts_preprocessed.csv` — Complete preprocessed dataset
- `data/processed/metadata.json` — Download statistics and configuration

### Final Splits
- `data/splits/X_train.csv` & `y_train.csv` — Training features and labels (172,395 samples)
- `data/splits/X_val.csv` & `y_val.csv` — Validation features and labels (36,942 samples)
- `data/splits/X_test.csv` & `y_test.csv` — Test features and labels (36,942 samples)
- `data/splits/split_metadata.json` — Stratification statistics and split configuration