import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import pandas as pd
from features.preprocessing_pipeline import PreprocessingPipeline
from data.data_loader import split_data, save_splits, verify_stratification, save_split_metadata

"""
This script preprocesses the dataset for Nutri-Score prediction.
It loads the dataset, applies the preprocessing pipeline and saves the preprocessed dataset.
It also creates the train/validation/test splits and saves the splits.

Note: We added some arguments to the script to allow for customization of the preprocessing pipeline
and to solve accuracy issues we encountered in first experiments.
"""

def main():
    parser = argparse.ArgumentParser(description='Preprocess Open Food Facts data for Nutri-Score prediction')
    parser.add_argument(
        '--no-pca',
        action='store_true',
        help='Skip PCA; keep all engineered features (useful to compare with PCA pipeline)'
    )
    parser.add_argument(
        '--scale-method',
        type=str,
        choices=['standard', 'minmax', 'robust', 'auto'],
        default='standard',
        help='Scaling method: standard, minmax, robust, or auto (by skewness). Default: standard'
    )
    parser.add_argument(
        '--remove-outliers',
        action='store_true',
        help='Remove statistical outliers (IQR-based). Default: keep outliers (include all samples).'
    )
    parser.add_argument(
        '--eda-only',
        action='store_true',
        help='Run preprocessing and save a single CSV for EDA only (no train/val/test splits).'
    )
    parser.add_argument(
        '--no-scaling',
        action='store_true',
        help='Skip feature scaling (StandardScaler/MinMaxScaler).'
    )
    parser.add_argument(
        '--no-encoding',
        action='store_true',
        help='Skip categorical encoding (countries, pnns_groups_1, pnns_groups_2). Categorical columns remain.'
    )
    parser.add_argument(
        '--no-feature-engineering',
        action='store_true',
        help='Skip derived/engineered features (e.g. ratios, composite nutritional features).'
    )
    args = parser.parse_args()

    data_path = Path("data/processed/openfoodfacts_filtered.csv")

    if not data_path.exists():
        print(f"      Error: {data_path} not found")
        print("      Run 'python scripts/download_data.py' first")
        sys.exit(1)

    print("      Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"      Loaded {len(df):,} products with {len(df.columns)} features")

    use_pca = not args.no_pca
    if args.no_pca:
        print("      PCA disabled (--no-pca): using full feature set")
    if args.scale_method == 'standard':
        print("      Scaling: StandardScaler for all features")
    if args.remove_outliers:
        print("      Outliers: IQR-based removal enabled (--remove-outliers)")
    if args.no_scaling:
        print("      Scaling disabled (--no-scaling)")
    if args.no_encoding:
        print("      Encoding disabled (--no-encoding): categorical columns kept as-is")
    if args.no_feature_engineering:
        print("      Feature engineering disabled (--no-feature-engineering): no derived features")

    print("      Applying transformations...")
    pipeline = PreprocessingPipeline(
        missing_threshold=0.95,
        top_n_countries=10,
        scaling_method=args.scale_method,
        scaling_skew_threshold=1.0,
        pca_variance_threshold=0.98,
        target_col='nutriscore_grade',
        include_feature_engineering=not args.no_feature_engineering,
        remove_statistical_outliers=args.remove_outliers,
        include_pca=use_pca,
        include_encoding=not args.no_encoding,
        include_scaling=not args.no_scaling
    )

    df_processed = pipeline.fit_transform(df)
    print(f"      Processed: {len(df_processed):,} products, {len(df_processed.columns)} features")

    if args.eda_only:
        print("      Saving EDA dataset (no splits)...")
        eda_path = Path("data/processed/openfoodfacts_eda.csv")
        df_processed.to_csv(eda_path, index=False)
        print(f"      EDA CSV saved: {eda_path}")
        print("      Preprocessing complete (EDA only)")
        return

    print("      Saving preprocessed dataset...")
    output_path = Path("data/processed/openfoodfacts_preprocessed.csv")
    df_processed.to_csv(output_path, index=False)

    print("      Creating train/validation/test splits...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        df_processed,
        target_col='nutriscore_grade',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        stratify=True
    )

    stats = verify_stratification(y_train, y_val, y_test)
    output_dir = Path("data/splits")
    save_splits(X_train, y_train, X_val, y_val, X_test, y_test, output_dir)

    config = {
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_state': 42,
        'stratify': True
    }
    save_split_metadata(stats, config, output_dir / 'split_metadata.json')

    print(f"      Train: {len(X_train):,} | Validation: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"      Preprocessing complete")


if __name__ == "__main__":
    main()
