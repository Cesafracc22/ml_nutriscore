import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json
import sys
import pycountry

try:
    from .countries_mappings import COUNTRY_OVERRIDES
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.countries_mappings import COUNTRY_OVERRIDES

"""
This class handles the missing values in the dataset.
It is used to analyze the missing values and to impute them.
The imputation strategy is to drop features with more than 95% missing values and to 
impute the remaining missing values with the median value.

Note: we choose the following methods to handle the missing values:
- Dropping features with more than 95% missing values
- Imputing the missing values with the median value
- Labeling the missing values as 'unknown'
- Removing the samples without Nutri-Score labels
"""
class MissingValueHandler:
    def __init__(self, threshold_drop_feature: float = 0.95):
        self.threshold_drop_feature = threshold_drop_feature
        self.imputation_stats = {}
        self.dropped_features = []
        self.missing_report = {}

    def analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        missing_info = {}

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100

            missing_info[col] = {
                'missing_count': int(missing_count),
                'missing_percentage': round(missing_pct, 2),
                'dtype': str(df[col].dtype),
                'unique_values': int(df[col].nunique()) if missing_count < len(df) else 0
            }

        missing_info = dict(sorted(missing_info.items(),
                                  key=lambda x: x[1]['missing_percentage'],
                                  reverse=True))

        return missing_info

    def handle_missing_values(self, df: pd.DataFrame,
                            target_col: str = 'nutriscore_grade') -> pd.DataFrame:
        from tqdm import tqdm

        df_processed = df.copy()
        initial_rows, initial_cols = len(df_processed), len(df_processed.columns)

        self.missing_report = self.analyze_missing_values(df_processed)

        threshold_pct = self.threshold_drop_feature * 100
        high_missing_cols = [col for col, info in self.missing_report.items()
                            if info['missing_percentage'] > threshold_pct and col != target_col]

        if high_missing_cols:
            df_processed = df_processed.drop(columns=high_missing_cols)
            self.dropped_features.extend(high_missing_cols)

        target_missing = df_processed[target_col].isnull().sum()
        if target_missing > 0:
            df_processed = df_processed[df_processed[target_col].notna()]

        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        imputed_numerical = []

        for col in numerical_cols:
            missing_count = df_processed[col].isnull().sum()
            if missing_count > 0:
                if col == 'additives_n':
                    df_processed[col] = df_processed[col].fillna(0)
                    method, value = 'constant', 0
                else:
                    value = df_processed[col].median()
                    df_processed[col] = df_processed[col].fillna(value)
                    method = 'median'

                self.imputation_stats[col] = {
                    'method': method,
                    'value': float(value) if method == 'median' else value,
                    'missing_count': int(missing_count),
                    'rationale': 'Assume no additives' if col == 'additives_n' else 'Median imputation'
                }
                imputed_numerical.append((col, method, value, missing_count))

        categorical_cols = [col for col in df_processed.select_dtypes(include=['object']).columns
                           if col != target_col]
        imputed_categorical = []

        for col in categorical_cols:
            missing_count = df_processed[col].isnull().sum()
            if missing_count > 0:
                df_processed[col] = df_processed[col].fillna('unknown')
                self.imputation_stats[col] = {
                    'method': 'constant',
                    'value': 'unknown',
                    'missing_count': int(missing_count),
                    'rationale': 'Placeholder for missing categorical data'
                }
                imputed_categorical.append((col, missing_count))

        final_rows, final_cols = len(df_processed), len(df_processed.columns)
        cols_dropped = initial_cols - final_cols
        rows_dropped = initial_rows - final_rows

        print("Operation: Missing value imputation and feature filtering")
        if cols_dropped > 0:
            dropped_names = ", ".join(["'" + col + "'" for col in high_missing_cols[:2]])
            if len(high_missing_cols) > 2:
                dropped_names += " (+" + str(len(high_missing_cols) - 2) + " more)"
            print(" - Dropped:", dropped_names)
        if rows_dropped > 0:
            print(" - Removed", rows_dropped, "rows with missing", "'" + target_col + "'")
        if len(imputed_numerical) > 0:
            num_examples = []
            for col, method, value, count in imputed_numerical[:2]:
                if method == 'median':
                    val_str = "%.1f" % value
                else:
                    val_str = str(value)
                num_examples.append("'" + col + "'=" + val_str)
            if len(imputed_numerical) > 2:
                num_examples.append("(+" + str(len(imputed_numerical) - 2) + " more)")
            print(" - Imputed numerical:", ", ".join(num_examples))
        if len(imputed_categorical) > 0:
            cat_names = ", ".join(["'" + col + "'" for col, _ in imputed_categorical[:2]])
            if len(imputed_categorical) > 2:
                cat_names += " (+" + str(len(imputed_categorical) - 2) + " more)"
            print(" - Imputed categorical:", cat_names, "-> 'unknown'")

        return df_processed

    def save_imputation_report(self, output_path: Path) -> None:
        report = {
            'strategy': {
                'threshold_drop_feature': self.threshold_drop_feature,
                'description': 'Drop features >95% missing, impute others based on type'
            },
            'dropped_features': self.dropped_features,
            'imputation_statistics': self.imputation_stats,
            'missing_value_analysis': self.missing_report
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print("Saved imputation report:", output_path)


def normalize_single_country(raw_name: str) -> str:
    name = str(raw_name).lower().strip()
    name = name.replace("en:", "").replace("-", " ")

    if not name:
        return None

    if name in COUNTRY_OVERRIDES:
        return COUNTRY_OVERRIDES[name]

    try:
        country = pycountry.countries.lookup(name)
        country_name = country.name

        country_name_lower = country_name.lower()

        if country_name_lower in COUNTRY_OVERRIDES:
            return COUNTRY_OVERRIDES[country_name_lower]

        standard_names = set(COUNTRY_OVERRIDES.values())
        for std_name in standard_names:
            if country_name_lower == std_name.lower():
                return std_name

        return country_name
    except LookupError:
        return None


def clean_countries_column(entry) -> str:
    if pd.isna(entry) or entry == "unknown":
        return "unknown"

    valid_countries = set()

    for raw_item in str(entry).split(','):
        clean_name = normalize_single_country(raw_item)
        if clean_name:
            valid_countries.add(clean_name)

    if not valid_countries:
        return "unknown"

    return ",".join(sorted(valid_countries))


def preprocess_dataset(input_path: Path,
                      output_path: Path,
                      report_path: Path = None) -> Tuple[pd.DataFrame, MissingValueHandler]:
    print()
    print("Loading data:", input_path)
    df = pd.read_csv(input_path)
    print("Loaded:", len(df), "rows x", len(df.columns), "columns")

    handler = MissingValueHandler(threshold_drop_feature=0.95)
    df_processed = handler.handle_missing_values(df)

    if 'countries' in df_processed.columns:
        print()
        print("Cleaning countries column...")
        df_processed['countries'] = df_processed['countries'].apply(clean_countries_column)
        print("Countries column cleaned")

    columns_to_remove = []

    if 'energy_100g' in df_processed.columns:
        columns_to_remove.append('energy_100g')

    for col in ['main_category', 'categories']:
        if col in df_processed.columns and df_processed[col].nunique() > 10000:
            columns_to_remove.append(col)

    if columns_to_remove:
        df_processed = df_processed.drop(columns=columns_to_remove)
        print("Removed", len(columns_to_remove), "unnecessary column(s)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print()
    print("Saved processed data:", output_path)

    if report_path:
        handler.save_imputation_report(report_path)

    return df_processed, handler


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / "data" / "processed" / "openfoodfacts_filtered.csv"
    output_file = project_root / "data" / "processed" / "openfoodfacts_preprocessed.csv"
    report_file = project_root / "data" / "processed" / "imputation_report.json"

    df_processed, handler = preprocess_dataset(input_file, output_file, report_file)
    print()
    print("Preprocessing complete:", df_processed.shape)
