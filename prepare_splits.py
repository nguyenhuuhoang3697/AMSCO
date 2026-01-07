import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# --- Loading Functions (Adapted from amsco_adult_v3 copy.py) ---

def load_adult_dataset(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")
    adult = pd.read_csv(csv_path)
    X = adult.drop(columns=['income'])
    y = adult['income'].str.strip().map({'<=50K': 0, '>50K': 1}).astype(int)
    return X, y

def load_breast_cancer_dataset():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target.astype(int))
    return X, y

def load_telco_dataset(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")
    df = pd.read_csv(csv_path)
    target_candidates = ['Churn', 'churn', 'target', 'label', 'y']
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError("Target column not found in telco.csv")
    y_raw = df[target_col]
    
    if y_raw.dtype == 'O' or str(y_raw.dtype).startswith('category'):
        mapping = {"Yes": 1, "No": 0, "True": 1, "False": 0, "Y": 1, "N": 0}
        y = y_raw.map(lambda v: mapping.get(str(v).strip(), np.nan)).astype(float)
        if y.isna().any():
             uniques = sorted(y_raw.dropna().unique().tolist())
             if len(uniques) == 2:
                 y = (y_raw == uniques[1]).astype(int)
             else:
                 raise ValueError("Telco target not binary")
        else:
            y = y.astype(int)
    else:
        y = y_raw.astype(int)
    X = df.drop(columns=[target_col])
    return X, y

def load_credit_dataset(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")
    df = pd.read_csv(csv_path)
    if 'Class' not in df.columns:
        raise ValueError("Class column not found in creditcard.csv")
    y = df['Class'].astype(int)
    X = df.drop(columns=['Class'])
    return X, y

def get_raw_data(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'adult':
        return load_adult_dataset(Path('datasets') / 'adult.csv')
    elif dataset_name == 'breast_cancer':
        return load_breast_cancer_dataset()
    elif dataset_name == 'telco':
        return load_telco_dataset(Path('datasets') / 'telco.csv')
    elif dataset_name == 'credit':
        return load_credit_dataset(Path('datasets') / 'creditcard.csv')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# --- Main Execution ---

if __name__ == "__main__":
    os.makedirs('splits', exist_ok=True)
    
    datasets = ['breast_cancer', 'telco', 'adult', 'credit']
    seeds = range(10) 

    for ds_name in datasets:
        print(f"Processing {ds_name}...")
        try:
            X, y = get_raw_data(ds_name)
            print(f"  Loaded {len(X)} samples.")
            
            for seed in seeds:
                # Stratified Split
                # Passing np.arange(len(y)) gives us the indices directly
                indices = np.arange(len(y))
                
                # We split X, y, and indices together to ensure alignment
                X_dev, X_test, y_dev, y_test, idx_dev, idx_test = train_test_split(
                    X, y, indices,
                    test_size=0.2,
                    random_state=seed,
                    stratify=y
                )
                
                split_info = {
                    'train_idx': idx_dev,
                    'test_idx': idx_test
                }
                
                filename = f'splits/{ds_name}_seed_{seed}.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(split_info, f)
                
            print(f"  Saved splits for {ds_name} (seeds 0-9).")
            
        except Exception as e:
            print(f"  Error processing {ds_name}: {e}")

    print("\nDone. All splits saved to splits/ directory.")
