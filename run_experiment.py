import argparse
import pickle
import gc
import os
import time
import pandas as pd
import numpy as np
import importlib.util
import sys
from sklearn.model_selection import train_test_split

# --- Import amsco_lib from "amsco_adult_v3 copy.py" ---
# We use importlib because the filename has spaces
spec = importlib.util.spec_from_file_location("amsco_lib", "amsco_adult_v4.py")
amsco_lib = importlib.util.module_from_spec(spec)
sys.modules["amsco_lib"] = amsco_lib
spec.loader.exec_module(amsco_lib)

# --- Shortcuts to library functions ---
get_data = amsco_lib.get_data
MASTER_SEARCH_SPACES = amsco_lib.MASTER_SEARCH_SPACES
create_objective = amsco_lib.create_objective
run_optuna_tpe = amsco_lib.run_optuna_tpe
run_amsco_optimizer = amsco_lib.run_amsco_optimizer
run_random_search = amsco_lib.run_random_search
run_hyperopt_tpe = amsco_lib.run_hyperopt_tpe
_nested_cv_metrics_for_method = amsco_lib._nested_cv_metrics_for_method
evaluate_on_holdout_test = amsco_lib.evaluate_on_holdout_test

# --- Configuration ---
MODELS = ['logistic_regression', 'random_forest']
METHODS = ['AMSCO', 'Optuna (TPE)'] 
# You can add 'Random Search', 'Hyperopt (TPE)' if desired

# Trials config (Adjust as needed or take from args)
N_TRIALS = 50
SLICE_BUDGET = 10
NESTED_INNER_TRIALS = 30
NESTED_OUTER_FOLDS = 5
NESTED_INNER_FOLDS = 3

def load_split(dataset, seed):
    """Loads raw data and split indices."""
    print(f"Loading data for {dataset}...")
    X, y, preprocessor, metric, sampler = get_data(dataset, quiet=True)
    
    split_path = f'splits/{dataset}_seed_{seed}.pkl'
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file {split_path} not found. Please run prepare_splits.py first.")
        
    with open(split_path, 'rb') as f:
        info = pickle.load(f)
        
    X_dev = X.iloc[info['train_idx']]
    y_dev = y.iloc[info['train_idx']]
    X_test = X.iloc[info['test_idx']]
    y_test = y.iloc[info['test_idx']]
    
    # === CREDIT UNDERSAMPLING (SAU KHI SPLIT) ===
    # Để tránh RAM overflow với credit dataset
    if dataset.lower() == 'credit':
        print(f"  Original Dev set: {len(X_dev)}, Test set: {len(X_test)}")
        X_dev, y_dev = amsco_lib.undersample_credit_data(X_dev, y_dev, 
                                                          undersample_class0_to=40000, 
                                                          random_state=seed)
        X_test, y_test = amsco_lib.undersample_credit_data(X_test, y_test, 
                                                            undersample_class0_to=10000, 
                                                            random_state=seed, quiet=True)
        print(f"  Undersampled Dev set: {len(X_dev)}, Test set: {len(X_test)}")
    else:
        print(f"  Dev set: {len(X_dev)}, Test set: {len(X_test)}")
    
    return X_dev, y_dev, X_test, y_test, preprocessor, metric, sampler

def run_holdout_task(dataset, seed):
    """Runs Holdout Validation (Train/Val split on Dev set) and evaluates on Test set."""
    print(f"\n=== Running Holdout Task: {dataset} (Seed {seed}) ===")
    X_dev, y_dev, X_test, y_test, preprocessor, metric, sampler = load_split(dataset, seed)
    
    # Split Dev into Train/Val (e.g. 75/25 of Dev -> 60/20 of Total)
    # This mimics the standard holdout procedure where we tune on Train/Val
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=seed
    )
    
    results = []
    
    for method in METHODS:
        for model_name in MODELS:
            print(f"  >> Method: {method}, Model: {model_name}")
            search_space = dict(MASTER_SEARCH_SPACES[model_name])
            
            # Create Objective for Optimization (Train on X_train, Eval on X_val)
            objective = create_objective(
                X_train, y_train, model_name, preprocessor, 
                metrics=('accuracy', 'f1', 'roc_auc'),
                use_cross_validation=False, # Holdout internal
                validation_data=(X_val, y_val),
                sampler=sampler
            )
            
            # Run Optimizer
            start_time = time.time()
            try:
                if method == 'AMSCO':
                    best_score, best_params, diag = run_amsco_optimizer(
                        objective, search_space, n_trials=N_TRIALS, slice_budget=SLICE_BUDGET
                    )
                elif method == 'Optuna (TPE)':
                    best_score, best_params, diag = run_optuna_tpe(
                        objective, search_space, n_trials=N_TRIALS
                    )
                elif method == 'Random Search':
                    best_score, best_params, diag = run_random_search(
                        objective, search_space, n_trials=N_TRIALS
                    )
                elif method == 'Hyperopt (TPE)':
                    best_score, best_params, diag = run_hyperopt_tpe(
                        objective, search_space, n_trials=N_TRIALS
                    )
                else:
                    print(f"Unknown method {method}")
                    continue
            except Exception as e:
                print(f"    [Error] Optimization failed: {e}")
                continue
            
            exec_time = time.time() - start_time
            
            # Evaluate on True Test (Retrain on full Dev set with best params)
            print("    Evaluating on True Test set...")
            try:
                test_metrics = evaluate_on_holdout_test(
                    best_params, model_name, preprocessor,
                    X_train=X_dev, y_train=y_dev, # Retrain on full Dev
                    X_test=X_test, y_test=y_test,
                    sampler=sampler
                )
                
                results.append({
                    'dataset': dataset,
                    'seed': seed,
                    'method': method,
                    'model': model_name,
                    'val_score': best_score,
                    'test_accuracy': test_metrics['accuracy'],
                    'test_f1': test_metrics['f1'],
                    'test_auc': test_metrics['roc_auc'],
                    'time': exec_time
                })
            except Exception as e:
                print(f"    [Error] Evaluation failed: {e}")

            # Cleanup
            del objective
            gc.collect()
            
    # Save Results
    os.makedirs('results_step2', exist_ok=True)
    out_file = f'results_step2/holdout_{dataset}_seed_{seed}.csv'
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"Saved {out_file}")

def run_nested_task(dataset, seed):
    """Runs Nested Cross-Validation on Dev set."""
    print(f"\n=== Running Nested CV Task: {dataset} (Seed {seed}) ===")
    X_dev, y_dev, X_test, y_test, preprocessor, metric, sampler = load_split(dataset, seed)
    
    results = []
    
    for method in METHODS:
        for model_name in MODELS:
            print(f"  >> Method: {method}, Model: {model_name}")
            search_space = dict(MASTER_SEARCH_SPACES[model_name])
            
            try:
                # Run Nested CV
                # _nested_cv_metrics_for_method handles the outer loop splitting of X_dev
                res = _nested_cv_metrics_for_method(
                    X_dev, y_dev, model_name, preprocessor, search_space,
                    method,
                    inner_trials=NESTED_INNER_TRIALS,
                    outer_folds=NESTED_OUTER_FOLDS,
                    inner_folds=NESTED_INNER_FOLDS,
                    slice_budget=SLICE_BUDGET,
                    sampler=sampler
                )
                
                # --- NEW: Calculate True Test Score for Comparison ---
                # We run the optimization on the full Dev set using Standard 5-Fold CV
                # to simulate the final model selection process, then evaluate on True Test.
                print("    [Comparison] Running Standard 5-Fold CV on full Dev set for True Test score...")
                
                objective_refit = create_objective(
                    X_dev, y_dev, model_name, preprocessor, 
                    metrics=('accuracy', 'f1', 'roc_auc'),
                    use_cross_validation=True, 
                    cv_folds=5, # Standard 5-Fold CV
                    sampler=sampler
                )
                
                # Use N_TRIALS (50) to match the standard experiment budget
                if method == 'AMSCO':
                    best_score_refit, best_params_refit, _ = run_amsco_optimizer(
                        objective_refit, search_space, n_trials=N_TRIALS, slice_budget=SLICE_BUDGET
                    )
                elif method == 'Optuna (TPE)':
                    best_score_refit, best_params_refit, _ = run_optuna_tpe(
                        objective_refit, search_space, n_trials=N_TRIALS
                    )
                elif method == 'Random Search':
                    best_score_refit, best_params_refit, _ = run_random_search(
                        objective_refit, search_space, n_trials=N_TRIALS
                    )
                elif method == 'Hyperopt (TPE)':
                    best_score_refit, best_params_refit, _ = run_hyperopt_tpe(
                        objective_refit, search_space, n_trials=N_TRIALS
                    )
                
                test_metrics = evaluate_on_holdout_test(
                    best_params_refit, model_name, preprocessor,
                    X_train=X_dev, y_train=y_dev,
                    X_test=X_test, y_test=y_test,
                    sampler=sampler
                )
                
                results.append({
                    'dataset': dataset,
                    'seed': seed,
                    'method': method,
                    'model': model_name,
                    'nested_accuracy': res['accuracy'],
                    'nested_f1': res['f1'],
                    'nested_auc': res['roc_auc'],
                    'avg_trials_to_95': res.get('avg_trials_to_95', np.nan),
                    'refit_val_score': best_score_refit, # Standard 5-Fold CV Estimate
                    'test_accuracy': test_metrics['accuracy'],
                    'test_f1': test_metrics['f1'],
                    'test_auc': test_metrics['roc_auc'],
                    'time': res['time']
                })
            except Exception as e:
                print(f"    [Error] Nested CV failed: {e}")
            
            gc.collect()
            
    # Save Results
    os.makedirs('results_step2', exist_ok=True)
    out_file = f'results_step2/nested_{dataset}_seed_{seed}.csv'
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"Saved {out_file}")

def run_standard_cv_task(dataset, seed):
    """Runs Standard 5-Fold CV (Tuning on Dev using CV) and evaluates on Test set."""
    print(f"\n=== Running Standard CV Task: {dataset} (Seed {seed}) ===")
    X_dev, y_dev, X_test, y_test, preprocessor, metric, sampler = load_split(dataset, seed)
    
    results = []
    
    for method in METHODS:
        for model_name in MODELS:
            print(f"  >> Method: {method}, Model: {model_name}")
            search_space = dict(MASTER_SEARCH_SPACES[model_name])
            
            # Create Objective for Optimization (Train on X_dev using 5-Fold CV)
            objective = create_objective(
                X_dev, y_dev, model_name, preprocessor, 
                metrics=('accuracy', 'f1', 'roc_auc'),
                use_cross_validation=True, 
                cv_folds=5, # Standard 5-Fold CV
                sampler=sampler
            )
            
            # Run Optimizer
            start_time = time.time()
            try:
                if method == 'AMSCO':
                    best_score, best_params, diag = run_amsco_optimizer(
                        objective, search_space, n_trials=N_TRIALS, slice_budget=SLICE_BUDGET
                    )
                elif method == 'Optuna (TPE)':
                    best_score, best_params, diag = run_optuna_tpe(
                        objective, search_space, n_trials=N_TRIALS
                    )
                elif method == 'Random Search':
                    best_score, best_params, diag = run_random_search(
                        objective, search_space, n_trials=N_TRIALS
                    )
                elif method == 'Hyperopt (TPE)':
                    best_score, best_params, diag = run_hyperopt_tpe(
                        objective, search_space, n_trials=N_TRIALS
                    )
                else:
                    print(f"Unknown method {method}")
                    continue
            except Exception as e:
                print(f"    [Error] Optimization failed: {e}")
                continue
            
            exec_time = time.time() - start_time
            
            # Evaluate on True Test (Retrain on full Dev set with best params)
            print("    Evaluating on True Test set...")
            try:
                test_metrics = evaluate_on_holdout_test(
                    best_params, model_name, preprocessor,
                    X_train=X_dev, y_train=y_dev, # Retrain on full Dev
                    X_test=X_test, y_test=y_test,
                    sampler=sampler
                )
                
                results.append({
                    'dataset': dataset,
                    'seed': seed,
                    'method': method,
                    'model': model_name,
                    'val_score': best_score,
                    'test_accuracy': test_metrics['accuracy'],
                    'test_f1': test_metrics['f1'],
                    'test_auc': test_metrics['roc_auc'],
                    'time': exec_time
                })
            except Exception as e:
                print(f"    [Error] Evaluation failed: {e}")

            # Cleanup
            del objective
            gc.collect()
            
    # Save Results
    os.makedirs('results_step2', exist_ok=True)
    out_file = f'results_step2/standard_cv_{dataset}_seed_{seed}.csv'
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (adult, breast_cancer, telco, credit)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed (0-9)')
    parser.add_argument('--task', type=str, choices=['holdout', 'nested', 'standard_cv'], required=True, help='Task to run')
    args = parser.parse_args()
    
    if args.task == 'holdout':
        run_holdout_task(args.dataset, args.seed)
    elif args.task == 'nested':
        run_nested_task(args.dataset, args.seed)
    elif args.task == 'standard_cv':
        run_standard_cv_task(args.dataset, args.seed)
