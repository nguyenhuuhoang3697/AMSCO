import pandas as pd
import numpy as np
import glob
import os
import ast
from scipy.stats import mannwhitneyu, ttest_rel, wilcoxon

def load_results():
    """Loads all CSV files from results_step2/ and combines them."""
    all_files = glob.glob("results_step2/*.csv")
    if not all_files:
        print("No result files found in results_step2/")
        return pd.DataFrame()
    
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            # Extract task type from filename if not in columns
            filename = os.path.basename(f)
            if 'nested' in filename:
                df['task'] = 'nested'
            elif 'holdout' in filename:
                df['task'] = 'holdout'
            elif 'standard_cv' in filename:
                df['task'] = 'standard_cv'
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def generate_tables(df):
    if df.empty:
        return

    # --- Table 4.3: Reliability of Validation Strategies ---
    print("\n=== Table 4.3: Reliability of Validation Strategies ===")
    
    strategies = []
    
    # 1. Holdout
    holdout_df = df[df['task'] == 'holdout'].copy()
    if not holdout_df.empty:
        holdout_df['strategy'] = 'Hold-out'
        holdout_df['est_acc'] = holdout_df['val_score']
        holdout_df['true_acc'] = holdout_df['test_accuracy']
        strategies.append(holdout_df)
        
    # 2. Standard CV
    # Try to get from 'standard_cv' task first
    std_cv_df = df[df['task'] == 'standard_cv'].copy()
    
    # If empty or we want to augment with data from 'nested' task (where we ran refit)
    # Check if 'nested' task has 'refit_val_score'
    if 'refit_val_score' in df.columns:
        nested_with_refit = df[(df['task'] == 'nested') & (df['refit_val_score'].notna())].copy()
    else:
        nested_with_refit = pd.DataFrame()
    
    if not nested_with_refit.empty:
        # Create a Standard CV view from nested results
        nested_std_view = nested_with_refit.copy()
        nested_std_view['val_score'] = nested_std_view['refit_val_score']
        # test_accuracy is already there
        
        # Combine with explicit standard_cv runs if any
        std_cv_df = pd.concat([std_cv_df, nested_std_view], ignore_index=True)
        
    if not std_cv_df.empty:
        std_cv_df['strategy'] = 'Standard 5-Fold'
        std_cv_df['est_acc'] = std_cv_df['val_score']
        std_cv_df['true_acc'] = std_cv_df['test_accuracy']
        strategies.append(std_cv_df)
        
    # 3. Nested CV
    nested_df_t43 = df[df['task'] == 'nested'].copy()
    if not nested_df_t43.empty:
        nested_df_t43['strategy'] = 'Nested CV (5x3)'
        nested_df_t43['est_acc'] = nested_df_t43['nested_accuracy']
        nested_df_t43['true_acc'] = nested_df_t43['test_accuracy']
        strategies.append(nested_df_t43)
    
    if strategies:
        full_df = pd.concat(strategies, ignore_index=True)
        
        # Ensure numeric
        full_df['est_acc'] = pd.to_numeric(full_df['est_acc'], errors='coerce')
        full_df['true_acc'] = pd.to_numeric(full_df['true_acc'], errors='coerce')
        
        # --- FIX: Drop rows where either est_acc or true_acc is NaN ---
        # This ensures we compare apples to apples (only runs that have both values)
        # and fixes the inconsistency in Bias calculation if mixed old/new results exist.
        before_len = len(full_df)
        full_df = full_df.dropna(subset=['est_acc', 'true_acc'])
        after_len = len(full_df)
        if before_len > after_len:
            print(f"Dropped {before_len - after_len} rows with missing Est or True accuracy for Table 4.3 consistency.")

        # Calculate Bias per row
        full_df['bias'] = full_df['est_acc'] - full_df['true_acc']
        
        # Calculate Std Dev across seeds for each (Dataset, Model, Method, Strategy)
        # This gives us the variability of the estimation method
        # We need at least 2 seeds to calculate std. If only 1, it returns NaN.
        std_devs = full_df.groupby(['strategy', 'dataset', 'model', 'method'])['est_acc'].std().reset_index()
        std_devs.rename(columns={'est_acc': 'std_dev'}, inplace=True)
        
        # Aggregate by Strategy
        # Mean Est Acc, Mean True Acc, Mean Bias
        main_stats = full_df.groupby('strategy')[['est_acc', 'true_acc', 'bias']].mean()
        
        # Aggregate Std Devs
        # We use nanmean to handle cases where some groups might have NaN std (single seed)
        std_stats = std_devs.groupby('strategy')['std_dev'].agg(
            mean=lambda x: np.nanmean(x) if not x.isna().all() else np.nan,
            min=lambda x: np.nanmin(x) if not x.isna().all() else np.nan,
            max=lambda x: np.nanmax(x) if not x.isna().all() else np.nan
        )
        
        # Combine
        table_4_3 = pd.concat([main_stats, std_stats], axis=1)
        
        # Format
        final_table = pd.DataFrame()
        final_table['Est. Acc.'] = table_4_3['est_acc'].map('{:.4f}'.format)
        final_table['True Test Acc.'] = table_4_3['true_acc'].map('{:.4f}'.format)
        
        # Recalculate Bias from the displayed means to ensure visual consistency (A - B = C)
        # Since we dropped NaNs, Mean(A-B) should equal Mean(A)-Mean(B) anyway, but this is safer for display.
        final_table['Avg Bias'] = (table_4_3['est_acc'] - table_4_3['true_acc']).map('{:+.4f}'.format)
        
        # Handle NaN in Std Dev (e.g. if only 1 seed run)
        final_table['Avg Std Dev'] = table_4_3['mean'].apply(lambda x: f"± {x:.4f}" if pd.notna(x) else "N/A")
        final_table['Min-Max Std'] = table_4_3.apply(
            lambda row: f"{row['min']:.4f} - {row['max']:.4f}" if pd.notna(row['min']) and pd.notna(row['max']) else "N/A", 
            axis=1
        )
        
        # Reorder rows if possible
        order = ['Hold-out', 'Standard 5-Fold', 'Nested CV (5x3)']
        final_table = final_table.reindex([o for o in order if o in final_table.index])
        
        print(final_table)
        final_table.to_csv("results/table_4_3_reliability.csv")
    else:
        print("No data available for Table 4.3")
    
    nested_df = df[df['task'] == 'nested'].copy()


    # --- Table 4.4: Detailed Bias and Variance Analysis per Dataset ---
    print("\n=== Table 4.4: Detailed Bias and Variance Analysis per Dataset ===")
    
    t44_strategies = []
    
    # 1. Standard CV (Reusing logic to combine explicit and implicit runs)
    std_cv_df = df[df['task'] == 'standard_cv'].copy()
    
    if 'refit_val_score' in df.columns:
        nested_with_refit = df[(df['task'] == 'nested') & (df['refit_val_score'].notna())].copy()
    else:
        nested_with_refit = pd.DataFrame()

    if not nested_with_refit.empty:
        nested_std_view = nested_with_refit.copy()
        nested_std_view['val_score'] = nested_std_view['refit_val_score']
        std_cv_df = pd.concat([std_cv_df, nested_std_view], ignore_index=True)
        
    if not std_cv_df.empty:
        std_cv_df['strategy'] = 'Standard 5-Fold'
        std_cv_df['est_acc'] = std_cv_df['val_score']
        std_cv_df['true_acc'] = std_cv_df['test_accuracy']
        t44_strategies.append(std_cv_df)

    # 2. Nested CV
    nested_df_t44 = df[df['task'] == 'nested'].copy()
    if not nested_df_t44.empty:
        nested_df_t44['strategy'] = 'Nested CV'
        nested_df_t44['est_acc'] = nested_df_t44['nested_accuracy']
        nested_df_t44['true_acc'] = nested_df_t44['test_accuracy']
        t44_strategies.append(nested_df_t44)
        
    if t44_strategies:
        t44_df = pd.concat(t44_strategies, ignore_index=True)
        t44_df['est_acc'] = pd.to_numeric(t44_df['est_acc'], errors='coerce')
        t44_df['true_acc'] = pd.to_numeric(t44_df['true_acc'], errors='coerce')
        t44_df.dropna(subset=['est_acc', 'true_acc'], inplace=True)
        t44_df['bias'] = t44_df['est_acc'] - t44_df['true_acc']
        
        summary = t44_df.groupby(['dataset', 'strategy']).agg(
            mean_est_acc=('est_acc', 'mean'),
            mean_true_acc=('true_acc', 'mean'),
            mean_bias=('bias', 'mean'),
            std_dev=('est_acc', 'std')
        ).reset_index()
        
        summary['Est. Acc.'] = summary['mean_est_acc'].map('{:.4f}'.format)
        summary['True Test Acc.'] = summary['mean_true_acc'].map('{:.4f}'.format)
        summary['Bias'] = summary['mean_bias'].map('{:+.4f}'.format)
        summary['Std Dev'] = summary['std_dev'].apply(lambda x: f"± {x:.4f}" if pd.notna(x) else "N/A")
        
        final_t44 = summary[['dataset', 'strategy', 'Est. Acc.', 'True Test Acc.', 'Bias', 'Std Dev']].copy()
        final_t44.rename(columns={'dataset': 'Dataset', 'strategy': 'Method'}, inplace=True)
        
        # Sort by Dataset and Method (Standard first)
        ds_order = {k: v for v, k in enumerate(['breast_cancer', 'telco', 'adult', 'credit'])}
        final_t44['ds_rank'] = final_t44['Dataset'].map(ds_order)
        final_t44.sort_values(['ds_rank', 'Method'], ascending=[True, False], inplace=True)
        final_t44.drop(columns=['ds_rank'], inplace=True)
        
        print(final_t44.to_string(index=False))
        final_t44.to_csv("results/table_4_4_detailed_bias_variance.csv", index=False)
    else:
        print("No data available for Table 4.4")


    # --- Table 4.5: Convergence Speed and Execution Time (Nested CV) ---
    print("\n=== Table 4.5: Convergence Speed and Execution Time (Nested CV) ===")
    
    nested_df = df[df['task'] == 'nested'].copy()
    if nested_df.empty:
        print("No nested CV data found.")
    else:
        # Filter for AMSCO and Optuna only
        target_methods = ['AMSCO', 'Optuna (TPE)']
        df_45 = nested_df[nested_df['method'].isin(target_methods)].copy()
        
        # Map method names to match image
        df_45['method'] = df_45['method'].replace({'Optuna (TPE)': 'Optuna'})
        
        datasets = df_45['dataset'].unique()
        
        rows = []
        
        # Accumulators for Average row
        avg_stats = {
            'time_optuna': [], 'time_amsco': [],
            'f1_optuna': [], 'f1_amsco': [],
            'auc_optuna': [], 'auc_amsco': [],
            'trials_optuna': [], 'trials_amsco': []
        }

        for ds in datasets:
            ds_df = df_45[df_45['dataset'] == ds]
            
            # Pivot to get paired data for t-test by (seed, model)
            # Include avg_trials_to_95 if available
            values_to_pivot = ['time', 'nested_f1', 'nested_auc']
            if 'avg_trials_to_95' in ds_df.columns:
                values_to_pivot.append('avg_trials_to_95')
                
            pivot = ds_df.pivot_table(index=['seed', 'model'], columns='method', values=values_to_pivot)
            
            if 'AMSCO' not in pivot['time'].columns or 'Optuna' not in pivot['time'].columns:
                continue
                
            time_amsco = pivot['time']['AMSCO'].dropna()
            time_optuna = pivot['time']['Optuna'].dropna()
            
            # Ensure indices match
            common_indices = time_amsco.index.intersection(time_optuna.index)
            time_amsco = time_amsco.loc[common_indices]
            time_optuna = time_optuna.loc[common_indices]
            
            if len(time_amsco) < 2:
                p_val = np.nan
            else:
                _, p_val = ttest_rel(time_optuna, time_amsco)
                
            # Means
            mean_time_amsco = time_amsco.mean()
            mean_time_optuna = time_optuna.mean()
            
            mean_f1_amsco = pivot['nested_f1']['AMSCO'].mean()
            mean_f1_optuna = pivot['nested_f1']['Optuna'].mean()
            
            mean_auc_amsco = pivot['nested_auc']['AMSCO'].mean()
            mean_auc_optuna = pivot['nested_auc']['Optuna'].mean()
            
            speedup = mean_time_optuna / mean_time_amsco if mean_time_amsco > 0 else np.nan
            
            # Trials to 99% (from pre-calculated column)
            if 'avg_trials_to_95' in pivot.columns:
                trials_optuna = pivot['avg_trials_to_95']['Optuna'].mean()
                trials_amsco = pivot['avg_trials_to_95']['AMSCO'].mean()
            else:
                trials_optuna = np.nan
                trials_amsco = np.nan
            
            avg_stats['time_optuna'].append(mean_time_optuna)
            avg_stats['time_amsco'].append(mean_time_amsco)
            avg_stats['f1_optuna'].append(mean_f1_optuna)
            avg_stats['f1_amsco'].append(mean_f1_amsco)
            avg_stats['auc_optuna'].append(mean_auc_optuna)
            avg_stats['auc_amsco'].append(mean_auc_amsco)
            if pd.notna(trials_optuna): avg_stats['trials_optuna'].append(trials_optuna)
            if pd.notna(trials_amsco): avg_stats['trials_amsco'].append(trials_amsco)

            # Row for Optuna
            rows.append({
                'Dataset': ds,
                'Method': 'Optuna',
                'Time (s)': mean_time_optuna,
                'Trials to 99%': f"{trials_optuna:.1f}" if pd.notna(trials_optuna) else "N/A", 
                'Speedup': '-',
                'F1-Score': mean_f1_optuna,
                'AUC-ROC': mean_auc_optuna,
                'P-value (Time)': '-'
            })
            
            # Row for AMSCO
            rows.append({
                'Dataset': ds,
                'Method': 'AMSCO',
                'Time (s)': mean_time_amsco,
                'Trials to 99%': f"{trials_amsco:.1f}" if pd.notna(trials_amsco) else "N/A",
                'Speedup': f"{speedup:.2f}x",
                'F1-Score': mean_f1_amsco,
                'AUC-ROC': mean_auc_amsco,
                'P-value (Time)': f"{p_val:.3f}" if pd.notna(p_val) else "N/A"
            })

        # Average Row
        if rows:
            mean_time_opt_avg = np.mean(avg_stats['time_optuna'])
            mean_time_amsco_avg = np.mean(avg_stats['time_amsco'])
            speedup_avg = mean_time_opt_avg / mean_time_amsco_avg
            
            mean_trials_opt_avg = np.mean(avg_stats['trials_optuna']) if avg_stats['trials_optuna'] else np.nan
            mean_trials_amsco_avg = np.mean(avg_stats['trials_amsco']) if avg_stats['trials_amsco'] else np.nan
            
            # P-value for average (paired t-test across all datasets)
            all_time_amsco = []
            all_time_optuna = []
            for ds in datasets:
                ds_df = df_45[df_45['dataset'] == ds]
                pivot = ds_df.pivot_table(index=['seed', 'model'], columns='method', values='time')
                if 'AMSCO' in pivot.columns and 'Optuna' in pivot.columns:
                    common = pivot.dropna()
                    all_time_amsco.extend(common['AMSCO'].values)
                    all_time_optuna.extend(common['Optuna'].values)
            
            if len(all_time_amsco) > 1:
                _, p_val_avg = ttest_rel(all_time_optuna, all_time_amsco)
                p_str_avg = f"{p_val_avg:.3f}" if p_val_avg >= 0.001 else "< 0.001"
            else:
                p_str_avg = "N/A"

            rows.append({
                'Dataset': 'Trung bình',
                'Method': 'Optuna',
                'Time (s)': mean_time_opt_avg,
                'Trials to 99%': f"{mean_trials_opt_avg:.1f}" if pd.notna(mean_trials_opt_avg) else "N/A",
                'Speedup': '-',
                'F1-Score': np.mean(avg_stats['f1_optuna']),
                'AUC-ROC': np.mean(avg_stats['auc_optuna']),
                'P-value (Time)': '-'
            })
            
            rows.append({
                'Dataset': 'Trung bình',
                'Method': 'AMSCO',
                'Time (s)': mean_time_amsco_avg,
                'Trials to 99%': f"{mean_trials_amsco_avg:.1f}" if pd.notna(mean_trials_amsco_avg) else "N/A",
                'Speedup': f"{speedup_avg:.2f}x",
                'F1-Score': np.mean(avg_stats['f1_amsco']),
                'AUC-ROC': np.mean(avg_stats['auc_amsco']),
                'P-value (Time)': p_str_avg
            })

        t45_df = pd.DataFrame(rows)
        
        # Formatting
        t45_df['Time (s)'] = t45_df['Time (s)'].map('{:,.0f}'.format)
        t45_df['F1-Score'] = t45_df['F1-Score'].map('{:.4f}'.format)
        t45_df['AUC-ROC'] = t45_df['AUC-ROC'].map('{:.4f}'.format)
        
        print(t45_df.to_string(index=False))
        t45_df.to_csv("results/table_4_5_convergence_speed.csv", index=False)


    # --- Table 4.6: Comprehensive Objective Function Score ---
    print("\n=== Table 4.6: Comprehensive Objective Function Score ===")
    
    nested_df = df[df['task'] == 'nested'].copy()
    if nested_df.empty:
        print("No nested CV data found.")
    else:
        # Filter for AMSCO and Optuna only
        target_methods = ['AMSCO', 'Optuna (TPE)']
        df_46 = nested_df[nested_df['method'].isin(target_methods)].copy()
        df_46['method'] = df_46['method'].replace({'Optuna (TPE)': 'Optuna'})
        
        # Dataset Name Mapping
        ds_map = {
            'breast_cancer': 'Breast Cancer',
            'telco': 'Telco Churn',
            'adult': 'Adult Income',
            'creditcard': 'Credit Fraud'
        }
        
        datasets = df_46['dataset'].unique()
        rows = []
        
        # Weights for the composite score
        W_F1 = 0.4
        W_AUC = 0.4
        W_TIME = 0.2
        
        avg_stats = {
            'f1_norm_opt': [], 'f1_norm_amsco': [],
            'auc_norm_opt': [], 'auc_norm_amsco': [],
            'time_norm_opt': [], 'time_norm_amsco': [],
            'score_opt': [], 'score_amsco': []
        }

        for ds in datasets:
            ds_df = df_46[df_46['dataset'] == ds]
            ds_name = ds_map.get(ds, ds)
            
            # Calculate Means
            means = ds_df.groupby('method')[['nested_f1', 'nested_auc', 'time']].mean()
            
            if 'AMSCO' not in means.index or 'Optuna' not in means.index:
                continue
                
            f1_amsco = means.loc['AMSCO', 'nested_f1']
            f1_optuna = means.loc['Optuna', 'nested_f1']
            
            auc_amsco = means.loc['AMSCO', 'nested_auc']
            auc_optuna = means.loc['Optuna', 'nested_auc']
            
            time_amsco = means.loc['AMSCO', 'time']
            time_optuna = means.loc['Optuna', 'time']
            
            # Normalize
            max_f1 = max(f1_amsco, f1_optuna)
            max_auc = max(auc_amsco, auc_optuna)
            max_time = max(time_amsco, time_optuna)
            
            f1_norm_amsco = f1_amsco / max_f1
            f1_norm_optuna = f1_optuna / max_f1
            
            auc_norm_amsco = auc_amsco / max_auc
            auc_norm_optuna = auc_optuna / max_auc
            
            time_norm_amsco = time_amsco / max_time
            time_norm_optuna = time_optuna / max_time
            
            # Calculate Score f(theta)
            # Formula: w1*F1_norm + w2*AUC_norm + w3*(1 - Time_norm)
            score_amsco = W_F1 * f1_norm_amsco + W_AUC * auc_norm_amsco + W_TIME * (1 - time_norm_amsco)
            score_optuna = W_F1 * f1_norm_optuna + W_AUC * auc_norm_optuna + W_TIME * (1 - time_norm_optuna)
            
            # Rank
            rank_amsco = 1 if score_amsco >= score_optuna else 2
            rank_optuna = 1 if score_optuna > score_amsco else 2
            
            rank_str_amsco = f"{rank_amsco} (*)" if rank_amsco == 1 else f"{rank_amsco}"
            rank_str_optuna = f"{rank_optuna} (*)" if rank_optuna == 1 else f"{rank_optuna}"
            
            # Collect for Average
            avg_stats['f1_norm_opt'].append(f1_norm_optuna)
            avg_stats['f1_norm_amsco'].append(f1_norm_amsco)
            avg_stats['auc_norm_opt'].append(auc_norm_optuna)
            avg_stats['auc_norm_amsco'].append(auc_norm_amsco)
            avg_stats['time_norm_opt'].append(time_norm_optuna)
            avg_stats['time_norm_amsco'].append(time_norm_amsco)
            avg_stats['score_opt'].append(score_optuna)
            avg_stats['score_amsco'].append(score_amsco)

            # Row Optuna
            rows.append({
                'Dataset': ds_name,
                'Method': 'Optuna',
                'F1-Score (Norm.)': f"{f1_optuna:.4f} ({f1_norm_optuna:.3f})",
                'AUC-ROC (Norm.)': f"{auc_optuna:.4f} ({auc_norm_optuna:.3f})",
                'Time (Norm.)': f"{time_optuna:,.0f}s ({time_norm_optuna:.3f})",
                'f(theta)': f"{score_optuna:.3f}",
                'Rank': rank_str_optuna
            })
            
            # Row AMSCO
            rows.append({
                'Dataset': ds_name,
                'Method': 'AMSCO',
                'F1-Score (Norm.)': f"{f1_amsco:.4f} ({f1_norm_amsco:.3f})",
                'AUC-ROC (Norm.)': f"{auc_amsco:.4f} ({auc_norm_amsco:.3f})",
                'Time (Norm.)': f"{time_amsco:,.0f}s ({time_norm_amsco:.3f})",
                'f(theta)': f"{score_amsco:.3f}",
                'Rank': rank_str_amsco
            })
            
        # Average Row
        if rows:
            score_opt_avg = np.mean(avg_stats['score_opt'])
            score_amsco_avg = np.mean(avg_stats['score_amsco'])
            
            rank_avg_amsco = 1 if score_amsco_avg >= score_opt_avg else 2
            rank_avg_optuna = 1 if score_opt_avg > score_amsco_avg else 2
            
            rows.append({
                'Dataset': 'Trung bình',
                'Method': 'Optuna',
                'F1-Score (Norm.)': '-',
                'AUC-ROC (Norm.)': '-',
                'Time (Norm.)': '-',
                'f(theta)': f"{score_opt_avg:.3f}",
                'Rank': f"{rank_avg_optuna} (*)" if rank_avg_optuna == 1 else f"{rank_avg_optuna}"
            })
            
            rows.append({
                'Dataset': 'Trung bình',
                'Method': 'AMSCO',
                'F1-Score (Norm.)': '-',
                'AUC-ROC (Norm.)': '-',
                'Time (Norm.)': '-',
                'f(theta)': f"{score_amsco_avg:.3f}",
                'Rank': f"{rank_avg_amsco} (*)" if rank_avg_amsco == 1 else f"{rank_avg_amsco}"
            })

        t46_df = pd.DataFrame(rows)
        print(t46_df.to_string(index=False))
        t46_df.to_csv("results/table_4_6_comprehensive_score.csv", index=False)

    # --- Table 4.7: Variance Comparison (Stability) ---
    print("\n=== Table 4.7: Variance Comparison of F1-Score (Stability) ===")
    
    nested_df = df[df['task'] == 'nested'].copy()
    if nested_df.empty:
        print("No nested CV data found.")
    else:
        target_methods = ['AMSCO', 'Optuna (TPE)']
        df_47 = nested_df[nested_df['method'].isin(target_methods)].copy()
        df_47['method'] = df_47['method'].replace({'Optuna (TPE)': 'Optuna'})
        
        # Dataset Name Mapping
        ds_map = {
            'breast_cancer': 'Breast Cancer',
            'telco': 'Telco Churn',
            'adult': 'Adult Income',
            'creditcard': 'Credit Fraud'
        }
        
        datasets = df_47['dataset'].unique()
        rows = []
        
        for ds in datasets:
            ds_df = df_47[df_47['dataset'] == ds]
            ds_name = ds_map.get(ds, ds)
            
            # Pivot to get paired data by seed
            pivot = ds_df.pivot_table(index='seed', columns='method', values='nested_f1')
            
            if 'AMSCO' not in pivot.columns or 'Optuna' not in pivot.columns:
                continue
                
            # Drop seeds where one method is missing
            pivot = pivot.dropna()
            
            f1_amsco = pivot['AMSCO']
            f1_optuna = pivot['Optuna']
            
            if len(f1_amsco) < 2:
                print(f"Not enough seeds for {ds_name} to calculate variance.")
                continue
                
            # Calculate Variance across seeds
            var_amsco = f1_amsco.var(ddof=1)
            var_optuna = f1_optuna.var(ddof=1)
            
            # Improvement
            improvement = (var_optuna - var_amsco) / var_optuna * 100
            
            # Wilcoxon p-value (comparing the distributions of F1 scores)
            # Note: With few seeds (e.g. 3), this p-value is not very powerful but we calculate it as requested.
            try:
                _, p_val = wilcoxon(f1_optuna, f1_amsco)
            except Exception:
                p_val = np.nan
                
            rows.append({
                'Dataset': ds_name,
                'Optuna Var': f"{var_optuna:.6f}",
                'AMSCO Var': f"{var_amsco:.6f}",
                'Improvement': f"✓ {improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%",
                'Wilcoxon p-value': f"{p_val:.3f}" if pd.notna(p_val) else "N/A"
            })
            
        t47_df = pd.DataFrame(rows)
        print(t47_df.to_string(index=False))
        t47_df.to_csv("results/table_4_7_variance_comparison.csv", index=False)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    df = load_results()
    generate_tables(df)
