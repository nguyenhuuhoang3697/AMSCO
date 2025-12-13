#!/usr/bin/env python
# coding: utf-8
"""
Script demo để test Multi-Objective Optimization trong AMSCO
f(θ) = α·F1 + β·ROC-AUC - γ·execution_time

Thiết lập:
- α = 0.4 (trọng số F1)
- β = 0.4 (trọng số ROC-AUC)  
- γ = 0.2 (penalty cho execution time)
"""

import os
import sys
import warnings
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Import các components từ amsco_adult_v2
import importlib.util
spec = importlib.util.spec_from_file_location("amsco", "/home/user/AMSCO/amsco_adult_v2.py")
amsco_module = importlib.util.module_from_spec(spec)
sys.modules["amsco"] = amsco_module
spec.loader.exec_module(amsco_module)

# Import các hàm cần thiết
MultiObjectiveNormalizer = amsco_module.MultiObjectiveNormalizer
create_objective = amsco_module.create_objective
run_amsco_optimizer = amsco_module.run_amsco_optimizer
run_optuna_tpe = amsco_module.run_optuna_tpe
MASTER_SEARCH_SPACES = amsco_module.MASTER_SEARCH_SPACES
preprocess_adult_columns = amsco_module.preprocess_adult_columns

print("="*70)
print(" "*10 + "SO SÁNH AMSCO vs OPTUNA (TPE)")
print(" "*15 + "MULTI-OBJECTIVE OPTIMIZATION")
print("="*70)
print("\nHàm mục tiêu: f(θ) = α·F1 + β·ROC-AUC - γ·execution_time")
print(f"Tham số: α=0.4, β=0.4, γ=0.2")
print(f"\nMục tiêu: Tối đa hóa f(θ)")
print(f"So sánh: Optimizer nào tìm được f(θ) lớn hơn?\n")

# Load dataset
print("1. Đang tải Breast Cancer dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target.astype(int))
print(f"   ✓ Dataset shape: {X.shape}, Classes: {y.value_counts().to_dict()}")

# Preprocessing
print("\n2. Tiền xử lý dữ liệu...")
X_processed, preprocessor = preprocess_adult_columns(X)
print("   ✓ Preprocessor created")

# Cấu hình
model_name = 'logistic_regression'
search_space = MASTER_SEARCH_SPACES[model_name]
metrics = ('f1', 'roc_auc')  # Primary metric = f1
n_trials = 50  # Số trials cho mỗi optimizer
slice_budget = 10  # Budget cho mỗi slice của AMSCO

print(f"\n3. Cấu hình thử nghiệm:")
print(f"   - Model: Logistic Regression")
print(f"   - Cross-validation: 3-fold")
print(f"   - Số trials: {n_trials}")
print(f"   - AMSCO slice budget: {slice_budget}")

# === TẠO NORMALIZER CHUNG ===
# Cả 2 optimizer sẽ dùng chung normalizer để công bằng
print("\n4. Khởi tạo Multi-Objective Normalizer chung...")
normalizer = MultiObjectiveNormalizer(alpha=0.4, beta=0.4, gamma=0.2)

objective_shared = create_objective(
    X_processed, y,
    model_name=model_name,
    preprocessor=preprocessor,
    metrics=metrics,
    use_cross_validation=True,
    cv_folds=3,
    sampler=None,
    multi_objective=True,
    normalizer=normalizer
)

# === TEST 1: OPTUNA TPE ===
print("\n" + "="*70)
print("TEST 1: OPTUNA (TPE)")
print("="*70)

print("\n  [Optuna TPE] Đang chạy tối ưu hóa...")
import time
start_optuna = time.time()
score_optuna, params_optuna, diag_optuna = run_optuna_tpe(
    objective_shared,
    search_space,
    n_trials=n_trials,
    normalizer=normalizer
)
time_optuna = time.time() - start_optuna

print(f"\n  ✓ Best f(θ): {score_optuna:.6f}")
print(f"  ✓ Thời gian: {time_optuna:.2f}s")
print(f"  ✓ Số trials: {diag_optuna.get('total_trials', n_trials)}")
print(f"  ✓ Best params: {params_optuna}")

# === TEST 2: AMSCO ===
print("\n" + "="*70)
print("TEST 2: AMSCO FRAMEWORK")
print("="*70)

print("\n  [AMSCO] Đang chạy tối ưu hóa...")
# Reset normalizer để công bằng (hoặc dùng cùng bounds)
# Ở đây ta sẽ dùng bounds đã tính từ Optuna để công bằng
normalizer_amsco = MultiObjectiveNormalizer(alpha=0.4, beta=0.4, gamma=0.2)
normalizer_amsco.set_bounds(
    normalizer.f1_min, normalizer.f1_max,
    normalizer.auc_min, normalizer.auc_max,
    normalizer.time_min, normalizer.time_max
)

objective_amsco = create_objective(
    X_processed, y,
    model_name=model_name,
    preprocessor=preprocessor,
    metrics=metrics,
    use_cross_validation=True,
    cv_folds=3,
    sampler=None,
    multi_objective=True,
    normalizer=normalizer_amsco
)

start_amsco = time.time()
score_amsco, params_amsco, diag_amsco = run_amsco_optimizer(
    objective_amsco,
    search_space,
    n_trials=n_trials,
    slice_budget=slice_budget,
    verbose=False,
    early_stopping_rounds=3,
    tolerance=1e-4,
    seed=42,
    normalizer=None  # Không cần warm-up lại, đã có bounds
)
time_amsco = time.time() - start_amsco

print(f"\n  ✓ Best f(θ): {score_amsco:.6f}")
print(f"  ✓ Thời gian: {time_amsco:.2f}s")
print(f"  ✓ Số trials: {diag_amsco.get('total_trials', n_trials)}")
print(f"  ✓ Agent pulls: {diag_amsco.get('agent_pulls', {})}")
print(f"  ✓ Best params: {params_amsco}")

# === SO SÁNH KẾT QUẢ ===
print("\n" + "="*70)
print("SO SÁNH KẾT QUẢ")
print("="*70)

comparison_data = {
    'Optimizer': ['Optuna TPE', 'AMSCO'],
    'Best f(θ)': [score_optuna, score_amsco],
    'Time (s)': [time_optuna, time_amsco],
    'Total Trials': [
        diag_optuna.get('total_trials', n_trials),
        diag_amsco.get('total_trials', n_trials)
    ]
}

df_comparison = pd.DataFrame(comparison_data)
print("\n" + df_comparison.to_string(index=False))

# Xác định winner
winner = "AMSCO" if score_amsco > score_optuna else "Optuna TPE"
improvement = abs(score_amsco - score_optuna)
improvement_pct = (improvement / max(score_optuna, score_amsco)) * 100

print(f"\n{'='*70}")
print(f"🏆 WINNER: {winner}")
print(f"{'='*70}")
print(f"• {winner} đạt f(θ) cao hơn: {max(score_amsco, score_optuna):.6f}")
print(f"• Chênh lệch: {improvement:.6f} ({improvement_pct:.2f}%)")
print(f"• Tốc độ: AMSCO {time_amsco:.2f}s vs Optuna {time_optuna:.2f}s")

if score_amsco > score_optuna:
    print(f"\n✅ AMSCO framework vượt trội nhờ:")
    print(f"   - Kết hợp đa chiến lược (Random, Bayesian, Grid)")
    print(f"   - Phân bổ ngân sách thông minh (UCB1)")
    print(f"   - Chia sẻ tri thức qua Knowledge Hub")
    agent_pulls = diag_amsco.get('agent_pulls', {})
    if agent_pulls:
        print(f"   - Agent usage: {agent_pulls}")
else:
    print(f"\n✅ Optuna TPE vượt trội trong trường hợp này")
    print(f"   - TPE sampler hiệu quả với không gian tìm kiếm đơn giản")
    print(f"   - Ít overhead hơn AMSCO")

print(f"\n{'='*70}")
print("ĐÁNH GIÁ TRÊN TEST SET")
print(f"{'='*70}")

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

def evaluate_on_test(params, label, normalizer_ref):
    """Đánh giá tham số trên test set"""
    model = LogisticRegression(random_state=42, max_iter=2000)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.set_params(**params)
    
    start = time.time()
    pipeline.fit(X_train_full, y_train_full)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    exec_time = time.time() - start
    
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Tính composite score
    f_theta = normalizer_ref.compute_objective(f1, roc_auc, exec_time)
    
    print(f"\n{label}:")
    print(f"  F1-score:       {f1:.4f}")
    print(f"  ROC-AUC:        {roc_auc:.4f}")
    print(f"  Execution time: {exec_time:.4f}s")
    print(f"  f(θ) composite: {f_theta:.6f}")
    
    return f1, roc_auc, exec_time, f_theta

print("\nSử dụng params tối ưu từ mỗi optimizer:")
f1_opt, auc_opt, time_opt, theta_opt = evaluate_on_test(
    params_optuna, 
    "Optuna TPE Best Params",
    normalizer
)
f1_ams, auc_ams, time_ams, theta_ams = evaluate_on_test(
    params_amsco,
    "AMSCO Best Params", 
    normalizer_amsco
)

print(f"\n{'='*70}")
print("KẾT LUẬN CUỐI CÙNG")
print(f"{'='*70}")

test_winner = "AMSCO" if theta_ams > theta_opt else "Optuna TPE"
print(f"\n🏆 Trên Test Set: {test_winner} cho f(θ) tốt hơn")
print(f"   - Optuna: f(θ) = {theta_opt:.6f}")
print(f"   - AMSCO:  f(θ) = {theta_ams:.6f}")
print(f"\n💡 AMSCO framework:")
print(f"   - Tối ưu đa mục tiêu: f(θ) = 0.4·F1 + 0.4·ROC-AUC - 0.2·time")
print(f"   - Cân bằng giữa accuracy, discrimination power và tốc độ")
print(f"   - Phù hợp cho production deployment")
print("="*70)
