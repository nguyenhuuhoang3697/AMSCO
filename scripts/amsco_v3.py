#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Early warning suppression to silence hyperopt/pkg_resources
import os
import warnings
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings(
    "ignore", message=r".*pkg_resources is deprecated as an API.*", category=UserWarning
)


import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
import math
import time
import random
import tracemalloc
import resource
from collections import defaultdict
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTE

# Tắt các cảnh báo không cần thiết
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# In[2]:


# ============================================================================
# BƯỚC 1: ĐỊNH NGHĨA KHÔNG GIAN TÌM KIẾM (SEARCH SPACES)
# Định nghĩa các không gian tìm kiếm cho cả 4 mô hình.
# Thu hẹp phạm vi theo adult_multi_model_nested_cv.py để chạy nhanh hơn.
# ============================================================================

MASTER_SEARCH_SPACES = {
    'logistic_regression': {
        'classifier__C': ('float', 1e-3, 1e2, 'log'),
        'classifier__penalty': ('categorical', ['l2']),
        'classifier__solver': ('categorical', ['liblinear'])
    },
    'random_forest': {
        'classifier__n_estimators': ('int', 50, 200),
        'classifier__max_depth': ('int', 5, 30),
        'classifier__min_samples_split': ('int', 2, 10),
        'classifier__min_samples_leaf': ('int', 1, 4),
        'classifier__max_features': ('categorical', ['sqrt', 'log2'])
    },
    'xgboost': {
        'classifier__n_estimators': ('int', 50, 200),
        'classifier__max_depth': ('int', 3, 8),
        'classifier__learning_rate': ('float', 0.01, 0.3, 'log'),
        'classifier__subsample': ('float', 0.6, 1.0),
        'classifier__colsample_bytree': ('float', 0.6, 1.0)
    },
    'lightgbm': {
        'classifier__n_estimators': ('int', 50, 200),
        'classifier__num_leaves': ('int', 20, 120),
        'classifier__max_depth': ('int', 3, 12),
        'classifier__learning_rate': ('float', 0.01, 0.3, 'log'),
        'classifier__subsample': ('float', 0.6, 1.0),
        'classifier__colsample_bytree': ('float', 0.6, 1.0)
    }
}


# In[3]:


# =============================================================================
# BƯỚC 2: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU (Adult, Breast Cancer, Telco)
# - Adult và Telco từ CSV nội bộ trong thư mục datasets/
# - Breast Cancer từ sklearn.datasets
# =============================================================================

from pathlib import Path


def make_one_hot_encoder():
    """Tạo OneHotEncoder tương thích với cả sklearn>=1.2 và phiên bản cũ."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=True)


def load_adult_dataset(csv_path: Path):
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
    df = pd.read_csv(csv_path)
    # Cố gắng suy đoán cột target thường gặp
    target_candidates = ['Churn', 'churn', 'target', 'label', 'y']
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError("Không tìm thấy cột đích trong telco.csv (kỳ vọng một trong: Churn/churn/target/label/y)")
    y_raw = df[target_col]
    if y_raw.dtype == 'O' or str(y_raw.dtype).startswith('category'):
        # map Yes/No hoặc True/False
        mapping = {"Yes": 1, "No": 0, "True": 1, "False": 0, "Y": 1, "N": 0}
        y = y_raw.map(lambda v: mapping.get(str(v).strip(), np.nan)).astype(float)
        if y.isna().any():
            # Thử map nhị phân bất kỳ khác
            uniques = sorted(y_raw.dropna().unique().tolist())
            if len(uniques) == 2:
                y = (y_raw == uniques[1]).astype(int)
            else:
                raise ValueError("Cột target của telco không phải nhị phân. Vui lòng chuẩn hóa về 0/1 hoặc Yes/No.")
        else:
            y = y.astype(int)
    else:
        y = y_raw.astype(int)
    X = df.drop(columns=[target_col])
    return X, y


def preprocess_adult_columns(X: pd.DataFrame):
    # Hàm tiền xử lý tổng quát dựa trên kiểu dữ liệu; giữ tên cũ cho tương thích
    numeric_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Xử lý gộp nhóm quốc gia hiếm chỉ khi có cột này (Adult)
    if 'native-country' in X.columns:
        value_counts = X['native-country'].value_counts(dropna=False)
        rare_mask = X['native-country'].isin(value_counts[value_counts < 100].index)
        X.loc[rare_mask, 'native-country'] = 'Other'

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('scaler', StandardScaler())]), numeric_cols),
            ('cat', Pipeline([('encoder', make_one_hot_encoder())]), categorical_cols),
        ]
    )
    return X, preprocessor


def load_credit_dataset(csv_path: Path):
    """Tải và tiền xử lý dữ liệu Credit Card Fraud."""
    df = pd.read_csv(csv_path)

    if 'Amount' in df.columns:
        scaler = StandardScaler()
        df['normAmount'] = scaler.fit_transform(df[['Amount']])
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])
    if 'Amount' in df.columns:
        df = df.drop(columns=['Amount'])

    if 'Class' not in df.columns:
        raise ValueError("Không tìm thấy cột 'Class' trong creditcard.csv")

    y = df['Class'].astype(int)
    X = df.drop(columns=['Class'])

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name='Class')

    return X_resampled, y_resampled


def get_data(dataset_name, quiet=False):
    dataset_name = dataset_name.lower()
    if dataset_name == 'adult':
        csv_path = Path('datasets') / 'adult.csv'
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {csv_path}. Vui lòng đặt adult.csv vào thư mục datasets/ như README hướng dẫn."
            )
        if not quiet:
            print("... Đang tải Adult Income")
        X, y = load_adult_dataset(csv_path)
        X, preprocessor = preprocess_adult_columns(X)
        metric = 'accuracy'
        return X, y, preprocessor, metric

    if dataset_name == 'breast_cancer':
        if not quiet:
            print("... Đang tải Breast Cancer (sklearn)")
        X, y = load_breast_cancer_dataset()
        X, preprocessor = preprocess_adult_columns(X)
        metric = 'accuracy'
        return X, y, preprocessor, metric

    if dataset_name == 'telco':
        csv_path = Path('datasets') / 'telco.csv'
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {csv_path}. Vui lòng đặt telco.csv vào thư mục datasets/ như README hướng dẫn."
            )
        if not quiet:
            print("... Đang tải Telco Customer Churn")
        X, y = load_telco_dataset(csv_path)
        X, preprocessor = preprocess_adult_columns(X)
        metric = 'accuracy'
        return X, y, preprocessor, metric

    if dataset_name == 'credit':
        csv_path = Path('datasets') / 'creditcard.csv'
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {csv_path}. Vui lòng đặt creditcard.csv vào thư mục datasets/ như README hướng dẫn."
            )
        if not quiet:
            print("... Đang tải Credit Card Fraud")
        X, y = load_credit_dataset(csv_path)
        X, preprocessor = preprocess_adult_columns(X)
        metric = 'roc_auc'
        return X, y, preprocessor, metric

    raise ValueError("Notebook này chỉ hỗ trợ dataset 'adult', 'breast_cancer', 'telco', hoặc 'credit'.")


# In[4]:


# =============================================================================
# BƯỚC 3: HÀM MỤC TIÊU (OBJECTIVE FUNCTION) TỔNG QUÁT
# - Hỗ trợ nhiều metric: accuracy, f1, roc_auc (roc_auc chỉ hoạt động khi dữ liệu nhị phân)
# =============================================================================

def create_objective(
    X,
    y,
    model_name,
    preprocessor,
    metrics=('accuracy',),  # tuple/list các metric cần tính
    use_cross_validation=True,
    validation_data=None,
    cv_folds=3
):
    """
    Xây dựng hàm objective cho optimizer.

    Nếu `use_cross_validation=True`, sử dụng StratifiedKFold (cv_folds) và trả về metric đầu tiên (primary) làm giá trị tối ưu.
    Các metric khác sẽ được tính và trả kèm (có thể lưu để phân tích, nhưng tối ưu vẫn dựa trên primary metric).

    Nếu `use_cross_validation=False`, yêu cầu `validation_data` = (X_valid, y_valid) để tính trên holdout.
    Primary metric = metrics[0].
    """

    if not metrics:
        raise ValueError("Phải cung cấp ít nhất một metric.")
    primary_metric = metrics[0]

    if not use_cross_validation and validation_data is None:
        raise ValueError("validation_data phải được cung cấp khi tắt cross validation.")

    def _build_classifier(name):
        if name == 'logistic_regression':
            return LogisticRegression(random_state=42, max_iter=2000)
        if name == 'random_forest':
            return RandomForestClassifier(random_state=42, n_jobs=-1)
        if name == 'xgboost':
            return xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        if name == 'lightgbm':
            return lgb.LGBMClassifier(random_state=42, verbosity=-1)
        raise ValueError(f"Mô hình '{name}' không được hỗ trợ.")

    def _normalize_params(name, params):
        p = dict(params)
        if name == 'logistic_regression':
            penalty = p.get('classifier__penalty', 'l2')
            solver = p.get('classifier__solver', 'liblinear')
            # Nếu elasticnet -> buộc dùng saga và yêu cầu l1_ratio
            if penalty == 'elasticnet':
                p['classifier__solver'] = 'saga'
                if 'classifier__l1_ratio' not in p:
                    p['classifier__l1_ratio'] = 0.5
            else:
                # Nếu không phải elasticnet thì loại bỏ l1_ratio nếu được gợi ý
                p.pop('classifier__l1_ratio', None)
            # liblinear không hỗ trợ elasticnet
            if solver == 'liblinear' and penalty == 'elasticnet':
                p['classifier__solver'] = 'saga'
        return p

    if not use_cross_validation:
        assert validation_data is not None
        X_valid, y_valid = validation_data

    def compute_metrics(y_true, y_pred, y_proba=None):
        results = {}
        for m in metrics:
            if m == 'accuracy':
                results['accuracy'] = accuracy_score(y_true, y_pred)
            elif m == 'f1':
                results['f1'] = f1_score(y_true, y_pred, average='binary')
            elif m == 'roc_auc':
                if y_proba is None:
                    results['roc_auc'] = np.nan
                else:
                    prob = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    try:
                        results['roc_auc'] = roc_auc_score(y_true, prob)
                    except Exception:
                        results['roc_auc'] = np.nan
            else:
                results[m] = np.nan
        return results

    def objective(params):
        model = _build_classifier(model_name)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        # Chuẩn hóa tham số cho các trường hợp đặc biệt (vd: LogisticRegression)
        params = _normalize_params(model_name, params)
        pipeline.set_params(**params)

        try:
            if use_cross_validation:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores_primary = []
                for train_idx, test_idx in cv.split(X, y):
                    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                    pipeline.fit(X_tr, y_tr)
                    y_pred = pipeline.predict(X_te)
                    y_proba = None
                    if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                        y_proba = pipeline.predict_proba(X_te)
                    fold_metrics = compute_metrics(y_te, y_pred, y_proba)
                    scores_primary.append(fold_metrics[primary_metric])
                score = np.mean(scores_primary)
            else:
                # Đảm bảo biến holdout đã được thiết lập
                assert 'X_valid' in locals() or 'X_valid' in globals()
                assert 'y_valid' in locals() or 'y_valid' in globals()
                pipeline.fit(X, y)
                y_pred = pipeline.predict(X_valid)
                y_proba = None
                if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                    y_proba = pipeline.predict_proba(X_valid)
                holdout_metrics = compute_metrics(y_valid, y_pred, y_proba)
                score = holdout_metrics[primary_metric]
        except Exception as e:
            print(f"Lỗi khi đánh giá {params}: {e}")
            return 0.0

        return score

    # Stepwise reporting để hỗ trợ Optuna Pruner
    def objective_stepwise(trial, params):
        model = _build_classifier(model_name)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        params = _normalize_params(model_name, dict(params))
        pipeline.set_params(**params)

        try:
            if use_cross_validation:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores_primary = []
                for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                    pipeline.fit(X_tr, y_tr)
                    y_pred = pipeline.predict(X_te)
                    y_proba = None
                    if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                        y_proba = pipeline.predict_proba(X_te)
                    fold_metrics = compute_metrics(y_te, y_pred, y_proba)
                    fold_score = float(fold_metrics[primary_metric])
                    scores_primary.append(fold_score)
                    # Báo cáo kết quả trung gian theo từng fold
                    try:
                        trial.report(fold_score, step=fold_idx)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                    except Exception:
                        # Nếu trial không hỗ trợ hoặc Optuna không sẵn có
                        pass
                return float(np.mean(scores_primary)) if scores_primary else 0.0
            else:
                assert 'X_valid' in locals() or 'X_valid' in globals()
                assert 'y_valid' in locals() or 'y_valid' in globals()
                pipeline.fit(X, y)
                y_pred = pipeline.predict(X_valid)
                y_proba = None
                if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                    y_proba = pipeline.predict_proba(X_valid)
                holdout_metrics = compute_metrics(y_valid, y_pred, y_proba)
                score = float(holdout_metrics[primary_metric])
                try:
                    trial.report(score, step=0)
                except Exception:
                    pass
                return score
        except optuna.exceptions.TrialPruned:
            # Bubbles up pruning
            raise
        except Exception as e:
            print(f"Lỗi khi đánh giá (stepwise) {params}: {e}")
            return 0.0

    # Gắn stepwise như thuộc tính của objective để agent có thể dùng
    try:
        objective._stepwise = objective_stepwise
    except Exception:
        pass

    return objective


# In[5]:


# ============================================================================
# BƯỚC 4: FRAMEWORK AMSCO (PHIÊN BẢN TỔNG QUÁT VÀ ĐÃ SỬA LỖI)
# ============================================================================

class KnowledgeHub:
    def __init__(self):
        self.trials = []
        self.best_score = -float('inf')
        self.best_params = None
        self.total_calls = 0
        self.best_iteration = None

    def store(self, agent_id, params, score):
        self.total_calls += 1
        record = {
            'iteration': self.total_calls,
            'agent_id': agent_id,
            'params': params,
            'score': score
        }
        self.trials.append(record)
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
            self.best_iteration = self.total_calls
            # print(f"  [KnowledgeHub] New best score: {self.best_score:.4f} from {agent_id}")

    def get_all_trials(self):
        return self.trials

    def get_best_trial(self):
        return {
            'params': self.best_params,
            'score': self.best_score,
            'iteration': self.best_iteration
        }

class StrategyAgent:
    """Lớp cơ sở, giờ nhận objective và search_space"""
    def __init__(self, agent_id, objective_func, search_space, knowledge_hub):
        self.agent_id = agent_id
        self.objective = objective_func
        self.search_space = search_space
        self.knowledge_hub = knowledge_hub

    def run(self, budget):
        raise NotImplementedError

class RandomAgent(StrategyAgent):
    """RandomAgent: Giờ đã hoàn toàn linh hoạt"""
    def run(self, budget):
        # print(f"    -> Running RandomAgent with budget: {budget}")
        for _ in range(budget):
            params = {}

            for name, details in self.search_space.items():
                type = details[0]  # Type là phần tử đầu tiên của tuple
                if type == 'float':
                    low, high = details[1], details[2]
                    dist_type = details[3] if len(details) > 3 else None
                    if dist_type == 'log':
                        params[name] = np.exp(random.uniform(np.log(low), np.log(high)))
                    else:
                        params[name] = random.uniform(low, high)
                elif type == 'int':
                    low, high = details[1], details[2]
                    params[name] = random.randint(low, high)
                elif type == 'categorical':
                    choices = details[1] # Lấy danh sách lựa chọn
                    params[name] = random.choice(choices)

            score = self.objective(params)
            self.knowledge_hub.store(self.agent_id, params, score)

class BayesianAgent(StrategyAgent):
    """BayesianAgent: Giờ đã hoàn toàn linh hoạt"""
    def run(self, budget):
        # print(f"    -> Running BayesianAgent with budget: {budget}")
        # (A) Objective Optuna tự động
        def optuna_objective(trial):
            params = {}
            for name, details in self.search_space.items():
                p_type = details[0]
                if p_type == 'float':
                    low, high = details[1], details[2]
                    dist_type = details[3] if len(details) > 3 else None
                    params[name] = trial.suggest_float(name, low, high, log=(dist_type == 'log'))
                elif p_type == 'int':
                    low, high = details[1], details[2]
                    params[name] = trial.suggest_int(name, low, high)
                elif p_type == 'categorical':
                    choices = details[1]
                    params[name] = trial.suggest_categorical(name, choices)
            # Dùng stepwise nếu có để pruner hoạt động
            if hasattr(self.objective, "_stepwise"):
                score = self.objective._stepwise(trial, params)
            else:
                score = self.objective(params)
            self.knowledge_hub.store(self.agent_id, params, score)
            return score

        # (B) Sampler nâng cấp + pruner (fallback nếu phiên bản không hỗ trợ)
        try:
            sampler = optuna.samplers.TPESampler(
                seed=42,
                multivariate=True,
                constant_liar=True,
                n_startup_trials=min(10, budget)
            )
        except Exception:
            sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

        existing_trials = self.knowledge_hub.get_all_trials()
        if existing_trials:
            full_distributions = {}
            for name, details in self.search_space.items():
                t = details[0]
                if t == 'float':
                    low, high = details[1], details[2]
                    dist_type = details[3] if len(details) > 3 else None
                    full_distributions[name] = optuna.distributions.FloatDistribution(low, high, log=(dist_type == 'log'))
                elif t == 'int':
                    low, high = details[1], details[2]
                    full_distributions[name] = optuna.distributions.IntDistribution(low, high)
                elif t == 'categorical':
                    choices = details[1]
                    full_distributions[name] = optuna.distributions.CategoricalDistribution(choices)

            filtered = [t for t in existing_trials if 'params' in t and 'score' in t and isinstance(t['score'], (int, float)) and not math.isnan(t['score'])]
            filtered.sort(key=lambda r: r['score'], reverse=True)
            if len(filtered) > 300:
                top_part = filtered[:150]
                remaining = filtered[150:]
                random_part = random.sample(remaining, min(50, len(remaining))) if remaining else []
                selected = top_part + random_part
            else:
                selected = filtered

            seen_param_keys = set()
            for t in selected:
                params_dict = t['params']
                valid_params = {k: v for k, v in params_dict.items() if k in full_distributions}
                if not valid_params:
                    continue
                key_tuple = tuple(sorted(valid_params.items()))
                if key_tuple in seen_param_keys:
                    continue
                seen_param_keys.add(key_tuple)
                sub_distributions = {k: full_distributions[k] for k in valid_params.keys()}
                try:
                    frozen_trial = optuna.trial.create_trial(
                        params=valid_params,
                        distributions=sub_distributions,
                        value=t['score']
                    )
                    study.add_trial(frozen_trial)
                except Exception:
                    continue

            # Perturbation quanh best
            best_record = self.knowledge_hub.get_best_trial()
            best_params = best_record.get('params') if best_record else None
            best_score = best_record.get('score') if best_record else None
            if best_params and isinstance(best_score, (int, float)) and math.isfinite(best_score):
                perturb_variants = []
                numeric_names = []
                for name, details in self.search_space.items():
                    if details[0] in ['float', 'int'] and name in best_params:
                        numeric_names.append(name)
                numeric_names = numeric_names[:3]
                for name in numeric_names:
                    details = self.search_space[name]
                    t = details[0]
                    low, high = details[1], details[2]
                    current_val = best_params.get(name)
                    if current_val is None:
                        continue
                    cand = []
                    if t == 'float':
                        for factor in [0.95, 1.0, 1.05]:
                            v = max(low, min(high, current_val * factor))
                            cand.append(v)
                    else:  # int
                        for d in [-1, 0, 1]:
                            v = int(round(current_val + d))
                            v = max(low, min(high, v))
                            cand.append(v)
                    unique = []
                    for v in cand:
                        if v not in unique:
                            unique.append(v)
                    for v in unique:
                        if v == current_val:
                            continue
                        new_p = dict(best_params)
                        new_p[name] = v
                        perturb_variants.append(new_p)
                seen_local = set()
                for pv in perturb_variants[:5]:
                    key_tuple = tuple(sorted(pv.items()))
                    if key_tuple in seen_param_keys or key_tuple in seen_local:
                        continue
                    seen_local.add(key_tuple)
                    sub_dist = {k: full_distributions[k] for k in pv.keys() if k in full_distributions}
                    try:
                        frozen_trial = optuna.trial.create_trial(
                            params=pv,
                            distributions=sub_dist,
                            value=best_score * 0.999
                        )
                        study.add_trial(frozen_trial)
                    except Exception:
                        continue

        # (C) Tối ưu với ngân sách còn lại
        study.optimize(optuna_objective, n_trials=budget, show_progress_bar=False)

class GridAgent(StrategyAgent):
        """GridAgent: Linh hoạt (tinh chỉnh 2 tham số quan trọng nhất)

        Lưu ý hiệu năng:
        - Giới hạn số vòng lặp tinh chỉnh cục bộ và dừng sớm nếu không cải thiện
            để tránh chiếm dụng slice quá lâu.
        """
        # Giới hạn để tránh bị "kẹt" quá lâu trong một slice
        MAX_LOOPS = 2                 # Số vòng lặp tinh chỉnh cục bộ tối đa trong một lần run()
        EARLY_STOP_NO_IMPROVE = 1     # Dừng nếu không cải thiện sau N vòng lặp

        def run(self, budget):
            # print(f"    -> Running GridAgent with budget: {budget}")
            trials_run = 0
            # Dùng set để tránh lặp lại tham số đã chạy (tuple hóa cho hashable)
            tried_param_sets = {
                tuple(sorted((k, v) for k, v in t['params'].items()))
                for t in self.knowledge_hub.get_all_trials() if 'params' in t
            }

            loops_count = 0
            no_improve_runs = 0

            while trials_run < budget:
                best_before = self.knowledge_hub.get_best_trial()['score']
                best_params = self.knowledge_hub.get_best_trial()['params']
                if not best_params:
                    return  # Không có gì để tinh chỉnh

                # Chọn tối đa 2 tham số kiểu số (float/int) để tinh chỉnh
                params_to_tune = []
                for name, details in self.search_space.items():
                    p_type = details[0] if isinstance(details, (list, tuple)) and details else None
                    if p_type in ['float', 'int']:
                        params_to_tune.append(name)
                    if len(params_to_tune) >= 2:
                        break
                if len(params_to_tune) == 0:
                    return  # Không có tham số số để tinh chỉnh

                # Khởi tạo lưới cục bộ mới quanh best_params hiện tại
                local_grid = [best_params]

                # Tinh chỉnh tham số đầu tiên
                p1_name = params_to_tune[0]
                p1_val = best_params.get(p1_name)
                p1_details = self.search_space[p1_name]
                p1_type = p1_details[0]
                p1_low = p1_details[1]
                p1_high = p1_details[2]

                if p1_type == 'float':
                    # 3 điểm quanh giá trị hiện tại (±10%)
                    p1_steps = np.linspace(max(p1_low, p1_val * 0.9), min(p1_high, p1_val * 1.1), 3)
                else:  # int
                    p1_steps = {p1_val - 1, p1_val, p1_val + 1}

                for p1 in p1_steps:
                    p1_clamped = max(p1_low, min(p1_high, p1))
                    if p1_type == 'int':
                        p1_clamped = int(round(p1_clamped))
                    new_params = {**best_params, p1_name: p1_clamped}
                    if new_params not in local_grid:
                        local_grid.append(new_params)

                # Chạy qua lưới, tôn trọng budget
                for params in local_grid:
                    if trials_run >= budget:
                        break
                    param_key = tuple(sorted((k, v) for k, v in params.items()))
                    if param_key in tried_param_sets:
                        continue  # Bỏ qua nếu đã thử
                    score = self.objective(params)
                    self.knowledge_hub.store(self.agent_id, params, score)
                    tried_param_sets.add(param_key)
                    trials_run += 1

                # Kiểm soát vòng lặp: dừng nếu không cải thiện hoặc vượt quá số vòng tối đa
                best_after = self.knowledge_hub.get_best_trial()['score']
                if not isinstance(best_before, (int, float)):
                    best_before = -float('inf')
                if not isinstance(best_after, (int, float)):
                    best_after = -float('inf')

                if best_after <= best_before + 1e-12:
                    no_improve_runs += 1
                else:
                    no_improve_runs = 0

                loops_count += 1

                if len(local_grid) <= 1 and trials_run < budget:
                    break
                if no_improve_runs >= self.EARLY_STOP_NO_IMPROVE:
                    break
                if loops_count >= self.MAX_LOOPS:
                    break


class PerformanceMonitor:
    def __init__(self, agent_ids):
        self.agent_ids = agent_ids
        self.history = defaultdict(list)

    def update(self, all_trials):
        self.history = defaultdict(list)
        for trial in all_trials:
            self.history[trial['agent_id']].append(trial['score'])

    def get_agent_rewards(self):
        rewards = {}
        for agent_id in self.agent_ids:
            scores = self.history.get(agent_id, [])  # Đặt giá trị mặc định là list rỗng
            if not scores or len(scores) < 2:  # Kiểm tra scores có tồn tại và đủ dài
                rewards[agent_id] = 0.5  # Giá trị mặc định cho agent mới
            else:
                recent_scores = scores[-5:]  # Lấy tối đa 5 điểm gần nhất
                reward = np.mean(np.diff(recent_scores)) if len(recent_scores) > 1 else 0
                normalized_reward = (math.tanh(reward * 100) + 1) / 2
                rewards[agent_id] = normalized_reward
        return rewards

class MetaController_UCB1:
    def __init__(self, agent_ids, verbose=False):
        self.agent_ids = agent_ids
        self.agent_pulls = {agent_id: 0 for agent_id in agent_ids}
        self.agent_rewards = {agent_id: 0.0 for agent_id in agent_ids}
        self.total_pulls = 0
        self.verbose = verbose

    def allocate(self, slice_budget):
        # Kiểm tra các agent chưa được khởi tạo
        uninitialized_agents = [aid for aid, pulls in self.agent_pulls.items() if pulls == 0]
        if uninitialized_agents:
            agent_to_run = uninitialized_agents[0]  # Lấy agent đầu tiên chưa được khởi tạo
            allocations = {agent_id: 0 for agent_id in self.agent_ids}
            allocations[agent_to_run] = slice_budget
            if self.verbose:
                print(f"  [MetaController] Initializing {agent_to_run}")
            return allocations

        # Tính toán UCB scores cho các agent đã được khởi tạo
        ucb_scores = {}

        ucb_scores = {}
        for agent_id in self.agent_ids:
            if self.agent_pulls[agent_id] == 0:
                ucb_scores[agent_id] = float('inf')
            else:
                avg_reward = self.agent_rewards[agent_id] / self.agent_pulls[agent_id]
                exploration_bonus = math.sqrt(2 * math.log(self.total_pulls) / self.agent_pulls[agent_id])
                ucb_scores[agent_id] = avg_reward + exploration_bonus

        # Dùng keys() để rõ ràng cho type checker
        best_agent = max(ucb_scores.keys(), key=lambda k: ucb_scores[k])
        if self.verbose:
            print(f"  [MetaController] UCB scores: { {k: f'{v:.2f}' for k, v in ucb_scores.items()} } -> Chose {best_agent}")

        allocations = {agent_id: 0 for agent_id in self.agent_ids}
        allocations[best_agent] = slice_budget
        return allocations

    def update(self, agent_id_to_update, reward):
        if reward >= 0:
            self.agent_rewards[agent_id_to_update] += reward
            self.agent_pulls[agent_id_to_update] += 1
            self.total_pulls += 1
            # print(f"  [MetaController] Updated {agent_id_to_update}: pulls={self.agent_pulls[agent_id_to_update]}, total_reward={self.agent_rewards[agent_id_to_update]:.2f}")


class AMSCO_Orchestrator:
    """Orchestrator: Giờ nhận objective và search_space"""
    def __init__(self, objective_func, search_space, total_budget, slice_budget, verbose=False):
        self.total_budget = total_budget
        self.slice_budget = slice_budget
        self.verbose = verbose

        self.knowledge_hub = KnowledgeHub()

        self.agents = {
            "Random": RandomAgent("Random", objective_func, search_space, self.knowledge_hub),
            "Bayesian": BayesianAgent("Bayesian", objective_func, search_space, self.knowledge_hub),
            "Grid": GridAgent("Grid", objective_func, search_space, self.knowledge_hub)
        }
        agent_ids = list(self.agents.keys())

        self.performance_monitor = PerformanceMonitor(agent_ids)
        self.meta_controller = MetaController_UCB1(agent_ids, verbose=self.verbose)
        self.agent_budget_usage = {agent_id: 0 for agent_id in agent_ids}

    def run(self):
        current_budget = self.total_budget
        slice_num = 1

        while current_budget > 0:
            # print(f"\n--- Slice {slice_num} | Budget remaining: {current_budget} ---")

            budget_for_slice = min(self.slice_budget, current_budget)

            allocations = self.meta_controller.allocate(budget_for_slice)

            for agent_id, budget in allocations.items():
                if budget > 0:
                    # print(f"... Giving budget to {agent_id}")
                    self.agents[agent_id].run(budget)
                    self.agent_budget_usage[agent_id] += budget

                    all_trials = self.knowledge_hub.get_all_trials()
                    self.performance_monitor.update(all_trials)

                    reward = self.performance_monitor.get_agent_rewards()[agent_id]
                    self.meta_controller.update(agent_id, reward)

            current_budget -= budget_for_slice
            slice_num += 1

        final_result = self.knowledge_hub.get_best_trial()
        return {
            'score': final_result.get('score'),
            'params': final_result.get('params'),
            'iteration_to_best': final_result.get('iteration'),
            'total_trials': self.knowledge_hub.total_calls,
            'agent_pulls': dict(self.meta_controller.agent_pulls),
            'agent_budget_usage': dict(self.agent_budget_usage)
        }


# In[6]:


# ============================================================================
# BƯỚC 5: CÁC TRÌNH TỐI ƯU HÓA BASELINE
# ============================================================================

def profile_optimizer_call(func):
    """Chạy hàm optimizer và ghi nhận thống kê tài nguyên sử dụng."""
    tracemalloc.start()
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    error = None
    result = None
    try:
        result = func()
    except Exception as exc:  # Lưu exception để re-raise sau khi thu thập thống kê
        error = exc
    finally:
        wall_time = time.perf_counter() - start_wall
        cpu_time = time.process_time() - start_cpu
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        usage_snapshot = resource.getrusage(resource.RUSAGE_SELF)

    stats = {
        'wall_time': wall_time,
        'cpu_time': cpu_time,
        'peak_memory_mb': peak / (1024 * 1024),
        'rss_memory_mb': usage_snapshot.ru_maxrss / 1024
    }

    if error is not None:
        raise error

    return result, stats

def run_random_search(objective, search_space, n_trials):
    """Chạy Random Search (sử dụng Optuna)"""
    sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(direction='maximize', sampler=sampler)

    # Hàm mục tiêu cho Optuna
    def optuna_objective(trial):
        params = {}
        for name, details in search_space.items():
            type = details[0]  # Type là phần tử đầu tiên của tuple
            if type == 'float':
                low, high = details[1], details[2]
                dist_type = details[3] if len(details) > 3 else None
                params[name] = trial.suggest_float(name, low, high, log=(dist_type == 'log'))
            elif type == 'int':
                low, high = details[1], details[2]
                params[name] = trial.suggest_int(name, low, high)
            elif type == 'categorical':
                choices = details[1]
                params[name] = trial.suggest_categorical(name, choices)
        return objective(params)

    study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)
    diagnostics = {
        'total_trials': len(study.trials),
        'iteration_to_best': study.best_trial.number + 1
    }
    return study.best_trial.value, study.best_params, diagnostics

def run_optuna_tpe(objective, search_space, n_trials, early_stopping_rounds=20, tolerance=1e-4):
    """Chạy Optuna TPE với MedianPruner và cơ chế dừng sớm tùy chọn.

    Ghi chú:
    - Sử dụng MedianPruner để dừng sớm dựa trên các báo cáo trung gian.
    - Nếu `objective` có thuộc tính `_stepwise(trial, params)`, hàm này sẽ được dùng
      để báo cáo theo từng fold, giúp pruner hoạt động hiệu quả.
    - `early_stopping_rounds` và `tolerance` điều khiển việc dừng tối ưu hóa khi không
      còn cải thiện đáng kể (theo tolerance) sau một số vòng liên tiếp.
    """
    # Sampler TPE cơ bản, có seed để tái lập.
    sampler = optuna.samplers.TPESampler(seed=42)
    # MedianPruner: warmup một vài trial đầu trước khi bắt đầu prune.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

    tolerance = 0.0 if tolerance is None else max(float(tolerance), 0.0)
    early_stop_state = {
        'best_value': None,
        'no_improve_rounds': 0,
        'stopped': False
    }

    def optuna_objective(trial):
        params = {}
        for name, details in search_space.items():
            p_type = details[0]
            if p_type == 'float':
                low, high = details[1], details[2]
                dist_type = details[3] if len(details) > 3 else None
                params[name] = trial.suggest_float(name, low, high, log=(dist_type == 'log'))
            elif p_type == 'int':
                low, high = details[1], details[2]
                params[name] = trial.suggest_int(name, low, high)
            elif p_type == 'categorical':
                choices = details[1]
                params[name] = trial.suggest_categorical(name, choices)

        # Dùng stepwise nếu có để pruner có dữ liệu trung gian theo từng fold
        if hasattr(objective, "_stepwise"):
            try:
                return objective._stepwise(trial, params)
            except optuna.exceptions.TrialPruned:
                # Cho phép Optuna ghi nhận trial bị prune
                raise
        # Fallback: đánh giá 1 lần, vẫn report 1 bước để pruner có dữ liệu
        score = objective(params)
        try:
            trial.report(float(score), step=0)
        except Exception:
            pass
        return score

    def _early_stop_callback(study_ref, trial):
        if not early_stopping_rounds or early_stopping_rounds <= 0:
            return
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        value = trial.value
        if not isinstance(value, (int, float)) or math.isnan(value):
            return

        best_val = early_stop_state['best_value']
        if best_val is None or (value - best_val) > tolerance:
            early_stop_state['best_value'] = value
            early_stop_state['no_improve_rounds'] = 0
        else:
            early_stop_state['no_improve_rounds'] += 1
            if early_stop_state['no_improve_rounds'] >= early_stopping_rounds:
                early_stop_state['stopped'] = True
                study_ref.stop()

    callbacks = []
    if early_stopping_rounds and early_stopping_rounds > 0:
        callbacks.append(_early_stop_callback)

    study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False, callbacks=callbacks)

    # Thống kê prune/completion
    try:
        pruned_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        completed_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    except Exception:
        pruned_trials = None
        completed_trials = None

    diagnostics = {
        'total_trials': len(study.trials),
        'iteration_to_best': study.best_trial.number + 1,
        'early_stopped': early_stop_state['stopped'],
        'patience_used': early_stop_state['no_improve_rounds'] if early_stop_state['stopped'] else None,
        'pruner': 'MedianPruner',
        'pruned_trials': pruned_trials,
        'completed_trials': completed_trials,
    }
    return study.best_trial.value, study.best_params, diagnostics

def run_hyperopt_tpe(objective, search_space, n_trials):
    """Chạy Hyperopt TPE (baseline)"""

    # 1. Chuyển đổi search_space sang định dạng Hyperopt
    hp_space = {}
    for name, details in search_space.items():
        type = details[0]  # Lấy phần tử đầu tiên của tuple

        if type == 'float':
            low, high = details[1], details[2]
            dist_type = details[3] if len(details) > 3 else None
            if dist_type == 'log':
                hp_space[name] = hp.loguniform(name, np.log(low), np.log(high))
            else:
                hp_space[name] = hp.uniform(name, low, high)
        elif type == 'int':
            low, high = details[1], details[2]
            # hp.quniform trả về float, nhưng được làm tròn theo 'q' (là 1). 
            # Chúng ta sẽ ép kiểu về int trong 'hyperopt_objective'
            hp_space[name] = hp.quniform(name, low, high, 1) 
        elif type == 'categorical':
            choices = details[1]
            hp_space[name] = hp.choice(name, choices)

    # 2. Định nghĩa hàm mục tiêu cho Hyperopt
    def hyperopt_objective(params):
        # Hyperopt trả về một số giá trị là float, cần ép kiểu về int
        # Cần kiểm tra lại logic này cho nhất quán
        params_copy = params.copy() # Làm việc trên bản sao để tránh lỗi
        for name, details in search_space.items():
            if details[0] == 'int' and name in params_copy:  # Sửa: kiểm tra type là phần tử đầu tiên
                params_copy[name] = int(params_copy[name])

        score = objective(params_copy)
        return {'loss': -score, 'status': STATUS_OK} # Hyperopt tối thiểu hóa

    # 3. Chạy fmin
    trials = Trials()
    best_raw = fmin(
        fn=hyperopt_objective,
        space=hp_space,
        algo=tpe.suggest, # [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        max_evals=n_trials,
        trials=trials,
        verbose=False
    )

    best_trial = trials.best_trial
    if not best_trial:
        return None, {}, {
            'total_trials': len(trials.trials),
            'iteration_to_best': np.nan
        }

    if not isinstance(best_raw, dict):
        best_raw = dict(best_raw) if best_raw is not None else {}

    best_score = -best_trial['result']['loss']
    best_params = {}
    for name, val in best_raw.items():
        details = search_space[name]
        type = details[0]  # Type là phần tử đầu tiên của tuple

        if type == 'int':
            best_params[name] = int(val)
        elif type == 'categorical':
            choices = details[1]
            # `val` từ hyperopt cho `hp.choice` là một *chỉ số* (index)
            best_params[name] = choices[int(val)] 
        else: # float
            best_params[name] = val
    best_tid = best_trial.get('tid')
    diagnostics = {
        'total_trials': len(trials.trials),
        'iteration_to_best': (best_tid + 1) if isinstance(best_tid, int) else len(trials.trials)
    }
    return best_score, best_params, diagnostics


def run_amsco_optimizer(objective, search_space, n_trials, slice_budget, verbose=False):
    """Chạy AMSCO (phương pháp của chúng ta)"""
    orchestrator = AMSCO_Orchestrator(
        objective_func=objective,
        search_space=search_space,
        total_budget=n_trials,
        slice_budget=slice_budget,
        verbose=verbose
    )
    result = orchestrator.run()
    diagnostics = {
        'total_trials': result.get('total_trials'),
        'iteration_to_best': result.get('iteration_to_best'),
        'agent_pulls': result.get('agent_pulls')
    }
    return result['score'], result['params'], diagnostics


# In[ ]:


# =============================================================================
# BƯỚC 6: BỘ CÔNG CỤ THỰC NGHIỆM (EXPERIMENTAL HARNESS)
# - Mở rộng chạy trên Adult, Breast Cancer, Telco
# - Thêm metrics: accuracy (primary), f1, roc_auc
# - Dùng StratifiedKFold (CV) mặc định
# =============================================================================

def _nested_cv_metrics_for_method(
    X, y, model_name, preprocessor, search_space,
    method_name, inner_trials, outer_folds, inner_folds, slice_budget
):
    """Tính Nested CV (k-outer, m-inner) cho một phương pháp (optimizer).
    Trả về dict: {accuracy, f1, roc_auc, time} (trung bình across outer folds; time = tổng thời gian tối ưu inner).
    """
    def _run_optimizer(objective):
        if method_name == 'Random Search':
            return run_random_search(objective, search_space, inner_trials)
        if method_name == 'Optuna (TPE)':
            return run_optuna_tpe(objective, search_space, inner_trials)
        if method_name == 'Hyperopt (TPE)':
            return run_hyperopt_tpe(objective, search_space, inner_trials)
        if method_name == 'AMSCO':
            return run_amsco_optimizer(objective, search_space, inner_trials, slice_budget, verbose=False)
        raise ValueError(f"Unknown method: {method_name}")

    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []
    total_time = 0.0

    for train_idx, test_idx in outer_cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        objective_inner = create_objective(
            X_tr, y_tr,
            model_name,
            preprocessor,
            metrics=('accuracy', 'f1', 'roc_auc'),
            use_cross_validation=True,
            cv_folds=inner_folds
        )

        start = time.time()
        _, best_params, _ = _run_optimizer(objective_inner)
        total_time += (time.time() - start)

        model = None
        if model_name == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=2000)
        elif model_name == 'random_forest':
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_name == 'xgboost':
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        elif model_name == 'lightgbm':
            model = lgb.LGBMClassifier(random_state=42, verbosity=-1)
        else:
            raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ.")

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.set_params(**best_params)
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        accs.append(accuracy_score(y_te, y_pred))
        try:
            f1s.append(f1_score(y_te, y_pred, average='binary'))
        except Exception:
            f1s.append(np.nan)
        try:
            if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                y_prob = pipeline.predict_proba(X_te)[:, 1]
            elif hasattr(pipeline.named_steps['classifier'], 'decision_function'):
                y_prob = pipeline.decision_function(X_te)
            else:
                y_prob = None
            aucs.append(float(roc_auc_score(y_te, y_prob)) if y_prob is not None else np.nan)
        except Exception:
            aucs.append(np.nan)

    return {
        'accuracy': np.nanmean(accs) if accs else np.nan,
        'f1': np.nanmean(f1s) if f1s else np.nan,
        'roc_auc': np.nanmean(aucs) if aucs else np.nan,
        'time': float(total_time)
    }

if __name__ == "__main__":
    # Log thời gian bắt đầu toàn bộ pipeline
    global_start_time = time.time()
    from datetime import datetime
    print(f"[INFO] Bắt đầu chạy lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ----- CẤU HÌNH THỬ NGHIỆM -----
    # Chạy nhanh: chỉ dùng bộ dữ liệu Breast Cancer
    DATASETS = ['breast_cancer', 'adult']  # Có thể thêm nhiều dataset hơn nếu muốn
    MODELS = ['logistic_regression', 'random_forest']  # Các mô hình để thử nghiệm
    # Giảm số trial để chạy nhanh khi thử nghiệm
    TOTAL_TRIALS = 100
    SLICE_BUDGET = 10   # Lát cắt cho AMSCO (phân bổ động)

    USE_CROSS_VALIDATION = True   # Bật Cross-Validation
    CV_FOLDS = 5                  # Số fold cho StratifiedKFold
    TEST_SIZE = 0.2               # Chỉ dùng nếu holdout
    METRICS = ('accuracy', 'f1', 'roc_auc')  # Primary = accuracy
    # Cấu hình Nested-CV
    NESTED_OUTER_FOLDS = 5
    NESTED_INNER_FOLDS = 3

    # Bảo đảm Nested CV tiêu tốn tối thiểu cùng cấp ngân sách huấn luyện như đánh giá chuẩn
    baseline_fit_budget = TOTAL_TRIALS * (CV_FOLDS if USE_CROSS_VALIDATION else 1)
    nested_fit_per_trial = NESTED_OUTER_FOLDS * NESTED_INNER_FOLDS
    NESTED_INNER_TRIALS = max(
        TOTAL_TRIALS,
        math.ceil(baseline_fit_budget / nested_fit_per_trial)
    )
    # --------------------------------

    def _build_model_for_eval(name):
        if name == 'logistic_regression':
            return LogisticRegression(random_state=42, max_iter=2000)
        if name == 'random_forest':
            return RandomForestClassifier(random_state=42, n_jobs=-1)
        if name == 'xgboost':
            return xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        if name == 'lightgbm':
            return lgb.LGBMClassifier(random_state=42, verbosity=-1)
        raise ValueError(f"Mô hình '{name}' không được hỗ trợ.")

    results = []

    for dataset_name in DATASETS:
        print(f"\n=======================================================")
        print(f"ĐANG THỬ NGHIỆM TRÊN BỘ DỮ LIỆU: {dataset_name.upper()}")
        print(f"=======================================================")

        try:
            X, y, preprocessor, metric_default = get_data(dataset_name)
        except Exception as e:
            print(f"  [WARNING] Bỏ qua dataset {dataset_name}: {e}")
            continue

        for model_name in MODELS:
            print(f"\n{'-'*60}")
            print(f"Đang tối ưu mô hình: {model_name.upper()}")
            print(f"{'-'*60}")

            search_space = dict(MASTER_SEARCH_SPACES[model_name])

            # Khởi tạo biến để tránh cảnh báo unbound trong phân tích tĩnh
            X_train = X_valid = y_train = y_valid = None
            if USE_CROSS_VALIDATION:
                print(f"  [Evaluation] StratifiedKFold ({CV_FOLDS}-fold)")
                objective_func = create_objective(
                    X,
                    y,
                    model_name,
                    preprocessor,
                    metrics=METRICS,
                    use_cross_validation=True,
                    cv_folds=CV_FOLDS
                )
                eval_label = f'cv_{CV_FOLDS}fold'
            else:
                print(f"  [Evaluation] Holdout split (test_size={TEST_SIZE})")
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X,
                    y,
                    test_size=TEST_SIZE,
                    stratify=y,
                    random_state=42
                )
                objective_func = create_objective(
                    X_train,
                    y_train,
                    model_name,
                    preprocessor,
                    metrics=METRICS,
                    use_cross_validation=False,
                    validation_data=(X_valid, y_valid),
                    cv_folds=CV_FOLDS
                )
                eval_label = f'holdout_{TEST_SIZE}'

            model_results = {
                'scores': {},
                'times': {},
                'params': {},
                'metrics': {},
                'diag': {}
            }

            optimizers = [
                ('Optuna (TPE)', run_optuna_tpe),
                ('AMSCO', run_amsco_optimizer)
            ]

            for optimizer_name, optimizer_func in optimizers:
                print(f"\nThực thi: {optimizer_name}...")
                def _invoke_optimizer():
                    if optimizer_name == 'AMSCO':
                        return optimizer_func(
                            objective_func,
                            search_space,
                            TOTAL_TRIALS,
                            SLICE_BUDGET,
                            verbose=False
                        )
                    return optimizer_func(objective_func, search_space, TOTAL_TRIALS)

                optimizer_return, resource_stats = profile_optimizer_call(_invoke_optimizer)

                if isinstance(optimizer_return, tuple) and len(optimizer_return) == 3:
                    primary_score, params, diag_stats = optimizer_return
                elif isinstance(optimizer_return, tuple) and len(optimizer_return) == 2:
                    primary_score, params = optimizer_return
                    diag_stats = {}
                else:
                    raise ValueError(f"Định dạng trả về không hợp lệ từ optimizer {optimizer_name}")

                diag_stats = diag_stats or {}
                combined_stats = {**resource_stats, **diag_stats}

                exec_time = combined_stats.get('wall_time', np.nan)
                total_trials_reported = combined_stats.get('total_trials', np.nan)
                iteration_to_best = combined_stats.get('iteration_to_best', np.nan)

                convergence_ratio = np.nan
                try:
                    if total_trials_reported is not None and iteration_to_best is not None:
                        tt = float(total_trials_reported)
                        ib = float(iteration_to_best)
                        if not math.isnan(tt) and not math.isnan(ib) and tt > 0:
                            convergence_ratio = ib / tt
                except (TypeError, ValueError):
                    convergence_ratio = np.nan

                combined_stats['opt_convergence_ratio'] = convergence_ratio

                # Đánh giá thêm các metric cho best params
                metric_scores = {m: np.nan for m in METRICS}
                # Các metric với CV=5
                acc_cv5 = np.nan
                prec_cv5 = np.nan
                rec_cv5 = np.nan
                bal_cv5 = np.nan
                acc_holdout = np.nan
                f1_holdout = np.nan
                auc_holdout = np.nan
                time_cv5_eval = np.nan
                time_holdout_eval = np.nan
                try:
                    model = _build_model_for_eval(model_name)
                    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
                    pipeline.set_params(**params)
                    if USE_CROSS_VALIDATION:
                        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
                        for m in METRICS:
                            try:
                                metric_scores[m] = cross_val_score(pipeline, X, y, cv=cv, scoring=m, n_jobs=-1).mean()
                            except Exception as e:
                                print(f"  [WARN] Không tính được metric {m} với CV cho {optimizer_name}: {e}")
                                metric_scores[m] = np.nan
                        # Đánh giá thêm CV k=5 cho Accuracy, Precision, Recall, Balanced Accuracy
                        try:
                            cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                            _t0 = time.time()
                            acc_cv5 = cross_val_score(pipeline, X, y, cv=cv5, scoring='accuracy', n_jobs=-1).mean()
                            prec_cv5 = cross_val_score(pipeline, X, y, cv=cv5, scoring='precision', n_jobs=-1).mean()
                            rec_cv5 = cross_val_score(pipeline, X, y, cv=cv5, scoring='recall', n_jobs=-1).mean()
                            bal_cv5 = cross_val_score(pipeline, X, y, cv=cv5, scoring='balanced_accuracy', n_jobs=-1).mean()
                            time_cv5_eval = time.time() - _t0
                        except Exception as e:
                            print(f"  [WARN] Không tính được metric CV=5 cho {optimizer_name}: {e}")
                            acc_cv5 = np.nan
                            prec_cv5 = np.nan
                            rec_cv5 = np.nan
                            bal_cv5 = np.nan
                        # Đánh giá thêm Without CV (holdout)
                        try:
                            X_tr2, X_va2, y_tr2, y_va2 = train_test_split(
                                X, y, test_size=TEST_SIZE, stratify=y, random_state=42
                            )
                            _t1 = time.time()
                            pipeline.fit(X_tr2, y_tr2)
                            y_pred2 = pipeline.predict(X_va2)
                            time_holdout_eval = time.time() - _t1
                            acc_holdout = accuracy_score(y_va2, y_pred2)
                            try:
                                f1_holdout = f1_score(y_va2, y_pred2, average='binary')
                            except Exception:
                                f1_holdout = np.nan
                            try:
                                if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                                    y_prob2 = pipeline.predict_proba(X_va2)[:, 1]
                                elif hasattr(pipeline.named_steps['classifier'], 'decision_function'):
                                    y_prob2 = pipeline.decision_function(X_va2)
                                else:
                                    y_prob2 = None
                                auc_holdout = float(roc_auc_score(y_va2, y_prob2)) if y_prob2 is not None else np.nan
                            except Exception:
                                auc_holdout = np.nan
                        except Exception as e:
                            print(f"  [WARN] Không tính được Accuracy holdout cho {optimizer_name}: {e}")
                            acc_holdout = np.nan
                    else:
                        assert X_train is not None and y_train is not None and X_valid is not None and y_valid is not None
                        _t2 = time.time()
                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_valid)
                        time_holdout_eval = time.time() - _t2
                        metric_scores['accuracy'] = float(accuracy_score(y_valid, y_pred))
                        try:
                            metric_scores['f1'] = float(f1_score(y_valid, y_pred, average='binary'))
                        except Exception:
                            metric_scores['f1'] = np.nan
                        try:
                            if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                                y_prob = pipeline.predict_proba(X_valid)[:, 1]
                            elif hasattr(pipeline.named_steps['classifier'], 'decision_function'):
                                y_prob = pipeline.decision_function(X_valid)
                            else:
                                y_prob = None
                            metric_scores['roc_auc'] = float(roc_auc_score(y_valid, y_prob)) if y_prob is not None else np.nan
                        except Exception:
                            metric_scores['roc_auc'] = np.nan
                        # Khi chế độ holdout, vẫn tính thêm CV=5 để so sánh (Accuracy, Precision, Recall, Balanced Accuracy)
                        try:
                            cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                            _t3 = time.time()
                            acc_cv5 = cross_val_score(pipeline, X_train, y_train, cv=cv5, scoring='accuracy', n_jobs=-1).mean()
                            prec_cv5 = cross_val_score(pipeline, X_train, y_train, cv=cv5, scoring='precision', n_jobs=-1).mean()
                            rec_cv5 = cross_val_score(pipeline, X_train, y_train, cv=cv5, scoring='recall', n_jobs=-1).mean()
                            bal_cv5 = cross_val_score(pipeline, X_train, y_train, cv=cv5, scoring='balanced_accuracy', n_jobs=-1).mean()
                            time_cv5_eval = time.time() - _t3
                        except Exception as e:
                            print(f"  [WARN] Không tính được metric CV=5 (holdout mode) cho {optimizer_name}: {e}")
                            acc_cv5 = np.nan
                            prec_cv5 = np.nan
                            rec_cv5 = np.nan
                            bal_cv5 = np.nan
                        # Và without CV chính là accuracy holdout đã tính ở trên
                        acc_holdout = metric_scores.get('accuracy', np.nan)
                        f1_holdout = metric_scores.get('f1', np.nan)
                        auc_holdout = metric_scores.get('roc_auc', np.nan)
                except Exception as e:
                    print(f"  [WARN] Lỗi khi tính metrics bổ sung cho {optimizer_name}: {e}")

                model_results['scores'][optimizer_name] = primary_score
                model_results['times'][optimizer_name] = exec_time
                model_results['params'][optimizer_name] = params
                model_results['metrics'][optimizer_name] = metric_scores
                model_results['diag'][optimizer_name] = combined_stats

                results.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'optimizer': optimizer_name,
                    'primary_metric': METRICS[0],
                    'primary_score': primary_score,
                    'accuracy': metric_scores.get('accuracy', np.nan),
                    'f1': metric_scores.get('f1', np.nan),
                    'roc_auc': metric_scores.get('roc_auc', np.nan),
                    'acc_holdout': acc_holdout,
                    'f1_holdout': f1_holdout,
                    'auc_holdout': auc_holdout,
                    'acc_cv5': acc_cv5,
                    'prec_cv5': prec_cv5,
                    'recall_cv5': rec_cv5,
                    'bal_acc_cv5': bal_cv5,
                    'time': exec_time,
                    'opt_wall_time': combined_stats.get('wall_time', np.nan),
                    'opt_cpu_time': combined_stats.get('cpu_time', np.nan),
                    'opt_peak_memory_mb': combined_stats.get('peak_memory_mb', np.nan),
                    'opt_rss_memory_mb': combined_stats.get('rss_memory_mb', np.nan),
                    'opt_total_trials': total_trials_reported,
                    'opt_iter_best': iteration_to_best,
                    'opt_convergence_ratio': convergence_ratio,
                    'time': exec_time,
                    'time_holdout': time_holdout_eval,
                    'time_cv5': time_cv5_eval,
                    'evaluation': eval_label
                })

            if model_results['scores']:
                print(f"\n{'='*60}")
                print(f"KẾT QUẢ CHO {model_name.upper()} (Primary metric: {METRICS[0]})")
                print(f"{'='*60}")

                diag_lookup = model_results['diag']
                results_table = pd.DataFrame({
                    'Optimizer': list(model_results['scores'].keys()),
                    'Accuracy': [model_results['metrics'][opt]['accuracy'] for opt in model_results['scores'].keys()],
                    'F1': [model_results['metrics'][opt]['f1'] for opt in model_results['scores'].keys()],
                    'ROC AUC': [model_results['metrics'][opt]['roc_auc'] for opt in model_results['scores'].keys()],
                    'Wall Time (s)': [diag_lookup[opt].get('wall_time', np.nan) for opt in model_results['scores'].keys()],
                    'CPU Time (s)': [diag_lookup[opt].get('cpu_time', np.nan) for opt in model_results['scores'].keys()],
                    'Peak Mem (MB)': [diag_lookup[opt].get('peak_memory_mb', np.nan) for opt in model_results['scores'].keys()],
                    'RSS Mem (MB)': [diag_lookup[opt].get('rss_memory_mb', np.nan) for opt in model_results['scores'].keys()],
                    'Total Trials': [diag_lookup[opt].get('total_trials', np.nan) for opt in model_results['scores'].keys()],
                    'Iter→Best': [diag_lookup[opt].get('iteration_to_best', np.nan) for opt in model_results['scores'].keys()],
                    'Conv. Ratio': [diag_lookup[opt].get('opt_convergence_ratio', np.nan) for opt in model_results['scores'].keys()]
                })

                for col in ['Accuracy', 'F1', 'ROC AUC']:
                    results_table[col] = results_table[col].map(lambda v: f"{v:.4f}" if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else "nan")
                def _fmt_time(val):
                    return f"{val:.2f}" if isinstance(val, (int, float, np.floating)) and not pd.isna(val) else 'nan'
                def _fmt_count(val):
                    if isinstance(val, (int, np.integer)) and not pd.isna(val):
                        return f"{int(val)}"
                    if isinstance(val, (float, np.floating)) and not pd.isna(val):
                        return f"{int(val)}" if float(val).is_integer() else f"{val:.1f}"
                    return 'nan'
                def _fmt_ratio_local(val):
                    try:
                        fval = float(val)
                        if math.isnan(fval) or math.isinf(fval):
                            return 'nan'
                        return f"{fval*100:.2f}%"
                    except Exception:
                        return 'nan'

                results_table['Wall Time (s)'] = results_table['Wall Time (s)'].map(_fmt_time)
                results_table['CPU Time (s)'] = results_table['CPU Time (s)'].map(_fmt_time)
                results_table['Peak Mem (MB)'] = results_table['Peak Mem (MB)'].map(_fmt_time)
                results_table['RSS Mem (MB)'] = results_table['RSS Mem (MB)'].map(_fmt_time)
                results_table['Total Trials'] = results_table['Total Trials'].map(_fmt_count)
                results_table['Iter→Best'] = results_table['Iter→Best'].map(_fmt_count)
                results_table['Conv. Ratio'] = results_table['Conv. Ratio'].map(_fmt_ratio_local)

                print("\n" + results_table.to_string(index=False))
                print("\n" + "-"*60 + "\n")

    if not results:
        print("Không có kết quả nào được ghi nhận.")
    else:
        print("\n\n" + "="*60)
        print(" "*20 + "KẾT QUẢ THỬ NGHIỆM TỔNG QUÁT" + " "*20)
        print("="*60 + "\n")

        results_df = pd.DataFrame(results)

        # (Đã lược bỏ các bảng pivot tổng hợp và thống kê chi tiết để output gọn hơn)
        # Vẫn giữ bảng tóm tắt theo yêu cầu phía dưới.

        # ------------------------------------------------------------
        # BẢNG TÓM TẮT THEO YÊU CẦU
        # Cột: Dataset, Method, Mean Accuracy, Std Dev, Mean Time (s), Convergence Speed
        # Convergence Speed được định nghĩa đơn giản: Mean Accuracy / Mean Time (s)
        # ------------------------------------------------------------
        try:
            grp = results_df.groupby(['dataset', 'optimizer'])
            summary = pd.DataFrame({
                'Mean Accuracy': grp['accuracy'].mean(),
                'Std Dev': grp['accuracy'].std(ddof=1),
                'Mean Time (s)': grp['time'].mean()
            }).reset_index()

            # Xử lý NaN std khi chỉ có 1 điểm
            summary['Std Dev'] = summary['Std Dev'].fillna(0.0)

            # Convergence Speed: độ chính xác trên mỗi giây, tránh chia 0
            summary['Convergence Speed'] = summary.apply(
                lambda r: (r['Mean Accuracy'] / r['Mean Time (s)']) if r['Mean Time (s)'] and r['Mean Time (s)'] > 0 else np.nan,
                axis=1
            )

            # Định dạng
            summary = summary.rename(columns={'dataset': 'Dataset', 'optimizer': 'Method'})
            # Sắp xếp: theo Dataset rồi Method để dễ đọc
            summary = summary.sort_values(['Dataset', 'Method']).reset_index(drop=True)

            # Làm tròn hiển thị
            disp = summary.copy()
            disp['Mean Accuracy'] = disp['Mean Accuracy'].map(lambda v: f"{v:.4f}")
            disp['Std Dev'] = disp['Std Dev'].map(lambda v: f"{v:.4f}")
            disp['Mean Time (s)'] = disp['Mean Time (s)'].map(lambda v: f"{v:.2f}")
            disp['Convergence Speed'] = disp['Convergence Speed'].map(lambda v: f"{v:.6f}" if pd.notna(v) else "nan")

            print("\nBẢNG TÓM TẮT (Dataset, Method, Mean Accuracy, Std Dev, Mean Time (s), Convergence Speed):")
            print("-"*60)
            print(disp.to_string(index=False))
            print("\n")
        except Exception as e:
            print(f"[WARN] Không thể tạo bảng tóm tắt: {e}")

        # ------------------------------------------------------------
        # BẢNG HIỆU NĂNG TỐI ƯU HÓA (Tập trung vào Time/Iteration/Resource)
        # ------------------------------------------------------------
        try:
            perf_cols = [
                'opt_wall_time',
                'opt_cpu_time',
                'opt_peak_memory_mb',
                'opt_total_trials',
                'opt_iter_best',
                'opt_convergence_ratio'
            ]
            available_cols = [c for c in perf_cols if c in results_df.columns]
            if available_cols:
                perf_grp = results_df.groupby(['dataset', 'optimizer'])[available_cols].mean().reset_index()
                perf_grp = perf_grp.rename(columns={'dataset': 'Dataset', 'optimizer': 'Method'})

                def _fmt_perf(val, nd=2):
                    if isinstance(val, (int, np.integer)) and not pd.isna(val):
                        return f"{int(val)}"
                    if isinstance(val, (float, np.floating)) and not pd.isna(val):
                        return f"{val:.{nd}f}"
                    return 'nan'

                display_perf = perf_grp.copy()
                if 'opt_wall_time' in display_perf:
                    display_perf['opt_wall_time'] = display_perf['opt_wall_time'].map(lambda v: _fmt_perf(v, 2))
                if 'opt_cpu_time' in display_perf:
                    display_perf['opt_cpu_time'] = display_perf['opt_cpu_time'].map(lambda v: _fmt_perf(v, 2))
                if 'opt_peak_memory_mb' in display_perf:
                    display_perf['opt_peak_memory_mb'] = display_perf['opt_peak_memory_mb'].map(lambda v: _fmt_perf(v, 2))
                if 'opt_total_trials' in display_perf:
                    display_perf['opt_total_trials'] = display_perf['opt_total_trials'].map(lambda v: _fmt_perf(v, 1))
                if 'opt_iter_best' in display_perf:
                    display_perf['opt_iter_best'] = display_perf['opt_iter_best'].map(lambda v: _fmt_perf(v, 1))
                if 'opt_convergence_ratio' in display_perf:
                    def _fmt_ratio_perf(v):
                        try:
                            fv = float(v)
                            if math.isnan(fv) or math.isinf(fv):
                                return 'nan'
                            return f"{fv*100:.2f}%"
                        except Exception:
                            return 'nan'
                    display_perf['opt_convergence_ratio'] = display_perf['opt_convergence_ratio'].map(_fmt_ratio_perf)

                display_perf = display_perf.rename(columns={
                    'opt_wall_time': 'Mean Wall Time (s)',
                    'opt_cpu_time': 'Mean CPU Time (s)',
                    'opt_peak_memory_mb': 'Mean Peak Mem (MB)',
                    'opt_total_trials': 'Mean Total Trials',
                    'opt_iter_best': 'Mean Iter→Best',
                    'opt_convergence_ratio': 'Mean Conv. Ratio'
                })

                display_perf = display_perf.sort_values(['Dataset', 'Method']).reset_index(drop=True)

                print("BẢNG HIỆU NĂNG TỐI ƯU HÓA (Time / Iterations / Resource):")
                print("-"*70)
                print(display_perf.to_string(index=False))
                print("\n")
        except Exception as e:
            print(f"[WARN] Không thể tạo bảng hiệu năng tối ưu hóa: {e}")

        # ------------------------------------------------------------
        # TÍNH NESTED CV CHỈ MỘT LẦN VÀ TÁI SỬ DỤNG CHO CÁC BẢNG
        # ------------------------------------------------------------
        nested_cache = {}
        try:
            methods = sorted(results_df['optimizer'].unique())
            for method in methods:
                acc_list, f1_list, auc_list = [], [], []
                total_time = 0.0
                count = 0
                for dataset_name in DATASETS:
                    try:
                        Xn, yn, prep_n, _ = get_data(dataset_name, quiet=True)
                    except Exception:
                        continue
                    for model_name in MODELS:
                        search_space = dict(MASTER_SEARCH_SPACES[model_name])
                        res = _nested_cv_metrics_for_method(
                            Xn, yn, model_name, prep_n, search_space,
                            method,
                            inner_trials=NESTED_INNER_TRIALS,
                            outer_folds=NESTED_OUTER_FOLDS,
                            inner_folds=NESTED_INNER_FOLDS,
                            slice_budget=SLICE_BUDGET
                        )
                        acc_list.append(res['accuracy'])
                        f1_list.append(res['f1'])
                        auc_list.append(res['roc_auc'])
                        total_time += res['time']
                        count += 1
                if acc_list:
                    nested_cache[method] = {
                        'accuracy': float(np.nanmean(acc_list)),
                        'f1': float(np.nanmean(f1_list)),
                        'roc_auc': float(np.nanmean(auc_list)),
                        'time': float(total_time / max(count, 1))
                    }
        except Exception as e:
            print(f"[WARN] Không thể tính Nested CV tổng hợp: {e}")
            nested_cache = {}

        # ------------------------------------------------------------
        # BẢNG SO SÁNH CV: Method, Without CV, With K-Fold (k=5), With Nested CV, Improvement (% K-Fold vs Holdout)
        # ------------------------------------------------------------
        try:
            if {'acc_holdout', 'acc_cv5'}.issubset(results_df.columns):
                comp = results_df.groupby('optimizer')[['acc_holdout', 'acc_cv5']].mean().reset_index()
                comp = comp.rename(columns={
                    'optimizer': 'Method',
                    'acc_holdout': 'Without CV',
                    'acc_cv5': 'With K-Fold (k=5)'
                })
                comp['With Nested CV'] = comp['Method'].map(
                    lambda m: nested_cache.get(m, {}).get('accuracy', np.nan)
                )

                def _improve(row):
                    base = row['Without CV']
                    cv5 = row['With K-Fold (k=5)']
                    if pd.isna(base) or pd.isna(cv5) or base == 0:
                        return np.nan
                    return (cv5 - base) / base * 100.0
                comp['Improvement (%)'] = comp.apply(_improve, axis=1)

                disp2 = comp.copy()
                disp2['Without CV'] = disp2['Without CV'].map(lambda v: f"{v:.4f}" if pd.notna(v) else 'nan')
                disp2['With K-Fold (k=5)'] = disp2['With K-Fold (k=5)'].map(lambda v: f"{v:.4f}" if pd.notna(v) else 'nan')
                disp2['With Nested CV'] = disp2['With Nested CV'].map(lambda v: f"{v:.4f}" if pd.notna(v) else 'nan')
                disp2['Improvement (%)'] = disp2['Improvement (%)'].map(lambda v: f"{v:.2f}" if pd.notna(v) else 'nan')

                print("BẢNG SO SÁNH CV (Method, Without CV, With K-Fold (k=5), With Nested CV, Improvement (%)):")
                print("-"*60)
                print(disp2.to_string(index=False))
                print("\n")
        except Exception as e:
            print(f"[WARN] Không thể tạo bảng so sánh CV: {e}")

        # ------------------------------------------------------------
        # BẢNG ĐÁNH GIÁ CV (GỘP TRUNG BÌNH QUA OPTIMIZERS)
        # Columns: Evaluation Method, Accuracy, F1-Score, AUC-ROC, Time
        # ------------------------------------------------------------
        try:
            # Trung bình toàn cục (qua mọi model và optimizer)
            hold_acc = float(np.nanmean(results_df['acc_holdout'])) if 'acc_holdout' in results_df else np.nan
            hold_f1 = float(np.nanmean(results_df['f1_holdout'])) if 'f1_holdout' in results_df else np.nan
            hold_auc = float(np.nanmean(results_df['auc_holdout'])) if 'auc_holdout' in results_df else np.nan
            hold_time = float(np.nanmean(results_df['time_holdout'])) if 'time_holdout' in results_df else (float(np.nanmean(results_df['time'])) if 'time' in results_df else np.nan)

            kfold_acc = float(np.nanmean(results_df['accuracy'])) if 'accuracy' in results_df else np.nan
            kfold_f1 = float(np.nanmean(results_df['f1'])) if 'f1' in results_df else np.nan
            kfold_auc = float(np.nanmean(results_df['roc_auc'])) if 'roc_auc' in results_df else np.nan
            kfold_time = float(np.nanmean(results_df['time_cv5'])) if 'time_cv5' in results_df else (float(np.nanmean(results_df['time'])) if 'time' in results_df else np.nan)

            # Trung bình Nested-CV qua optimizers sử dụng nested_cache
            if nested_cache:
                nested_acc = float(np.nanmean([v['accuracy'] for v in nested_cache.values()]))
                nested_f1 = float(np.nanmean([v['f1'] for v in nested_cache.values()]))
                nested_auc = float(np.nanmean([v['roc_auc'] for v in nested_cache.values()]))
                nested_time = float(np.nanmean([v['time'] for v in nested_cache.values()]))
            else:
                nested_acc = nested_f1 = nested_auc = nested_time = np.nan

            # Improvement (%) của Nested so với Holdout
            def _pct(new, base):
                if pd.isna(new) or pd.isna(base) or base == 0:
                    return np.nan
                return (new - base) / base * 100.0

            imp_acc = _pct(nested_acc, hold_acc)
            imp_f1 = _pct(nested_f1, hold_f1)
            imp_auc = _pct(nested_auc, hold_auc)

            display_df = pd.DataFrame([
                {
                    'Evaluation Method': 'Simple Holdout (80-20 split)',
                    'Accuracy': hold_acc,
                    'F1-Score': hold_f1,
                    'AUC-ROC': hold_auc,
                    'Time': hold_time
                },
                {
                    'Evaluation Method': f'Standard K-Fold (k={CV_FOLDS})',
                    'Accuracy': kfold_acc,
                    'F1-Score': kfold_f1,
                    'AUC-ROC': kfold_auc,
                    'Time': kfold_time
                },
                {
                    'Evaluation Method': f'Nested CV ({NESTED_OUTER_FOLDS}-outer, {NESTED_INNER_FOLDS}-inner)',
                    'Accuracy': nested_acc,
                    'F1-Score': nested_f1,
                    'AUC-ROC': nested_auc,
                    'Time': nested_time
                },
                {
                    'Evaluation Method': 'Improvement (Nested vs Holdout)',
                    'Accuracy': imp_acc,
                    'F1-Score': imp_f1,
                    'AUC-ROC': imp_auc,
                    'Time': np.nan
                }
            ])

            # Định dạng số
            def _fmt_num(v, nd):
                return f"{v:.{nd}f}" if isinstance(v, (int, float, np.floating)) and pd.notna(v) else 'nan'

            for col in ['Accuracy', 'F1-Score', 'AUC-ROC']:
                # Với hàng Improvement -> hiển thị %
                mask_imp = display_df['Evaluation Method'] == 'Improvement (Nested vs Holdout)'
                display_df.loc[~mask_imp, col] = display_df.loc[~mask_imp, col].map(lambda v: _fmt_num(v, 4))
                display_df.loc[mask_imp, col] = display_df.loc[mask_imp, col].map(lambda v: (f"{float(v):.2f}%" if pd.notna(v) else 'nan'))

            display_df['Time'] = display_df['Time'].map(lambda v: f"{v:.2f}" if isinstance(v, (int, float, np.floating)) and pd.notna(v) else '-')

            print("BẢNG ĐÁNH GIÁ CV (Evaluation Method, Accuracy, F1-Score, AUC-ROC, Time):")
            print("-"*60)
            print(display_df.to_string(index=False))
            print("\n")
        except Exception as e:
            print(f"[WARN] Không thể tạo bảng đánh giá theo định dạng yêu cầu: {e}")

        # ------------------------------------------------------------
        # BẢNG SO SÁNH THỐNG KÊ GIỮA AMSCO VÀ CÁC BASELINE (THEO TÀI NGUYÊN)
        #  - Đánh giá các chỉ số tối ưu hóa: thời gian, số trial, hội tụ, bộ nhớ
        #  - Sử dụng Mann-Whitney U test (phi tham số) và Cohen's d (effect size)
        # ------------------------------------------------------------
        try:
            from scipy.stats import mannwhitneyu

            def _cohen_d_paired(x, y):
                x = np.asarray(x, dtype=float)
                y = np.asarray(y, dtype=float)
                mask = ~np.isnan(x) & ~np.isnan(y)
                x, y = x[mask], y[mask]
                if len(x) < 2:
                    return np.nan
                diff = x - y
                return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))

            def _effect_label(d):
                if pd.isna(d):
                    return "Unknown"
                ad = abs(d)
                if ad < 0.2:
                    return "Negligible"
                if ad < 0.5:
                    return "Small"
                if ad < 0.8:
                    return "Medium"
                if ad < 1.2:
                    return "Large"
                return "Very Large"

            resource_metrics = [
                ('opt_wall_time', 'Wall Time (s)', 'lower'),
                ('opt_cpu_time', 'CPU Time (s)', 'lower'),
                ('opt_total_trials', 'Total Trials', 'lower'),
                ('opt_iter_best', 'Iter→Best', 'lower'),
                ('opt_convergence_ratio', 'Convergence Ratio', 'lower'),
                ('opt_peak_memory_mb', 'Peak Memory (MB)', 'lower'),
                ('opt_rss_memory_mb', 'RSS Memory (MB)', 'lower')
            ]

            comparisons = [
                ("AMSCO", "Random Search"),
                ("AMSCO", "Optuna (TPE)"),
                ("AMSCO", "Hyperopt (TPE)")
            ]

            rows_stat = []
            for metric_col, metric_label, better in resource_metrics:
                if metric_col not in results_df.columns:
                    continue
                pivot = results_df.pivot_table(
                    index=['dataset', 'model'],
                    columns='optimizer',
                    values=metric_col,
                    aggfunc='mean'
                )

                for a, b in comparisons:
                    if a not in pivot.columns or b not in pivot.columns:
                        continue
                    xa = pivot[a].to_numpy()
                    xb = pivot[b].to_numpy()
                    mask = ~np.isnan(xa) & ~np.isnan(xb)
                    xa, xb = xa[mask], xb[mask]
                    if len(xa) < 2:
                        continue

                    u_stat, p_val = mannwhitneyu(xa, xb, alternative='two-sided')
                    d_val = _cohen_d_paired(xa, xb)

                    mean_a = float(np.mean(xa)) if len(xa) else np.nan
                    mean_b = float(np.mean(xb)) if len(xb) else np.nan
                    diff = mean_b - mean_a  # baseline minus AMSCO

                    if better == 'lower':
                        better_method = a if mean_a < mean_b else b
                    else:
                        better_method = a if mean_a > mean_b else b

                    rows_stat.append({
                        "Metric": metric_label,
                        "Comparison": f"{a} vs {b}",
                        "Mean AMSCO": mean_a,
                        "Mean Baseline": mean_b,
                        "Δ (Baseline-AMSCO)": diff,
                        "U-statistic": u_stat,
                        "p-value": p_val,
                        "Significant (α=0.05)": "Yes" if (p_val < 0.05) else "No",
                        "Cohen's d": d_val,
                        "Effect Size": _effect_label(d_val),
                        "Preferred": better_method
                    })

            if rows_stat:
                stat_df = pd.DataFrame(rows_stat)

                def _fmt_num_local(v, nd=4):
                    return f"{v:.{nd}f}" if isinstance(v, (int, float, np.floating)) and pd.notna(v) else "nan"

                stat_df["Mean AMSCO"] = stat_df["Mean AMSCO"].map(lambda v: _fmt_num_local(v, 3))
                stat_df["Mean Baseline"] = stat_df["Mean Baseline"].map(lambda v: _fmt_num_local(v, 3))
                stat_df["Δ (Baseline-AMSCO)"] = stat_df["Δ (Baseline-AMSCO)"].map(lambda v: _fmt_num_local(v, 3))
                stat_df["U-statistic"] = stat_df["U-statistic"].map(lambda v: _fmt_num_local(v, 3))
                stat_df["p-value"] = stat_df["p-value"].map(lambda v: _fmt_num_local(v, 4))
                stat_df["Cohen's d"] = stat_df["Cohen's d"].map(lambda v: _fmt_num_local(v, 3))

                print("BẢNG SO SÁNH THỐNG KÊ (TÀI NGUYÊN TỐI ƯU HÓA – AMSCO vs BASELINES):")
                print("-"*80)
                print(stat_df.to_string(index=False))
                print("\n")
        except Exception as e:
            print(f"[WARN] Không thể tạo bảng so sánh thống kê tài nguyên: {e}")

        # ------------------------------------------------------------
        # BẢNG TỔNG HỢP THEO YÊU CẦU (CỘT = OPTIMIZER, HÀNG = METRIC)
        # Columns: AMSCO, Optuna (TPE), Hyperopt (TPE), Random Search
        # Rows: Accuracy, Precision, Recall, F1-score, AUC-ROC,
        #       Balanced Acc, Optimization Time
        # ------------------------------------------------------------
        try:
            rows = [
                ("Accuracy (CV=5)", "accuracy"),
                ("Precision (CV=5)", "precision"),
                ("Recall (CV=5)", "recall"),
                ("F1-score (CV metric)", "f1"),
                ("AUC-ROC (CV metric)", "roc_auc"),
                ("Balanced Acc (CV=5)", "balanced_accuracy"),
                ("Optimization Time (s)", "opt_time"),
            ]

            optim_order = ["AMSCO", "Optuna (TPE)", "Hyperopt (TPE)", "Random Search"]
            data = {"Metric": [r[0] for r in rows]}

            for opt in optim_order:
                sub = results_df[results_df["optimizer"] == opt]
                # Chuẩn bị các cột metric
                acc = float(np.nanmean(sub["acc_cv5"])) if "acc_cv5" in sub else np.nan
                prec = float(np.nanmean(sub["prec_cv5"])) if "prec_cv5" in sub else np.nan
                rec = float(np.nanmean(sub["recall_cv5"])) if "recall_cv5" in sub else np.nan
                bal = float(np.nanmean(sub["bal_acc_cv5"])) if "bal_acc_cv5" in sub else np.nan
                f1 = float(np.nanmean(sub["f1"])) if "f1" in sub else np.nan
                auc = float(np.nanmean(sub["roc_auc"])) if "roc_auc" in sub else np.nan
                opt_time = float(np.nanmean(sub["time"])) if "time" in sub else np.nan

                col_vals = []
                for label, key in rows:
                    if key == "accuracy":
                        col_vals.append(acc)
                    elif key == "precision":
                        col_vals.append(prec)
                    elif key == "recall":
                        col_vals.append(rec)
                    elif key == "f1":
                        col_vals.append(f1)
                    elif key == "roc_auc":
                        col_vals.append(auc)
                    elif key == "balanced_accuracy":
                        col_vals.append(bal)
                    elif key == "opt_time":
                        col_vals.append(opt_time)
                    else:
                        col_vals.append(np.nan)

                data[opt] = col_vals

            summary_opt_df = pd.DataFrame(data)

            def _fmt(v, is_time=False):
                if not isinstance(v, (int, float, np.floating)) or pd.isna(v):
                    return "nan"
                return f"{v:.2f}" if is_time else f"{v:.4f}"

            # Định dạng từng dòng
            for i, (label, key) in enumerate(rows):
                is_time = (key == "opt_time")
                for opt in optim_order:
                    summary_opt_df.loc[i, opt] = _fmt(summary_opt_df.loc[i, opt], is_time=is_time)

            print("\nBẢNG TỔNG HỢP THEO OPTIMIZER (HÀNG = METRIC):")
            print("-"*60)
            print(summary_opt_df.to_string(index=False))
            print("\n")
        except Exception as e:
            print(f"[WARN] Không thể tạo bảng tổng hợp theo optimizer: {e}")

        # In thời gian kết thúc toàn bộ pipeline
        total_time = time.time() - global_start_time
        print("\n=======================================================")
        print("         KẾT QUẢ THỬ NGHIỆM TỔNG QUÁT")
        print("=======================================================")
        print(results_df.to_string(index=False))
        print(f"\n[INFO] Kết thúc chạy lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (tổng thời gian: {total_time:.2f} s)")


# In[ ]:




