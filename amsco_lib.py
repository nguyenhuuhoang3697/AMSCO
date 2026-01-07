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
import json
import gc
import matplotlib.pyplot as plt
from collections import defaultdict
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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
        'classifier__n_estimators': ('int', 50, 150),
        'classifier__max_depth': ('int', 5, 20),
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
        # 'classifier__max_depth': ('int', 3, 12), # Loại bỏ để tránh xung đột với num_leaves
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

    return X, y


def get_data(dataset_name, quiet=False):
    dataset_name = dataset_name.lower()
    sampler = None
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
        return X, y, preprocessor, metric, sampler

    if dataset_name == 'breast_cancer':
        if not quiet:
            print("... Đang tải Breast Cancer (sklearn)")
        X, y = load_breast_cancer_dataset()
        X, preprocessor = preprocess_adult_columns(X)
        metric = 'accuracy'
        return X, y, preprocessor, metric, sampler

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
        return X, y, preprocessor, metric, sampler

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
        sampler = SMOTE(random_state=42)
        return X, y, preprocessor, metric, sampler

    raise ValueError("Notebook này chỉ hỗ trợ dataset 'adult', 'breast_cancer', 'telco', hoặc 'credit'.")


def prepare_data_with_holdout_test(dataset_name, test_size=0.2, random_state=42):
    """
    Tải dữ liệu và tách thành: 
    - Train+Val set (80%): Dùng cho Nested CV và tối ưu hóa
    - Test set (20%): Dùng làm ground truth, KHÔNG tham gia training
    
    Args:
        dataset_name:  Tên dataset
        test_size: Tỷ lệ test set (default:  0.2)
        random_state: Seed để tái lập
    
    Returns:
        tuple: (X_train_val, X_test, y_train_val, y_test, preprocessor, metric, sampler)
    """
    # Load full data
    X_full, y_full, preprocessor, metric, sampler = get_data(dataset_name, quiet=True)
    
    # Split:  80% Train+Val, 20% Test (độc lập)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_full, y_full,
        test_size=test_size,
        stratify=y_full,
        random_state=random_state
    )
    
    print(f"  [Data Split] {dataset_name}:")
    print(f"    Train+Val: {len(X_train_val)} samples ({(1-test_size)*100:.0f}%)")
    print(f"    Test (held-out): {len(X_test)} samples ({test_size*100:.0f}%)")
    print(f"    Class distribution (Test): {dict(pd.Series(y_test).value_counts())}")
    
    return X_train_val, X_test, y_train_val, y_test, preprocessor, metric, sampler

def get_optimal_n_jobs(n_samples):
    """
    Xác định số cores tối ưu dựa trên kích thước dataset.
    Small datasets: dùng nhiều cores (nhanh, ít RAM)
    Large datasets: giới hạn cores (tránh RAM explosion)
    """
    if n_samples < 5000:
        return -1  # Dùng tất cả cores (breast_cancer: ~500 samples)
    elif n_samples < 10000:
        return 8   # Dùng 8 cores (telco: ~7K samples)
    elif n_samples < 30000:
        return 6   # Dùng 6 cores
    elif n_samples < 100000:
        return 4   # Dùng 4 cores (adult: ~40K samples)
    else:
        return 2   # Dùng 2 cores (credit: 284K samples - CỰC LỚN!)

def evaluate_on_holdout_test(best_params, model_name, preprocessor, 
                              X_train, y_train, X_test, y_test, sampler=None):
    """
    Đánh giá mô hình trên test set độc lập (ground truth).
    
    Args:
        best_params: Tham số tốt nhất từ optimizer
        model_name: Tên mô hình
        preprocessor: Preprocessor
        X_train: Dữ liệu train (80%)
        y_train: Label train
        X_test:  Dữ liệu test độc lập (20%)
        y_test: Label test
        sampler:  SMOTE sampler (nếu có)
    
    Returns:
        dict: {'accuracy', 'f1', 'roc_auc', 'precision', 'recall', 'balanced_accuracy'}
    """
    # Xác định số cores tối ưu dựa trên kích thước dataset
    n_jobs_optimal = get_optimal_n_jobs(len(X_train))
    
    # Build model (n_jobs=2 để TRÁNH TRÀN RAM - mỗi process copy toàn bộ data)
    if model_name == 'logistic_regression':
        if sampler is None:
            model = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced')
        else:
            model = LogisticRegression(random_state=42, max_iter=2000)
    elif model_name == 'random_forest':
        # n_jobs=2 thay vì n_jobs_optimal để tránh tràn RAM
        if sampler is None:
            model = RandomForestClassifier(random_state=42, n_jobs=2, class_weight='balanced_subsample')
        else:
            model = RandomForestClassifier(random_state=42, n_jobs=2)
    elif model_name == 'xgboost':
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    elif model_name == 'lightgbm':
        model = lgb.LGBMClassifier(random_state=42, verbosity=-1)
    else:
        raise ValueError(f"Model '{model_name}' not supported")
    
    # Build pipeline
    if sampler is not None:
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('sampler', clone(sampler)),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    
    # Set best params and train on full train set
    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)
    
    # Predict on held-out test set
    y_pred = pipeline.predict(X_test)
    
    # Predict proba trước khi delete pipeline
    y_proba = None
    try:
        if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline.named_steps['classifier'], 'decision_function'):
            y_proba = pipeline.decision_function(X_test)
    except Exception:
        pass
    
    # Giải phóng pipeline ngay để tiết kiệm RAM
    del pipeline
    gc.collect()
    
    # Calculate metrics
    results = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='binary', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='binary', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred))
    }
    
    # F1-Score
    try:
        results['f1'] = float(f1_score(y_test, y_pred, average='binary'))
    except Exception: 
        results['f1'] = np.nan
    
    # ROC-AUC
    try:
        if y_proba is not None:
            results['roc_auc'] = float(roc_auc_score(y_test, y_proba))
        else:
            results['roc_auc'] = np.nan
    except Exception:
        results['roc_auc'] = np.nan
    
    return results

# In[4]:


# =============================================================================
# BƯỚC 3: NORMALIZER CHO MULTI-OBJECTIVE OPTIMIZATION
# =============================================================================

class MultiObjectiveNormalizer:
    """
    Chuẩn hóa Min-Max cho các metrics trong multi-objective optimization.
    f(θ) = α·F1 + β·ROC-AUC - γ·execution_time (normalized)
    """
    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Storage for normalization bounds
        self.f1_min = None
        self.f1_max = None
        self.auc_min = None
        self.auc_max = None
        self.time_min = None
        self.time_max = None
        
        # Collected data for computing bounds
        self.f1_values = []
        self.auc_values = []
        self.time_values = []
    
    def collect(self, f1, roc_auc, exec_time):
        """Thu thập giá trị để tính min/max sau này."""
        if not math.isnan(f1):
            self.f1_values.append(f1)  # Collect F1 score
        if not math.isnan(roc_auc):
            self.auc_values.append(roc_auc)  # Collect ROC AUC score
        if not math.isnan(exec_time) and exec_time > 0:
            self.time_values.append(exec_time)  # Collect execution time
    
    def compute_bounds(self):
        """Tính min/max từ dữ liệu đã thu thập."""
        if self.f1_values:
            self.f1_min = min(self.f1_values)
            self.f1_max = max(self.f1_values)
        if self.auc_values:
            self.auc_min = min(self.auc_values)
            self.auc_max = max(self.auc_values)
        if self.time_values:
            self.time_min = min(self.time_values)
            self.time_max = max(self.time_values)
    
    def normalize(self, value, vmin, vmax):
        """Chuẩn hóa Min-Max: (x - min) / (max - min)."""
        if vmin is None or vmax is None or math.isnan(value):
            return 0.0
        if vmax - vmin < 1e-9:  # Tránh chia cho 0
            return 0.5
        return (value - vmin) / (vmax - vmin)
    
    def compute_objective(self, f1, roc_auc, exec_time):
        """
        Tính hàm mục tiêu đa chiều:
        f(θ) = α·F1_norm + β·ROC-AUC_norm - γ·time_norm
        """
        f1_norm = self.normalize(f1, self.f1_min, self.f1_max)
        auc_norm = self.normalize(roc_auc, self.auc_min, self.auc_max)
        time_norm = self.normalize(exec_time, self.time_min, self.time_max)
        
        # Công thức mục tiêu
        objective = (self.alpha * f1_norm + 
                    self.beta * auc_norm - 
                    self.gamma * time_norm)
        return objective
    
    def set_bounds(self, f1_min, f1_max, auc_min, auc_max, time_min, time_max):
        """Đặt bounds thủ công (dùng khi đã biết trước từ warm-up)."""
        self.f1_min = f1_min
        self.f1_max = f1_max
        self.auc_min = auc_min
        self.auc_max = auc_max
        self.time_min = time_min
        self.time_max = time_max


# =============================================================================
# BƯỚC 4: HÀM MỤC TIÊU (OBJECTIVE FUNCTION) TỔNG QUÁT
# - Hỗ trợ nhiều metric: accuracy, f1, roc_auc (roc_auc chỉ hoạt động khi dữ liệu nhị phân)
# - Hỗ trợ multi-objective với normalizer
# =============================================================================

def create_objective(
    X,
    y,
    model_name,
    preprocessor,
    metrics=('accuracy',),  # tuple/list các metric cần tính
    use_cross_validation=True,
    validation_data=None,
    cv_folds=3,
    sampler=None,
    multi_objective=False,  # Bật multi-objective optimization
    normalizer=None  # MultiObjectiveNormalizer instance
):
    """
    Xây dựng hàm objective cho optimizer.

    Nếu `use_cross_validation=True`, sử dụng StratifiedKFold (cv_folds) và trả về metric đầu tiên (primary) làm giá trị tối ưu.
    Các metric khác sẽ được tính và trả kèm (có thể lưu để phân tích, nhưng tối ưu vẫn dựa trên primary metric).

    Nếu `use_cross_validation=False`, yêu cầu `validation_data` = (X_valid, y_valid) để tính trên holdout.
    Primary metric = metrics[0].
    
    Nếu `multi_objective=True`, sử dụng normalizer để tính:
    f(θ) = α·F1 + β·ROC-AUC - γ·execution_time
    """

    if not metrics:
        raise ValueError("Phải cung cấp ít nhất một metric.")
    primary_metric = metrics[0]

    if not use_cross_validation and validation_data is None:
        raise ValueError("validation_data phải được cung cấp khi tắt cross validation.")
    
    if multi_objective and normalizer is None:
        raise ValueError("normalizer phải được cung cấp khi bật multi_objective.")

    def _build_classifier(name):
        if name == 'logistic_regression':
            # Chỉ dùng class_weight='balanced' khi KHÔNG có SMOTE
            # (SMOTE đã balance data rồi, tránh over-correction)
            if sampler is None:
                return LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced')
            return LogisticRegression(random_state=42, max_iter=2000)
        if name == 'random_forest':
            # Dùng class_weight='balanced_subsample' khi KHÔNG có SMOTE để xử lý imbalance
            # (balanced_subsample: balance tại mỗi bootstrap sample, phù hợp với RF)
            if sampler is None:
                return RandomForestClassifier(random_state=42, n_jobs=4, class_weight='balanced_subsample')
            return RandomForestClassifier(random_state=42, n_jobs=4)
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

    def _build_pipeline():
        steps = [('preprocessor', preprocessor)]
        pipeline_cls = Pipeline
        if sampler is not None:
            steps.append(('sampler', clone(sampler)))
            pipeline_cls = ImbPipeline
        steps.append(('classifier', _build_classifier(model_name)))
        return pipeline_cls(steps=steps)

    def _clean_numeric(value):
        try:
            if value is None:
                return None
            val = float(value)
            if math.isnan(val):
                return None
            return val
        except Exception:
            return None

    def objective(params):
        pipeline = _build_pipeline()
        # Chuẩn hóa tham số cho các trường hợp đặc biệt (vd: LogisticRegression)
        params = _normalize_params(model_name, params)
        pipeline.set_params(**params)

        trial_wall_start = time.perf_counter()
        avg_f1 = np.nan
        avg_auc = np.nan
        avg_exec_time = np.nan
        primary_value = np.nan

        try:
            if use_cross_validation:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores_primary = []
                all_f1 = []
                all_auc = []
                all_time = []
                
                for train_idx, test_idx in cv.split(X, y):
                    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Đo thời gian thực thi
                    start_time = time.time()
                    pipeline.fit(X_tr, y_tr)
                    y_pred = pipeline.predict(X_te)
                    exec_time = time.time() - start_time
                    
                    y_proba = None
                    if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                        y_proba = pipeline.predict_proba(X_te)
                    fold_metrics = compute_metrics(y_te, y_pred, y_proba)
                    scores_primary.append(fold_metrics[primary_metric])

                    # Thu thập metrics cho trace/multi-objective
                    all_f1.append(fold_metrics.get('f1', np.nan))
                    all_auc.append(fold_metrics.get('roc_auc', np.nan))
                    all_time.append(exec_time)
                
                avg_f1 = np.nanmean(all_f1) if all_f1 else np.nan
                avg_auc = np.nanmean(all_auc) if all_auc else np.nan
                avg_exec_time = np.mean(all_time) if all_time else np.nan
                primary_value = np.mean(scores_primary) if scores_primary else np.nan

                if multi_objective:
                    normalizer.collect(avg_f1, avg_auc, avg_exec_time)
                    score = normalizer.compute_objective(avg_f1, avg_auc, avg_exec_time)
                else:
                    score = primary_value
            else:
                # Đảm bảo biến holdout đã được thiết lập
                assert 'X_valid' in locals() or 'X_valid' in globals()
                assert 'y_valid' in locals() or 'y_valid' in globals()
                
                start_time = time.time()
                pipeline.fit(X, y)
                y_pred = pipeline.predict(X_valid)
                exec_time = time.time() - start_time
                
                y_proba = None
                if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                    y_proba = pipeline.predict_proba(X_valid)
                holdout_metrics = compute_metrics(y_valid, y_pred, y_proba)
                avg_f1 = holdout_metrics.get('f1', np.nan)
                avg_auc = holdout_metrics.get('roc_auc', np.nan)
                avg_exec_time = exec_time
                primary_value = holdout_metrics.get(primary_metric, np.nan)
                
                if multi_objective:
                    normalizer.collect(avg_f1, avg_auc, exec_time)
                    score = normalizer.compute_objective(avg_f1, avg_auc, exec_time)
                else:
                    score = primary_value
        except Exception as e:
            print(f"Lỗi khi đánh giá {params}: {e}")
            return 0.0
        finally:
            # Giải phóng pipeline và bộ nhớ sau mỗi trial
            if 'pipeline' in locals():
                del pipeline
            gc.collect()

        trial_wall = time.perf_counter() - trial_wall_start
        trace_meta = getattr(objective, "_trace_meta", None)
        if trace_meta and trace_meta.get('enabled'):
            trace_entry = {
                'f1': _clean_numeric(avg_f1),
                'roc_auc': _clean_numeric(avg_auc),
                'avg_fold_time': _clean_numeric(avg_exec_time),
                'primary': _clean_numeric(primary_value),
                'composite': _clean_numeric(score),
                'trial_wall_time': _clean_numeric(trial_wall)
            }
            trace_meta['buffer'].append(trace_entry)

        return score

    # Stepwise reporting để hỗ trợ Optuna Pruner
    def objective_stepwise(trial, params):
        pipeline = _build_pipeline()
        params = _normalize_params(model_name, dict(params))
        pipeline.set_params(**params)

        try:
            trial_wall_start = time.perf_counter()
            avg_f1 = np.nan
            avg_auc = np.nan
            avg_exec_time = np.nan
            primary_value = np.nan
            if use_cross_validation:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores_primary = []
                all_f1 = []
                all_auc = []
                all_time = []
                
                for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                    
                    start_time = time.time()
                    pipeline.fit(X_tr, y_tr)
                    y_pred = pipeline.predict(X_te)
                    exec_time = time.time() - start_time
                    
                    y_proba = None
                    if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                        y_proba = pipeline.predict_proba(X_te)
                    fold_metrics = compute_metrics(y_te, y_pred, y_proba)

                    all_f1.append(fold_metrics.get('f1', np.nan))
                    all_auc.append(fold_metrics.get('roc_auc', np.nan))
                    all_time.append(exec_time)

                    if multi_objective:
                        fold_score = normalizer.compute_objective(
                            fold_metrics.get('f1', np.nan),
                            fold_metrics.get('roc_auc', np.nan),
                            exec_time
                        )
                    else:
                        fold_score = float(fold_metrics[primary_metric])
                    
                    scores_primary.append(fold_score)
                    # Báo cáo kết quả trung gian theo từng fold
                    try:
                        trial.report(fold_score, step=fold_idx)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                    except optuna.exceptions.TrialPruned:
                        raise
                    except Exception:
                        # Nếu trial không hỗ trợ hoặc Optuna không sẵn có
                        pass
                
                if all_f1:
                    avg_f1 = np.nanmean(all_f1)
                if all_auc:
                    avg_auc = np.nanmean(all_auc)
                if all_time:
                    avg_exec_time = np.mean(all_time)
                if scores_primary:
                    primary_value = float(np.mean(scores_primary))

                if multi_objective and all_f1:
                    normalizer.collect(avg_f1, avg_auc, avg_exec_time)
                
                score = primary_value
                if multi_objective:
                    score = normalizer.compute_objective(avg_f1, avg_auc, avg_exec_time)
                if isinstance(score, (int, float, np.floating)) and not math.isnan(score):
                    result = float(score)
                else:
                    result = 0.0
            else:
                assert 'X_valid' in locals() or 'X_valid' in globals()
                assert 'y_valid' in locals() or 'y_valid' in globals()
                
                start_time = time.time()
                pipeline.fit(X, y)
                y_pred = pipeline.predict(X_valid)
                exec_time = time.time() - start_time
                
                y_proba = None
                if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                    y_proba = pipeline.predict_proba(X_valid)
                holdout_metrics = compute_metrics(y_valid, y_pred, y_proba)
                
                avg_f1 = holdout_metrics.get('f1', np.nan)
                avg_auc = holdout_metrics.get('roc_auc', np.nan)
                avg_exec_time = exec_time
                primary_value = float(holdout_metrics.get(primary_metric, np.nan))

                if multi_objective:
                    f1_val = holdout_metrics.get('f1', np.nan)
                    auc_val = holdout_metrics.get('roc_auc', np.nan)
                    normalizer.collect(f1_val, auc_val, exec_time)
                    score = normalizer.compute_objective(f1_val, auc_val, exec_time)
                else:
                    result = primary_value
                
                try:
                    trial.report(result, step=0)
                except Exception:
                    pass
            trial_wall = time.perf_counter() - trial_wall_start
            trace_meta = getattr(objective, "_trace_meta", None)
            if trace_meta and trace_meta.get('enabled'):
                trace_meta['buffer'].append({
                    'f1': _clean_numeric(avg_f1),
                    'roc_auc': _clean_numeric(avg_auc),
                    'avg_fold_time': _clean_numeric(avg_exec_time),
                    'primary': _clean_numeric(primary_value),
                    'composite': _clean_numeric(result),
                    'trial_wall_time': _clean_numeric(trial_wall)
                })
            return result
        except optuna.exceptions.TrialPruned:
            # Bubbles up pruning
            raise
        except Exception as e:
            print(f"Lỗi khi đánh giá (stepwise) {params}: {e}")
            return 0.0
        finally:
            # Giải phóng bộ nhớ sau mỗi trial
            gc.collect()

    # Gắn stepwise như thuộc tính của objective để agent có thể dùng
    try:
        objective._stepwise = objective_stepwise
    except Exception:
        pass

    objective._trace_meta = {'enabled': False, 'buffer': []}

    def _enable_trace():
        meta = getattr(objective, '_trace_meta', None)
        if meta is None:
            return
        meta['buffer'].clear()
        meta['enabled'] = True

    def _disable_trace():
        meta = getattr(objective, '_trace_meta', None)
        if meta is None:
            return
        meta['enabled'] = False

    def _consume_trace():
        meta = getattr(objective, '_trace_meta', None)
        if meta is None:
            return []
        data = list(meta['buffer'])
        meta['buffer'].clear()
        return data

    def _peek_trace():
        meta = getattr(objective, '_trace_meta', None)
        if meta is None:
            return []
        return list(meta['buffer'])

    objective.enable_trace = _enable_trace
    objective.disable_trace = _disable_trace
    objective.consume_trace = _consume_trace
    objective.peek_trace = _peek_trace

    return objective


# In[5]:


# ============================================================================
# BƯỚC 4: FRAMEWORK AMSCO (PHIÊN BẢN TỔNG QUÁT VÀ ĐÃ SỬA LỖI)
# ============================================================================

class KnowledgeHub:
    def __init__(self, max_trials_kept=100):
        self.trials = []
        self.best_score = -float('inf')
        self.best_params = None
        self.total_calls = 0
        self.best_iteration = None
        self.max_trials_kept = max_trials_kept  # Giới hạn trials để tránh tràn RAM

    def store(self, agent_id, params, score):
        self.total_calls += 1
        record = {
            'iteration': self.total_calls,
            'agent_id': agent_id,
            'params': params,
            'score': score
        }
        self.trials.append(record)
        
        # Giới hạn kích thước trials list để tránh tràn RAM (giữ N trials gần nhất)
        if len(self.trials) > self.max_trials_kept:
            self.trials = self.trials[-self.max_trials_kept:]
        
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
    def __init__(self, agent_id, objective_func, search_space, knowledge_hub, tolerance=0.0, patience=0):
        super().__init__(agent_id, objective_func, search_space, knowledge_hub)
        self.tolerance = 0.0 if tolerance is None else max(float(tolerance), 0.0)
        self.patience = max(int(patience or 0), 0)

    def run(self, budget):
        # print(f"    -> Running RandomAgent with budget: {budget}")
        best_record = self.knowledge_hub.get_best_trial()
        best_local = best_record.get('score') if isinstance(best_record, dict) else None
        if not isinstance(best_local, (int, float)) or not math.isfinite(best_local):
            best_local = -float('inf')
        no_improve = 0

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

            best_current = self.knowledge_hub.get_best_trial().get('score')
            if not isinstance(best_current, (int, float)) or not math.isfinite(best_current):
                best_current = -float('inf')
class BayesianAgent(StrategyAgent):
    """BayesianAgent: Giờ đã hoàn toàn linh hoạt"""

    def __init__(
        self,
        agent_id,
        objective_func,
        search_space,
        knowledge_hub,
        tolerance=0.0,
        patience=0,
        seed=42,
        warm_start_decay=0.95,
        warm_start_quality_percentile=60
    ):
        super().__init__(agent_id, objective_func, search_space, knowledge_hub)
        self.tolerance = 0.0 if tolerance is None else max(float(tolerance), 0.0)
        self.patience = max(int(patience or 0), 0)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.warm_start_decay = max(min(float(warm_start_decay), 0.9999), 1e-3)
        self.warm_start_quality_percentile = min(max(float(warm_start_quality_percentile), 0.0), 100.0)
        self._last_early_stop = False

    def _param_fingerprint(self, params):
        fingerprint = []
        for key, value in sorted(params.items()):
            if isinstance(value, float) or isinstance(value, np.floating):
                fingerprint.append((key, round(float(value), 10)))
            elif isinstance(value, np.integer):
                fingerprint.append((key, int(value)))
            else:
                fingerprint.append((key, value))
        return tuple(fingerprint)

    def run(self, budget):
        # print(f"    -> Running BayesianAgent with budget: {budget}")
        best_record = self.knowledge_hub.get_best_trial()
        best_local = best_record.get('score') if isinstance(best_record, dict) else None
        if not isinstance(best_local, (int, float)) or not math.isfinite(best_local):
            best_local = -float('inf')

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

            if hasattr(self.objective, "_stepwise"):
                score = self.objective._stepwise(trial, params)
            else:
                score = self.objective(params)

            self.knowledge_hub.store(self.agent_id, params, score)
            return score

        try:
            sampler = optuna.samplers.TPESampler(
                seed=self.seed,
                multivariate=True,
                constant_liar=True,
                n_startup_trials=min(10, budget)
            )
        except Exception:
            sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"BayesianAgent_{self.agent_id}_{self.seed}",
            load_if_exists=False
        )

        self._last_early_stop = False
        early_stop_state = {
            'best': best_local,
            'no_improve': 0
        }

        def _early_stop_callback(study_ref, trial):
            if not self.patience:
                return
            if trial.state != optuna.trial.TrialState.COMPLETE:
                return
            value = trial.value
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                return
            if (value - early_stop_state['best']) > self.tolerance:
                early_stop_state['best'] = value
                early_stop_state['no_improve'] = 0
            else:
                early_stop_state['no_improve'] += 1
                if early_stop_state['no_improve'] >= self.patience:
                    self._last_early_stop = True
                    study_ref.stop()

        callbacks = [_early_stop_callback] if self.patience else []

        existing_trials = self.knowledge_hub.get_all_trials()
        seen_param_keys = set(
            self._param_fingerprint(t['params'])
            for t in existing_trials
            if isinstance(t, dict) and 'params' in t
        )

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

            filtered = [
                t for t in existing_trials
                if isinstance(t, dict)
                and 'params' in t
                and 'score' in t
                and isinstance(t['score'], (int, float))
                and math.isfinite(t['score'])
            ]
            filtered.sort(key=lambda r: r['score'], reverse=True)

            scores_array = np.array([trial['score'] for trial in filtered], dtype=float)
            quality_threshold = -float('inf')
            if scores_array.size:
                percentile = np.percentile(scores_array, self.warm_start_quality_percentile)
                quality_threshold = float(percentile)

            if len(filtered) > 300:
                top_part = [t for t in filtered[:150] if t['score'] >= quality_threshold]
                remaining = [t for t in filtered[150:] if t['score'] >= quality_threshold]
                random_part = self._rng.choice(remaining, size=min(50, len(remaining)), replace=False).tolist() if remaining else []
                selected = top_part + random_part
            else:
                selected = [t for t in filtered if t['score'] >= quality_threshold]

            best_score_record = filtered[0]['score'] if filtered else None

            for idx, t in enumerate(selected):
                params_dict = t['params']
                valid_params = {k: v for k, v in params_dict.items() if k in full_distributions}
                if not valid_params:
                    continue
                key_tuple = self._param_fingerprint(valid_params)
                if key_tuple in seen_param_keys:
                    continue
                seen_param_keys.add(key_tuple)
                sub_distributions = {k: full_distributions[k] for k in valid_params.keys()}
                try:
                    weight = self.warm_start_decay ** idx
                    if isinstance(best_score_record, (int, float)) and math.isfinite(best_score_record):
                        weighted_value = best_score_record + (t['score'] - best_score_record) * weight
                    else:
                        weighted_value = t['score'] * weight
                    frozen_trial = optuna.trial.create_trial(
                        params=valid_params,
                        distributions=sub_distributions,
                        value=weighted_value
                    )
                    study.add_trial(frozen_trial)
                except Exception:
                    continue

            best_params = best_record.get('params') if isinstance(best_record, dict) else None
            best_score = best_record.get('score') if isinstance(best_record, dict) else None
            if best_params and isinstance(best_score, (int, float)) and math.isfinite(best_score):
                perturb_variants = []
                numeric_names = [
                    name for name, details in self.search_space.items()
                    if details[0] in ['float', 'int'] and name in best_params
                ][:3]
                for name in numeric_names:
                    details = self.search_space[name]
                    param_type = details[0]
                    low, high = details[1], details[2]
                    current_val = best_params.get(name)
                    if current_val is None:
                        continue
                    if param_type == 'float':
                        scale = 0.1 * (high - low) or (abs(current_val) * 0.1) or 1e-3
                        draws = self._rng.normal(loc=current_val, scale=scale, size=5)
                        candidates = [float(np.clip(v, low, high)) for v in draws]
                    else:
                        scale = max(1.0, 0.1 * (high - low + 1))
                        draws = self._rng.normal(loc=current_val, scale=scale, size=5)
                        candidates = [int(np.clip(round(v), low, high)) for v in draws]
                    for cand in candidates:
                        if cand == current_val:
                            continue
                        new_params = dict(best_params)
                        new_params[name] = cand
                        perturb_variants.append(new_params)

                seen_local = set()
                for pv in perturb_variants:
                    key_tuple = self._param_fingerprint(pv)
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
                        seen_param_keys.add(key_tuple)
                    except Exception:
                        continue

        study.optimize(optuna_objective, n_trials=budget, show_progress_bar=False, callbacks=callbacks)

class GridAgent(StrategyAgent):
        """GridAgent: Linh hoạt (tinh chỉnh 2 tham số quan trọng nhất)

        Lưu ý hiệu năng:
        - Giới hạn số vòng lặp tinh chỉnh cục bộ và dừng sớm nếu không cải thiện
          để tránh chiếm dụng slice quá lâu.
        """
        MAX_LOOPS = 2                 # Số vòng lặp tinh chỉnh cục bộ tối đa trong một lần run()
        EARLY_STOP_NO_IMPROVE = 1     # Dừng nếu không cải thiện sau N vòng lặp

        def __init__(
            self,
            agent_id,
            objective_func,
            search_space,
            knowledge_hub,
            tolerance=0.0,
            patience=0,
            seed=1234
        ):
            super().__init__(agent_id, objective_func, search_space, knowledge_hub)
            self.tolerance = 0.0 if tolerance is None else max(float(tolerance), 0.0)
            self.patience = max(int(patience or 0), 0)
            self._rng = np.random.default_rng(int(seed))

        def _param_fingerprint(self, params):
            fp = []
            for key, value in sorted(params.items()):
                if isinstance(value, (float, np.floating)):
                    fp.append((key, round(float(value), 10)))
                elif isinstance(value, np.integer):
                    fp.append((key, int(value)))
                else:
                    fp.append((key, value))
            return tuple(fp)

        def _generate_candidates(self, name, details, current_val):
            param_type = details[0]
            low, high = details[1], details[2]
            candidates = []
            if current_val is None:
                current_val = (low + high) / 2.0 if param_type == 'float' else int(round((low + high) / 2.0))

            if param_type == 'float':
                span = max(high - low, abs(current_val), 1e-3)
                local_scale = 0.15 * span
                samples = self._rng.normal(loc=current_val, scale=local_scale, size=4)
                lin_points = np.linspace(max(low, current_val - local_scale), min(high, current_val + local_scale), num=3)
                mix = list(lin_points) + samples.tolist()
                for val in mix:
                    clipped = float(np.clip(val, low, high))
                    if abs(clipped - current_val) < 1e-12:
                        continue
                    candidates.append(clipped)
            elif param_type == 'int':
                span = max(high - low, 1)
                local_scale = max(1.0, 0.2 * span)
                samples = self._rng.normal(loc=current_val, scale=local_scale, size=5)
                base_neighbors = [current_val - 1, current_val, current_val + 1]
                for val in list(samples) + base_neighbors:
                    clipped = int(np.clip(round(val), low, high))
                    if clipped == current_val:
                        continue
                    candidates.append(clipped)
            return candidates

        def run(self, budget):
            # print(f"    -> Running GridAgent with budget: {budget}")
            trials_run = 0
            tried_param_sets = {
                self._param_fingerprint(t['params'])
                for t in self.knowledge_hub.get_all_trials() if 'params' in t
            }

            loops_count = 0
            no_improve_runs = 0
            best_record = self.knowledge_hub.get_best_trial()
            best_local = best_record.get('score') if isinstance(best_record, dict) else None
            if not isinstance(best_local, (int, float)) or not math.isfinite(best_local):
                best_local = -float('inf')
            no_improve_evals = 0

            while trials_run < budget:
                current_best = self.knowledge_hub.get_best_trial()
                best_before = current_best.get('score') if isinstance(current_best, dict) else None
                best_params = current_best.get('params') if isinstance(current_best, dict) else None
                if not best_params:
                    return  # Không có gì để tinh chỉnh

                params_to_tune = []
                for name, details in self.search_space.items():
                    p_type = details[0] if isinstance(details, (list, tuple)) and details else None
                    if p_type in ['float', 'int']:
                        params_to_tune.append(name)
                    if len(params_to_tune) >= 2:
                        break
                if len(params_to_tune) == 0:
                    return  # Không có tham số số để tinh chỉnh

                seen_local = set()
                local_grid = []

                def _add_candidate(candidate_params):
                    fingerprint = self._param_fingerprint(candidate_params)
                    if fingerprint in seen_local:
                        return
                    seen_local.add(fingerprint)
                    local_grid.append(candidate_params)

                _add_candidate(dict(best_params))

                p1_name = params_to_tune[0]
                p1_details = self.search_space[p1_name]
                p1_val = best_params.get(p1_name)
                for cand in self._generate_candidates(p1_name, p1_details, p1_val):
                    new_params = dict(best_params)
                    new_params[p1_name] = cand
                    _add_candidate(new_params)

                for params in local_grid:
                    if trials_run >= budget:
                        break
                    param_key = self._param_fingerprint(params)
                    if param_key in tried_param_sets:
                        continue
                    score = self.objective(params)
                    self.knowledge_hub.store(self.agent_id, params, score)
                    tried_param_sets.add(param_key)
                    trials_run += 1

                    best_current = self.knowledge_hub.get_best_trial().get('score')
                    if isinstance(best_current, (int, float)) and math.isfinite(best_current):
                        if (best_current - best_local) > self.tolerance:
                            best_local = best_current
                            no_improve_evals = 0
                        else:
                            no_improve_evals += 1
                            if self.patience and no_improve_evals >= self.patience:
                                return

                best_after = self.knowledge_hub.get_best_trial().get('score')
                if not isinstance(best_before, (int, float)) or not math.isfinite(best_before):
                    best_before = -float('inf')
                if not isinstance(best_after, (int, float)) or not math.isfinite(best_after):
                    best_after = -float('inf')

                if (best_after - best_before) <= self.tolerance:
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
    def __init__(self, agent_ids, max_history_per_agent=30):
        self.agent_ids = agent_ids
        self.history = defaultdict(list)
        self.max_history_per_agent = max_history_per_agent  # Giới hạn history

    def update(self, all_trials):
        # Không reset toàn bộ history, chỉ cập nhật incremental
        for trial in all_trials:
            agent_id = trial['agent_id']
            if agent_id not in self.history:
                self.history[agent_id] = []
            # Chỉ thêm trial mới chưa có trong history
            if not self.history[agent_id] or trial['iteration'] > len(self.history[agent_id]):
                self.history[agent_id].append(trial['score'])
                # Giới hạn kích thước history
                if len(self.history[agent_id]) > self.max_history_per_agent:
                    self.history[agent_id] = self.history[agent_id][-self.max_history_per_agent:]

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
    """
    Meta-controller với Performance Score và Power-weighted Budget Allocation
    
    Cơ chế theo lý thuyết AMSCO:
    - Performance Score: S_{i,t} = α·(Δ_{i,t}/(C_{i,t}+ε)) + β·sqrt(ln(ΣN_j)/(N_i+ε))
    - Budget Allocation: b_{i,t} = B_t · (S_{i,t})^γ / Σ(S_{j,t})^γ
    """
    def __init__(self, agent_ids, verbose=False, alpha=0.7, beta=0.3, gamma=1.0, epsilon=1e-5):
        self.agent_ids = agent_ids
        self.agent_pulls = {agent_id: 0 for agent_id in agent_ids}  # N_{i,t}: Số trials
        self.agent_rewards = {agent_id: 0.0 for agent_id in agent_ids}
        self.agent_improvements = {agent_id: 0.0 for agent_id in agent_ids}  # Δ_{i,t}: Mức cải thiện
        self.agent_costs = {agent_id: [] for agent_id in agent_ids}  # C_{i,t}: Chi phí tính toán
        self.total_pulls = 0
        self.verbose = verbose
        
        # Các tham số theo lý thuyết
        self.alpha = alpha  # Trọng số cho hiệu suất chi phí (exploitation)
        self.beta = beta    # Trọng số cho tiềm năng khám phá (exploration - UCB1)
        self.gamma = gamma  # Hệ số áp lực khai thác (exploitation pressure)
        self.epsilon = epsilon  # Hằng số ổn định số học (10^-5)

    def allocate(self, slice_budget, performance_monitor=None):
        """
        Phân bổ budget theo lý thuyết AMSCO với Performance Score và Power-weighted Allocation
        
        Args:
            slice_budget: Tổng budget B_t cho chu kỳ hiện tại
            performance_monitor: PerformanceMonitor để lấy thông tin cải thiện
        """
        # === KHỞI TẠO AGENTS CHƯA CHẠY ===
        uninitialized_agents = [aid for aid, pulls in self.agent_pulls.items() if pulls == 0]
        if uninitialized_agents:
            agent_to_run = uninitialized_agents[0]
            allocations = {agent_id: 0 for agent_id in self.agent_ids}
            allocations[agent_to_run] = slice_budget
            if self.verbose:
                print(f"  [MetaController] Khởi tạo agent: {agent_to_run}")
            return allocations

        # === TÍNH TOÁN PERFORMANCE SCORE S_{i,t} ===
        performance_scores = {}
        
        for agent_id in self.agent_ids:
            N_i = self.agent_pulls[agent_id]  # Số trials của agent i
            
            # Thành phần 1: Hiệu suất chi phí = Δ_{i,t} / (C_{i,t} + ε)
            Delta_i = self.agent_improvements.get(agent_id, 0.0)  # Mức cải thiện
            C_i = np.mean(self.agent_costs[agent_id]) if self.agent_costs[agent_id] else 1.0  # Chi phí trung bình
            cost_efficiency = Delta_i / (C_i + self.epsilon)
            
            # Thành phần 2: Tiềm năng khám phá (UCB1) = sqrt(ln(Σ N_j) / (N_i + ε))
            total_pulls = sum(self.agent_pulls.values())
            if total_pulls > 0 and N_i > 0:
                exploration_potential = math.sqrt(math.log(total_pulls) / (N_i + self.epsilon))
            else:
                # Agent chưa được thử: gán giá trị cao nhưng được giới hạn để tránh overflow khi lũy thừa
                exploration_potential = 10.0  # Giảm từ 1e10 xuống 10.0
            
            # Performance Score: S_{i,t} = α · cost_efficiency + β · exploration_potential
            S_i = self.alpha * cost_efficiency + self.beta * exploration_potential
            performance_scores[agent_id] = max(S_i, self.epsilon)  # Đảm bảo S_i > 0

        # === PHÂN BỔ BUDGET THEO POWER-WEIGHTED ALLOCATION ===
        # b_{i,t} = B_t · (S_{i,t})^γ / Σ(S_{j,t})^γ
        
        # Tính tổng (S_{j,t})^γ
        powered_scores = {agent_id: (score ** self.gamma) for agent_id, score in performance_scores.items()}
        total_powered = sum(powered_scores.values())
        
        # Kiểm tra và xử lý trường hợp total_powered không hợp lệ
        if not np.isfinite(total_powered) or total_powered <= 0:
            # Fallback: phân bổ đều cho tất cả agents
            num_agents = len(self.agent_ids)
            base_allocation = slice_budget // num_agents
            allocations = {agent_id: base_allocation for agent_id in self.agent_ids}
            # Phần dư cho agent đầu tiên
            allocations[self.agent_ids[0]] += slice_budget - (base_allocation * num_agents)
            if self.verbose:
                print(f"  [MetaController] Warning: total_powered={total_powered}, using equal allocation")
            return allocations
        
        # Phân bổ budget theo tỷ lệ
        allocations = {}
        remaining_budget = slice_budget
        
        for i, (agent_id, powered_score) in enumerate(powered_scores.items()):
            if i < len(powered_scores) - 1:
                # Kiểm tra powered_score hợp lệ trước khi tính budget
                if not np.isfinite(powered_score):
                    budget = 0
                else:
                    # Tính budget theo công thức
                    budget = int(slice_budget * (powered_score / total_powered))
                allocations[agent_id] = budget
                remaining_budget -= budget
            else:
                # Agent cuối cùng nhận phần còn lại (tránh làm tròn)
                allocations[agent_id] = remaining_budget
        
        if self.verbose:
            score_str = {k: f"S={v:.3f}" for k, v in performance_scores.items()}
            alloc_str = {k: v for k, v in allocations.items() if v > 0}
            print(f"  [MetaController] Performance Scores: {score_str}")
            print(f"  [MetaController] Budget Allocation: {alloc_str}")

        return allocations

    def update(self, agent_id_to_update, reward, improvement=0.0, cost=1.0):
        """
        Cập nhật thông tin agent sau khi chạy
        
        Args:
            agent_id_to_update: ID của agent
            reward: Phần thưởng (normalized score)
            improvement: Δ_{i,t} - Mức cải thiện hiệu suất (e.g., ΔAUC)
            cost: C_{i,t} - Chi phí tính toán (thời gian thực thi)
        """
        if reward >= 0:
            self.agent_rewards[agent_id_to_update] += reward
            self.agent_pulls[agent_id_to_update] += 1
            self.total_pulls += 1
            
            # Cập nhật mức cải thiện và chi phí cho công thức Performance Score
            self.agent_improvements[agent_id_to_update] = max(improvement, 0.0)
            self.agent_costs[agent_id_to_update].append(cost)
            
            # Giữ lại tối đa 15 giá trị chi phí gần nhất để tính trung bình
            if len(self.agent_costs[agent_id_to_update]) > 15:
                self.agent_costs[agent_id_to_update] = self.agent_costs[agent_id_to_update][-15:]


class AMSCO_Orchestrator:
    """Orchestrator: Giờ nhận objective và search_space"""
    def __init__(
        self,
        objective_func,
        search_space,
        total_budget,
        slice_budget,
        verbose=False,
        early_stopping_rounds=0,
        tolerance=1e-4,
        base_seed=42
    ):
        self.total_budget = total_budget
        self.slice_budget = slice_budget
        self.verbose = verbose
        self.early_stopping_rounds = max(int(early_stopping_rounds or 0), 0)
        self.tolerance = 0.0 if tolerance is None else max(float(tolerance), 0.0)

        # Giới hạn trials lưu trữ dựa trên total_budget
        max_trials = min(200, total_budget)  # Tối đa 200 trials trong memory
        self.knowledge_hub = KnowledgeHub(max_trials_kept=max_trials)

        self._base_seed = int(base_seed)
        self.agents = {
            "Random": RandomAgent(
                "Random",
                objective_func,
                search_space,
                self.knowledge_hub,
                tolerance=self.tolerance,
                patience=self.early_stopping_rounds
            ),
            "Bayesian": BayesianAgent(
                "Bayesian",
                objective_func,
                search_space,
                self.knowledge_hub,
                tolerance=self.tolerance,
                patience=self.early_stopping_rounds,
                seed=self._base_seed + 1
            ),
            "Grid": GridAgent(
                "Grid",
                objective_func,
                search_space,
                self.knowledge_hub,
                tolerance=self.tolerance,
                patience=self.early_stopping_rounds,
                seed=self._base_seed + 2
            )
        }
        agent_ids = list(self.agents.keys())

        self.performance_monitor = PerformanceMonitor(agent_ids, max_history_per_agent=50)
        self.meta_controller = MetaController_UCB1(
            agent_ids, 
            verbose=self.verbose,
            alpha=0.7,      # Trọng số cho hiệu suất chi phí (exploitation)
            beta=0.3,       # Trọng số cho tiềm năng khám phá (exploration)
            gamma=1.0,      # Hệ số áp lực khai thác ban đầu (sẽ tăng dần)
            epsilon=1e-5    # Hằng số ổn định số học
        )
        self.agent_budget_usage = {agent_id: 0 for agent_id in agent_ids}
        self.agent_costs = {agent_id: [] for agent_id in agent_ids}  # Lưu chi phí thực thi
        self._early_stopped = False
        self._patience_used = 0

    def run(self):
        current_budget = self.total_budget
        slice_num = 1
        total_slices = max(1, self.total_budget // self.slice_budget)  # Ước tính tổng số slices

        no_improve_slices = 0

        while current_budget > 0:
            # print(f"\n--- Slice {slice_num} | Budget remaining: {current_budget} ---")

            budget_for_slice = min(self.slice_budget, current_budget)

            best_before = self.knowledge_hub.get_best_trial().get('score')
            best_before = best_before if isinstance(best_before, (int, float)) else -float('inf')

            # === TĂNG DẦN γ THEO THỜI GIAN (Exploration → Exploitation) ===
            # γ tăng từ 1.0 → 3.0 theo tiến trình: γ(t) = 1.0 + 2.0 * (t / T)
            progress = min(1.0, slice_num / total_slices)
            self.meta_controller.gamma = 1.0 + 2.0 * progress
            
            # === PHÂN BỔ BUDGET ===
            allocations = self.meta_controller.allocate(budget_for_slice, self.performance_monitor)

            for agent_id, budget in allocations.items():
                if budget > 0:
                    # print(f"... Giving budget to {agent_id}")
                    
                    # Đo thời gian thực thi (cost)
                    start_time = time.time()
                    self.agents[agent_id].run(budget)
                    execution_time = time.time() - start_time
                    
                    self.agent_budget_usage[agent_id] += budget
                    self.agent_costs[agent_id].append(execution_time)
                    
                    # Giới hạn kích thước list chi phí để tránh tràn RAM (giữ 20 giá trị gần nhất)
                    if len(self.agent_costs[agent_id]) > 20:
                        self.agent_costs[agent_id] = self.agent_costs[agent_id][-20:]

                    all_trials = self.knowledge_hub.get_all_trials()
                    self.performance_monitor.update(all_trials)

                    # Tính mức cải thiện (improvement)
                    best_after_agent = self.knowledge_hub.get_best_trial().get('score')
                    best_after_agent = best_after_agent if isinstance(best_after_agent, (int, float)) else -float('inf')
                    improvement = max(0.0, best_after_agent - best_before)
                    
                    # Cập nhật Meta-Controller với đầy đủ thông tin
                    reward = self.performance_monitor.get_agent_rewards()[agent_id]
                    self.meta_controller.update(
                        agent_id, 
                        reward=reward,
                        improvement=improvement,
                        cost=execution_time
                    )

            current_budget -= budget_for_slice

            best_after = self.knowledge_hub.get_best_trial().get('score')
            best_after = best_after if isinstance(best_after, (int, float)) else -float('inf')
            if best_after - best_before > self.tolerance:
                no_improve_slices = 0
            else:
                no_improve_slices += 1
                if self.early_stopping_rounds and no_improve_slices >= self.early_stopping_rounds:
                    self._early_stopped = True
                    self._patience_used = no_improve_slices
                    break

            slice_num += 1
            
            # Giải phóng bộ nhớ định kỳ sau mỗi slice
            if slice_num % 5 == 0:
                gc.collect()

        final_result = self.knowledge_hub.get_best_trial()
        return {
            'score': final_result.get('score'),
            'params': final_result.get('params'),
            'iteration_to_best': final_result.get('iteration'),
            'total_trials': self.knowledge_hub.total_calls,
            'agent_pulls': dict(self.meta_controller.agent_pulls),
            'agent_budget_usage': dict(self.agent_budget_usage),
            'agent_improvements': dict(self.meta_controller.agent_improvements),  # Mức cải thiện của agents
            'agent_avg_costs': {k: np.mean(v) if v else 0.0 for k, v in self.agent_costs.items()},  # Chi phí trung bình
            'final_gamma': self.meta_controller.gamma,  # Giá trị γ cuối cùng
            'early_stopped': self._early_stopped,
            'patience_used': self._patience_used if self._early_stopped else None
        }


# In[6]:


# ============================================================================
# BƯỚC 5: WARM-UP VÀ CÁC TRÌNH TỐI ƯU HÓA
# ============================================================================

def warmup_normalizer(objective, search_space, n_warmup_trials=5):
    """
    Chạy warm-up để thu thập dữ liệu và tính bounds cho normalizer.
    Thực hiện random sampling để khám phá không gian tìm kiếm.
    Giảm từ 10 xuống 5 trials để tiết kiệm RAM.
    """
    print(f"  [Warm-up] Đang chạy {n_warmup_trials} trials để tính bounds...")
    trace_meta = getattr(objective, '_trace_meta', None)
    trace_was_enabled = False
    if trace_meta and trace_meta.get('enabled'):
        trace_was_enabled = True
        trace_meta['enabled'] = False
    for _ in range(n_warmup_trials):
        params = {}
        for name, details in search_space.items():
            type = details[0]
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
                choices = details[1]
                params[name] = random.choice(choices)
        
        # Chạy objective để thu thập metrics
        try:
            objective(params)
        except Exception as e:
            print(f"  [Warm-up] Lỗi: {e}")
            continue
    if trace_meta is not None and trace_was_enabled:
        trace_meta['enabled'] = True

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

def run_random_search(objective, search_space, n_trials, normalizer=None):
    """Chạy Random Search (sử dụng Optuna)"""
    # Warm-up nếu dùng multi-objective
    if normalizer is not None:
        warmup_normalizer(objective, search_space, n_warmup_trials=min(10, n_trials // 10))
        normalizer.compute_bounds()
        print(f"  [Warm-up] Bounds computed: F1[{normalizer.f1_min:.4f}, {normalizer.f1_max:.4f}], "
              f"AUC[{normalizer.auc_min:.4f}, {normalizer.auc_max:.4f}], "
              f"Time[{normalizer.time_min:.4f}, {normalizer.time_max:.4f}]")
    
    sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(direction='maximize', sampler=sampler)

    # Hàm mục tiêu cho Optuna
    def optuna_objective(trial):
        trial_wall_start = time.perf_counter()
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

def run_optuna_tpe(objective, search_space, n_trials, early_stopping_rounds=15, tolerance=1e-3, normalizer=None):
    """
    Chạy Optuna TPE với MedianPruner và cơ chế dừng sớm tối ưu.
    
    Cải tiến: 
    - Multivariate TPE sampler để tối ưu hóa hiệu quả
    - MedianPruner với warmup steps
    - Early stopping callback cải tiến với logging
    - Hỗ trợ stepwise reporting cho pruning
    
    Args:
        objective:  Hàm mục tiêu
        search_space: Không gian tìm kiếm
        n_trials: Số trials tối đa
        early_stopping_rounds: Số trials không cải thiện trước khi dừng (default: 15)
        tolerance: Ngưỡng cải thiện tối thiểu (default: 1e-3)
        normalizer:  MultiObjectiveNormalizer (optional)
    
    Returns:
        tuple: (best_score, best_params, diagnostics)
    """
    
    # === WARM-UP CHO MULTI-OBJECTIVE ===
    if normalizer is not None:
        warmup_trials = min(10, max(5, n_trials // 10))
        print(f"  [Warm-up] Running {warmup_trials} trials to compute bounds...")
        warmup_normalizer(objective, search_space, n_warmup_trials=warmup_trials)
        normalizer.compute_bounds()
        print(f"  [Warm-up] Bounds:  F1[{normalizer.f1_min:.4f}, {normalizer.f1_max:.4f}], "
              f"AUC[{normalizer.auc_min:.4f}, {normalizer.auc_max:.4f}], "
              f"Time[{normalizer.time_min:.4f}s, {normalizer.time_max:.4f}s]")
    
    # === SAMPLER:  TPE với Multivariate ===
    try:
        sampler = optuna.samplers.TPESampler(
            seed=42,
            multivariate=True,  # Tối ưu hóa nhiều tham số cùng lúc
            group=True,  # Nhóm các tham số liên quan
            constant_liar=True,  # Xử lý parallel trials
            n_startup_trials=min(10, n_trials // 10)  # Khám phá ban đầu
        )
        # print(f"  [Optuna] Using Multivariate TPE sampler")
    except TypeError:
        # Fallback cho phiên bản Optuna cũ
        sampler = optuna.samplers.TPESampler(seed=42)
        # print(f"  [Optuna] Using Standard TPE sampler (multivariate not supported)")
    
    # === PRUNER: MedianPruner với warmup ===
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Không prune 5 trials đầu
        n_warmup_steps=1,    # Chờ ít nhất 1 step trước khi prune
        interval_steps=1     # Kiểm tra sau mỗi step
    )
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=f"optuna_tpe_{int(time.time())}"
    )
    
    # === EARLY STOPPING STATE ===
    tolerance = max(float(tolerance or 0.0), 0.0)
    early_stop_state = {
        'best_value': -float('inf'),
        'no_improve_rounds': 0,
        'stopped':  False,
        'best_trial_number': -1
    }
    
    # === OBJECTIVE FUNCTION ===
    def optuna_objective(trial):
        """Wrapper objective với stepwise reporting."""
        trial_start = time.perf_counter()
        
        # Suggest parameters
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
        
        # Evaluate với stepwise reporting (nếu có)
        if hasattr(objective, "_stepwise"):
            try:
                score = objective._stepwise(trial, params)
            except optuna.exceptions.TrialPruned:
                # Trial bị pruned - đây là hành vi mong muốn
                raise
        else:
            # Fallback:  đánh giá thông thường
            score = objective(params)
            # Report 1 step để pruner có dữ liệu (dù không hiệu quả bằng stepwise)
            try:
                trial.report(float(score), step=0)
            except Exception:
                pass
        
        trial_time = time.perf_counter() - trial_start
        
        # Lưu metadata
        trial.set_user_attr("execution_time", trial_time)
        
        return score
    
    # === EARLY STOPPING CALLBACK ===
    def _early_stop_callback(study_ref, trial):
        """
        Dừng tối ưu hóa khi không cải thiện sau `early_stopping_rounds` trials.
        
        Cải tiến:
        - Chỉ kiểm tra completed trials
        - Logging chi tiết
        - So sánh với tolerance
        """
        if not early_stopping_rounds or early_stopping_rounds <= 0:
            return
        
        # Chỉ xử lý completed trials
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        
        value = trial.value
        if not isinstance(value, (int, float)) or math.isnan(value):
            return
        
        # Kiểm tra cải thiện
        improvement = value - early_stop_state['best_value']
        
        if improvement > tolerance:
            # Có cải thiện đáng kể
            early_stop_state['best_value'] = value
            early_stop_state['no_improve_rounds'] = 0
            early_stop_state['best_trial_number'] = trial. number
            # print(f"  [Optuna] ✓ New best:  {value:.6f} (trial {trial.number}, +{improvement:.6f})")
        else:
            # Không cải thiện
            early_stop_state['no_improve_rounds'] += 1
            
            if early_stop_state['no_improve_rounds'] >= early_stopping_rounds:
                early_stop_state['stopped'] = True
                # print(f"  [Optuna] ⊗ Early stopping at trial {trial.number}")
                # print(f"           No improvement for {early_stopping_rounds} trials (tolerance={tolerance})")
                # print(f"           Best score: {early_stop_state['best_value']:.6f} (trial {early_stop_state['best_trial_number']})")
                study_ref.stop()
    
    # Setup callbacks
    callbacks = [_early_stop_callback] if early_stopping_rounds > 0 else []
    
    # === RUN OPTIMIZATION ===
    # print(f"  [Optuna] Starting optimization (max_trials={n_trials}, patience={early_stopping_rounds})")
    opt_start = time.perf_counter()
    
    study.optimize(
        optuna_objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=callbacks
    )
    
    opt_time = time.perf_counter() - opt_start
    
    # === STATISTICS ===
    completed_trials = [t for t in study.trials if t.state == optuna.trial. TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    # print(f"  [Optuna] Optimization finished in {opt_time:.2f}s")
    # print(f"           Completed: {len(completed_trials)}, Pruned: {len(pruned_trials)}, Failed: {len(failed_trials)}")
    # print(f"           Early stopped: {early_stop_state['stopped']}")
    # print(f"           Best score: {study.best_trial.value:.6f} (trial {study.best_trial.number})")
    
    # === CONVERGENCE HISTORY ===
    best_so_far = -float('inf')
    conv_history = []
    for t in sorted(completed_trials, key=lambda tr: tr.number):
        if isinstance(t.value, (int, float)) and not math.isnan(t.value):
            best_so_far = max(best_so_far, float(t.value))
            conv_history.append(best_so_far)
    
    # === DIAGNOSTICS ===
    diagnostics = {
        'total_trials': len(study.trials),
        'completed_trials': len(completed_trials),
        'pruned_trials': len(pruned_trials),
        'failed_trials': len(failed_trials),
        'iteration_to_best': study.best_trial.number + 1,
        'early_stopped': early_stop_state['stopped'],
        'patience_used': early_stop_state['no_improve_rounds'],
        'pruner':  'MedianPruner',
        'convergence_history': conv_history,
        'optimization_time': opt_time,
        'avg_trial_time': opt_time / max(len(completed_trials), 1)
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


def run_amsco_optimizer(
    objective,
    search_space,
    n_trials,
    slice_budget,
    verbose=False,
    early_stopping_rounds=0,
    tolerance=1e-4,
    seed=None,
    normalizer=None
):
    """Chạy AMSCO (phương pháp của chúng ta)"""
    # Warm-up nếu dùng multi-objective
    if normalizer is not None:
        warmup_normalizer(objective, search_space, n_warmup_trials=min(10, n_trials // 10))
        normalizer.compute_bounds()
        print(f"  [Warm-up] Bounds computed: F1[{normalizer.f1_min:.4f}, {normalizer.f1_max:.4f}], "
              f"AUC[{normalizer.auc_min:.4f}, {normalizer.auc_max:.4f}], "
              f"Time[{normalizer.time_min:.4f}, {normalizer.time_max:.4f}]")
    
    orchestrator = AMSCO_Orchestrator(
        objective_func=objective,
        search_space=search_space,
        total_budget=n_trials,
        slice_budget=slice_budget,
        verbose=verbose,
        early_stopping_rounds=early_stopping_rounds,
        tolerance=tolerance,
        base_seed=seed if seed is not None else 42
    )
    result = orchestrator.run()

    # Xây dựng convergence history từ KnowledgeHub (best-so-far theo iteration)
    all_trials = sorted(
        orchestrator.knowledge_hub.get_all_trials(),
        key=lambda x: x.get('iteration', 0)
    )
    best_so_far = -float('inf')
    conv_history = []
    for tr in all_trials:
        s = tr.get('score')
        if isinstance(s, (int, float)) and not math.isnan(s):
            best_so_far = max(best_so_far, float(s))
            conv_history.append(best_so_far)

    diagnostics = {
        'total_trials': result.get('total_trials'),
        'iteration_to_best': result.get('iteration_to_best'),
        'agent_pulls': result.get('agent_pulls'),
        'early_stopped': result.get('early_stopped'),
        'patience_used': result.get('patience_used'),
        'convergence_history': conv_history,
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
    method_name, inner_trials, outer_folds, inner_folds, slice_budget,
    sampler=None
):
    """Tính Nested CV (k-outer, m-inner) cho một phương pháp (optimizer).
    Trả về dict: {accuracy, f1, roc_auc, time} (trung bình across outer folds; time = tổng thời gian tối ưu inner).
    """
    def _run_optimizer(objective):
        if method_name == 'Random Search':
            return run_random_search(objective, search_space, inner_trials)
        if method_name == 'Optuna (TPE)':
            return run_optuna_tpe(
                objective, 
                search_space, 
                inner_trials,
                early_stopping_rounds=OPTUNA_EARLY_STOP_TRIALS,
                tolerance=AMSCO_EARLY_STOP_TOLERANCE
            )
        if method_name == 'Hyperopt (TPE)':
            return run_hyperopt_tpe(objective, search_space, inner_trials)
        if method_name == 'AMSCO':
            return run_amsco_optimizer(
                objective,
                search_space,
                inner_trials,
                slice_budget,
                verbose=False,
                early_stopping_rounds=AMSCO_EARLY_STOP_SLICES,
                tolerance=AMSCO_EARLY_STOP_TOLERANCE
            )
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
            cv_folds=inner_folds,
            sampler=sampler
        )

        start = time.time()
        _, best_params, _ = _run_optimizer(objective_inner)
        total_time += (time.time() - start)

        model = None
        if model_name == 'logistic_regression':
            if sampler is None:
                model = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced')
            else:
                model = LogisticRegression(random_state=42, max_iter=2000)
        elif model_name == 'random_forest':
            if sampler is None:
                model = RandomForestClassifier(random_state=42, n_jobs=4, class_weight='balanced_subsample')
            else:
                model = RandomForestClassifier(random_state=42, n_jobs=4)
        elif model_name == 'xgboost':
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        elif model_name == 'lightgbm':
            model = lgb.LGBMClassifier(random_state=42, verbosity=-1)
        else:
            raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ.")

        if sampler is not None:
            pipeline = ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('sampler', clone(sampler)),
                ('classifier', model)
            ])
        else:
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
    QUICK_MODE = False  # Đặt False để chạy full pipeline với Nested CV
    # Chạy full: breast_cancer và telco
    DATASETS = ['adult']  # 2 datasets
    MODELS = ['random_forest', 'logistic_regression']
    TOTAL_TRIALS = 50  # Giảm từ 100 → 50 để tiết kiệm RAM
    SLICE_BUDGET = 10

    USE_CROSS_VALIDATION = False
    CV_FOLDS = 3  # Giảm từ 5 → 3 để tiết kiệm RAM (3 folds = 60% RAM so với 5)
    TEST_SIZE = 0.2               # Chỉ dùng nếu holdout
    METRICS = ('accuracy', 'f1', 'roc_auc')  # Primary = accuracy
    
    # === CẤU HÌNH MULTI-OBJECTIVE ===
    USE_MULTI_OBJECTIVE = True    # Bật multi-objective optimization
    ALPHA = 0.4                   # Trọng số F1
    BETA = 0.4                    # Trọng số ROC-AUC
    GAMMA = 0.2                   # Trọng số execution_time (penalty)
    # =================================
    
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
    
    # === CÂN BẰNG PATIENCE ===
    # AMSCO: 2 slices × 10 trials/slice = 20 trials effective
    # Optuna: 20 trials
    AMSCO_EARLY_STOP_SLICES = 2
    AMSCO_EARLY_STOP_TOLERANCE = 1e-3
    OPTUNA_EARLY_STOP_TRIALS = 15
    # =========================
    # --------------------------------

    def _build_model_for_eval(name, dataset_size):
        # Xác định số cores tối ưu dựa trên kích thước dataset
        n_jobs_optimal = get_optimal_n_jobs(dataset_size)
        
        if name == 'logistic_regression':
            # Dùng class_weight='balanced' khi không có SMOTE
            if sampler is None:
                return LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced')
            return LogisticRegression(random_state=42, max_iter=2000)
        if name == 'random_forest':
            # Dùng class_weight='balanced_subsample' khi không có SMOTE
            if sampler is None:
                return RandomForestClassifier(random_state=42, n_jobs=n_jobs_optimal, class_weight='balanced_subsample')
            return RandomForestClassifier(random_state=42, n_jobs=n_jobs_optimal)
        if name == 'xgboost':
            return xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        if name == 'lightgbm':
            return lgb.LGBMClassifier(random_state=42, verbosity=-1)
        raise ValueError(f"Mô hình '{name}' không được hỗ trợ.")

    results = []

    # QUICK_MODE điều chỉnh cấu hình để chạy nhanh hơn
    if QUICK_MODE:
        SEEDS = list(range(1, 3))  # chạy 2 seed thay vì 10 để test nhanh
        TOTAL_TRIALS = 15          # giảm số trial
        CV_FOLDS = 3               # giảm số folds
    else:
        SEEDS = list(range(1, 3))  # Chạy 10 seeds cho full pipeline

    OPTUNA_RESULTS = []
    AMSCO_RESULTS = []
    CONV_LOG = []  # Lưu history hội tụ cho Hình 4.1
    SAVE_CONVERGENCE_HISTORY = False  # Tắt để tiết kiệm RAM, bật nếu cần vẽ Hình 4.1
    
    # === TỐI ƯU RAM: SAVE VÀ CLEAR SAU MỖI SEED ===
    SAVE_PER_SEED = True  # Lưu kết quả sau mỗi seed và clear để tiết kiệm RAM

    def _prepare_trace_series(trace_records):
        """Chuyển trace thô thành chuỗi thời gian để vẽ."""
        if not trace_records:
            return None

        def _to_float(value):
            try:
                if value is None:
                    return None
                val = float(value)
                if math.isnan(val):
                    return None
                return val
            except Exception:
                return None

        cumulative_time = []
        trial_durations = []
        raw_f1 = []
        best_f1 = []
        raw_auc = []
        best_auc = []
        best_f1_val = -float('inf')
        best_auc_val = -float('inf')
        cum_time = 0.0

        for rec in trace_records:
            t = _to_float(rec.get('trial_wall_time'))
            if t is None:
                t = 0.0
            t = max(t, 0.0)
            trial_durations.append(t)
            cum_time += t
            cumulative_time.append(cum_time)

            f1_val = _to_float(rec.get('f1'))
            raw_f1.append(f1_val)
            if f1_val is not None:
                best_f1_val = f1_val if best_f1_val == -float('inf') else max(best_f1_val, f1_val)
            best_f1.append(best_f1_val if best_f1_val != -float('inf') else None)

            auc_val = _to_float(rec.get('roc_auc'))
            raw_auc.append(auc_val)
            if auc_val is not None:
                best_auc_val = auc_val if best_auc_val == -float('inf') else max(best_auc_val, auc_val)
            best_auc.append(best_auc_val if best_auc_val != -float('inf') else None)

        return {
            'cumulative_time': cumulative_time,
            'trial_durations': trial_durations,
            'raw_f1': raw_f1,
            'best_f1': best_f1,
            'raw_auc': raw_auc,
            'best_auc': best_auc
        }

    true_test_cache = {}
    for seed in SEEDS:
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"\n########## SEED {seed} ##########")

        for dataset_name in DATASETS:
            print(f"\n=======================================================")
            print(f"ĐANG THỬ NGHIỆM TRÊN BỘ DỮ LIỆU: {dataset_name.upper()} (seed={seed})")
            print(f"=======================================================")
            # === TẠO TEST SET ĐỘC LẬP ===
            try:
                X_train_val, X_test_holdout, y_train_val, y_test_holdout, preprocessor, _metric_default, sampler = \
                    prepare_data_with_holdout_test(dataset_name, test_size=0.2, random_state=seed)
                
                # Khởi tạo cache cho dataset (lưu nhẹ chỉ cần thiết)
                if dataset_name not in true_test_cache:
                    true_test_cache[dataset_name] = {}
                
                # Giải phóng bộ nhớ sau khi chuẩn bị data
                gc.collect()

            except Exception as e:
                print(f"  [WARNING] Bỏ qua dataset {dataset_name}: {e}")
                continue

            for model_name in MODELS:
                print(f"\n{'-'*60}")
                print(f"Đang tối ưu mô hình: {model_name.upper()}")
                print(f"{'-'*60}")

                search_space = dict(MASTER_SEARCH_SPACES[model_name])
                
                X = X_train_val  # ← Thay đổi này
                y = y_train_val  # ← Thay đổi này
                primary_metric = _metric_default if _metric_default else METRICS[0]
                metrics_to_use = (primary_metric,) + tuple(m for m in METRICS if m != primary_metric)
                
                # Tạo normalizer cho multi-objective
                normalizer = None
                if USE_MULTI_OBJECTIVE:
                    normalizer = MultiObjectiveNormalizer(alpha=ALPHA, beta=BETA, gamma=GAMMA)
                    print(f"  [Multi-Objective] Sử dụng f(θ) = {ALPHA}·F1 + {BETA}·ROC-AUC - {GAMMA}·time")
                
                if USE_CROSS_VALIDATION:
                    print(f"  [Evaluation] StratifiedKFold ({CV_FOLDS}-fold), Primary metric: {primary_metric}")
                    objective_func = create_objective(
                        X,
                        y,
                        model_name,
                        preprocessor,
                        metrics=metrics_to_use,
                        use_cross_validation=True,
                        cv_folds=CV_FOLDS,
                        sampler=sampler,
                        multi_objective=USE_MULTI_OBJECTIVE,
                        normalizer=normalizer
                    )
                    eval_label = f'cv_{CV_FOLDS}fold'
                else:
                    print(f"  [Evaluation] Holdout split (test_size={TEST_SIZE}), Primary metric: {primary_metric}")
                    X_train, X_valid, y_train, y_valid = train_test_split(
                        X,
                        y,
                        test_size=TEST_SIZE,
                        stratify=y,
                        random_state=seed
                    )
                    objective_func = create_objective(
                        X_train,
                        y_train,
                        model_name,
                        preprocessor,
                        metrics=metrics_to_use,
                        use_cross_validation=False,
                        validation_data=(X_valid, y_valid),
                        cv_folds=CV_FOLDS,
                        sampler=sampler,
                        multi_objective=USE_MULTI_OBJECTIVE,
                        normalizer=normalizer
                    )
                    eval_label = f'holdout_{TEST_SIZE}'

                model_results = { 'scores': {}, 'times': {}, 'params': {}, 'metrics': {}, 'diag': {} }
                optimizers = [('Optuna (TPE)', run_optuna_tpe), ('AMSCO', run_amsco_optimizer)]

                for optimizer_name, optimizer_func in optimizers:
                    print(f"\nThực thi: {optimizer_name}...")
                    def _invoke_optimizer():
                        if optimizer_name == 'AMSCO':
                            return optimizer_func(
                                objective_func,
                                search_space,
                                TOTAL_TRIALS,
                                SLICE_BUDGET,
                                verbose=False,
                                early_stopping_rounds=AMSCO_EARLY_STOP_SLICES,
                                tolerance=AMSCO_EARLY_STOP_TOLERANCE,
                                seed=seed,
                                normalizer=normalizer
                            )
                        return optimizer_func(
                            objective_func, 
                            search_space, 
                            TOTAL_TRIALS,
                            early_stopping_rounds=OPTUNA_EARLY_STOP_TRIALS,
                            tolerance=AMSCO_EARLY_STOP_TOLERANCE,
                            normalizer=normalizer
                        )

                    trace_entries = []
                    trace_supported = hasattr(objective_func, 'enable_trace')
                    if trace_supported:
                        objective_func.enable_trace()
                    try:
                        optimizer_return, resource_stats = profile_optimizer_call(_invoke_optimizer)
                    finally:
                        if trace_supported:
                            trace_entries = objective_func.consume_trace()
                            objective_func.disable_trace()

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

                    metric_scores = {m: np.nan for m in METRICS}
                    acc_cv5 = prec_cv5 = rec_cv5 = bal_cv5 = acc_val = np.nan
                    acc_holdout = f1_holdout = auc_holdout = acc_indep = np.nan
                    time_cv5_eval = time_holdout_eval = np.nan
                    
                    try:
                        # 1. Đánh giá CV/Validation trên X_train_val
                        model = _build_model_for_eval(model_name, len(X))
                        if sampler is not None:
                            pipeline = ImbPipeline(steps=[('preprocessor', preprocessor), ('sampler', clone(sampler)), ('classifier', model)])
                        else:
                            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
                        pipeline.set_params(**params)

                        if USE_CROSS_VALIDATION:
                            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
                            fold_metrics_all = {m: [] for m in METRICS}
                            for train_idx, test_idx in cv.split(X, y):
                                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                                pipeline.fit(X_tr, y_tr)
                                y_pred = pipeline.predict(X_te)
                                y_proba = pipeline.predict_proba(X_te) if hasattr(pipeline.named_steps['classifier'], 'predict_proba') else None
                                for m in METRICS:
                                    if m == 'accuracy':
                                        fold_metrics_all[m].append(accuracy_score(y_te, y_pred))
                                    elif m == 'f1':
                                        try:
                                            fold_metrics_all[m].append(f1_score(y_te, y_pred, average='binary'))
                                        except Exception:
                                            fold_metrics_all[m].append(np.nan)
                                    elif m == 'roc_auc':
                                        if y_proba is None:
                                            fold_metrics_all[m].append(np.nan)
                                        else:
                                            prob = y_proba[:,1] if y_proba.ndim == 2 else y_proba
                                            try:
                                                fold_metrics_all[m].append(roc_auc_score(y_te, prob))
                                            except Exception:
                                                fold_metrics_all[m].append(np.nan)
                            for m in METRICS:
                                metric_scores[m] = np.nanmean(fold_metrics_all[m]) if fold_metrics_all[m] else np.nan
                            
                            # Fit toàn bộ X, y để chuẩn bị cho holdout test evaluation
                            pipeline.fit(X, y)
                            
                            if not QUICK_MODE:
                                try:
                                    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                                    _t0 = time.time()
                                    acc_cv5 = cross_val_score(pipeline, X, y, cv=cv5, scoring='accuracy', n_jobs=4).mean()
                                    prec_cv5 = cross_val_score(pipeline, X, y, cv=cv5, scoring='precision', n_jobs=4).mean()
                                    rec_cv5 = cross_val_score(pipeline, X, y, cv=cv5, scoring='recall', n_jobs=4).mean()
                                    bal_cv5 = cross_val_score(pipeline, X, y, cv=cv5, scoring='balanced_accuracy', n_jobs=4).mean()
                                    time_cv5_eval = time.time() - _t0
                                except Exception as e:
                                    print(f"  [WARN] Không tính được metric CV=5 cho {optimizer_name}: {e}")
                        else:
                            # Holdout optimization mode (X is X_train_val)
                            X_train, X_valid, y_train, y_valid = train_test_split(
                                X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
                            )
                            pipeline.fit(X_train, y_train)
                            y_pred = pipeline.predict(X_valid)
                            metric_scores['accuracy'] = float(accuracy_score(y_valid, y_pred))
                            try:
                                metric_scores['f1'] = float(f1_score(y_valid, y_pred, average='binary'))
                            except Exception:
                                metric_scores['f1'] = np.nan
                            try:
                                y_proba = pipeline.predict_proba(X_valid)[:,1] if hasattr(pipeline.named_steps['classifier'], 'predict_proba') else None
                                metric_scores['roc_auc'] = float(roc_auc_score(y_valid, y_proba)) if y_proba is not None else np.nan
                            except Exception:
                                metric_scores['roc_auc'] = np.nan

                        # 2. Đánh giá True Test trên X_test_holdout (ĐỘC LẬP)
                        print(f"  [True Test] Evaluating on held-out test set ({len(X_test_holdout)} samples)...")
                        true_test_metrics = evaluate_on_holdout_test(
                            best_params=params,
                            model_name=model_name,
                            preprocessor=preprocessor,
                            X_train=X, # X_train_val
                            y_train=y, # y_train_val
                            X_test=X_test_holdout,
                            y_test=y_test_holdout,
                            sampler=sampler
                        )
                        acc_holdout = true_test_metrics['accuracy']
                        f1_holdout = true_test_metrics['f1']
                        auc_holdout = true_test_metrics['roc_auc']
                        acc_indep = acc_holdout
                        print(f"  [True Test] Accuracy: {acc_holdout:.4f}, F1: {f1_holdout:.4f}, AUC: {auc_holdout:.4f}")

                        # Lưu vào cache
                        if optimizer_name not in true_test_cache[dataset_name]:
                            true_test_cache[dataset_name][optimizer_name] = {}
                        true_test_cache[dataset_name][optimizer_name][model_name] = true_test_metrics

                    except Exception as e:
                        import traceback
                        print(f"  [WARN] Lỗi khi tính metrics: {e}")
                        print(f"  [DEBUG] Traceback:\n{traceback.format_exc()}")

                    # Lưu convergence history (nếu optimizer có trả về)
                    conv_hist = diag_stats.get('convergence_history') if isinstance(diag_stats, dict) else None
                    trace_payload = _prepare_trace_series(trace_entries)
                    if conv_hist is not None and SAVE_CONVERGENCE_HISTORY:
                        entry = {
                            'seed': seed,
                            'dataset': dataset_name,
                            'model': model_name,
                            'optimizer': optimizer_name,
                            'history': json.dumps(list(conv_hist)),
                            'raw_f1_history': json.dumps(trace_payload['best_f1']) if trace_payload else json.dumps([]),
                            'raw_auc_history': json.dumps(trace_payload['best_auc']) if trace_payload else json.dumps([]),
                            'raw_f1_trials': json.dumps(trace_payload['raw_f1']) if trace_payload else json.dumps([]),
                            'raw_auc_trials': json.dumps(trace_payload['raw_auc']) if trace_payload else json.dumps([]),
                            'cumulative_time': json.dumps(trace_payload['cumulative_time']) if trace_payload else json.dumps([]),
                            'trial_durations': json.dumps(trace_payload['trial_durations']) if trace_payload else json.dumps([])
                        }
                        CONV_LOG.append(entry)

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
                        'acc_val': acc_val,
                        'acc_cv5': acc_cv5,
                        'prec_cv5': prec_cv5,
                        'recall_cv5': rec_cv5,
                        'bal_acc_cv5': bal_cv5,
                        'acc_indep': acc_indep,  # Thêm kết quả trên tập test độc lập
                        'time': exec_time,
                        'opt_wall_time': combined_stats.get('wall_time', np.nan),
                        'opt_cpu_time': combined_stats.get('cpu_time', np.nan),
                        'opt_peak_memory_mb': combined_stats.get('peak_memory_mb', np.nan),
                        'opt_rss_memory_mb': combined_stats.get('rss_memory_mb', np.nan),
                        'opt_total_trials': total_trials_reported,
                        'opt_iter_best': iteration_to_best,
                        'opt_convergence_ratio': convergence_ratio,
                        'time_holdout': time_holdout_eval,
                        'time_cv5': time_cv5_eval,
                        'evaluation': eval_label
                    })
                    
                    # Giải phóng bộ nhớ sau mỗi optimizer (sau khi đã dùng xong combined_stats)
                    del optimizer_return, resource_stats, diag_stats, combined_stats
                    if trace_supported:
                        del trace_entries
                    gc.collect()

                if model_results['scores']:
                    print(f"\n{'='*60}")
                    print(f"KẾT QUẢ CHO {model_name.upper()} (Primary metric: {primary_metric})")
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
                
                # Lưu điểm vào hai danh sách kết quả theo seed cho thống kê (TRƯỚC KHI DELETE)
                if 'Optuna (TPE)' in model_results['scores']:
                    OPTUNA_RESULTS.append({
                        'seed': seed,
                        'dataset': dataset_name,
                        'model': model_name,
                        'score': model_results['scores']['Optuna (TPE)'],
                        'accuracy': model_results['metrics']['Optuna (TPE)'].get('accuracy', np.nan),
                        'f1': model_results['metrics']['Optuna (TPE)'].get('f1', np.nan),
                        'roc_auc': model_results['metrics']['Optuna (TPE)'].get('roc_auc', np.nan),
                        'exec_time': float(model_results['times'].get('Optuna (TPE)', np.nan))
                    })
                if 'AMSCO' in model_results['scores']:
                    AMSCO_RESULTS.append({
                        'seed': seed,
                        'dataset': dataset_name,
                        'model': model_name,
                        'score': model_results['scores']['AMSCO'],
                        'accuracy': model_results['metrics']['AMSCO'].get('accuracy', np.nan),
                        'f1': model_results['metrics']['AMSCO'].get('f1', np.nan),
                        'roc_auc': model_results['metrics']['AMSCO'].get('roc_auc', np.nan),
                        'exec_time': float(model_results['times'].get('AMSCO', np.nan))
                    })
                
                # Giải phóng objective function và model resources sau khi hoàn tất model
                del objective_func, model_results
                gc.collect()

            # Giải phóng bộ nhớ sau mỗi model
            gc.collect()
        
        # === CLEANUP SAU MỖI DATASET ===
        print(f"\n[Memory] Cleaning up after dataset {dataset_name}...")
        
        # Giải phóng test holdout data sau khi xử lý xong dataset
        try:
            del X_train_val, X_test_holdout, y_train_val, y_test_holdout
            print(f"[Memory] Freed X_train_val, X_test_holdout, y_train_val, y_test_holdout")
        except:
            pass
        
        try:
            del preprocessor
            print(f"[Memory] Freed preprocessor")
        except:
            pass
        
        if sampler is not None:
            try:
                del sampler
                print(f"[Memory] Freed sampler")
            except:
                pass
        
        # KHÔNG xóa true_test_cache ở đây - cần giữ để tính Bias/Variance (Bảng 4.3, 4.4)
        # Cache sẽ được clear sau khi save seed results ra file
        # if dataset_name in true_test_cache:
        #     true_test_cache[dataset_name].clear()
        #     print(f"[Memory] Cleared true_test_cache[{dataset_name}]")
        
        # Force garbage collection
        collected = gc.collect()
        print(f"[Memory] Completed dataset cleanup (GC collected {collected} objects)")
    
    # === CLEANUP SAU MỖI SEED ===
    # Giải phóng bộ nhớ sau khi xử lý xong seed
    print(f"\n[Memory] Cleaning up after seed {seed}...")
    
    # Save kết quả seed này ra file nếu cần
    if SAVE_PER_SEED:
        os.makedirs('results', exist_ok=True)
        
        # Save main results
        seed_results = [r for r in results if r.get('seed') == seed]
        if seed_results:
            seed_df = pd.DataFrame(seed_results)
            seed_file = f'results/seed_{seed}_results.csv'
            seed_df.to_csv(seed_file, index=False)
            print(f"[Memory] Saved {len(seed_results)} results to {seed_file}")
            
            # Clear results của seed này khỏi memory
            results = [r for r in results if r.get('seed') != seed]
            print(f"[Memory] Cleared seed {seed} from results (kept {len(results)} entries)")
        
        # Save OPTUNA_RESULTS for this seed
        seed_optuna = [r for r in OPTUNA_RESULTS if r.get('seed') == seed]
        if seed_optuna:
            optuna_file = f'results/seed_{seed}_optuna.csv'
            pd.DataFrame(seed_optuna).to_csv(optuna_file, index=False)
            print(f"[Memory] Saved {len(seed_optuna)} OPTUNA_RESULTS to {optuna_file}")
        
        # Save AMSCO_RESULTS for this seed
        seed_amsco = [r for r in AMSCO_RESULTS if r.get('seed') == seed]
        if seed_amsco:
            amsco_file = f'results/seed_{seed}_amsco.csv'
            pd.DataFrame(seed_amsco).to_csv(amsco_file, index=False)
            print(f"[Memory] Saved {len(seed_amsco)} AMSCO_RESULTS to {amsco_file}")
        
        # Save CONV_LOG for this seed
        seed_conv = [r for r in CONV_LOG if r.get('seed') == seed]
        if seed_conv:
            conv_file = f'results/seed_{seed}_conv_log.csv'
            pd.DataFrame(seed_conv).to_csv(conv_file, index=False)
            print(f"[Memory] Saved {len(seed_conv)} CONV_LOG to {conv_file}")
        
        # Save true_test_cache for this seed (cần cho Bảng 4.3, 4.4)
        if true_test_cache:
            cache_file = f'results/seed_{seed}_true_test_cache.json'
            with open(cache_file, 'w') as f:
                json.dump(true_test_cache, f)
            print(f"[Memory] Saved true_test_cache ({len(true_test_cache)} datasets) to {cache_file}")
        
        # Clear all these lists from memory after saving
        OPTUNA_RESULTS.clear()
        AMSCO_RESULTS.clear()
        CONV_LOG.clear()
        true_test_cache.clear()  # Clear after saving to disk
        print(f"[Memory] Cleared OPTUNA_RESULTS, AMSCO_RESULTS, CONV_LOG, true_test_cache from memory")
    
    gc.collect()
    print(f"[Memory] Cleanup completed for seed {seed}")

# === KẾT THÚC VÒNG LẶP SEED ===
# Từ đây trở đi, code chỉ chạy 1 LẦN sau khi TẤT CẢ seeds đã hoàn thành

# === MERGE CÁC FILE SEED RESULTS ===
if SAVE_PER_SEED:
    print("\n" + "="*60)
    print("MERGING SEED RESULTS FILES")
    print("="*60)
        
    # Tìm tất cả các file seed_*_results.csv
    import glob
    seed_files = sorted(glob.glob('results/seed_*_results.csv'))
    
    # Reload OPTUNA_RESULTS from per-seed files
    optuna_files = sorted(glob.glob('results/seed_*_optuna.csv'))
    if optuna_files:
        print(f"\n[Merge] Loading {len(optuna_files)} OPTUNA seed files...")
        for f in optuna_files:
            try:
                df = pd.read_csv(f)
                OPTUNA_RESULTS.extend(df.to_dict('records'))
            except Exception as e:
                print(f"  [WARNING] Could not load {f}: {e}")
        print(f"[Merge] Loaded {len(OPTUNA_RESULTS)} OPTUNA_RESULTS entries")
    
    # Reload AMSCO_RESULTS from per-seed files
    amsco_files = sorted(glob.glob('results/seed_*_amsco.csv'))
    if amsco_files:
        print(f"[Merge] Loading {len(amsco_files)} AMSCO seed files...")
        for f in amsco_files:
            try:
                df = pd.read_csv(f)
                AMSCO_RESULTS.extend(df.to_dict('records'))
            except Exception as e:
                print(f"  [WARNING] Could not load {f}: {e}")
        print(f"[Merge] Loaded {len(AMSCO_RESULTS)} AMSCO_RESULTS entries")
    
    # Reload CONV_LOG from per-seed files
    conv_files = sorted(glob.glob('results/seed_*_conv_log.csv'))
    if conv_files:
        print(f"[Merge] Loading {len(conv_files)} CONV_LOG seed files...")
        for f in conv_files:
            try:
                df = pd.read_csv(f)
                CONV_LOG.extend(df.to_dict('records'))
            except Exception as e:
                print(f"  [WARNING] Could not load {f}: {e}")
        print(f"[Merge] Loaded {len(CONV_LOG)} CONV_LOG entries\n")
    
    # Reload true_test_cache from per-seed files (cần cho Bảng 4.3, 4.4)
    cache_files = sorted(glob.glob('results/seed_*_true_test_cache.json'))
    if cache_files:
        print(f"[Merge] Loading {len(cache_files)} true_test_cache seed files...")
        for f in cache_files:
            try:
                with open(f, 'r') as file:
                    seed_cache = json.load(file)
                    # Merge into global true_test_cache
                    for dataset_name, dataset_data in seed_cache.items():
                        if dataset_name not in true_test_cache:
                            true_test_cache[dataset_name] = {}
                        for optimizer_name, optimizer_data in dataset_data.items():
                            if optimizer_name not in true_test_cache[dataset_name]:
                                true_test_cache[dataset_name][optimizer_name] = {}
                            true_test_cache[dataset_name][optimizer_name].update(optimizer_data)
            except Exception as e:
                print(f"  [WARNING] Could not load {f}: {e}")
        total_datasets = len(true_test_cache)
        total_entries = sum(len(opts) for opts in true_test_cache.values())
        print(f"[Merge] Loaded true_test_cache: {total_datasets} datasets, {total_entries} optimizer entries\n")
        
    if seed_files:
        print(f"Found {len(seed_files)} seed result files:")
        for f in seed_files:
            print(f"  - {f}")
            
        # Merge tất cả các files
        all_seed_dfs = []
        for seed_file in seed_files:
            try:
                df = pd.read_csv(seed_file)
                all_seed_dfs.append(df)
                print(f"  Loaded {len(df)} rows from {seed_file}")
            except Exception as e:
                print(f"  [WARNING] Could not load {seed_file}: {e}")
            
        if all_seed_dfs:
            results_df = pd.concat(all_seed_dfs, ignore_index=True)
            print(f"\n[Merge] Total merged: {len(results_df)} rows from {len(all_seed_dfs)} files")
                
            # Convert back to results list for compatibility
            results = results_df.to_dict('records')
            print(f"[Merge] Converted to {len(results)} result entries")
        else:
            print("[WARNING] No seed files could be loaded")
    else:
        print("[INFO] No seed result files found to merge")

if not results:
    print("Không có kết quả nào được ghi nhận.")
else:
    print("\n\n" + "="*60)
    print(" "*20 + "KẾT QUẢ THỬ NGHIỆM TỔNG QUÁT" + " "*20)
    print("="*60 + "\n")

    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------
    # HÌNH 4.2: Boxplot so sánh thời gian thực thi giữa các optimizer
    # ------------------------------------------------------------
    try:
        if {'dataset', 'optimizer', 'opt_wall_time'}.issubset(results_df.columns):
            fig, ax = plt.subplots(figsize=(8, 5))
            datasets_for_plot = sorted(results_df['dataset'].unique())
            all_data = []
            labels = []
            colors = []

            color_map = {
                'Optuna (TPE)': '#1f77b4',  # xanh dương
                'AMSCO': '#2ca02c',        # xanh lá
            }

            for ds in datasets_for_plot:
                for opt in ['Optuna (TPE)', 'AMSCO']:
                    subset = results_df[(results_df['dataset'] == ds) & (results_df['optimizer'] == opt)]['opt_wall_time'].dropna().values
                    if subset.size == 0:
                        continue
                    all_data.append(subset)
                    labels.append(f"{ds.title()}\n{opt.split()[0]}")
                    colors.append(color_map.get(opt, '#999999'))

            if all_data:
                bp = ax.boxplot(all_data, patch_artist=True, labels=labels, showfliers=False)
                for patch, c in zip(bp['boxes'], colors):
                    patch.set_facecolor(c)
                ax.set_ylabel('Optimization Wall Time (s)')
                ax.set_title('Hình 4.2: Boxplot so sánh thời gian thực thi')
                plt.xticks(rotation=20, ha='right')
                plt.tight_layout()
                plt.savefig('figure_4_2_boxplot_time.png', dpi=300)
                plt.close(fig)
                print("[INFO] Đã lưu Hình 4.2: figure_4_2_boxplot_time.png")
    except Exception as e:
        print(f"[WARN] Không thể vẽ Hình 4.2: {e}")

    def _print_per_seed_table(df_source, label):
        display_cols = ['seed', 'dataset', 'model', 'score', 'f1', 'roc_auc', 'exec_time']
        existing_cols = [c for c in display_cols if c in df_source.columns]
        table = df_source[existing_cols].copy().sort_values(['dataset', 'model', 'seed'])
        rename_map = {
            'seed': 'Seed',
            'dataset': 'Dataset',
            'model': 'Model',
            'score': 'Primary Score',
            'f1': 'F1-score',
            'roc_auc': 'ROC AUC',
            'exec_time': 'Exec Time (s)'
        }
        table = table.rename(columns=rename_map)
        for col in ['Primary Score', 'F1-score', 'ROC AUC']:
            if col in table:
                table[col] = table[col].map(lambda v: f"{float(v):.4f}" if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else 'nan')
        if 'Exec Time (s)' in table:
            table['Exec Time (s)'] = table['Exec Time (s)'].map(lambda v: f"{float(v):.2f}" if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else 'nan')
        print(f"\nKẾT QUẢ LẶP THEO SEED - {label}:")
        print(table.to_string(index=False))

        stat_rows = []
        stats_targets = [
            ('score', 'Primary Score'),
            ('f1', 'F1-score'),
            ('roc_auc', 'ROC AUC'),
            ('exec_time', 'Exec Time (s)')
        ]
        for col, pretty in stats_targets:
            if col not in df_source:
                continue
            arr = pd.to_numeric(df_source[col], errors='coerce').to_numpy(dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                stat_rows.append({'Metric': pretty, 'Mean': np.nan, 'Std': np.nan, 'n': 0})
                continue
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan
            stat_rows.append({'Metric': pretty, 'Mean': mean_val, 'Std': std_val, 'n': int(arr.size)})

        if stat_rows:
            stat_df = pd.DataFrame(stat_rows)
            stat_df['Mean'] = stat_df['Mean'].map(lambda v: f"{v:.4f}" if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else 'nan')
            stat_df['Std'] = stat_df['Std'].map(lambda v: f"{v:.4f}" if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else 'nan')
            stat_df['n'] = stat_df['n'].astype(int)
            if 'Exec Time (s)' in stat_df['Metric'].values:
                idx = stat_df['Metric'] == 'Exec Time (s)'
                stat_df.loc[idx, 'Mean'] = stat_df.loc[idx, 'Mean'].map(lambda v: f"{float(v):.2f}" if v != 'nan' else 'nan')
                stat_df.loc[idx, 'Std'] = stat_df.loc[idx, 'Std'].map(lambda v: f"{float(v):.2f}" if v != 'nan' else 'nan')
            print("\nThống kê mô tả (" + label + "):")
            print(stat_df.to_string(index=False))

    df_optuna = pd.DataFrame(OPTUNA_RESULTS) if OPTUNA_RESULTS else pd.DataFrame()
    df_amsco = pd.DataFrame(AMSCO_RESULTS) if AMSCO_RESULTS else pd.DataFrame()

    if not df_optuna.empty:
        _print_per_seed_table(df_optuna, 'OPTUNA (TPE)')
    else:
        print("\nKhông có kết quả Optuna để thống kê.")

    if not df_amsco.empty:
        _print_per_seed_table(df_amsco, 'AMSCO')
    else:
        print("\nKhông có kết quả AMSCO để thống kê.")
    
    # Cleanup df_optuna và df_amsco sau khi dùng xong
    del df_optuna, df_amsco
    gc.collect()
    print("[Memory] Cleaned up df_optuna and df_amsco")

    # Recreate only if needed for comparison
    df_optuna = pd.DataFrame(OPTUNA_RESULTS) if OPTUNA_RESULTS else pd.DataFrame()
    df_amsco = pd.DataFrame(AMSCO_RESULTS) if AMSCO_RESULTS else pd.DataFrame()

    if not df_optuna.empty and not df_amsco.empty:
        overlap = df_optuna.merge(
            df_amsco,
            on=['seed', 'dataset', 'model'],
            suffixes=('_optuna', '_amsco')
        )
        if overlap.empty:
            print("\n[INFO] Không tìm thấy cặp seed/dataset/model chung giữa hai optimizer để kiểm định thống kê.")
        else:
            try:
                from scipy.stats import ttest_rel, mannwhitneyu
            except ImportError:
                ttest_rel = mannwhitneyu = None
                print("\n[WARN] Không thể import scipy.stats.ttest_rel/mannwhitneyu. Bỏ qua kiểm định thống kê.")

            if ttest_rel and mannwhitneyu:
                def _paired_ttest(series_a, series_b):
                    arr_a = pd.to_numeric(series_a, errors='coerce').to_numpy(dtype=float)
                    arr_b = pd.to_numeric(series_b, errors='coerce').to_numpy(dtype=float)
                    mask = ~np.isnan(arr_a) & ~np.isnan(arr_b)
                    arr_a = arr_a[mask]
                    arr_b = arr_b[mask]
                    if arr_a.size < 2:
                        return np.nan, np.nan, int(arr_a.size)
                    stat_val, p_val = ttest_rel(arr_a, arr_b)
                    return float(stat_val), float(p_val), int(arr_a.size)

                def _mannwhitney(series_a, series_b):
                    arr_a = pd.to_numeric(series_a, errors='coerce').to_numpy(dtype=float)
                    arr_b = pd.to_numeric(series_b, errors='coerce').to_numpy(dtype=float)
                    arr_a = arr_a[~np.isnan(arr_a)]
                    arr_b = arr_b[~np.isnan(arr_b)]
                    if arr_a.size == 0 or arr_b.size == 0:
                        return np.nan, np.nan, int(arr_a.size), int(arr_b.size)
                    stat_val, p_val = mannwhitneyu(arr_a, arr_b, alternative='two-sided')
                    return float(stat_val), float(p_val), int(arr_a.size), int(arr_b.size)

                test_rows = []

                stat_f1, p_f1, n_f1 = _paired_ttest(overlap['f1_optuna'], overlap['f1_amsco'])
                test_rows.append({
                    'Metric': 'F1-score',
                    'Test': 'Paired t-test',
                    'Statistic': stat_f1,
                    'p-value': p_f1,
                    '#Pairs': n_f1
                })

                stat_auc, p_auc, n_auc = _paired_ttest(overlap['roc_auc_optuna'], overlap['roc_auc_amsco'])
                test_rows.append({
                    'Metric': 'ROC AUC',
                    'Test': 'Paired t-test',
                    'Statistic': stat_auc,
                    'p-value': p_auc,
                    '#Pairs': n_auc
                })

                u_time, p_time, n_time_opt, n_time_ams = _mannwhitney(overlap['exec_time_optuna'], overlap['exec_time_amsco'])
                test_rows.append({
                    'Metric': 'Exec Time (s)',
                    'Test': 'Mann-Whitney U',
                    'Statistic': u_time,
                    'p-value': p_time,
                    '#Pairs': f"{n_time_opt}/{n_time_ams}"
                })

                test_df = pd.DataFrame(test_rows)
                def _fmt_stat(val, is_time=False):
                    if isinstance(val, str):
                        return val
                    if not isinstance(val, (int, float, np.floating)) or pd.isna(val):
                        return 'nan'
                    precision = 4 if not is_time else 2
                    return f"{val:.{precision}f}"

                test_df['Statistic'] = test_df['Statistic'].map(_fmt_stat)
                test_df['p-value'] = test_df['p-value'].map(_fmt_stat)

                print("\nKIỂM ĐỊNH THỐNG KÊ GIỮA OPTUNA (TPE) VÀ AMSCO:")
                print(test_df.to_string(index=False))
                
                # Cleanup test_df and overlap
                del test_df, overlap
                gc.collect()
                print("[Memory] Cleaned up test_df and overlap")

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

    # Nested CV (bỏ qua nếu QUICK_MODE)
    nested_cache = {} # Structure: nested_cache[dataset][method] = {'accuracy': ..., ...}
    # Cache lưu True Test Accuracy (độc lập hoàn toàn)
    # true_test_cache đã được populate trong vòng lặp chính

    if not QUICK_MODE and USE_CROSS_VALIDATION:
        try:
            # Khởi tạo cache cho từng dataset
            for d in DATASETS:
                nested_cache[d] = {}

            methods = sorted(results_df['optimizer'].unique())
            for dataset_name in DATASETS:
                try:
                    Xn, yn, prep_n, _, sampler_n = get_data(dataset_name, quiet=True)
                except Exception:
                    continue
                        
                for method in methods:
                    acc_list, f1_list, auc_list = [], [], []
                    total_time = 0.0
                    count = 0
                        
                    for model_name in MODELS:
                        search_space = dict(MASTER_SEARCH_SPACES[model_name])
                        res = _nested_cv_metrics_for_method(
                            Xn, yn, model_name, prep_n, search_space,
                            method,
                            inner_trials=NESTED_INNER_TRIALS,
                            outer_folds=NESTED_OUTER_FOLDS,
                            inner_folds=NESTED_INNER_FOLDS,
                            slice_budget=SLICE_BUDGET,
                            sampler=sampler_n
                        )
                        acc_list.append(res['accuracy'])
                        f1_list.append(res['f1'])
                        auc_list.append(res['roc_auc'])
                        total_time += res['time']
                        count += 1
                        
                    if acc_list:
                        nested_cache[dataset_name][method] = {
                            'accuracy': float(np.nanmean(acc_list)),
                            'f1': float(np.nanmean(f1_list)),
                            'roc_auc': float(np.nanmean(auc_list)),
                            'time': float(total_time / max(count, 1))
                        }
        except Exception as e:
            print(f"[WARN] Không thể tính Nested CV tổng hợp: {e}")
            nested_cache = {}

    # Tính Nested CV estimates (để so sánh)
    nested_cv_estimates = {}
    if nested_cache:
        for dataset_name, method_dict in nested_cache.items():
            nested_vals = []
            for method_val in method_dict.values():
                if isinstance(method_val, dict):
                    acc_val_nested = method_val.get('accuracy', np.nan)
                else:
                    acc_val_nested = method_val
                if isinstance(acc_val_nested, (int, float, np.floating)) and not np.isnan(acc_val_nested):
                    nested_vals.append(float(acc_val_nested))
            if nested_vals:
                nested_cv_estimates[dataset_name] = float(np.mean(nested_vals))

    # Tính ground truth thực sự (True Test Acc) dựa trên tập độc lập (acc_holdout)
    ground_truth_by_dataset = {}
    for dataset_name in DATASETS:
        if 'acc_holdout' in results_df.columns:
            holdout_vals = results_df[results_df['dataset'] == dataset_name]['acc_holdout'].dropna().values
            if len(holdout_vals) > 0:
                ground_truth_by_dataset[dataset_name] = float(np.mean(holdout_vals))

    # ============================================================
    # BẢNG 4.3: So sánh Bias và Variance trung bình trên 4 datasets
    # ============================================================
    try:
        print("\n" + "="*80)
        print("BẢNG 4.3: So sánh Bias và Variance trung bình trên 4 datasets")
        print("="*80)
            
        eval_methods = [
            ('Hold-out (70/30)', 'acc_holdout'),
            ('Standard 5-Fold', 'acc_cv5'),
            ('Nested CV (5x3)', 'nested_cv')
        ]
            
        bias_var_rows = []
        for method_name, metric_key in eval_methods:
            # Tính tổng và count thay vì lưu tất cả giá trị
            sum_estimates = 0.0
            count_estimates = 0
            sum_biases = 0.0
            sum_truths = 0.0

            # Process từng dataset một để tiết kiệm memory
            for dataset_name in DATASETS:
                # Lấy estimates
                estimates = []
                if metric_key == 'nested_cv':
                    if dataset_name in nested_cache:
                        for res in nested_cache[dataset_name].values():
                            if isinstance(res, dict) and 'accuracy' in res:
                                estimates.append(float(res['accuracy']))
                            elif isinstance(res, (int, float)):
                                estimates.append(float(res))
                else:
                    if metric_key in results_df.columns:
                        dataset_values = results_df[results_df['dataset'] == dataset_name][metric_key].dropna().values
                        estimates = [float(v) for v in dataset_values]
                
                if not estimates:
                    continue

                # Lấy truth value cho dataset này
                truth_val = None
                if true_test_cache and dataset_name in true_test_cache:
                    true_test_vals = []
                    for opt_dict in true_test_cache[dataset_name].values():
                        for model_metrics in opt_dict.values():
                            if isinstance(model_metrics, dict) and 'accuracy' in model_metrics:
                                true_test_vals.append(model_metrics['accuracy'])
                    
                    if true_test_vals:
                        truth_val = float(np.mean(true_test_vals))
                    del true_test_vals
                
                if truth_val is None:
                    continue
                
                # Cập nhật tổng thay vì lưu list
                n = len(estimates)
                sum_estimates += sum(estimates)
                count_estimates += n
                sum_truths += truth_val * n
                
                for est in estimates:
                    bias = est - truth_val
                    sum_biases += bias
                
                # Xóa estimates để giải phóng memory
                del estimates
            
            if count_estimates == 0:
                continue

            # Tính mean values
            est_acc = sum_estimates / count_estimates
            true_test_acc = sum_truths / count_estimates
            avg_bias = sum_biases / count_estimates
            
            # Tính std dev - cần pass qua data lần nữa
            sum_sq_dev = 0.0
            min_dev = float('inf')
            max_dev = 0.0
            
            for dataset_name in DATASETS:
                estimates = []
                if metric_key == 'nested_cv':
                    if dataset_name in nested_cache:
                        for res in nested_cache[dataset_name].values():
                            if isinstance(res, dict) and 'accuracy' in res:
                                estimates.append(float(res['accuracy']))
                            elif isinstance(res, (int, float)):
                                estimates.append(float(res))
                else:
                    if metric_key in results_df.columns:
                        dataset_values = results_df[results_df['dataset'] == dataset_name][metric_key].dropna().values
                        estimates = [float(v) for v in dataset_values]
                
                for est in estimates:
                    dev = abs(est - est_acc)
                    sum_sq_dev += (est - est_acc) ** 2
                    min_dev = min(min_dev, dev)
                    max_dev = max(max_dev, dev)
                del estimates
            
            avg_std = float(np.sqrt(sum_sq_dev / (count_estimates - 1))) if count_estimates > 1 else 0.0
            min_std = min_dev if min_dev != float('inf') else 0.0

            bias_var_rows.append({
                'Phương pháp': method_name,
                'Est. Acc.': f"{est_acc:.4f}",
                'True Test Acc.': f"{true_test_acc:.4f}",
                'Avg Bias': f"{avg_bias:+.4f}",
                'Avg Std Dev': f"± {avg_std:.4f}",
                'Min-Max Std': f"{min_std:.4f}-{max_std:.4f}"
            })
            
        if bias_var_rows:
            bias_var_df = pd.DataFrame(bias_var_rows)
            print(bias_var_df.to_string(index=False))
            print("\nGhi chú: True Test Acc. là accuracy trên tập test độc lập (20%), KHÔNG tham gia Nested CV.")
            print()
            del bias_var_df
            del bias_var_rows
            gc.collect()
        else:
            print("Không đủ dữ liệu để tạo bảng (cần chạy với nhiều seeds hơn)")
    except Exception as e:
        print(f"[WARN] Không thể tạo Bảng 4.3: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # BẢNG 4.4: Chi tiết Bias và Variance theo từng Dataset
    # ============================================================
    try:
        print("\n" + "="*90)
        print("BẢNG 4.4: Chi tiết Bias và Variance theo từng Dataset (Độ lệch giữa CV và Test)")
        print("="*90)
            
        dataset_bias_rows = []
            
        # Cấu hình tên cột (Sửa lại cho khớp với dataframe của bạn)
        COL_CV_SCORE = 'acc_cv5'     # Cột điểm Validation/CV
            
        for dataset in DATASETS:
            # Lọc dữ liệu theo dataset - sử dụng view thay vì copy
            dataset_mask = results_df['dataset'] == dataset
            dataset_results = results_df[dataset_mask]

            if len(dataset_results) == 0:
                continue

            # 1. True Test Acc (Ground Truth) lấy từ true_test_cache
            true_acc = None
            if dataset in true_test_cache:
                # Tính tổng thay vì lưu list
                sum_vals = 0.0
                count_vals = 0
                for opt_dict in true_test_cache[dataset].values():
                    for model_metrics in opt_dict.values():
                        if isinstance(model_metrics, dict) and 'accuracy' in model_metrics:
                            sum_vals += model_metrics['accuracy']
                            count_vals += 1
                if count_vals > 0:
                    true_acc = float(sum_vals / count_vals)
                
            if true_acc is None or pd.isna(true_acc):
                continue
                
            for method_label in ['Standard 5-Fold', 'Nested CV']:
                est_acc = np.nan
                std_dev = 0.0
                    
                # --- TRƯỜNG HỢP 1: STANDARD 5-FOLD ---
                if method_label == 'Standard 5-Fold':
                    # Kiểm tra xem cột Validation Score có tồn tại không
                    if COL_CV_SCORE in dataset_results.columns:
                        vals = dataset_results[COL_CV_SCORE].dropna().values
                        if len(vals) > 0:
                            est_acc = np.mean(vals)
                            std_dev = np.std(vals, ddof=1) if len(vals) > 1 else 0
                    else:
                        # Fallback nếu không tìm thấy tên cột
                        print(f"[WARN] Không tìm thấy cột '{COL_CV_SCORE}' trong dataset {dataset}")

                # --- TRƯỜNG HỢP 2: NESTED CV ---
                else:
                    # Lấy dữ liệu từ cache
                    if dataset in nested_cache:
                        # Tính tổng và count trực tiếp thay vì lưu list
                        sum_vals = 0.0
                        count_vals = 0
                        sum_sq = 0.0
                        
                        for res in nested_cache[dataset].values():
                            val = None
                            if isinstance(res, dict) and 'accuracy' in res:
                                val = res['accuracy']
                            elif isinstance(res, (int, float)):
                                val = res
                            
                            if val is not None and not np.isnan(val):
                                val = float(val)
                                sum_vals += val
                                count_vals += 1
                        
                        if count_vals > 0:
                            est_acc = sum_vals / count_vals
                            
                            # Tính std dev
                            if count_vals > 1:
                                for res in nested_cache[dataset].values():
                                    val = None
                                    if isinstance(res, dict) and 'accuracy' in res:
                                        val = res['accuracy']
                                    elif isinstance(res, (int, float)):
                                        val = res
                                    if val is not None and not np.isnan(val):
                                        sum_sq += (float(val) - est_acc) ** 2
                                std_dev = float(np.sqrt(sum_sq / (count_vals - 1)))
                            else:
                                std_dev = 0.0

                # --- TÍNH BIAS & LƯU KẾT QUẢ ---
                if not np.isnan(est_acc):
                    bias = est_acc - true_acc
                        
                    dataset_bias_rows.append({
                        'Dataset': dataset.replace('_', ' ').title(),
                        'Method': method_label,
                        'Est. Acc.': f"{est_acc:.4f}",
                        'True Test Acc.': f"{true_acc:.4f}",
                        'Bias': f"{bias:+.4f}",          # Dấu + thể hiện Bias dương (Optimistic)
                        'Std Dev': f"± {std_dev:.4f}"
                    })
            
        if dataset_bias_rows:
            # Sắp xếp theo Dataset để dễ nhìn
            dataset_bias_df = pd.DataFrame(dataset_bias_rows)
            # Format lại bảng in ra
            print(dataset_bias_df.to_string(index=False))
            print()
            
            # Cleanup
            del dataset_bias_df
            del dataset_bias_rows
            gc.collect()
        else:
            print("Không đủ dữ liệu để tạo bảng 4.4")
    except Exception as e:
        print(f"[WARN] Không thể tạo Bảng 4.4: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Force cleanup
        gc.collect()

    # ============================================================
    # BẢNG 4.5: Tốc độ Hội tụ và Thời gian Thực thi (5 outer folds)
    # ============================================================
    try:
        from scipy.stats import mannwhitneyu
            
        print("\n" + "="*80)
        print("BẢNG 4.5: Tốc độ Hội tụ và Thời gian Thực thi (5 outer folds)")
        print("="*80)
            
        convergence_rows = []
        for dataset in DATASETS:
            dataset_results = results_df[results_df['dataset'] == dataset]
                
            # Lấy dữ liệu time cho Mann-Whitney test
            optuna_times = dataset_results[dataset_results['optimizer'] == 'Optuna (TPE)']['opt_wall_time'].dropna().values
            amsco_times = dataset_results[dataset_results['optimizer'] == 'AMSCO']['opt_wall_time'].dropna().values
                
            # Thời gian baseline của Optuna cho dataset này (dùng cho Speedup)
            optuna_baseline_time = dataset_results[dataset_results['optimizer'] == 'Optuna (TPE)']['opt_wall_time'].mean() if len(optuna_times) > 0 else np.nan
                
            # Tính p-value bằng Mann-Whitney U test
            if len(optuna_times) > 0 and len(amsco_times) > 0:
                try:
                    _, p_value_mw = mannwhitneyu(optuna_times, amsco_times, alternative='two-sided')
                except Exception:
                    p_value_mw = None
            else:
                p_value_mw = None
                
            for optimizer in ['Optuna (TPE)', 'AMSCO']:
                opt_mask = dataset_results['optimizer'] == optimizer
                opt_results = dataset_results[opt_mask]
                    
                if len(opt_results) > 0:
                    # Tính mean values - chỉ lấy giá trị cần thiết
                    avg_time = opt_results['opt_wall_time'].mean() if 'opt_wall_time' in opt_results.columns else 0
                    avg_iter_to_best = opt_results['opt_iter_best'].mean() if 'opt_iter_best' in opt_results.columns else 0
                    f1_score = opt_results['f1'].mean() if 'f1' in opt_results.columns else 0
                    auc_roc = opt_results['roc_auc'].mean() if 'roc_auc' in opt_results.columns else 0
                        
                    # Tính speedup và gán p-value
                    if optimizer == 'AMSCO':
                        # Nếu thiếu dữ liệu Optuna cho dataset này, không tính speedup
                        if not np.isnan(optuna_baseline_time) and avg_time > 0:
                            speedup = optuna_baseline_time / avg_time
                        else:
                            speedup = None
                        p_value = p_value_mw
                    else:
                        speedup = None
                        p_value = None
                        
                    convergence_rows.append({
                        'Dataset': dataset.replace('_', ' ').title(),
                        'Method': optimizer.split()[0],  # Optuna hoặc AMSCO
                        'Time (s)': f"{int(avg_time)}",
                        'Trials to Best': f"{int(avg_iter_to_best)}",  # Sửa: dùng iter_to_best thay vì total_trials
                        'Speedup': f"{speedup:.2f}x" if speedup else '-',
                        'F1-Score': f"{f1_score:.4f}",
                        'AUC-ROC': f"{auc_roc:.4f}",
                        'P-value': f"{p_value:.3f}" if p_value and p_value < 0.05 else ('-' if not p_value else f"{p_value:.3f}")
                    })
            
        if convergence_rows:
            convergence_df = pd.DataFrame(convergence_rows)
            print(convergence_df.to_string(index=False))
            print("\nGhi chú: Speedup = Thời gian Optuna / Thời gian AMSCO.")
            print("'Trials to Best' = Số trial để đạt được best score.")
            print("P-value được tính bằng Mann-Whitney U test (cần nhiều seeds để có ý nghĩa thống kê).")
            print()
            
            # Cleanup
            del convergence_df
            del convergence_rows
            gc.collect()
    except Exception as e:
        print(f"[WARN] Không thể tạo Bảng 4.5: {e}")
    finally:
        gc.collect()
    
    # ============================================================
    # BẢNG 4.6: Điểm Hàm Mục tiêu Toàn diện
    # ============================================================
    try:
        print("\n")
        print("="*80)
        print("BẢNG 4.6: Điểm Hàm Mục tiêu Toàn diện")
        print("="*80)
            
        objective_rows = []
        # Tính f(θ) trước để xác định rank
        temp_results = {}
        for dataset in DATASETS:
            dataset_results = results_df[results_df['dataset'] == dataset]
            temp_results[dataset] = {}
                
            for optimizer in ['Optuna (TPE)', 'AMSCO']:
                opt_results = dataset_results[dataset_results['optimizer'] == optimizer]
                    
                if len(opt_results) > 0:
                    f1_mean = opt_results['f1'].mean() if 'f1' in opt_results.columns else 0
                    auc_mean = opt_results['roc_auc'].mean() if 'roc_auc' in opt_results.columns else 0
                    time_mean = opt_results['opt_wall_time'].mean() if 'opt_wall_time' in opt_results.columns else 0
                        
                    # Chuẩn hóa và tính f(θ)
                    if optimizer == 'Optuna (TPE)':
                        optuna_time = time_mean
                        f1_norm = 1.0
                        auc_norm = 1.0
                        time_norm = 1.0
                    else:
                        f1_norm = f1_mean / dataset_results[dataset_results['optimizer'] == 'Optuna (TPE)']['f1'].mean() if len(dataset_results[dataset_results['optimizer'] == 'Optuna (TPE)']) > 0 else 1.0
                        auc_norm = auc_mean / dataset_results[dataset_results['optimizer'] == 'Optuna (TPE)']['roc_auc'].mean() if len(dataset_results[dataset_results['optimizer'] == 'Optuna (TPE)']) > 0 else 1.0
                        time_norm = time_mean / optuna_time if optuna_time > 0 else 1.0
                        
                    f_theta = 0.4 * f1_norm + 0.4 * auc_norm - 0.2 * time_norm
                        
                    temp_results[dataset][optimizer] = {
                        'f1_mean': f1_mean,
                        'f1_norm': f1_norm,
                        'auc_mean': auc_mean,
                        'auc_norm': auc_norm,
                        'time_mean': time_mean,
                        'time_norm': time_norm,
                        'f_theta': f_theta
                    }
            
        # Xác định rank dựa trên f(θ) và tạo rows
        for dataset in DATASETS:
            if dataset not in temp_results or len(temp_results[dataset]) == 0:
                continue
                    
            # So sánh f(θ) để xác định rank
            optuna_f_theta = temp_results[dataset].get('Optuna (TPE)', {}).get('f_theta', 0)
            amsco_f_theta = temp_results[dataset].get('AMSCO', {}).get('f_theta', 0)
                
            for optimizer in ['Optuna (TPE)', 'AMSCO']:
                if optimizer not in temp_results[dataset]:
                    continue
                    
                data = temp_results[dataset][optimizer]
                    
                # Rank dựa trên f(θ): cao hơn = rank 1
                if optimizer == 'Optuna (TPE)':
                    rank = 1 if optuna_f_theta >= amsco_f_theta else 2
                else:  # AMSCO
                    rank = 1 if amsco_f_theta >= optuna_f_theta else 2
                    
                objective_rows.append({
                    'Dataset': dataset.replace('_', ' ').title(),
                    'Method': optimizer.split()[0],
                    'F1-Score': f"{data['f1_mean']:.4f} ({data['f1_norm']:.3f})",
                    'AUC-ROC': f"{data['auc_mean']:.4f} ({data['auc_norm']:.3f})",
                    'Time': f"{int(data['time_mean'])}s ({data['time_norm']:.3f})",
                    'f(θ)': f"{data['f_theta']:.3f}",
                    'Rank': f"{rank} ({'⋆' if rank == 1 else ''})"
                })
            
        if objective_rows:
            objective_df = pd.DataFrame(objective_rows)
            print(objective_df.to_string(index=False))
            print("\nGhi chú: f(θ) = 0.4·F1 + 0.4·AUC-ROC - 0.2·Time (chuẩn hóa). ⋆ = Phương pháp tốt nhất.")
            print()
            
            # Cleanup
            del objective_df
            del objective_rows
            del temp_results
            gc.collect()
    except Exception as e:
        print(f"[WARN] Không thể tạo Bảng 4.6: {e}")
    finally:
        gc.collect()

    # ============================================================
    # BẢNG 4.7: So sánh Variance của F1-Score qua 5 Outer Folds
    # ============================================================
    try:
        print("\n" + "="*80)
        print("BẢNG 4.7: So sánh Variance của F1-Score qua 5 Outer Folds")
        print("="*80)
            
        variance_rows = []
        from scipy.stats import levene
            
        for dataset in DATASETS:
            dataset_results = results_df[results_df['dataset'] == dataset]
                
            optuna_f1 = dataset_results[dataset_results['optimizer'] == 'Optuna (TPE)']['f1'].dropna().values
            amsco_f1 = dataset_results[dataset_results['optimizer'] == 'AMSCO']['f1'].dropna().values
                
            if len(optuna_f1) > 1 and len(amsco_f1) > 1:
                optuna_var = np.var(optuna_f1, ddof=1)
                amsco_var = np.var(amsco_f1, ddof=1)
                improvement = ((optuna_var - amsco_var) / optuna_var * 100) if optuna_var > 0 else 0
                    
                # Levene's test để so sánh phương sai (variance) giữa hai nhóm
                try:
                    stat_lev, p_val = levene(optuna_f1, amsco_f1, center='median')
                except Exception:
                    stat_lev, p_val = np.nan, np.nan
                    
                variance_rows.append({
                    'Dataset': dataset.replace('_', ' ').title(),
                    'Optuna Var': f"{optuna_var:.6f}",
                    'AMSCO Var': f"{amsco_var:.6f}",
                    'Reduction': f"✓ {improvement:.1f}%" if improvement > 0 else f"✗ {abs(improvement):.1f}%",
                    'Levene p-value': f"{p_val:.3f}" if not np.isnan(p_val) else 'nan'
                })
            
        if variance_rows:
            variance_df = pd.DataFrame(variance_rows)
            print(variance_df.to_string(index=False))
            print("\nGhi chú: Improvement = (Optuna Var - AMSCO Var) / Optuna Var × 100%")
            print("Levene's test kiểm định sự khác biệt phương sai giữa hai nhóm (Optuna vs AMSCO).")
            print("P-value cao khi có ít seeds (cần >= 5 seeds để có ý nghĩa thống kê).")
            print()
            
            # Cleanup
            del variance_df
            del variance_rows
            gc.collect()
    except Exception as e:
        print(f"[WARN] Không thể tạo Bảng 4.7: {e}")
    finally:
        gc.collect()
    
    # ============================================================
    # CLEANUP CACHES NGAY SAU CÁC BẢNG PHÂN TÍCH CHÍNH
    # Xóa nested_cache và true_test_cache để giải phóng RAM
    # ============================================================
    print("\n[Memory] Cleaning up analysis caches (nested_cache, true_test_cache)...")
    initial_nested_size = len(nested_cache) if nested_cache else 0
    initial_true_test_size = len(true_test_cache) if true_test_cache else 0
    
    nested_cache.clear()
    # Không xóa true_test_cache ở đây vì còn cần cho các bảng sau
    
    gc.collect()
    print(f"[Memory] Cleared nested_cache ({initial_nested_size} datasets), keeping true_test_cache for remaining tables")
    print(f"[Memory] Current true_test_cache size: {initial_true_test_size} datasets")

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
            disp2['Without CV'] = disp2['Without CV'].map(lambda v: f"{v:.4f}" if pd.notna(v) else '-')
            disp2['With K-Fold (k=5)'] = disp2['With K-Fold (k=5)'].map(lambda v: f"{v:.4f}" if pd.notna(v) else '-')
            disp2['With Nested CV'] = disp2['With Nested CV'].map(lambda v: f"{v:.4f}" if pd.notna(v) else '-')
            disp2['Improvement (%)'] = disp2['Improvement (%)'].map(lambda v: f"{v:.2f}" if pd.notna(v) else '-')

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
            return f"{v:.{nd}f}" if isinstance(v, (int, float, np.floating)) and pd.notna(v) else '-'

        for col in ['Accuracy', 'F1-Score', 'AUC-ROC']:
            # Với hàng Improvement -> hiển thị %
            mask_imp = display_df['Evaluation Method'] == 'Improvement (Nested vs Holdout)'
            display_df.loc[~mask_imp, col] = display_df.loc[~mask_imp, col].map(lambda v: _fmt_num(v, 4))
            display_df.loc[mask_imp, col] = display_df.loc[mask_imp, col].map(lambda v: (f"{float(v):.2f}%" if pd.notna(v) else '-'))

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

            metric_subset = results_df[['dataset', 'model', 'optimizer', metric_col]].copy()
            metric_subset = metric_subset.dropna(subset=[metric_col])
            if metric_subset.empty:
                continue
            metric_subset['rep_idx'] = metric_subset.groupby(['dataset', 'model', 'optimizer']).cumcount()
            pivot = metric_subset.pivot_table(
                index=['dataset', 'model', 'rep_idx'],
                columns='optimizer',
                values=metric_col,
                aggfunc='first'
            )

            for a, b in comparisons:
                if a not in pivot.columns or b not in pivot.columns:
                    continue
                pair_df = pivot[[a, b]].dropna()
                if pair_df.empty:
                    continue
                xa = pair_df[a].to_numpy(dtype=float)
                xb = pair_df[b].to_numpy(dtype=float)
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
                    "Preferred": better_method,
                    "Sample Size": len(xa)
                })

        if rows_stat and not QUICK_MODE:
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
            
            # Cleanup stat_df
            del stat_df
            gc.collect()
    except Exception as e:
        print(f"[WARN] Không thể tạo bảng so sánh thống kê tài nguyên: {e}")
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
        
        # Cleanup summary_opt_df
        del summary_opt_df
        gc.collect()
    except Exception as e:
        print(f"[WARN] Không thể tạo bảng tổng hợp theo optimizer: {e}")

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Xuất CSV kết quả seed nếu có
    if OPTUNA_RESULTS:
        try:
            pd.DataFrame(OPTUNA_RESULTS).to_csv('results/optuna_seed_results.csv', index=False)
            print('[INFO] Đã lưu results/optuna_seed_results.csv')
        except Exception as e:
            print(f'[WARN] Không thể lưu results/optuna_seed_results.csv: {e}')

    # Xuất CSV convergence history cho Hình 4.1
    if CONV_LOG:
        try:
            pd.DataFrame(CONV_LOG).to_csv('results/convergence_history.csv', index=False)
            print('[INFO] Đã lưu results/convergence_history.csv')
        except Exception as e:
            print(f'[WARN] Không thể lưu results/convergence_history.csv: {e}')
    
    # === FINAL CLEANUP: Xóa true_test_cache SAU KHI đã tạo xong TẤT CẢ bảng phân tích ===
    print("\n[Memory] Final cleanup: Clearing true_test_cache...")
    true_test_cache.clear()
    gc.collect()
    print("[Memory] true_test_cache cleared")
        
    # === CLEANUP: XÓA CÁC FILE SEED TẠM SAU KHI ĐÃ MERGE ===
    if SAVE_PER_SEED:
        import glob
        # Clean up all temporary per-seed files
        temp_patterns = ['results/seed_*_results.csv', 'results/seed_*_optuna.csv', 
                        'results/seed_*_amsco.csv', 'results/seed_*_conv_log.csv',
                        'results/seed_*_true_test_cache.json']  # Thêm true_test_cache files
        all_temp_files = []
        for pattern in temp_patterns:
            all_temp_files.extend(glob.glob(pattern))
        
        if all_temp_files:
            print(f"\n[Cleanup] Removing {len(all_temp_files)} temporary seed files...")
            for seed_file in all_temp_files:
                try:
                    os.remove(seed_file)
                    print(f"  Removed: {seed_file}")
                except Exception as e:
                    print(f"  [WARNING] Could not remove {seed_file}: {e}")
            print("[Cleanup] Temporary seed files cleaned up")
            
    print(f"\n[INFO] Pipeline completed in {time.time() - global_start_time:.2f} seconds")
    print(f"[INFO] Results saved to results/ directory")
