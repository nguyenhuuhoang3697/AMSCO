#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
import math
import time
import random
from collections import defaultdict
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.datasets import load_breast_cancer

# Tắt các cảnh báo không cần thiết
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
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


def get_data(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'adult':
        csv_path = Path('datasets') / 'adult.csv'
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {csv_path}. Vui lòng đặt adult.csv vào thư mục datasets/ như README hướng dẫn."
            )
        print("... Đang tải Adult Income")
        X, y = load_adult_dataset(csv_path)
        X, preprocessor = preprocess_adult_columns(X)
        metric = 'accuracy'
        return X, y, preprocessor, metric

    if dataset_name == 'breast_cancer':
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
        print("... Đang tải Telco Customer Churn")
        X, y = load_telco_dataset(csv_path)
        X, preprocessor = preprocess_adult_columns(X)
        metric = 'accuracy'
        return X, y, preprocessor, metric

    raise ValueError("Notebook này chỉ hỗ trợ dataset 'adult', 'breast_cancer', hoặc 'telco'.")


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
    validation_data=None
):
    """
    Xây dựng hàm objective cho optimizer.

    Nếu `use_cross_validation=True`, sử dụng StratifiedKFold (3-fold) và trả về metric đầu tiên (primary) làm giá trị tối ưu.
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
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
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

    def store(self, agent_id, params, score):
        self.trials.append({'agent_id': agent_id, 'params': params, 'score': score})
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
            # print(f"  [KnowledgeHub] New best score: {self.best_score:.4f} from {agent_id}")

    def get_all_trials(self):
        return self.trials

    def get_best_trial(self):
        return {'params': self.best_params, 'score': self.best_score}

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

        # (A) Tự động xây dựng hàm objective cho Optuna
        def optuna_objective(trial):
            params = {}
            for name, details in self.search_space.items():
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

            score = self.objective(params)
            self.knowledge_hub.store(self.agent_id, params, score) # Vẫn báo cáo về Hub chung
            return score

        # (B) Khởi tạo và Warm-start
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())

        existing_trials = self.knowledge_hub.get_all_trials()
        if existing_trials:
            # print(f"  ... BayesianAgent warm-starting with {len(existing_trials)} previous trials.")

            # Tự động tạo 'distributions' cho Optuna
            distributions = {}
            for name, details in self.search_space.items():
                type = details[0]  # Type là phần tử đầu tiên của tuple
                if type == 'float':
                    low, high = details[1], details[2]
                    dist_type = details[3] if len(details) > 3 else None
                    distributions[name] = optuna.distributions.FloatDistribution(low, high, log=(dist_type == 'log'))
                elif type == 'int':
                    low, high = details[1], details[2]
                    distributions[name] = optuna.distributions.IntDistribution(low, high)
                elif type == 'categorical':
                    choices = details[1]
                    distributions[name] = optuna.distributions.CategoricalDistribution(choices)

            for t in existing_trials:
                if not t or 'params' not in t or 'score' not in t:
                    continue  # Bỏ qua trial không hợp lệ

                try:
                    # Đảm bảo các tham số từ agent khác nằm trong không gian
                    valid_params = {}
                    for k, v in t['params'].items():
                        if k in distributions:
                            # Đặc biệt xử lý cho categorical parameters
                            if isinstance(distributions[k], optuna.distributions.CategoricalDistribution):
                                if v in distributions[k].choices:
                                    valid_params[k] = v
                            else:
                                valid_params[k] = v

                    if not valid_params or len(valid_params) != len(t['params']):
                        continue  # Bỏ qua trial có tham số lạ hoặc giá trị không hợp lệ

                    # Đảm bảo score là một số hợp lệ
                    if not isinstance(t['score'], (int, float)) or math.isnan(t['score']):
                        continue

                    frozen_trial = optuna.trial.create_trial(
                        params=valid_params,
                        distributions=distributions,
                        value=t['score']
                    )
                    study.add_trial(frozen_trial)
                except Exception as e:
                    print(f"Lỗi khi thêm trial: {e}")  # In ra lỗi để debug
                    continue  # Bỏ qua các trial không tương thích

        study.optimize(optuna_objective, n_trials=budget, show_progress_bar=False)

class GridAgent(StrategyAgent):
    """GridAgent: Linh hoạt (tinh chỉnh 2 tham số quan trọng nhất)"""
    def run(self, budget):
        # print(f"    -> Running GridAgent with budget: {budget}")
        best_params = self.knowledge_hub.get_best_trial()['params']
        if not best_params:
            # print("  ... GridAgent skipped (no best params yet).")
            return

        # Chọn 2 tham số đầu tiên (hoặc float/int) để tinh chỉnh
        params_to_tune = []
        for name, details in self.search_space.items():
            if details[0] in ['float', 'int']:  # Kiểm tra type ở vị trí đầu tiên của tuple
                params_to_tune.append(name)
            if len(params_to_tune) >= 2:
                break

        if len(params_to_tune) == 0:
            return 

        local_grid = [best_params]

        # Lấy tham số đầu tiên để tinh chỉnh
        p1_name = params_to_tune[0]
        p1_val = best_params.get(p1_name)

        p1_details = self.search_space[p1_name]
        p1_type = p1_details[0]  # Type là phần tử đầu tiên của tuple
        p1_low = p1_details[1]   # Low là phần tử thứ hai
        p1_high = p1_details[2]  # High là phần tử thứ ba
        # --- KẾT THÚC SỬA LỖI ---

        # Tạo 3-5 điểm cho p1
        if p1_type == 'float':
            p1_steps = np.linspace(p1_val * 0.9, p1_val * 1.1, 3)
        else: # int
            p1_steps = {p1_val - 1, p1_val, p1_val + 1}

        for p1 in p1_steps:
            # "Kẹp" giá trị trong phạm vi
            p1_clamped = max(p1_low, min(p1_high, p1))
            if p1_type == 'int': p1_clamped = int(p1_clamped)

            new_params = {**best_params, p1_name: p1_clamped}
            if new_params not in local_grid:
                local_grid.append(new_params)

        # Giới hạn số lần chạy theo budget
        for i in range(min(budget, len(local_grid))):
            params = local_grid[i]
            score = self.objective(params)
            self.knowledge_hub.store(self.agent_id, params, score)


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
    def __init__(self, agent_ids):
        self.agent_ids = agent_ids
        self.agent_pulls = {agent_id: 0 for agent_id in agent_ids}
        self.agent_rewards = {agent_id: 0.0 for agent_id in agent_ids}
        self.total_pulls = 0

    def allocate(self, slice_budget):
        # Kiểm tra các agent chưa được khởi tạo
        uninitialized_agents = [aid for aid, pulls in self.agent_pulls.items() if pulls == 0]
        if uninitialized_agents:
            agent_to_run = uninitialized_agents[0]  # Lấy agent đầu tiên chưa được khởi tạo
            allocations = {agent_id: 0 for agent_id in self.agent_ids}
            allocations[agent_to_run] = slice_budget
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

        best_agent = max(ucb_scores, key=ucb_scores.get)
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
    def __init__(self, objective_func, search_space, total_budget, slice_budget):
        self.total_budget = total_budget
        self.slice_budget = slice_budget

        self.knowledge_hub = KnowledgeHub()

        self.agents = {
            "Random": RandomAgent("Random", objective_func, search_space, self.knowledge_hub),
            "Bayesian": BayesianAgent("Bayesian", objective_func, search_space, self.knowledge_hub),
            "Grid": GridAgent("Grid", objective_func, search_space, self.knowledge_hub)
        }
        agent_ids = list(self.agents.keys())

        self.performance_monitor = PerformanceMonitor(agent_ids)
        self.meta_controller = MetaController_UCB1(agent_ids)

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

                    all_trials = self.knowledge_hub.get_all_trials()
                    self.performance_monitor.update(all_trials)

                    reward = self.performance_monitor.get_agent_rewards()[agent_id]
                    self.meta_controller.update(agent_id, reward)

            current_budget -= budget_for_slice
            slice_num += 1

        final_result = self.knowledge_hub.get_best_trial()
        return final_result


# In[6]:


# ============================================================================
# BƯỚC 5: CÁC TRÌNH TỐI ƯU HÓA BASELINE
# ============================================================================

def run_random_search(objective, search_space, n_trials):
    """Chạy Random Search (sử dụng Optuna)"""
    print("  Running Random Search...")
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
    return study.best_trial.value, study.best_params

def run_optuna_tpe(objective, search_space, n_trials):
    """Chạy Optuna TPE (baseline)"""
    print("  Running Optuna (TPE)...")
    sampler = optuna.samplers.TPESampler() # Sử dụng TPE sampler [17, 18, 19, 20]
    study = optuna.create_study(direction='maximize', sampler=sampler)

    # Hàm mục tiêu (giống hệt Random Search)
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
    return study.best_trial.value, study.best_params

def run_hyperopt_tpe(objective, search_space, n_trials):
    """Chạy Hyperopt TPE (baseline)"""
    print("  Running Hyperopt (TPE)...")

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
        verbose=0
    )

    best_score = -trials.best_trial['result']['loss']
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
    return best_score, best_params


def run_amsco_optimizer(objective, search_space, n_trials, slice_budget):
    """Chạy AMSCO (phương pháp của chúng ta)"""
    print("  Running AMSCO...")
    orchestrator = AMSCO_Orchestrator(
        objective_func=objective,
        search_space=search_space,
        total_budget=n_trials,
        slice_budget=slice_budget
    )
    result = orchestrator.run()
    return result['score'], result['params']


# In[ ]:


# =============================================================================
# BƯỚC 6: BỘ CÔNG CỤ THỰC NGHIỆM (EXPERIMENTAL HARNESS)
# - Mở rộng chạy trên Adult, Breast Cancer, Telco
# - Thêm metrics: accuracy (primary), f1, roc_auc
# - Dùng StratifiedKFold (CV) mặc định
# =============================================================================

if __name__ == "__main__":

    # ----- CẤU HÌNH THỬ NGHIỆM -----
    DATASETS = ['adult', 'breast_cancer', 'telco']
    MODELS = ['logistic_regression', 'random_forest', 'lightgbm', 'xgboost']
    TOTAL_TRIALS = 100  # Tăng lên để TPE và AMSCO có cơ hội thể hiện
    SLICE_BUDGET = 10   # Lát cắt cho AMSCO (phân bổ động)

    USE_CROSS_VALIDATION = True   # Bật CV 3-fold
    TEST_SIZE = 0.2               # Chỉ dùng nếu holdout
    METRICS = ('accuracy', 'f1', 'roc_auc')  # Primary = accuracy
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

            if USE_CROSS_VALIDATION:
                print("  [Evaluation] StratifiedKFold (3-fold)")
                objective_func = create_objective(
                    X,
                    y,
                    model_name,
                    preprocessor,
                    metrics=METRICS,
                    use_cross_validation=True
                )
                eval_label = 'cv_3fold'
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
                    validation_data=(X_valid, y_valid)
                )
                eval_label = f'holdout_{TEST_SIZE}'

            model_results = {
                'scores': {},
                'times': {},
                'params': {},
                'metrics': {}
            }

            optimizers = [
                ('Random Search', run_random_search),
                ('Optuna (TPE)', run_optuna_tpe),
                ('Hyperopt (TPE)', run_hyperopt_tpe),
                ('AMSCO', run_amsco_optimizer)
            ]

            for optimizer_name, optimizer_func in optimizers:
                print(f"\nThực thi: {optimizer_name}...")
                start_time = time.time()

                if optimizer_name == 'AMSCO':
                    primary_score, params = optimizer_func(objective_func, search_space, TOTAL_TRIALS, SLICE_BUDGET)
                else:
                    primary_score, params = optimizer_func(objective_func, search_space, TOTAL_TRIALS)

                exec_time = time.time() - start_time

                # Đánh giá thêm các metric cho best params
                metric_scores = {m: np.nan for m in METRICS}
                try:
                    model = _build_model_for_eval(model_name)
                    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
                    pipeline.set_params(**params)
                    if USE_CROSS_VALIDATION:
                        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                        for m in METRICS:
                            try:
                                metric_scores[m] = cross_val_score(pipeline, X, y, cv=cv, scoring=m, n_jobs=-1).mean()
                            except Exception as e:
                                print(f"  [WARN] Không tính được metric {m} với CV cho {optimizer_name}: {e}")
                                metric_scores[m] = np.nan
                    else:
                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_valid)
                        metric_scores['accuracy'] = accuracy_score(y_valid, y_pred)
                        try:
                            metric_scores['f1'] = f1_score(y_valid, y_pred, average='binary')
                        except Exception:
                            metric_scores['f1'] = np.nan
                        try:
                            if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                                y_prob = pipeline.predict_proba(X_valid)[:, 1]
                            elif hasattr(pipeline.named_steps['classifier'], 'decision_function'):
                                y_prob = pipeline.decision_function(X_valid)
                            else:
                                y_prob = None
                            metric_scores['roc_auc'] = roc_auc_score(y_valid, y_prob) if y_prob is not None else np.nan
                        except Exception:
                            metric_scores['roc_auc'] = np.nan
                except Exception as e:
                    print(f"  [WARN] Lỗi khi tính metrics bổ sung cho {optimizer_name}: {e}")

                model_results['scores'][optimizer_name] = primary_score
                model_results['times'][optimizer_name] = exec_time
                model_results['params'][optimizer_name] = params
                model_results['metrics'][optimizer_name] = metric_scores

                results.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'optimizer': optimizer_name,
                    'primary_metric': METRICS[0],
                    'primary_score': primary_score,
                    'accuracy': metric_scores.get('accuracy', np.nan),
                    'f1': metric_scores.get('f1', np.nan),
                    'roc_auc': metric_scores.get('roc_auc', np.nan),
                    'time': exec_time,
                    'evaluation': eval_label
                })

            if model_results['scores']:
                print(f"\n{'='*60}")
                print(f"KẾT QUẢ CHO {model_name.upper()} (Primary metric: {METRICS[0]})")
                print(f"{'='*60}")

                results_table = pd.DataFrame({
                    'Optimizer': list(model_results['scores'].keys()),
                    'Accuracy': [model_results['metrics'][opt]['accuracy'] for opt in model_results['scores'].keys()],
                    'F1': [model_results['metrics'][opt]['f1'] for opt in model_results['scores'].keys()],
                    'ROC AUC': [model_results['metrics'][opt]['roc_auc'] for opt in model_results['scores'].keys()],
                    'Time (s)': list(model_results['times'].values())
                })

                for col in ['Accuracy', 'F1', 'ROC AUC']:
                    results_table[col] = results_table[col].map(lambda v: f"{v:.4f}" if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else "nan")
                results_table['Time (s)'] = results_table['Time (s)'].map('{:.2f}'.format)

                print("\n" + results_table.to_string(index=False))
                print("\n" + "-"*60 + "\n")

    if not results:
        print("Không có kết quả nào được ghi nhận.")
    else:
        print("\n\n" + "="*60)
        print(" "*20 + "KẾT QUẢ THỬ NGHIỆM TỔNG QUÁT" + " "*20)
        print("="*60 + "\n")

        results_df = pd.DataFrame(results)

        # Pivots theo từng metric
        for metric_name in ['primary_score', 'accuracy', 'f1', 'roc_auc']:
            try:
                pivot_metric = results_df.pivot_table(
                    values=metric_name,
                    index=['dataset', 'model'],
                    columns='optimizer',
                    aggfunc='mean'
                )
                title = 'PRIMARY SCORE' if metric_name == 'primary_score' else metric_name.upper()
                print(f"ĐIỂM SỐ TRUNG BÌNH THEO DATASET VÀ MÔ HÌNH ({title}):")
                print("-"*60)
                print(pivot_metric.round(4).to_string())
                print("\n")
            except Exception as e:
                print(f"[WARN] Không thể tạo pivot cho {metric_name}: {e}")

        pivot_times = results_df.pivot_table(
            values='time',
            index=['dataset', 'model'],
            columns='optimizer',
            aggfunc='mean'
        )

        print("THỜI GIAN CHẠY TRUNG BÌNH (GIÂY):")
        print("-"*60)
        print(pivot_times.round(2).to_string())
        print("\n")

        print("THỐNG KÊ TỔNG HỢP (Primary metric):")
        print("-"*60)
        stats_df = pd.DataFrame({
            'Optimizer': results_df['optimizer'].unique(),
            'Avg Score': [results_df[results_df['optimizer'] == opt]['primary_score'].mean() for opt in results_df['optimizer'].unique()],
            'Std Score': [results_df[results_df['optimizer'] == opt]['primary_score'].std() for opt in results_df['optimizer'].unique()],
            'Avg Time': [results_df[results_df['optimizer'] == opt]['time'].mean() for opt in results_df['optimizer'].unique()],
            'Total Time': [results_df[results_df['optimizer'] == opt]['time'].sum() for opt in results_df['optimizer'].unique()]
        })
        stats_df = stats_df.round(4)
        print(stats_df.to_string(index=False))

        print("\nGhi chú:")
        if USE_CROSS_VALIDATION:
            print("- Điểm số là mean của StratifiedKFold (3-fold) cho các metric: accuracy, f1, roc_auc")
        else:
            print(f"- Điểm số là trên tập holdout (test_size={TEST_SIZE}) cho các metric: accuracy, f1, roc_auc")
        print("- Tối ưu dựa trên primary metric: {}".format(METRICS[0]))

        print("\n\n=======================================================")
        print("         KẾT QUẢ THỬ NGHIỆM TỔNG QUÁT")
        print("=======================================================")

        print(results_df.to_string(index=False))


# In[ ]:




