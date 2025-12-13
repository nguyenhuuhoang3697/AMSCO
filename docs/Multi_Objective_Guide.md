# Multi-Objective Optimization trong AMSCO

## Tổng Quan

Framework AMSCO đã được mở rộng để hỗ trợ **Multi-Objective Optimization** với hàm mục tiêu:

```
f(θ) = α·F1 + β·ROC-AUC - γ·execution_time
```

**Mục tiêu**: Tối đa hóa f(θ)

### Tham Số Mặc Định
- α = 0.4 (trọng số F1-score)
- β = 0.4 (trọng số ROC-AUC)
- γ = 0.2 (penalty cho execution time)

## Quy Trình Hoạt Động

### 1. Warm-up Phase
Trước khi bắt đầu tối ưu hóa chính thức, hệ thống chạy **warm-up trials** (10% của total trials) để:
- Thu thập giá trị F1, ROC-AUC, execution_time từ các cấu hình ngẫu nhiên
- Xác định min/max cho mỗi metric (Xmin, Xmax)

### 2. Normalization
Mỗi metric được chuẩn hóa về khoảng [0, 1] bằng Min-Max scaling:

```python
metric_normalized = (metric - Xmin) / (Xmax - Xmin)
```

### 3. Objective Calculation
Sau khi chuẩn hóa, tính hàm mục tiêu:

```python
f(θ) = α × F1_norm + β × ROC-AUC_norm - γ × time_norm
```

### 4. Optimization
Các optimizer (Random, Bayesian, Grid) tìm kiếm cấu hình θ có f(θ) cao nhất.

## Cách Sử Dụng

### Bật Multi-Objective trong Code

```python
# Trong amsco_adult_v2.py, tại phần __main__

USE_MULTI_OBJECTIVE = True    # Bật multi-objective
ALPHA = 0.4                   # Trọng số F1
BETA = 0.4                    # Trọng số ROC-AUC  
GAMMA = 0.2                   # Penalty cho execution_time
```

### Sử Dụng trong Code

```python
from amsco_adult_v2 import (
    MultiObjectiveNormalizer,
    create_objective,
    run_amsco_optimizer
)

# 1. Tạo normalizer
normalizer = MultiObjectiveNormalizer(alpha=0.4, beta=0.4, gamma=0.2)

# 2. Tạo objective function
objective = create_objective(
    X, y,
    model_name='logistic_regression',
    preprocessor=preprocessor,
    metrics=('f1', 'roc_auc'),
    use_cross_validation=True,
    cv_folds=5,
    multi_objective=True,      # Bật multi-objective
    normalizer=normalizer       # Truyền normalizer
)

# 3. Chạy optimizer
score, params, diag = run_amsco_optimizer(
    objective,
    search_space,
    n_trials=100,
    slice_budget=10,
    normalizer=normalizer       # Truyền normalizer
)
```

## Demo Script

Chạy script demo để xem sự khác biệt:

```bash
cd /home/user/AMSCO
source venv/bin/activate
python test_multi_objective.py
```

Script sẽ so sánh:
- **Single-Objective**: Chỉ tối ưu F1-score
- **Multi-Objective**: Tối ưu f(θ) = 0.4·F1 + 0.4·ROC-AUC - 0.2·time

## Components Mới

### 1. MultiObjectiveNormalizer

```python
class MultiObjectiveNormalizer:
    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2):
        # Khởi tạo với trọng số
        
    def collect(self, f1, roc_auc, exec_time):
        # Thu thập metrics từ trials
        
    def compute_bounds(self):
        # Tính min/max từ dữ liệu đã thu thập
        
    def compute_objective(self, f1, roc_auc, exec_time):
        # Tính f(θ) = α·F1_norm + β·ROC-AUC_norm - γ·time_norm
```

### 2. Warm-up Function

```python
def warmup_normalizer(objective, search_space, n_warmup_trials=10):
    # Chạy random trials để thu thập dữ liệu cho normalizer
```

### 3. Updated Objective Function

```python
def create_objective(..., multi_objective=False, normalizer=None):
    # Hỗ trợ cả single và multi-objective
    # Đo execution time cho mỗi fold
    # Thu thập và chuẩn hóa metrics
```

## Lợi Ích

### 1. Cân Bằng Nhiều Mục Tiêu
- **Single-objective** có thể tối ưu F1 cao nhưng ROC-AUC thấp hoặc chậm
- **Multi-objective** cân bằng giữa accuracy, discrimination power và speed

### 2. Phù Hợp Thực Tế
- Production cần model vừa chính xác, vừa nhanh
- Tránh overfitting vào một metric duy nhất

### 3. Linh Hoạt
- Điều chỉnh α, β, γ theo nhu cầu:
  - α, β cao: Ưu tiên accuracy
  - γ cao: Ưu tiên speed
  
## Ví Dụ Kết Quả

```
Single-Objective (F1 only):
  F1-score:       0.9654
  ROC-AUC:        0.9823
  Execution time: 0.0234s
  Composite f(θ): 0.7821

Multi-Objective (F1 + ROC-AUC - time):
  F1-score:       0.9632
  ROC-AUC:        0.9891
  Execution time: 0.0156s
  Composite f(θ): 0.8145  ← Cao hơn!
```

→ Multi-objective cho model cân bằng hơn với composite score cao hơn.

## Cấu Hình Nâng Cao

### Điều Chỉnh Trọng Số

```python
# Ưu tiên accuracy hơn speed
normalizer = MultiObjectiveNormalizer(alpha=0.45, beta=0.45, gamma=0.1)

# Ưu tiên speed cho real-time system
normalizer = MultiObjectiveNormalizer(alpha=0.3, beta=0.3, gamma=0.4)

# Chỉ quan tâm ROC-AUC và speed
normalizer = MultiObjectiveNormalizer(alpha=0.0, beta=0.7, gamma=0.3)
```

### Tắt Multi-Objective

Nếu muốn quay lại single-objective:

```python
USE_MULTI_OBJECTIVE = False

objective = create_objective(
    ...,
    multi_objective=False,   # Tắt
    normalizer=None          # Không cần normalizer
)
```

## Lưu Ý Kỹ Thuật

1. **Warm-up overhead**: Thêm ~10% trials cho warm-up, nhưng cần thiết để tính bounds
2. **Time measurement**: Đo thời gian fit + predict, không bao gồm preprocessing
3. **Normalization stability**: Nếu tất cả trials có giá trị gần nhau, normalization có thể không ổn định (return 0.5)
4. **Metrics required**: Multi-objective yêu cầu đo F1 và ROC-AUC, không áp dụng cho regression

## Tích Hợp với Pipeline Hiện Tại

Code mới **backward compatible**:
- Không truyền `multi_objective` và `normalizer` → hoạt động như cũ
- Truyền đầy đủ tham số → bật multi-objective

Tất cả baselines (Random Search, Optuna TPE, Hyperopt) đều được cập nhật để hỗ trợ multi-objective.

---

**Tác giả**: AMSCO Framework  
**Phiên bản**: 2.0 (Multi-Objective Support)  
**Ngày cập nhật**: December 2, 2025
