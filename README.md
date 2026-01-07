# AMSCO (Adaptive Multi-Strategy Coordinated Optimization)

AMSCO là một framework tối ưu hóa siêu tham số tự động, kết hợp nhiều phương pháp tối ưu hóa khác nhau như Random Search, Bayesian Optimization (TPE), và Grid Search trong một hệ thống đa tác tử thông minh.

## Các tính năng chính

- Tự động tối ưu hóa siêu tham số cho nhiều loại mô hình khác nhau
- Hỗ trợ nhiều thuật toán tối ưu hóa:
  - Random Search
  - Bayesian Optimization (TPE via Optuna)
  - Grid Search (tinh chỉnh local)
- Hệ thống đa tác tử thông minh với cơ chế học tăng cường
- Tự động điều chỉnh ngân sách tính toán giữa các tác tử

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/nguyenhuuhoang3697/AMSCO.git
cd AMSCO
```

2. Tạo và kích hoạt virtual environment:
```bash
# Trên Windows
python -m venv venv
venv\Scripts\activate

# Trên Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

4. Tải datasets:

- Breast Cancer: Được tích hợp sẵn trong scikit-learn (không cần tải)
- Adult Income (UCI): Có thể tải tự động bằng script hoặc thủ công theo hướng dẫn dưới.
- Telco Customer Churn (Kaggle): Cần tài khoản Kaggle và API key.
- Credit Card Fraud (Kaggle): Cần tài khoản Kaggle và API key.

Tùy chọn A — Tải nhanh (Linux/macOS):
```bash
# Tạo thư mục datasets nếu chưa có
mkdir -p datasets

# Chạy script tải dữ liệu Adult từ UCI (đã kèm sẵn)
# Script sẽ GHÉP train/test thành một file duy nhất: datasets/adult.csv
# và loại bỏ thư mục con datasets/adult/ cùng các file gốc adult.data / adult.test / adult.names
bash scripts/download_datasets.sh
```

Tùy chọn B — Hướng dẫn chi tiết theo từng bộ dữ liệu:

1) Adult Income (UCI)
```bash
mkdir -p datasets/adult
BASE_UCI="https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
curl -fsSL "$BASE_UCI/adult.data" -o datasets/adult/adult.data
curl -fsSL "$BASE_UCI/adult.test" -o datasets/adult/adult.test
curl -fsSL "$BASE_UCI/adult.names" -o datasets/adult/adult.names || true
# Gộp và chuyển về datasets/adult.csv, đồng thời xóa thư mục con datasets/adult
python3 scripts/prepare_adult.py
```

2) Telco Customer Churn (Kaggle)
- Cài đặt Kaggle CLI (chỉ cần 1 lần, ngoài môi trường dự án):
```bash
pip install kaggle
mkdir -p ~/.kaggle
# Tải API token từ https://www.kaggle.com/settings (kaggle.json)
# Sau đó đặt vào ~/.kaggle/kaggle.json và phân quyền an toàn
chmod 600 ~/.kaggle/kaggle.json
```
- Tải và đổi tên file:
```bash
mkdir -p datasets
kaggle datasets download -d blastchar/telco-customer-churn -p datasets
unzip -o datasets/telco-customer-churn.zip -d datasets
# File gốc: WA_Fn-UseC_-Telco-Customer-Churn.csv
mv -f "datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv" datasets/telco.csv
```

3) Credit Card Fraud (Kaggle)
```bash
mkdir -p datasets
kaggle datasets download -d mlg-ulb/creditcardfraud -p datasets
unzip -o datasets/creditcardfraud.zip -d datasets
mv -f datasets/creditcard.csv datasets/creditcard.csv  # đảm bảo tên đích đúng
```

Lưu ý:
- Các tập dữ liệu Kaggle yêu cầu đăng nhập và tạo API token theo điều khoản Kaggle.
- Nếu không muốn dùng Kaggle CLI, bạn có thể tải thủ công từ trang Kaggle và đặt file vào:
  - `datasets/telco.csv`
  - `datasets/creditcard.csv`

Tùy chọn C — Tải nhanh 2 bộ Kaggle (nếu đã có API token):
```bash
# Sau khi đã có ~/.kaggle/kaggle.json (chmod 600)
bash scripts/download_kaggle_datasets.sh
# Kết quả:
# - datasets/telco.csv
# - datasets/creditcard.csv
```

## Cấu trúc thư mục
```
AMSCO/
├── amsco.ipynb          # Notebook chính với toàn bộ implementation
├── datasets/            # Thư mục chứa datasets (cần tạo)
│   ├── telco.csv       # (cần tải về)
│   └── creditcard.csv  # (cần tải về)
├── main.py             # Script version của implementation
├── requirements.txt    # Các thư viện phụ thuộc
└── README.md          # File này
```

## Sử dụng

1. Mở Jupyter Notebook:
```bash
jupyter notebook amsco.ipynb
nohup python -u amsco_adult_v2.py > output.log 2>&1 &
pip install pillow python-telegram-bot==13.15
export TELEGRAM_TOKEN/TELEGRAM_CHAT_ID
nohup python -u send_log_to_telegram.py > log_watcher.log 2>&1 &     #send noti to telegram
ps -eo pid,cmd | grep python
pkill -f "python -u amsco_adult_v2.py"
```

2. Chạy các cell theo thứ tự để:
- Import thư viện và cài đặt môi trường
- Định nghĩa không gian tìm kiếm cho các mô hình
- Tải và tiền xử lý dữ liệu
- Chạy thử nghiệm với các optimizer khác nhau

## Kết quả

AMSCO được thử nghiệm trên 4 bộ dữ liệu với 2 loại mô hình khác nhau, so sánh với baseline Optuna (TPE)

## Tài liệu tham khảo

[1] Bergstra, J., Yamins, D., & Cox, D. D. (2013). Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures.
[2] Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical bayesian optimization of machine learning algorithms.
[3] Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017). Hyperband: A novel bandit-based approach to hyperparameter optimization.

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng:
1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## Giấy phép

Phân phối theo giấy phép MIT. Xem `LICENSE` để biết thêm thông tin.
