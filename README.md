# AMSCO (Automated Model Selection and Configuration Optimization)

AMSCO là một framework tối ưu hóa siêu tham số tự động, kết hợp nhiều phương pháp tối ưu hóa khác nhau như Random Search, Bayesian Optimization (TPE), và Grid Search trong một hệ thống đa tác tử thông minh.

## Các tính năng chính

- Tự động tối ưu hóa siêu tham số cho nhiều loại mô hình khác nhau
- Hỗ trợ nhiều thuật toán tối ưu hóa:
  - Random Search
  - Bayesian Optimization (TPE via Optuna)
  - Hyperopt TPE
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
- Breast Cancer: Được tích hợp sẵn trong scikit-learn
- Adult Income: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- Telco Customer Churn: [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
  - Tải file và lưu vào `datasets/telco.csv`
- Credit Card Fraud: [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
  - Tải file và lưu vào `datasets/creditcard.csv`

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
```

2. Chạy các cell theo thứ tự để:
- Import thư viện và cài đặt môi trường
- Định nghĩa không gian tìm kiếm cho các mô hình
- Tải và tiền xử lý dữ liệu
- Chạy thử nghiệm với các optimizer khác nhau

## Kết quả

AMSCO được thử nghiệm trên 4 bộ dữ liệu với 4 loại mô hình khác nhau, so sánh với các baseline:
- Random Search
- Optuna (TPE)
- Hyperopt (TPE)

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
