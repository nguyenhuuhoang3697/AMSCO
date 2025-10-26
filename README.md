# Hướng dẫn cài đặt và chạy dự án

## 1. Tạo virtual environment
```bash
# On some systems use `python3` if `python` is not available
python3 -m venv venv
```

## 2. Kích hoạt virtual environment

### Trên Linux/Mac:
```bash
source venv/bin/activate
```

### Trên Windows:
```bash
venv\Scripts\activate
```

## 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## 4. Chạy chương trình
```bash
python main.py
```

## 5. Tắt virtual environment
```bash
deactivate
```

## Lưu ý:
- Luôn kích hoạt virtual environment trước khi làm việc với dự án
- File `requirements.txt` chứa danh sách các thư viện cần thiết
- Virtual environment giúp tránh xung đột thư viện giữa các dự án khác nhau
 
Lưu ý: trên một số hệ thống (như Ubuntu/Debian) lệnh `python` có thể không được cài hoặc trỏ tới Python 2.
Nếu lệnh `python` báo lỗi "command not found", dùng `python3` thay thế: `python3 -m venv venv`.
