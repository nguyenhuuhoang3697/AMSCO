import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Đọc tất cả các file CSV từ thư mục results_step2
# Sử dụng nested CV vì đó là dữ liệu trong Table 4.5 (output1.log)
results_path = 'results_step2/'
all_files = glob.glob(os.path.join(results_path, 'nested_*.csv'))

# Đọc và gộp tất cả các file
df_list = []
for file in all_files:
    df = pd.read_csv(file)
    df_list.append(df)

# Gộp tất cả dữ liệu
data = pd.concat(df_list, ignore_index=True)

# Lọc chỉ lấy cột cần thiết
data = data[['dataset', 'method', 'time']]

# Đổi tên method cho ngắn gọn
data['method'] = data['method'].replace({'Optuna (TPE)': 'Optuna'})

# Thứ tự hiển thị các dataset
dataset_order = ['adult', 'breast_cancer', 'credit', 'telco']

# Tạo figure với kích thước phù hợp
plt.figure(figsize=(14, 6))

# Tạo boxplot
sns.boxplot(data=data, x='dataset', y='time', hue='method', 
            order=dataset_order, palette=['blue', 'green'])

# Tùy chỉnh biểu đồ
plt.title('So sánh thời gian chạy giữa AMSCO và Optuna trên 4 bộ dữ liệu (Nested CV)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Bộ dữ liệu', fontsize=12, fontweight='bold')
plt.ylabel('Thời gian (giây)', fontsize=12, fontweight='bold')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.legend(title='Phương pháp', fontsize=11, title_fontsize=12)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

# Lưu biểu đồ
output_path = 'charts/time_comparison_boxplot.png'
os.makedirs('charts', exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Đã lưu biểu đồ tại: {output_path}")

# Hiển thị biểu đồ
plt.show()

# In thống kê tóm tắt
print("\n" + "="*80)
print("THỐNG KÊ THỜI GIAN CHẠY (giây)")
print("="*80)
summary = data.groupby(['dataset', 'method'])['time'].agg(['mean', 'median', 'std', 'min', 'max'])
print(summary.round(2))
print("="*80)
