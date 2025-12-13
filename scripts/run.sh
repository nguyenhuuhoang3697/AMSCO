#!/bin/bash

echo "🚀 Starting AMSCO job and Telegram watcher..."

# Khởi động bot Telegram ngay từ đầu

nohup python -u send_log_to_telegram.py > log_watcher.log 2>&1 &
# Ghi log chung cho tất cả model
# nohup python amsco_breast.py   >> output1.log 2>&1
nohup python -u amsco_adult_v2.py   >> output1.log 2>&1
# nohup python amsco_telco.py  >> output1.log 2>&1


echo "✅ All models finished training!"
