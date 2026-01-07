import os
import textwrap
import time
import psutil
import threading
from typing import Iterable
from datetime import datetime

import requests
from PIL import Image, ImageDraw, ImageFont


def _first_available_env(*names: Iterable[str]):
    """Return the first non-empty environment variable value from names."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


TELEGRAM_TOKEN = _first_available_env("TELEGRAM_TOKEN", "TELEGRAM_BOT_TOKEN", "BOT_TOKEN")
CHAT_ID = _first_available_env("TELEGRAM_CHAT_ID", "CHAT_ID")
LOG_FILE = os.getenv("AMSCO_LOG_FILE", "output.log")
BATCH_SIZE = 25
POLL_INTERVAL = 1.0
SYSTEM_STATUS_INTERVAL = 5 * 60  # 5 phút = 300 giây
FONT = ImageFont.load_default()


def log_to_image(text: str, output_path: str = "log_preview.png") -> str:
    """Render plain-text log snippet into a dark themed PNG."""
    wrapped_lines = []
    for line in text.splitlines():
        wrapped = textwrap.wrap(line, width=90)
        if not wrapped:
            wrapped_lines.append(" ")
        else:
            wrapped_lines.extend(wrapped)

    line_height = 16
    padding = 20
    width = 900
    height = max(padding * 2 + line_height * len(wrapped_lines), line_height + padding * 2)

    image = Image.new("RGB", (width, height), color=(20, 20, 20))
    draw = ImageDraw.Draw(image)

    y = padding
    for wrapped_line in wrapped_lines:
        draw.text((padding, y), wrapped_line, fill=(200, 200, 200), font=FONT)
        y += line_height

    image.save(output_path)
    return output_path


def send_photo(path: str) -> None:
    """Send image at path to Telegram chat if credentials are available."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("[TELEGRAM] Missing env vars: set TELEGRAM_TOKEN/TELEGRAM_BOT_TOKEN/BOT_TOKEN and TELEGRAM_CHAT_ID/CHAT_ID.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(path, "rb") as photo_file:
        try:
            response = requests.post(url, data={"chat_id": CHAT_ID}, files={"photo": photo_file}, timeout=10)
            if response.status_code != 200:
                print(f"[TELEGRAM] sendPhoto failed: {response.status_code} {response.text}")
        except Exception as exc:
            print(f"[TELEGRAM] Exception while sending photo: {exc}")


def send_message(text: str) -> None:
    """Send text message to Telegram chat."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(
            url,
            data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
        if response.status_code != 200:
            print(f"[TELEGRAM] sendMessage failed: {response.status_code} {response.text}")
    except Exception as exc:
        print(f"[TELEGRAM] Exception while sending message: {exc}")


def get_system_stats() -> str:
    """Get current CPU, RAM, Disk and uptime statistics."""
    try:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # RAM
        memory = psutil.virtual_memory()
        ram_total_gb = memory.total / (1024**3)
        ram_used_gb = memory.used / (1024**3)
        ram_available_gb = memory.available / (1024**3)
        ram_percent = memory.percent
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / (1024**3)
        disk_used_gb = disk.used / (1024**3)
        disk_free_gb = disk.free / (1024**3)
        disk_percent = disk.percent
        
        # Uptime
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        uptime_hours = uptime_seconds / 3600
        uptime_days = uptime_hours / 24
        
        # Current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        stats = (
            f"📊 <b>System Status Report</b>\n"
            f"🕐 Time: {current_time}\n\n"
            f"🖥️ <b>CPU</b>\n"
            f"  • Usage: {cpu_percent:.1f}% ({cpu_count} cores)\n\n"
            f"💾 <b>RAM</b>\n"
            f"  • Used: {ram_used_gb:.1f}/{ram_total_gb:.1f} GB ({ram_percent:.1f}%)\n"
            f"  • Available: {ram_available_gb:.1f} GB\n\n"
            f"💿 <b>Disk (/)</b>\n"
            f"  • Used: {disk_used_gb:.1f}/{disk_total_gb:.1f} GB ({disk_percent:.1f}%)\n"
            f"  • Free: {disk_free_gb:.1f} GB\n\n"
            f"⏱️ <b>Uptime</b>\n"
            f"  • {uptime_days:.1f} days ({uptime_hours:.1f} hours)"
        )
        return stats
    except Exception as e:
        return f"⚠️ Could not fetch system stats: {e}"


def send_system_status_periodically():
    """Send system status report every SYSTEM_STATUS_INTERVAL seconds."""
    while True:
        try:
            time.sleep(SYSTEM_STATUS_INTERVAL)
            stats = get_system_stats()
            send_message(stats)
            print(f"[TELEGRAM] Sent periodic system status report.")
        except Exception as e:
            print(f"[TELEGRAM] Error in periodic status thread: {e}")


def tail_and_send() -> None:
    """Tail log file and send latest chunks when batch size is reached."""
    # Start background thread for periodic system status
    status_thread = threading.Thread(target=send_system_status_periodically, daemon=True)
    status_thread.start()
    print(f"[TELEGRAM] Started periodic system status reporting (every {SYSTEM_STATUS_INTERVAL/60:.0f} minutes).")
    
    try:
        with open(LOG_FILE, "r") as handle:
            # Đọc tất cả dòng có sẵn trước, gửi ảnh đầu tiên ngay lập tức
            all_lines = handle.readlines()
            if all_lines:
                # Lấy BATCH_SIZE dòng cuối để gen ảnh ban đầu
                initial_chunk = [line.rstrip() for line in all_lines[-BATCH_SIZE:]]
                img_path = log_to_image("\n".join(initial_chunk))
                send_photo(img_path)
                print(f"[TELEGRAM] Sent initial snapshot ({len(initial_chunk)} lines).")
                # Gửi thông tin hệ thống sau ảnh đầu tiên
                send_message(get_system_stats())
            
            # Sau đó theo dõi các dòng mới
            buffer = []
            while True:
                line = handle.readline()
                if line:
                    buffer.append(line.rstrip())
                    if len(buffer) > BATCH_SIZE:
                        buffer = buffer[-BATCH_SIZE:]
                    if len(buffer) == BATCH_SIZE:
                        img_path = log_to_image("\n".join(buffer))
                        send_photo(img_path)
                        # Không gửi system status sau mỗi ảnh nữa, để thread định kỳ xử lý
                        buffer = []
                else:
                    time.sleep(POLL_INTERVAL)
    except FileNotFoundError:
        print(f"[TELEGRAM] Log file '{LOG_FILE}' not found.")
    except KeyboardInterrupt:
        print("[TELEGRAM] Stopped by user.")


if __name__ == "__main__":
    print("Watching log…")
    print(f"TELEGRAM_TOKEN set: {bool(TELEGRAM_TOKEN)}, CHAT_ID set: {bool(CHAT_ID)}")
    print(f"Following log file: {LOG_FILE}")
    tail_and_send()
