import os
import textwrap
import time
from typing import Iterable

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


def tail_and_send() -> None:
    """Tail log file and send latest chunks when batch size is reached."""
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
