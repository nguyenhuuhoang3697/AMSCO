import os
import re
import time
import requests
from typing import Optional
from datetime import datetime


def _first_available_env(*names):
    """Return the first non-empty environment variable value from names."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


# Get Telegram credentials from environment variables
TELEGRAM_TOKEN = _first_available_env("TELEGRAM_TOKEN", "TELEGRAM_BOT_TOKEN", "BOT_TOKEN")
CHAT_ID = _first_available_env("TELEGRAM_CHAT_ID", "CHAT_ID")
LOG_FILE = os.getenv("AMSCO_LOG_FILE", "output.log")
POLL_INTERVAL = 1.0  # Check log file every second


def send_telegram_message(text: str, silent: bool = False) -> bool:
    """
    Send text message to Telegram chat.
    
    Args:
        text: Message content to send
        silent: If True, send notification silently
        
    Returns:
        True if message was sent successfully, False otherwise
    """
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("[TELEGRAM] Missing credentials: set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID environment variables.")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_notification": silent
            },
            timeout=10
        )
        if response.status_code == 200:
            return True
        else:
            print(f"[TELEGRAM] Failed to send message: {response.status_code} {response.text}")
            return False
    except Exception as exc:
        print(f"[TELEGRAM] Exception while sending message: {exc}")
        return False


def send_experiment_start(
    dataset: str,
    seed: int,
    method: str = "AMSCO",
    total_seeds: Optional[int] = None,
    additional_info: Optional[str] = None
) -> bool:
    """
    Send notification that an experiment has started.
    
    Args:
        dataset: Name of the dataset being processed
        seed: Current seed number
        method: Optimization method (default: "AMSCO")
        total_seeds: Total number of seeds (optional)
        additional_info: Additional information to include (optional)
        
    Returns:
        True if message was sent successfully
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    seed_info = f"Seed {seed}"
    if total_seeds:
        seed_info += f"/{total_seeds}"
    
    message = (
        f"🚀 <b>Experiment Started</b>\n\n"
        f"📊 Dataset: <code>{dataset}</code>\n"
        f"🎲 {seed_info}\n"
        f"⚙️ Method: <b>{method}</b>\n"
        f"🕐 Time: {current_time}"
    )
    
    if additional_info:
        message += f"\n\n💡 Info: {additional_info}"
    
    return send_telegram_message(message)


def send_experiment_complete(
    dataset: str,
    seed: int,
    method: str = "AMSCO",
    duration: Optional[float] = None,
    best_score: Optional[float] = None,
    additional_info: Optional[str] = None
) -> bool:
    """
    Send notification that an experiment has completed.
    
    Args:
        dataset: Name of the dataset being processed
        seed: Current seed number
        method: Optimization method (default: "AMSCO")
        duration: Duration in seconds (optional)
        best_score: Best score achieved (optional)
        additional_info: Additional information to include (optional)
        
    Returns:
        True if message was sent successfully
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    message = (
        f"✅ <b>Experiment Completed</b>\n\n"
        f"📊 Dataset: <code>{dataset}</code>\n"
        f"🎲 Seed: {seed}\n"
        f"⚙️ Method: <b>{method}</b>\n"
        f"🕐 Time: {current_time}"
    )
    
    if duration is not None:
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        if hours > 0:
            duration_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            duration_str = f"{minutes}m {seconds}s"
        else:
            duration_str = f"{seconds}s"
        message += f"\n⏱️ Duration: {duration_str}"
    
    if best_score is not None:
        message += f"\n🎯 Best Score: {best_score:.4f}"
    
    if additional_info:
        message += f"\n\n💡 Info: {additional_info}"
    
    return send_telegram_message(message)


def send_experiment_error(
    dataset: str,
    seed: int,
    method: str = "AMSCO",
    error_message: Optional[str] = None
) -> bool:
    """
    Send notification that an experiment encountered an error.
    
    Args:
        dataset: Name of the dataset being processed
        seed: Current seed number
        method: Optimization method (default: "AMSCO")
        error_message: Error message (optional)
        
    Returns:
        True if message was sent successfully
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    message = (
        f"❌ <b>Experiment Error</b>\n\n"
        f"📊 Dataset: <code>{dataset}</code>\n"
        f"🎲 Seed: {seed}\n"
        f"⚙️ Method: <b>{method}</b>\n"
        f"🕐 Time: {current_time}"
    )
    
    if error_message:
        # Truncate long error messages
        if len(error_message) > 200:
            error_message = error_message[:200] + "..."
        message += f"\n\n⚠️ Error:\n<code>{error_message}</code>"
    
    return send_telegram_message(message)


def send_progress_update(
    dataset: str,
    seed: int,
    progress: float,
    method: str = "AMSCO",
    current_metric: Optional[float] = None,
    iterations: Optional[int] = None
) -> bool:
    """
    Send progress update for a running experiment.
    
    Args:
        dataset: Name of the dataset being processed
        seed: Current seed number
        progress: Progress percentage (0-100)
        method: Optimization method (default: "AMSCO")
        current_metric: Current best metric value (optional)
        iterations: Number of iterations completed (optional)
        
    Returns:
        True if message was sent successfully
    """
    progress_bar_length = 10
    filled = int(progress / 100 * progress_bar_length)
    bar = "█" * filled + "░" * (progress_bar_length - filled)
    
    message = (
        f"⏳ <b>Progress Update</b>\n\n"
        f"📊 Dataset: <code>{dataset}</code>\n"
        f"🎲 Seed: {seed}\n"
        f"⚙️ Method: <b>{method}</b>\n"
        f"📈 Progress: {bar} {progress:.1f}%"
    )
    
    if iterations is not None:
        message += f"\n🔄 Iterations: {iterations}"
    
    if current_metric is not None:
        message += f"\n🎯 Current Best: {current_metric:.4f}"
    
    # Send silently for progress updates to avoid spam
    return send_telegram_message(message, silent=True)


def send_batch_summary(
    completed_experiments: list,
    total_duration: Optional[float] = None
) -> bool:
    """
    Send summary of a batch of completed experiments.
    
    Args:
        completed_experiments: List of dicts with keys: dataset, seed, method, score
        total_duration: Total duration for all experiments in seconds (optional)
        
    Returns:
        True if message was sent successfully
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    message = (
        f"📋 <b>Batch Summary</b>\n"
        f"🕐 Time: {current_time}\n"
        f"📊 Completed: {len(completed_experiments)} experiments\n\n"
    )
    
    if total_duration is not None:
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        if hours > 0:
            duration_str = f"{hours}h {minutes}m"
        else:
            duration_str = f"{minutes}m"
        message += f"⏱️ Total Duration: {duration_str}\n\n"
    
    message += "<b>Results:</b>\n"
    for exp in completed_experiments:
        dataset = exp.get('dataset', 'unknown')
        seed = exp.get('seed', '?')
        score = exp.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        message += f"  • {dataset} (seed {seed}): {score_str}\n"
    
    return send_telegram_message(message)


def parse_log_line(line: str) -> Optional[dict]:
    """
    Parse log line to extract dataset and seed information.
    
    Returns:
        dict with 'type', 'dataset', 'seed' keys, or None if not a relevant line
    """
    line = line.strip()
    
    # Match "Processing {dataset} - Seed {seed}"
    match_start = re.match(r'Processing\s+(\w+)\s+-\s+Seed\s+(\d+)', line)
    if match_start:
        return {
            'type': 'start',
            'dataset': match_start.group(1),
            'seed': int(match_start.group(2))
        }
    
    # Match "Completed {dataset} - Seed {seed}"
    match_complete = re.match(r'Completed\s+(\w+)\s+-\s+Seed\s+(\d+)', line)
    if match_complete:
        return {
            'type': 'complete',
            'dataset': match_complete.group(1),
            'seed': int(match_complete.group(2))
        }
    
    # Match progress indicators
    if re.search(r'\[(\d+)/(\d+)\]\s+Running\s+(\w+)', line):
        match_progress = re.search(r'\[(\d+)/(\d+)\]\s+Running\s+(\w+)', line)
        return {
            'type': 'progress',
            'step': int(match_progress.group(1)),
            'total': int(match_progress.group(2)),
            'task': match_progress.group(3)
        }
    
    return None


def tail_log_and_notify():
    """
    Tail log file and send notifications when dataset/seed changes are detected.
    """
    print(f"[INFO] Starting log watcher...")
    print(f"[INFO] TELEGRAM_TOKEN set: {bool(TELEGRAM_TOKEN)}")
    print(f"[INFO] CHAT_ID set: {bool(CHAT_ID)}")
    print(f"[INFO] Watching log file: {LOG_FILE}")
    print(f"[INFO] Poll interval: {POLL_INTERVAL}s")
    print("-" * 60)
    
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("[WARNING] Telegram credentials not set. Notifications will not be sent.")
        print("[WARNING] Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID environment variables.")
    
    current_dataset = None
    current_seed = None
    start_time = None
    
    try:
        with open(LOG_FILE, "r") as handle:
            # Read existing lines first to get to the end
            existing_lines = handle.readlines()
            
            # Process recent lines to get current state
            for line in existing_lines[-20:]:
                parsed = parse_log_line(line)
                if parsed and parsed['type'] == 'start':
                    current_dataset = parsed['dataset']
                    current_seed = parsed['seed']
            
            if current_dataset and current_seed is not None:
                print(f"[INFO] Current state: {current_dataset} - Seed {current_seed}")
            
            print("[INFO] Now monitoring for new changes...")
            print("-" * 60)
            
            # Now follow new lines
            while True:
                line = handle.readline()
                if line:
                    line = line.strip()
                    if line:  # Only process non-empty lines
                        print(f"[LOG] {line}")
                        
                        parsed = parse_log_line(line)
                        if parsed:
                            if parsed['type'] == 'start':
                                dataset = parsed['dataset']
                                seed = parsed['seed']
                                
                                print(f"[NOTIFY] 🚀 Starting: {dataset} - Seed {seed}")
                                send_experiment_start(
                                    dataset=dataset,
                                    seed=seed,
                                    method="AMSCO"
                                )
                                
                                current_dataset = dataset
                                current_seed = seed
                                start_time = time.time()
                            
                            elif parsed['type'] == 'complete':
                                dataset = parsed['dataset']
                                seed = parsed['seed']
                                duration = None
                                if start_time:
                                    duration = time.time() - start_time
                                
                                print(f"[NOTIFY] ✅ Completed: {dataset} - Seed {seed}")
                                send_experiment_complete(
                                    dataset=dataset,
                                    seed=seed,
                                    method="AMSCO",
                                    duration=duration
                                )
                                
                                start_time = None
                            
                            elif parsed['type'] == 'progress':
                                step = parsed['step']
                                total = parsed['total']
                                task = parsed['task']
                                progress = (step / total) * 100
                                
                                print(f"[PROGRESS] {task}: {step}/{total} ({progress:.0f}%)")
                                # Only send progress for major steps to avoid spam
                                if step == 1 or step == total:
                                    send_progress_update(
                                        dataset=current_dataset or "unknown",
                                        seed=current_seed if current_seed is not None else 0,
                                        progress=progress,
                                        method="AMSCO",
                                        iterations=step
                                    )
                else:
                    time.sleep(POLL_INTERVAL)
    
    except FileNotFoundError:
        print(f"[ERROR] Log file '{LOG_FILE}' not found.")
        print(f"[INFO] Waiting for file to be created...")
        while not os.path.exists(LOG_FILE):
            time.sleep(POLL_INTERVAL)
        print(f"[INFO] Log file found. Starting monitoring...")
        tail_log_and_notify()  # Retry
    
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C)")
        if current_dataset and current_seed is not None:
            send_telegram_message(
                f"⏸️ <b>Log Watcher Stopped</b>\n\n"
                f"Last state: {current_dataset} - Seed {current_seed}"
            )
    
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


# Example usage
if __name__ == "__main__":
    # Start tailing and notifying
    tail_log_and_notify()
