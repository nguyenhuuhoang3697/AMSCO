#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/datasets"
mkdir -p "$DATA_DIR"

# 1) Check Kaggle token (prefer ~/.kaggle/kaggle.json; fallback to repo's kaggle/kaggle.json)
if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  if [[ -f "$ROOT_DIR/kaggle/kaggle.json" ]]; then
    echo "==> No ~/.kaggle/kaggle.json found; using repo token at kaggle/kaggle.json"
    mkdir -p "$HOME/.kaggle"
    install -m 600 "$ROOT_DIR/kaggle/kaggle.json" "$HOME/.kaggle/kaggle.json"
  else
    cat >&2 <<'EOF'
[ERROR] Kaggle API token not found at ~/.kaggle/kaggle.json
- Get it from https://www.kaggle.com/settings ("Create New API Token")
- Save to ~/.kaggle/kaggle.json and set permissions:
    mkdir -p ~/.kaggle
    chmod 600 ~/.kaggle/kaggle.json
Then re-run this script.
EOF
    exit 1
  fi
fi
chmod 600 "$HOME/.kaggle/kaggle.json" || true

# 2) Ensure kaggle CLI is installed
if ! command -v kaggle >/dev/null 2>&1; then
  echo "==> Installing kaggle CLI (pip)"
  PIP_BIN="pip3"
  if [[ -x "$ROOT_DIR/venv/bin/pip" ]]; then
    PIP_BIN="$ROOT_DIR/venv/bin/pip"
  elif command -v pip >/dev/null 2>&1; then
    PIP_BIN="pip"
  fi
  "$PIP_BIN" install --quiet --upgrade kaggle
fi

# 3) Ensure unzip is available
if ! command -v unzip >/dev/null 2>&1; then
  echo "==> Installing unzip (requires sudo)"
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -y >/dev/null
    sudo apt-get install -y unzip >/dev/null
  else
    echo "[WARN] 'unzip' not found and sudo unavailable. Please install unzip and re-run." >&2
    exit 1
  fi
fi

set -x
# 4) Telco Customer Churn
telco_zip="$DATA_DIR/telco-customer-churn.zip"
kaggle datasets download -d blastchar/telco-customer-churn -p "$DATA_DIR" -f telco-customer-churn.zip || kaggle datasets download -d blastchar/telco-customer-churn -p "$DATA_DIR"
unzip -o "$telco_zip" -d "$DATA_DIR"
# Source file in zip: WA_Fn-UseC_-Telco-Customer-Churn.csv
if [[ -f "$DATA_DIR/WA_Fn-UseC_-Telco-Customer-Churn.csv" ]]; then
  mv -f "$DATA_DIR/WA_Fn-UseC_-Telco-Customer-Churn.csv" "$DATA_DIR/telco.csv"
fi
rm -f "$telco_zip"

# 5) Credit Card Fraud
credit_zip="$DATA_DIR/creditcardfraud.zip"
kaggle datasets download -d mlg-ulb/creditcardfraud -p "$DATA_DIR" -f creditcardfraud.zip || kaggle datasets download -d mlg-ulb/creditcardfraud -p "$DATA_DIR"
unzip -o "$credit_zip" -d "$DATA_DIR"
# The extracted file is creditcard.csv (keep the same name)
# Ensure file is present
if [[ ! -f "$DATA_DIR/creditcard.csv" ]]; then
  echo "[WARN] Expected $DATA_DIR/creditcard.csv not found after unzip" >&2
fi
rm -f "$credit_zip"
set +x

echo "==> Finished. Produced:"
echo "    $DATA_DIR/telco.csv"
echo "    $DATA_DIR/creditcard.csv"