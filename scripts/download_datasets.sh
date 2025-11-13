#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/datasets"
mkdir -p "$DATA_DIR"

echo "==> Downloading Adult Income dataset from UCI"
ADULT_DIR="$DATA_DIR/adult"
mkdir -p "$ADULT_DIR"
# UCI Adult dataset files
BASE_UCI="https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
curl -fsSL "$BASE_UCI/adult.data" -o "$ADULT_DIR/adult.data" || {
  echo "Failed to download adult.data from UCI. Check your network and try again." >&2
  exit 1
}
curl -fsSL "$BASE_UCI/adult.test" -o "$ADULT_DIR/adult.test" || {
  echo "Failed to download adult.test from UCI. Check your network and try again." >&2
  exit 1
}
# Optional: names file for reference
curl -fsSL "$BASE_UCI/adult.names" -o "$ADULT_DIR/adult.names" || true

echo "==> Adult dataset saved under $ADULT_DIR"

echo "==> Merging Adult train/test into a single CSV with headers (datasets/adult.csv)"
PY_BIN="python3"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python"
fi
"$PY_BIN" "$ROOT_DIR/scripts/prepare_adult.py"

echo "==> Telco and Credit Card datasets require Kaggle access. Skipping automatic download."
echo "   Follow README instructions to use kaggle CLI to fetch:" 
echo "   - Telco Customer Churn (blastchar/telco-customer-churn)"
echo "   - Credit Card Fraud (mlg-ulb/creditcardfraud)"

echo "Done."