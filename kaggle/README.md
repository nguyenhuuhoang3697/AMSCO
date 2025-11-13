# Kaggle Credentials (Not in Git)

Place your Kaggle API token here as `kaggle.json` but do NOT commit it.

How to set up:
1. Go to https://www.kaggle.com/settings and click "Create New API Token".
2. Save the downloaded `kaggle.json` to this folder (or to `~/.kaggle/kaggle.json`).
3. Set secure permissions:
   chmod 600 ~/.kaggle/kaggle.json

The project scripts will prefer `~/.kaggle/kaggle.json`. If missing, they can copy from this folder into place automatically. 