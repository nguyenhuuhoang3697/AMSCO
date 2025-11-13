#!/usr/bin/env python3
from __future__ import annotations
import sys
import shutil
from pathlib import Path
import pandas as pd

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def read_adult_file(path: Path, is_test: bool = False) -> pd.DataFrame:
    # adult.test has one non-data header-ish line beginning with '|' and labels ending with '.'
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        comment="|",
    )
    # drop completely empty lines if any
    df = df.dropna(how="all")

    # strip whitespace on string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    if is_test and "income" in df.columns:
        # remove trailing '.' in income labels from adult.test
        df["income"] = df["income"].str.replace(".", "", regex=False).str.strip()
    return df


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    datasets_dir = root / "datasets"
    adult_dir = datasets_dir / "adult"
    data_path = adult_dir / "adult.data"
    test_path = adult_dir / "adult.test"
    out_path = datasets_dir / "adult.csv"

    if not data_path.exists() or not test_path.exists():
        print("adult.data or adult.test not found. Run scripts/download_datasets.sh first.", file=sys.stderr)
        return 1

    df_train = read_adult_file(data_path, is_test=False)
    df_test = read_adult_file(test_path, is_test=True)

    df_all = pd.concat([df_train, df_test], ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_path, index=False)
    print(f"Wrote merged CSV: {out_path} (rows={len(df_all)})")

    # Remove original directory to keep only datasets/adult.csv
    try:
        shutil.rmtree(adult_dir, ignore_errors=True)
    except Exception as e:
        print(f"Warning: could not remove directory {adult_dir}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
