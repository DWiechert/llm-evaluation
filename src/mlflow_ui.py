#!/usr/bin/env python3
"""
mlflow_ui.py — Launch the MLflow UI pointed at this project's results/mlflow.db.

Plain `mlflow ui` defaults to a tracking DB at the repo root, not the
results/-scoped one mlflow_eval.py writes to. This wraps that with the
correct --backend-store-uri so there's nothing to remember.

Usage:
    uv run src/mlflow_ui.py [extra mlflow ui args, e.g. --port 5001]
"""

import os
import sys
from pathlib import Path


def main():
    db_path = Path(__file__).parent.parent / "results" / "mlflow.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    args = ["mlflow", "ui", "--backend-store-uri", f"sqlite:///{db_path}", *sys.argv[1:]]
    os.execvp(args[0], args)


if __name__ == "__main__":
    main()
