from __future__ import annotations

import argparse
import sys

import pandas as pd

from .paths import (
    ensure_directories,
    PROCESSED_DATA_PATH,
)
from .logging_utils import configure_logging
from .preprocess import clean_training_dataframe
from .training import train_and_eval_multiclass


def cmd_clean(args: argparse.Namespace) -> int:
    ensure_directories()
    if args.input and args.input != "auto":
        df = pd.read_csv(args.input)
    else:
        # auto: concat session_data_*.csv
        from .paths import list_session_year_files

        parts = []
        for path in list_session_year_files():
            try:
                parts.append(pd.read_csv(path))
            except Exception:
                pass
        if not parts:
            print("No session_data_*.csv files found.")
            return 1
        df = pd.concat(parts)

    cleaned = clean_training_dataframe(df, use_existing_dicts=args.use_existing)
    cleaned.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Wrote cleaned data to {PROCESSED_DATA_PATH}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    ensure_directories()
    if args.data:
        dataset = pd.read_csv(args.data)
    else:
        dataset = pd.read_csv(PROCESSED_DATA_PATH)
    res = train_and_eval_multiclass(dataset)
    print(
        f"Saved pipeline to {res['pipeline_path']}; label encoder to {res['label_encoder_path']} | "
        f"mean_acc={res['mean_accuracy']}, mean_auc={res['mean_auc']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="f1pred")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_clean = sub.add_parser("clean", help="Clean raw session data into training dataset")
    p_clean.add_argument("--input", default="auto", help="Input CSV; default auto-concat session_data_*.csv")
    p_clean.add_argument("--use-existing", action="store_true", help="Use existing encoding dicts if available")
    p_clean.set_defaults(func=cmd_clean)

    p_train = sub.add_parser("train", help="Train model on processed data")
    p_train.add_argument("--data", default=None, help="Processed data CSV; defaults to data/processed_data.csv")
    p_train.set_defaults(func=cmd_train)

    return p


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())


