import pandas as pd
from f1pred.paths import PROCESSED_DATA_PATH
from f1pred.training import train_and_eval_multiclass


def main() -> None:
    dataset = pd.read_csv(PROCESSED_DATA_PATH)
    res = train_and_eval_multiclass(dataset)
    print("Model pipeline saved.")
    if res.get("mean_accuracy") is not None:
        print(f"Average accuracy across folds: {res['mean_accuracy']:.4f}")
    if res.get("mean_auc") is not None:
        print(f"Average AUC across folds: {res['mean_auc']:.4f}")


if __name__ == "__main__":
    main()