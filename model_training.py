import pandas as pd
from f1pred.paths import PROCESSED_DATA_PATH
from f1pred.training import train_and_eval_multiclass
import numpy as np
import json


def main() -> None:
    dataset = pd.read_csv(PROCESSED_DATA_PATH)
    import xgboost as xgb

    with open('xgb_best_params.json', 'r') as f:
        params = json.load(f)

    n_classes = len(np.unique(dataset["session_5_final_position"]))
    classifier = xgb.XGBClassifier(**params)
    res = train_and_eval_multiclass(dataset, classifier=classifier)
    print("Model pipeline saved.")
    if res.get("mean_accuracy") is not None:
        print(f"Average accuracy across folds: {res['mean_accuracy']:.4f}")
    if res.get("mean_auc") is not None:
        print(f"Average AUC across folds: {res['mean_auc']:.4f}")


if __name__ == "__main__":
    main()