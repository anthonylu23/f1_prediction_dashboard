import pandas as pd
import numpy as np

from f1pred.paths import LATEST_RACE_DATA_PATH, ENCODINGS_DIR
from f1pred.predict_utils import load_artifacts, evaluate_position_prop_with_classes


def main() -> None:
    model, le = load_artifacts()
    df = pd.read_csv(LATEST_RACE_DATA_PATH)
    final_position = df.get("session_5_final_position")
    start_position = df.get("session_4_final_position")
    input_features = df.drop(columns=[c for c in ["session_5_final_position"] if c in df.columns])
    predictions_idx = model.predict(input_features)
    predictions = le.inverse_transform(predictions_idx)
    predict_probabilities = model.predict_proba(input_features)
    drivers = input_features["driver"]
    drivers_dict = pd.read_csv(ENCODINGS_DIR / 'driver_names_dict.csv')
    driver_names = [
        drivers_dict[drivers_dict["encoded_value"] == driver]["driver_name"].values[0].replace(" ", "")
        for driver in drivers
    ]
    driver_mapping = {driver: i for i, driver in enumerate(drivers)}
    line = 6.5
    prob_under_list = []
    prob_over_list = []
    classes = le.classes_
    for driver in drivers:
        idx = driver_mapping[driver]
        pu, po = evaluate_position_prop_with_classes(predict_probabilities[idx], classes, line)
        prob_under_list.append(pu)
        prob_over_list.append(po)
    over_under_results = pd.DataFrame(
        {
            "driver": driver_names,
            "line": line,
            "prob_under": prob_under_list,
            "prob_over": prob_over_list,
        }
    )
    print(over_under_results.sort_values(by="prob_over", ascending=False))

    results = pd.DataFrame(columns=["driver", "predicted_position", "actual_position"])
    results["driver"] = driver_names
    results["predicted_position"] = predictions
    if final_position is not None:
        results["actual_position"] = final_position
    if start_position is not None:
        results["start_position"] = start_position
    results = results.sort_values(by="predicted_position", ascending=True)
    print(results)


if __name__ == "__main__":
    main()