from __future__ import annotations

import hashlib
import math
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from .paths import MODEL_PIPELINE_PATH, LABEL_ENCODER_PATH


def load_artifacts():
    model = joblib.load(MODEL_PIPELINE_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, label_encoder


def hash_dataframe(df: pd.DataFrame) -> str:
    try:
        return hashlib.sha1(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
    except Exception:
        return hashlib.sha1(df.to_csv(index=True).encode("utf-8")).hexdigest()


def evaluate_position_prop_with_classes(proba_row: np.ndarray, classes: np.ndarray, line_half: float) -> tuple[float, float]:
    cut = math.floor(line_half)
    mask_under = classes <= cut
    p_under = np.sum(proba_row[mask_under])
    p_over = 1 - p_under
    odds_under = float(1 / p_under)
    odds_over = float(1 / p_over)
    return odds_under, odds_over


def compute_prop_table(
    proba_matrix: np.ndarray,
    classes: np.ndarray,
    line_half: float,
    driver_ids: pd.Series,
    id_to_name: dict,
) -> pd.DataFrame:
    odds_unders = []
    odds_overs = []
    for i in range(proba_matrix.shape[0]):
        pu, po = evaluate_position_prop_with_classes(proba_matrix[i], classes, line_half)
        odds_unders.append(pu)
        odds_overs.append(po)

    df = pd.DataFrame(
        {
            "driver_id": driver_ids.values,
            "driver": driver_ids.map(lambda x: id_to_name.get(int(x), str(x))).astype(str),
            "line": [line_half] * len(driver_ids),
            "odds_under": np.round(odds_unders, 4),
            "odds_over": np.round(odds_overs, 4),
        }
    )
    return df


