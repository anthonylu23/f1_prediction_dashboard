from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

from .paths import MODEL_PIPELINE_PATH, LABEL_ENCODER_PATH


@dataclass
class FeatureSets:
    numerical: list[str]
    categorical: list[str]


def infer_feature_sets(df: pd.DataFrame, target_col: str) -> FeatureSets:
    features = [c for c in df.columns if c != target_col]
    categorical_features = [
        "year",
        "driver",
        "team_name",
        "team_lineage",
        "circuit_key",
    ]
    for col in features:
        for token in ["type", "rainfall", "final_position", "missing", "missed"]:
            if token in col:
                categorical_features.append(col)
                break
    categorical_features = sorted(set(categorical_features))
    numerical_features = [c for c in features if c not in categorical_features]
    return FeatureSets(numerical=numerical_features, categorical=categorical_features)


def build_preprocessor(feature_sets: FeatureSets) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), feature_sets.numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feature_sets.categorical),
        ]
    )


def sliding_window_splits(
    data: pd.DataFrame,
    *,
    group_col: str = "year",
    window_size: int = 2,
    include_final_train: bool = True,
    final_train_strategy: str = "window",
    return_weights: bool = True,
    weight_decay: float = 0.5,
) -> Iterator[tuple]:
    groups = np.sort(data[group_col].unique())
    if len(groups) < window_size + 1:
        raise ValueError(
            f"Cannot create splits with window_size={window_size} and only {len(groups)} groups."
        )
    for i in range(window_size, len(groups)):
        train_groups = groups[i - window_size : i]
        test_group = groups[i]
        train_idx = data[data[group_col].isin(train_groups)].index
        test_idx = data[data[group_col] == test_group].index
        print(f"training on year {train_groups} and testing on year {test_group}")
        if return_weights and weight_decay is not None and weight_decay != 1.0:
            group_weights = {g: weight_decay ** (len(train_groups) - 1 - j) for j, g in enumerate(train_groups)}
            train_weights = data.loc[train_idx, group_col].map(group_weights).to_numpy()
            yield list(train_idx), list(test_idx), train_weights
        else:
            yield list(train_idx), list(test_idx)

    if include_final_train:
        if final_train_strategy == "window":
            final_groups = groups[-window_size:]
        else:
            final_groups = groups
        final_idx = data[data[group_col].isin(final_groups)].index
        print(f"training on year {final_groups}")
        if return_weights and weight_decay is not None and weight_decay != 1.0:
            group_weights = {g: weight_decay ** (len(final_groups) - 1 - j) for j, g in enumerate(final_groups)}
            train_weights = data.loc[final_idx, group_col].map(group_weights).to_numpy()
            
            yield list(final_idx), [], train_weights
        else:
            yield list(final_idx), []


def train_and_eval_multiclass(
    dataset: pd.DataFrame,
    *,
    target_col: str = "session_5_final_position",
) -> dict:
    features = infer_feature_sets(dataset, target_col)
    preprocessor = build_preprocessor(features)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(dataset[target_col])
    X = dataset[[c for c in dataset.columns if c != target_col]]

    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    n_classes = len(np.unique(y))
    classifier = xgb.XGBClassifier(
        objective="multi:softprob",
        n_estimators=100,
        random_state=42,
        num_class=n_classes,
    )
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    scores: list[float] = []
    auc_scores: list[float] = []

    for split in sliding_window_splits(dataset):
        if len(split) == 3:
            train_idx, test_idx, recency_w = split
        else:
            train_idx, test_idx = split
            recency_w = None
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]

        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(zip(np.unique(y_train), class_weights))
        class_w = np.array([class_weights_dict[i] for i in y_train])
        sample_w = class_w * (recency_w if recency_w is not None else 1.0)
        model_pipeline.fit(X_train, y_train, classifier__sample_weight=sample_w)

        if len(test_idx) > 0:
            X_test = X.iloc[test_idx]
            y_test = y[test_idx]
            scores.append(model_pipeline.score(X_test, y_test))
            proba = model_pipeline.predict_proba(X_test)
            pred = model_pipeline.predict(X_test)
            auc_score = roc_auc_score(y_test, proba, multi_class="ovr")
            accuracy = accuracy_score(y_test, pred)
            f1score = f1_score(y_test, pred, average="macro")
            print(f"AUC score: {auc_score}")
            print(f"Accuracy score: {accuracy}")
            print(f"F1 score: {f1score}")
            auc_scores.append(auc_score)

    joblib.dump(model_pipeline, MODEL_PIPELINE_PATH)
    return {
        "pipeline_path": str(MODEL_PIPELINE_PATH),
        "label_encoder_path": str(LABEL_ENCODER_PATH),
        "mean_accuracy": float(np.mean(scores)) if scores else None,
        "mean_auc": float(np.mean(auc_scores)) if auc_scores else None,
    }


