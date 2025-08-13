from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .paths import ENCODINGS_DIR


def compute_team_lineage(team_name: str) -> str:
    lineage_groups = {
        "Mclaren": {"Mclaren"},
        "Red Bull Racing": {"Red Bull Racing"},
        "Ferrari": {"Ferrari"},
        "Haas F1 Team": {"Haas F1 Team"},
        "Sauber": {"Sauber", "Alfa Romeo", "Alfa Romeo Racing", "Kick Sauber"},
        "Toro Rosso": {"Toro Rosso", "AlphaTauri", "Racing Bulls", "RB"},
        "Aston Martin": {"Force India", "Racing Point", "Aston Martin"},
        "Williams": {"Williams"},
        "Renault": {"Renault", "Alpine"},
        "Mercedes": {"Mercedes"},
    }
    for lineage, aliases in lineage_groups.items():
        if team_name in aliases:
            return lineage
    return team_name


def add_session_columns_subset(df: pd.DataFrame, columns: list[str]) -> Tuple[pd.DataFrame, list[str]]:
    cols: list[str] = []
    for session in range(1, 6):
        cols.extend([c for c in columns if c.startswith(f"session_{session}")])
    return df[cols], cols


def _handle_missing_laps(df: pd.DataFrame, col: str, session: int) -> pd.DataFrame:
    if "avg_lap_time" in col:
        df[f"session_{session}_missing_avg_lap_time"] = df[col].isna() & (df[f"session_{session}_type"] != "Qualifying")
        df.loc[df[f"session_{session}_missing_avg_lap_time"], col] = df.loc[df[f"session_{session}_missing_avg_lap_time"], col].fillna(-2)
        mask = ~df[f"session_{session}_missing_avg_lap_time"]
        df.loc[mask, col] = df.loc[mask, col].fillna(-1)
    elif "best_lap" in col:
        mask = (df[f"session_{session}_type"] != "Qualifying") & df[col].isna()
        df.loc[mask, col] = df.loc[mask, col].fillna(-1)
        qmask = (df[col].isna()) & (df[f"session_{session}_type"] == "Qualifying")
        if "q1" in col:
            df[f"session_{session}_missed_qualifying"] = qmask
            df.loc[qmask, col] = df.loc[qmask, col].fillna(-2)
        else:
            df.loc[qmask, col] = df.loc[qmask, col].fillna(-3)
    elif "final_position" in col:
        df[f"session_{session}_missing_final_position"] = df[col].isna() & ((df[f"session_{session}_type"] != "Race") | (df[f"session_{session}_type"] != "Qualifying"))
        df.loc[df[f"session_{session}_missing_final_position"], col] = (
            df.loc[df[f"session_{session}_missing_final_position"], col].fillna(-1)
        )
        qmask = (df[col].isna()) & (df[f"session_{session}_type"] == "Qualifying")
        if "q1" in col:
            df[f"session_{session}_missed_qualifying"] = qmask
            df.loc[qmask, col] = df.loc[qmask, col].fillna(-2)
        else:
            df.loc[qmask, col] = df.loc[qmask, col].fillna(-3)
    elif "gap_to_fastest" in col:
        df[f"session_{session}_missing_fastest_lap"] = df[col].isna() & (df[f"session_{session}_type"] != "Qualifying")
        df.loc[df[f"session_{session}_missing_fastest_lap"], col] = (
            df.loc[df[f"session_{session}_missing_fastest_lap"], col].fillna(-1)
        )
        qmask = (df[col].isna()) & (df[f"session_{session}_type"] == "Qualifying")
        if "q1" in col:
            df[f"session_{session}_missed_qualifying"] = qmask
            df.loc[qmask, col] = df.loc[qmask, col].fillna(-2)
        else:
            df.loc[qmask, col] = df.loc[qmask, col].fillna(-3)
    return df


def impute_session_data(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for session in range(1, 6):
        for col in columns:
            if col.startswith(f"session_{session}"):
                df = _handle_missing_laps(df, col, session)
    return df


def impute_session_type(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for session in range(1, 6):
        for col in columns:
            if "type" in col:
                df[col] = df[col].fillna(-1)
    return df


def _handle_weather_col(df: pd.DataFrame, col: str, session: int) -> pd.DataFrame:
    session_type = f"session_{session}_type"
    mask_missing_session = df[session_type] == -1
    df.loc[mask_missing_session, col] = -1
    is_same_circuit = df["circuit_key"] == df["circuit_key"].shift(1)
    ffill_mask = df[col].isnull() & is_same_circuit
    df.loc[ffill_mask, col] = df[col].ffill()[ffill_mask]
    df[col].fillna(method="bfill", inplace=True)
    return df


def impute_environment_data(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    weather_keys = [
        "air_temp",
        "humidity",
        "pressure",
        "rainfall",
        "track_temp",
        "wind_speed",
        "wind_direction",
    ]
    for session in range(1, 6):
        for col in columns:
            for key in weather_keys:
                if key in col:
                    _handle_weather_col(df, col, session)
    return df


def _save_mapping(mapping: dict[str, int], file_path: Path, columns: tuple[str, str]) -> None:
    df = pd.DataFrame(list(mapping.items()), columns=list(columns))
    df.to_csv(file_path, index=False)


def encode_categorical_data(df: pd.DataFrame, columns: list[str], *, use_existing_dicts: bool = False) -> pd.DataFrame:
    cols = df.columns.tolist()
    session_types: list[str] = []
    for col in cols:
        if "type" in col:
            session_types.extend(df[col].unique().tolist())
    team_names = df["team_name"].unique().tolist()
    team_lineages = df["team_lineage"].unique().tolist()

    session_types_dict = {val: i for i, val in enumerate(session_types)}
    if not use_existing_dicts:
        _save_mapping(session_types_dict, ENCODINGS_DIR / "session_types_dict.csv", ("session_type", "encoded_value"))
        team_names_dict = {team: i for i, team in enumerate(team_names)}
        _save_mapping(team_names_dict, ENCODINGS_DIR / "team_names_dict.csv", ("team_name", "encoded_value"))
        team_lineages_dict = {tl: i for i, tl in enumerate(team_lineages)}
        _save_mapping(team_lineages_dict, ENCODINGS_DIR / "team_lineages_dict.csv", ("team_lineage", "encoded_value"))

    def session_type_encoder(val: str) -> int:
        if use_existing_dicts:
            session_dict = pd.read_csv(ENCODINGS_DIR / "session_types_dict.csv")
            session_dict = dict(zip(session_dict["session_type"], session_dict["encoded_value"]))
            return int(session_dict[val])
        return int(session_types_dict[val])

    def team_name_encoder(team: str) -> int:
        if use_existing_dicts:
            team_dict = pd.read_csv(ENCODINGS_DIR / "team_names_dict.csv")
            team_dict = dict(zip(team_dict["team_name"], team_dict["encoded_value"]))
            return int(team_dict[team])
        return int(team_names_dict[team])

    def team_lineage_encoder(tl: str) -> int:
        if use_existing_dicts:
            lin_dict = pd.read_csv(ENCODINGS_DIR / "team_lineages_dict.csv")
            lin_dict = dict(zip(lin_dict["team_lineage"], lin_dict["encoded_value"]))
            return int(lin_dict[tl])
        return int(team_lineages_dict[tl])

    df["team_name"] = df["team_name"].map(team_name_encoder)
    df["team_lineage"] = df["team_lineage"].map(team_lineage_encoder)

    for col in cols:
        if "type" in col:
            df[col] = df[col].map(session_type_encoder)
        if "rainfall" in col:
            df[col] = df[col].astype(bool)

    return df


def encode_driver_names(df: pd.DataFrame, *, use_existing_dicts: bool = False) -> pd.DataFrame:
    driver_names = df["driver_name"].unique().tolist()
    driver_names_dict = {name: i for i, name in enumerate(driver_names)}
    path = ENCODINGS_DIR / "driver_names_dict.csv"
    if not use_existing_dicts:
        _save_mapping(driver_names_dict, path, ("driver_name", "encoded_value"))

    exceptions = {"K ANTONELLI": "A ANTONELLI"}

    def driver_name_encoder(name: str) -> int:
        if use_existing_dicts:
            driver_df = pd.read_csv(path)
            driver_map = dict(zip(driver_df["driver_name"], driver_df["encoded_value"]))
            if name in exceptions:
                return int(driver_map[exceptions[name]])
            return int(driver_map[name])
        return int(driver_names_dict[name])

    df["driver_name"] = df["driver_name"].map(driver_name_encoder)
    return df


def clean_training_dataframe(session_data: pd.DataFrame, *, use_existing_dicts: bool = False) -> pd.DataFrame:
    columns = session_data.columns.tolist()
    df = session_data.copy()

    df = encode_driver_names(df, use_existing_dicts=use_existing_dicts)

    processed = pd.DataFrame()
    processed["year"] = df["year"]
    processed["driver"] = df["driver_name"]
    processed["team_name"] = df["team_name"]
    processed["team_lineage"] = df["team_name"].map(compute_team_lineage)
    processed["circuit_key"] = df["circuit_key"]

    _, cols = add_session_columns_subset(df, columns)
    processed[cols] = df[cols]

    columns = processed.columns.tolist()
    processed = impute_session_data(processed, columns)
    processed = impute_session_type(processed, columns)
    processed = impute_environment_data(processed, columns)

    # Drop rows where team_name is missing
    processed = processed.dropna(subset=["team_name"])  # type: ignore[arg-type]

    # Remove known leakage/problematic columns for training
    cols_drop = [
        "session_5_missing_final_position",
        "session_5_missing_avg_lap_time",
        "session_5_avg_lap_time",
    ]
    processed = processed.drop([c for c in cols_drop if c in processed.columns], axis=1)

    processed = encode_categorical_data(processed, processed.columns.tolist(), use_existing_dicts=use_existing_dicts)
    return processed


