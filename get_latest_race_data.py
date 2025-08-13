import pandas as pd
import numpy as np
import fastf1
import datetime
from f1pred.paths import PROCESSED_DATA_PATH, LATEST_RACE_DATA_PATH
from f1pred.preprocess import clean_training_dataframe
from f1pred.fetching import event_dataframe as get_event_data

def get_latest_race_data():
    year = datetime.datetime.now().year
    columns = pd.read_csv(PROCESSED_DATA_PATH).columns.tolist()

    def get_latest_event():
        events_remaining = fastf1.get_events_remaining()
        next_event = events_remaining["RoundNumber"].iloc[0]
        schedule = fastf1.get_event_schedule(year)
        for rnd in schedule["RoundNumber"]:
            if rnd == next_event:
                return schedule["EventName"].iloc[rnd - 1]
        return None

    latest_event = get_latest_event()
    event = fastf1.get_event(year, latest_event)
    event_data = get_event_data(event, save=False)
    cleaned_event_data = clean_training_dataframe(event_data, use_existing_dicts=True)
    df = pd.DataFrame(columns=columns)
    df = pd.concat([df, cleaned_event_data], axis=0)
    if "session_2_final_position" not in cleaned_event_data.columns:
        df["session_2_final_position"] = [-1] * len(df)
    if "session_2_q1_best_lap" not in cleaned_event_data.columns:
        df["session_2_q1_best_lap"] = [-1] * len(df)
    if "session_2_q2_best_lap" not in cleaned_event_data.columns:
        df["session_2_q2_best_lap"] = [-1] * len(df)
    if "session_2_q3_best_lap" not in cleaned_event_data.columns:
        df["session_2_q3_best_lap"] = [-1] * len(df)
    if "session_2_gap_to_fastest" not in cleaned_event_data.columns:
        df["session_2_gap_to_fastest"] = [-1] * len(df)
    if "session_3_final_position" not in cleaned_event_data.columns:
        df["session_3_final_position"] = [-1] * len(df)
    if "session_4_avg_lap_time" not in cleaned_event_data.columns:
        df["session_4_avg_lap_time"] = [-1] * len(df)
    if "session_2_missing_final_position" not in cleaned_event_data.columns:
        df["session_2_missing_final_position"] = [True] * len(df)
    if "session_2_missed_qualifying" not in cleaned_event_data.columns:
        df["session_2_missed_qualifying"] = [False] * len(df)
    if "session_2_missing_fastest_lap" not in cleaned_event_data.columns:
        df["session_2_missing_fastest_lap"] = [True] * len(df)
    if "session_3_missing_final_position" not in cleaned_event_data.columns:
        df["session_3_missing_final_position"] = [True] * len(df)
    if "session_4_missing_avg_lap_time" not in cleaned_event_data.columns:
        df["session_4_missing_avg_lap_time"] = [False] * len(df)

    df.to_csv(LATEST_RACE_DATA_PATH, index=False)


if __name__ == "__main__":
    get_latest_race_data()