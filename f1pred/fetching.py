from __future__ import annotations

import time
import datetime as dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import fastf1

from .paths import session_year_file


def _attempt_load(session, *, weather: bool, laps: bool, info: bool, attempts: int = 0):
    try:
        if weather:
            return session.weather_data
        if laps:
            return session.laps
        if info:
            return session.session_info
    except Exception:
        if attempts > 3:
            return None
        session.load(weather=True, laps=True)
        return _attempt_load(session, weather=weather, laps=laps, info=info, attempts=attempts + 1)


def _gap_to_fastest(df: pd.DataFrame, pole_time, session_number: int) -> list[float]:
    gaps: list[float] = []
    for _i in range(len(df)):
        if not np.isnan(df[f"session_{session_number}_q3_best_lap"].iloc[_i]):
            gaps.append(df[f"session_{session_number}_q3_best_lap"].iloc[_i] - pole_time)
        elif not np.isnan(df[f"session_{session_number}_q2_best_lap"].iloc[_i]):
            gaps.append(df[f"session_{session_number}_q2_best_lap"].iloc[_i] - pole_time)
        elif not np.isnan(df[f"session_{session_number}_q1_best_lap"].iloc[_i]):
            gaps.append(df[f"session_{session_number}_q1_best_lap"].iloc[_i] - pole_time)
        else:
            gaps.append(np.nan)
    return gaps


def _qualifying_dataframe(session, session_number: int) -> pd.DataFrame:
    results = session.results
    qual = pd.DataFrame()
    qual["driver_number"] = results["DriverNumber"]
    results["Q1"] = results["Q1"].map(lambda x: x.total_seconds())
    results["Q2"] = results["Q2"].map(lambda x: x.total_seconds())
    results["Q3"] = results["Q3"].map(lambda x: x.total_seconds())
    qual[f"session_{session_number}_q1_best_lap"] = results["Q1"]
    qual[f"session_{session_number}_q2_best_lap"] = results["Q2"]
    qual[f"session_{session_number}_q3_best_lap"] = results["Q3"]
    pole_time = session.laps.pick_fastest()["LapTime"]
    qual[f"session_{session_number}_gap_to_fastest"] = _gap_to_fastest(qual, pole_time, session_number)
    return qual


def _lap_dataframe(session, session_number: int, session_type: Optional[str]) -> pd.DataFrame:
    lap_data = _attempt_load(session, weather=False, laps=True, info=False)
    if lap_data is None or not session_type:
        drivers = session.results["DriverNumber"].unique()
        data = pd.DataFrame(columns=["driver_number"])  # empty columns for fallbacks
        data["driver_number"] = drivers
        return data

    drivers = lap_data["DriverNumber"].unique()
    lap_data["LapTime"] = lap_data["LapTime"].map(lambda x: x.total_seconds())
    if session_type == "Qualifying":
        qual_df = _qualifying_dataframe(session, session_number)
        data = pd.DataFrame()
        data["driver_number"] = drivers
        data = data.merge(qual_df, on="driver_number")
        return data

    avg_lap_times: list[float] = []
    for driver in drivers:
        driver_laps = lap_data[lap_data["DriverNumber"] == driver]
        avg_lap_times.append(float(np.nanmean(driver_laps["LapTime"])))
    data = pd.DataFrame(columns=["driver_number", f"session_{session_number}_avg_lap_time"])
    data["driver_number"] = drivers
    data[f"session_{session_number}_avg_lap_time"] = avg_lap_times
    return data


def _weather_dataframe(session, session_number: int) -> pd.DataFrame:
    w = _attempt_load(session, weather=True, laps=False, info=False)
    drivers = session.results["DriverNumber"].unique()
    cols = [
        f"session_{session_number}_starting_air_temp",
        f"session_{session_number}_starting_humidity",
        f"session_{session_number}_starting_pressure",
        f"session_{session_number}_starting_rainfall",
        f"session_{session_number}_starting_track_temp",
        f"session_{session_number}_starting_wind_direction",
        f"session_{session_number}_starting_wind_speed",
        f"session_{session_number}_ending_air_temp",
        f"session_{session_number}_ending_humidity",
        f"session_{session_number}_ending_pressure",
        f"session_{session_number}_ending_rainfall",
        f"session_{session_number}_ending_track_temp",
        f"session_{session_number}_ending_wind_direction",
        f"session_{session_number}_ending_wind_speed",
    ]
    data = pd.DataFrame(columns=["driver_number", *cols])
    data["driver_number"] = drivers
    if w is None:
        for c in cols:
            data[c] = np.nan
        return data

    data[f"session_{session_number}_starting_air_temp"] = [w["AirTemp"].iloc[0]] * len(drivers)
    data[f"session_{session_number}_starting_humidity"] = [w["Humidity"].iloc[0]] * len(drivers)
    data[f"session_{session_number}_starting_pressure"] = [w["Pressure"].iloc[0]] * len(drivers)
    data[f"session_{session_number}_starting_rainfall"] = [w["Rainfall"].iloc[0]] * len(drivers)
    data[f"session_{session_number}_starting_track_temp"] = [w["TrackTemp"].iloc[0]] * len(drivers)
    data[f"session_{session_number}_starting_wind_direction"] = [w["WindDirection"].iloc[0]] * len(drivers)
    data[f"session_{session_number}_starting_wind_speed"] = [w["WindSpeed"].iloc[0]] * len(drivers)

    data[f"session_{session_number}_ending_air_temp"] = [w["AirTemp"].iloc[-1]] * len(drivers)
    data[f"session_{session_number}_ending_humidity"] = [w["Humidity"].iloc[-1]] * len(drivers)
    data[f"session_{session_number}_ending_pressure"] = [w["Pressure"].iloc[-1]] * len(drivers)
    data[f"session_{session_number}_ending_rainfall"] = [w["Rainfall"].iloc[-1]] * len(drivers)
    data[f"session_{session_number}_ending_track_temp"] = [w["TrackTemp"].iloc[-1]] * len(drivers)
    data[f"session_{session_number}_ending_wind_direction"] = [w["WindDirection"].iloc[-1]] * len(drivers)
    data[f"session_{session_number}_ending_wind_speed"] = [w["WindSpeed"].iloc[-1]] * len(drivers)
    return data


def session_dataframe(session, session_number: int) -> pd.DataFrame:
    df = pd.DataFrame()
    info = _attempt_load(session, weather=False, laps=False, info=True)
    df["driver_name"] = session.results["BroadcastName"]
    df["driver_number"] = session.results["DriverNumber"]
    df["team_name"] = session.results["TeamName"]
    if info is None:
        df[f"session_{session_number}_type"] = [np.nan] * len(df)
        session_type = None
    else:
        df[f"session_{session_number}_type"] = info["Type"]
        session_type = info["Type"]
        df["circuit_name"] = info["Meeting"]["Circuit"]["ShortName"]
        df["circuit_key"] = info["Meeting"]["Circuit"]["Key"]
        df["event_name"] = info["Meeting"]["Name"]

    weather_df = _weather_dataframe(session, session_number)
    lap_df = _lap_dataframe(session, session_number, session_type)
    if session_type != "Practice":
        df[f"session_{session_number}_final_position"] = session.results["Position"]
    df = df.merge(lap_df, on="driver_number", how="left")
    df = df.merge(weather_df, on="driver_number", how="left")
    return df


def event_dataframe(event, *, testing: bool = False, save: bool = True) -> pd.DataFrame:
    event_df = pd.DataFrame(columns=["year"])  # placeholder col for concat
    event_sessions = [event.Session1, event.Session2, event.Session3, event.Session4, event.Session5]
    for i in range(len(event_sessions)):
        round_number = event.RoundNumber
        year = event.EventDate.year
        if testing:
            try:
                session = fastf1.get_testing_session(year, round_number, i + 1)
            except Exception:
                break
        elif event_sessions[i] is not None and event_sessions[i] != "" and event_sessions[i] == event_sessions[i]:
            session = event.get_session(i + 1)
            session_df = session_dataframe(session, i + 1)
            if event_df.empty:
                event_df = session_df
            else:
                cols_to_merge = [c for c in session_df.columns if c not in event_df.columns or c == "driver_number"]
                event_df = event_df.merge(session_df[cols_to_merge], on="driver_number")
    event_df["year"] = [event.EventDate.year] * len(event_df)
    if save:
        file_path = session_year_file(event.EventDate.year)
        try:
            prev = pd.read_csv(file_path)
            event_df = pd.concat([prev, event_df])
            event_df.to_csv(file_path, index=False)
        except Exception:
            event_df.to_csv(file_path, index=False)
    time.sleep(3)
    return event_df


def season_dataframe(year: int, *, next_event_round: Optional[int]) -> pd.DataFrame:
    file_name = session_year_file(year)
    try:
        # Clear file for re-generation
        pd.DataFrame().to_csv(file_name, index=False)
    except Exception:
        pass

    schedule = fastf1.get_event_schedule(year)
    data = pd.DataFrame(columns=["year"])  # schema placeholder
    testing_event_number = 1

    for round_number in schedule["RoundNumber"]:
        if next_event_round == round_number and year == dt.datetime.now().year:
            break
        round_diff = 0 if round_number == 0 else 1 - int(sum(schedule["RoundNumber"] == 0))
        if schedule.is_testing()[round_number - round_diff]:
            event = fastf1.get_testing_event(year, testing_event_number)
            testing_event_number += 1
            testing = True
        else:
            event = schedule.get_event_by_round(round_number)
            testing = False
        evt_df = event_dataframe(event, testing=testing)
        data = pd.concat([data, evt_df])
    return data


def multi_season_dataframe(start_year: int, end_year: Optional[int] = None) -> pd.DataFrame:
    if end_year is None:
        end_year = dt.datetime.now().year
    data = pd.DataFrame()
    events_remaining = fastf1.get_events_remaining()
    next_event = int(events_remaining["RoundNumber"].iloc[0])
    for year in range(start_year, end_year + 1):
        season_df = season_dataframe(year, next_event_round=next_event)
        data = pd.concat([data, season_df])
    return data


