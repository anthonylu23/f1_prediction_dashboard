import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from f1pred.paths import (
    PROCESSED_DATA_PATH,
    LATEST_RACE_DATA_PATH,
    ENCODINGS_DIR,
)
from f1pred.predict_utils import (
    load_artifacts,
    hash_dataframe,
    evaluate_position_prop_with_classes,
    compute_prop_table,
)


# -----------------------------
# Caching helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_encoder():
    return load_artifacts()


@st.cache_data(show_spinner=False)
def load_training_feature_columns():
    df_cols = pd.read_csv(PROCESSED_DATA_PATH, nrows=1).columns.tolist()
    feature_cols = [c for c in df_cols if c != 'session_5_final_position']
    return feature_cols


@st.cache_data(show_spinner=False)
def load_driver_name_mapping() -> dict:
    try:
        df = pd.read_csv(ENCODINGS_DIR / 'driver_names_dict.csv')
        mapping = {int(row['encoded_value']): str(row['driver_name']) for _, row in df.iterrows()}
        return mapping
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_encoding_dataframe(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def get_driver_names_from_encoded(encoded_series: pd.Series, mapping: dict) -> pd.Series:
    if not mapping:
        return encoded_series.astype(str)
    return encoded_series.map(lambda x: mapping.get(int(x), str(x)))


# Probability utilities are imported from f1pred.predict_utils


# -----------------------------
# Data alignment helpers
# -----------------------------
def ensure_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> tuple[bool, list[str]]:
    missing = [c for c in feature_cols if c not in df.columns]
    return (len(missing) == 0), missing


# hash_dataframe imported from f1pred.predict_utils


def select_latest_template_row(feature_cols: list[str]) -> pd.DataFrame:
    # Try to use latest_race_data.csv to provide a convenient, editable template row
    template_df = None
    if os.path.exists(LATEST_RACE_DATA_PATH):
        try:
            latest = pd.read_csv(LATEST_RACE_DATA_PATH)
            # Drop target if present
            latest = latest[[c for c in latest.columns if c in feature_cols]]
            if not latest.empty:
                template_df = latest
        except Exception:
            template_df = None

    if template_df is None:
        # Fallback: single empty row with correct columns
        template_df = pd.DataFrame(columns=feature_cols)
        template_df.loc[0] = [0] * len(feature_cols)

    return template_df


# -----------------------------
# Feature glossary helpers
# -----------------------------
def _weather_metric_description(metric_key: str) -> str:
    mapping = {
        'air_temp': 'Air temperature (°C).',
        'humidity': 'Relative humidity (%).',
        'pressure': 'Atmospheric pressure (hPa).',
        'rainfall': 'Rainfall present (boolean).',
        'track_temp': 'Track temperature (°C).',
        'wind_direction': 'Wind direction (degrees).',
        'wind_speed': 'Wind speed (units).',
    }
    return mapping.get(metric_key, metric_key.replace('_', ' ').capitalize())


def describe_feature(feature_name: str) -> str:
    if feature_name == 'year':
        return 'Season year.'
    if feature_name == 'driver':
        return 'Encoded driver identifier (see Encoding Guide).'
    if feature_name == 'team_name':
        return 'Encoded team name identifier (see Encoding Guide).'
    if feature_name == 'team_lineage':
        return 'Encoded team lineage identifier (see Encoding Guide).'
    if feature_name == 'circuit_key':
        return 'Encoded circuit identifier (see Encoding Guide).'

    m = re.match(r'^session_(\d+)_(.+)$', feature_name)
    if not m:
        return feature_name.replace('_', ' ').capitalize()

    session_num = m.group(1)
    tail = m.group(2)

    # Qualifying laps
    if tail in ['q1_best_lap', 'q2_best_lap', 'q3_best_lap']:
        return f'Best lap time (s) in {tail[:2].upper()} for session {session_num}.'

    # Type and core metrics
    if tail == 'type':
        return f'Session {session_num} type (Practice/Qualifying/Race), encoded (see Encoding Guide).'
    if tail == 'avg_lap_time':
        return f'Average lap time (s) in session {session_num}.'
    if tail == 'gap_to_fastest':
        return f'Gap to fastest (s) in session {session_num}.'
    if tail == 'final_position':
        return f'Finishing position in session {session_num}.'

    # Missing indicators
    if tail.startswith('missing_'):
        return f'Missing indicator for {tail[len("missing_"):].replace("_", " ")} in session {session_num}.'
    if tail.startswith('missed_'):
        return f'Indicator for missed portion of session {session_num} ({tail[len("missed_"):].replace("_", " ")}).'

    # Starting/ending weather blocks
    if tail.startswith('starting_'):
        metric = tail[len('starting_'):]
        return f'Start-of-session {session_num} ' + _weather_metric_description(metric)
    if tail.startswith('ending_'):
        metric = tail[len('ending_'):]
        return f'End-of-session {session_num} ' + _weather_metric_description(metric)

    return feature_name.replace('_', ' ').capitalize()


def generate_feature_glossary(feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        'feature': feature_columns,
        'description': [describe_feature(col) for col in feature_columns],
    })


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title='F1 Predictions and Props', layout='wide')
st.title('F1 Predictions and Props')
st.caption('Predict finishing positions and compute over/under position prop probabilities.')

model, label_encoder = load_model_and_encoder()
feature_cols = load_training_feature_columns()
driver_name_map = load_driver_name_mapping()
classes = label_encoder.classes_

tab_batch, tab_manual, tab_glossary, tab_encodings = st.tabs(["Batch Predictions", "Manual Single Prediction", "Feature Glossary", "Encoding Guide"]) 


with tab_batch:
    st.subheader('Data source')
    source = st.radio('Choose source', options=['Latest race data', 'Upload CSV'], horizontal=True)

    input_df = None
    if source == 'Latest race data':
        # Optionally allow refreshing latest data (if available in the environment)
        col1, col2 = st.columns([1, 3])
        with col1:
            refresh = st.button('Fetch/refresh latest data', use_container_width=True)
        if refresh:
            try:
                # Lazy import to avoid hard dependency unless used
                from get_latest_race_data import get_latest_race_data
                get_latest_race_data()
                st.success('Latest race data refreshed.')
            except Exception as e:
                st.warning(f'Could not refresh latest data: {e}')

        if os.path.exists(LATEST_RACE_DATA_PATH):
            try:
                df_latest = pd.read_csv(LATEST_RACE_DATA_PATH)
                # Ensure we feed only feature columns to the model
                input_df = df_latest[[c for c in df_latest.columns if c in feature_cols]].copy()
            except Exception as e:
                st.error(f'Error reading latest data: {e}')
        else:
            st.info('No latest data found. Use "Upload CSV" or click refresh if available.')

    else:
        uploaded = st.file_uploader('Upload a CSV with the same schema as training features', type=['csv'])
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                input_df = df_up
            except Exception as e:
                st.error(f'Failed to read uploaded CSV: {e}')

    if input_df is not None:
        ok, missing = ensure_feature_columns(input_df, feature_cols)
        if not ok:
            st.error('Missing required columns for the model: ' + ', '.join(missing))
        else:
            st.caption(f"{len(input_df)} rows loaded")
            st.dataframe(input_df, use_container_width=True, height=600)
            current_hash = hash_dataframe(input_df[feature_cols])
            if st.button('Predict for all rows', type='primary'):
                try:
                    proba = model.predict_proba(input_df[feature_cols])
                    pred_idx = np.argmax(proba, axis=1)
                    pred_pos = label_encoder.inverse_transform(pred_idx)

                    results = pd.DataFrame({
                        'driver_id': input_df['driver'].values if 'driver' in input_df.columns else np.arange(len(pred_pos)),
                        'driver': get_driver_names_from_encoded(input_df['driver'], driver_name_map) if 'driver' in input_df.columns else np.arange(len(pred_pos)).astype(str),
                        'predicted_position': pred_pos
                    })

                    st.session_state['batch_proba'] = proba
                    st.session_state['batch_results'] = results
                    st.session_state['batch_hash'] = current_hash
                except Exception as e:
                    st.error(f'Prediction failed: {e}')

            if (
                'batch_proba' in st.session_state
                and 'batch_results' in st.session_state
                and st.session_state.get('batch_hash') == current_hash
            ):
                proba = st.session_state['batch_proba']
                results = st.session_state['batch_results']

                st.subheader('Predicted finishing positions')
                st.dataframe(results.sort_values(by=['predicted_position']).reset_index(drop=True))

                st.subheader('Over/Under position prop probabilities')
                st.caption('Lower finish number is better (P1 best). Under = better (finishing ≤ floor(line)); Over = worse (finishing > floor(line)).')
                half_lines = [x + 0.5 for x in range(1, 20)]
                selected_line = st.select_slider('Prop line (finish position)', options=half_lines, value=6.5, key='batch_line')
                prop_table = compute_prop_table(proba, classes, selected_line, results['driver_id'], driver_name_map)
                st.dataframe(prop_table.sort_values(by='odds_over', ascending=False).reset_index(drop=True))

                with st.expander('Show predicted probabilities by finish position'):
                    top_idx = prop_table.sort_values(by='odds_under', ascending=False).index
                    display_positions = sorted(list(classes))
                    dist_rows = []
                    for i in top_idx:
                        row = {'driver': prop_table.loc[i, 'driver']}
                        for pos in display_positions:
                            j = int(np.where(classes == pos)[0][0])
                            row[f'P{int(pos)}'] = round(float(proba[i, j]), 4)
                        dist_rows.append(row)
                    probabilities_table = pd.DataFrame(dist_rows)
                    st.dataframe(probabilities_table)
                    driver_names = prop_table['driver']
                    if len(driver_names) > 1:
                        choice = st.selectbox('Choose a driver to view distribution', options=list(range(len(driver_names))), format_func=lambda i: driver_names[i] if i < len(driver_names) else f'Row {i}')
                        positions = [int(pos[1:]) for pos in probabilities_table.columns[1:]]
                        probabilities = probabilities_table.loc[probabilities_table['driver'] == driver_names[choice], probabilities_table.columns[1:]].values[0]
                        df = pd.DataFrame({
                            'position': positions,
                            'probability': probabilities
                        })
                        st.bar_chart(df, x='position', y='probability')
                    else: 
                        positions = [int(pos[1:]) for pos in probabilities_table.columns[1:]]
                        probabilities = probabilities_table.loc[probabilities_table['driver'] == driver_names[0], probabilities_table.columns[1:]].values[0]
                        df = pd.DataFrame({
                            'position': positions,
                            'probability': probabilities
                        })
                        st.bar_chart(df, x='position', y='probability')

with tab_manual:
    st.subheader('Edit a single row and predict')
    st.caption('The following features are target encoded (team name/lineage, driver, and session type); use the Encoding Guide tab to map data.')
    template_df = select_latest_template_row(feature_cols)

    # Choose a starting row if multiple present
    start_idx = 0
    if len(template_df) > 1:
        # If driver column exists, show driver names to pick a base row
        if 'driver' in template_df.columns:
            names = get_driver_names_from_encoded(template_df['driver'], driver_name_map)
            choice = st.selectbox('Choose a base driver row to edit', options=list(range(len(template_df))), format_func=lambda i: names.iloc[i] if i < len(names) else f'Row {i}')
            start_idx = int(choice)
        else:
            start_idx = st.number_input('Choose base row index', min_value=0, max_value=len(template_df)-1, value=0, step=1)

    base_row = template_df.iloc[[start_idx]].copy()
    st.caption('Edit values below to form a single prediction row:')
    edited = st.data_editor(base_row.reset_index(drop=True), num_rows='dynamic', use_container_width=True)

    current_manual_hash = None
    try:
        current_manual_hash = hash_dataframe(edited[feature_cols])
    except Exception:
        pass

    if st.button('Predict this row', type='primary'):
        try:
            # Validate required columns
            ok, missing = ensure_feature_columns(edited, feature_cols)
            if not ok:
                st.error('Missing required columns for the model: ' + ', '.join(missing))
            else:
                proba = model.predict_proba(edited[feature_cols])
                pred_idx = int(np.argmax(proba, axis=1)[0])
                pred_pos = int(label_encoder.inverse_transform([pred_idx])[0])

                st.session_state['manual_proba'] = proba
                st.session_state['manual_pred_pos'] = pred_pos
                st.session_state['manual_hash'] = current_manual_hash
        except Exception as e:
            st.error(f'Prediction failed: {e}')

    if (
        'manual_proba' in st.session_state
        and st.session_state.get('manual_hash') is not None
        and current_manual_hash is not None
        and st.session_state['manual_hash'] == current_manual_hash
    ):
        proba = st.session_state['manual_proba']
        pred_pos = st.session_state['manual_pred_pos']

        st.markdown(f"**Predicted finish position:** P{pred_pos}")

        half_lines = [x + 0.5 for x in range(1, 20)]
        selected_line = st.select_slider('Prop line (finish position)', options=half_lines, value=6.5, key='manual_line')
        st.caption('Lower finish number is better (P1 best). Under = better (finishing ≤ floor(line)); Over = worse (finishing > floor(line)).')
        pu, po = evaluate_position_prop_with_classes(proba[0], classes, selected_line)
        st.dataframe(pd.DataFrame([
            {
                'line': selected_line,
                'odds_under': round(pu, 4),
                'odds_over': round(po, 4),
            }
        ]), use_container_width=True)

        dist = []
        for pos in sorted(classes):
            j = int(np.where(classes == pos)[0][0])
            dist.append({'position': int(pos), 'probability': round(float(proba[0, j]), 4)})
        dist_df = pd.DataFrame(dist).sort_values(by='position')
        st.dataframe(dist_df)
        
        st.bar_chart(dist_df, x='position', y='probability')

with tab_encodings:
    st.subheader('Encoding dictionaries (input guide)')
    st.caption('Use these tables to map categorical inputs to the encoded integers expected by the model.')

    c1, c2 = st.columns(2)
    with c1:
        df_driver = load_encoding_dataframe((ENCODINGS_DIR / 'driver_names_dict.csv').as_posix())
        if df_driver is not None:
            st.markdown('**Driver names**')
            st.dataframe(df_driver.rename(columns={'driver_name': 'driver_name', 'encoded_value': 'encoded_value'}), use_container_width=True)
        else:
            st.info('driver_names_dict.csv not found')

        df_team = load_encoding_dataframe((ENCODINGS_DIR / 'team_names_dict.csv').as_posix())
        if df_team is not None:
            st.markdown('**Team names**')
            st.dataframe(df_team.rename(columns={'team_name': 'team_name', 'encoded_value': 'encoded_value'}), use_container_width=True)
        else:
            st.info('team_names_dict.csv not found')

    with c2:
        df_lineage = load_encoding_dataframe((ENCODINGS_DIR / 'team_lineages_dict.csv').as_posix())
        if df_lineage is not None:
            st.markdown('**Team lineages**')
            st.dataframe(df_lineage.rename(columns={'team_lineage': 'team_lineage', 'encoded_value': 'encoded_value'}), use_container_width=True)
        else:
            st.info('team_lineages_dict.csv not found')

        df_session = load_encoding_dataframe((ENCODINGS_DIR / 'session_types_dict.csv').as_posix())
        if df_session is not None:
            st.markdown('**Session types**')
            st.dataframe(df_session.rename(columns={'session_type': 'session_type', 'encoded_value': 'encoded_value'}), use_container_width=True)
        else:
            st.info('session_types_dict.csv not found')

        df_circuit = load_encoding_dataframe((ENCODINGS_DIR / 'circuit_dict.csv').as_posix())
        if df_circuit is not None:
            st.markdown('**Circuits**')
            st.dataframe(df_circuit, use_container_width=True)
        else:
            st.info('circuit_dict.csv not found')


with tab_glossary:
    st.subheader('Feature glossary')
    st.caption('Short descriptions for each feature used by the model.')
    glossary_df = generate_feature_glossary(feature_cols)
    filter_text = st.text_input('Filter features (optional)', '')
    if filter_text:
        mask = glossary_df['feature'].str.contains(filter_text, case=False, na=False) | glossary_df['description'].str.contains(filter_text, case=False, na=False)
        st.dataframe(glossary_df[mask].reset_index(drop=True), use_container_width=True, height=600)
    else:
        st.dataframe(glossary_df.reset_index(drop=True), use_container_width=True, height=600)


st.markdown('---')
st.caption('To run locally: `streamlit run app.py`')


