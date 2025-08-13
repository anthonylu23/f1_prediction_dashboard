import pandas as pd

from f1pred.paths import PROCESSED_DATA_PATH, ENCODINGS_DIR
from f1pred.preprocess import clean_training_dataframe


def _read_year(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def main() -> None:
    parts = [
        _read_year('data/session_data_2018.csv'),
        _read_year('data/session_data_2019.csv'),
        _read_year('data/session_data_2020.csv'),
        _read_year('data/session_data_2021.csv'),
        _read_year('data/session_data_2022.csv'),
        _read_year('data/session_data_2023.csv'),
        _read_year('data/session_data_2024.csv'),
        _read_year('data/session_data_2025.csv'),
    ]
    session_data = pd.concat([p for p in parts if not p.empty])

    processed_data = clean_training_dataframe(session_data)
    processed_data.to_csv(PROCESSED_DATA_PATH, index=False)

    # Build circuit dictionary
    circuit_dict = {}
    for _, row in session_data.iterrows():
        circuit_key = row.get('circuit_key')
        circuit_name = row.get('circuit_name')
        if pd.notna(circuit_key) and circuit_key not in circuit_dict:
            circuit_dict[circuit_key] = circuit_name

    circuit_df = (
        pd.DataFrame(list(circuit_dict.items()), columns=['circuit_key', 'circuit_name'])
        .sort_values(by='circuit_key')
    )
    ENCODINGS_DIR.mkdir(parents=True, exist_ok=True)
    circuit_df.to_csv((ENCODINGS_DIR / 'circuit_dict.csv'), index=False)


if __name__ == '__main__':
    main()

