### Formula 1 Prediction Dashboard

A Streamlit dashboard and ML pipeline that predicts Formula 1 race finishing positions from session data and computes over/under prop probabilities. Built end-to-end: data ingestion with FastF1, preprocessing, model training (XGBoost + scikit-learn pipeline), and interactive exploration.

### Project structure

- `app.py`: Streamlit UI for batch/manual predictions, prop odds, feature glossary, and encoding guide.
- `predict.py`: Simple script to run predictions and over/under odds on `data/latest_race_data.csv`.
- `model_training.py`: Trains the model on `data/processed_data.csv` and saves artifacts.
- `fetch_training.py`: Builds `data/processed_data.csv` from the included `data/session_data_*.csv` files and writes encoding dictionaries.
- `get_latest_race_data.py`: Fetches and cleans the next-race snapshot using FastF1 (for app/CLI predictions).
- `f1pred/`: Library code (paths, preprocessing, fetching, training, CLI).
  - `f1pred/cli.py`: CLI for cleaning and training.
  - `f1pred/preprocess.py`: Feature engineering, imputations, and categorical encodings.
  - `f1pred/training.py`: Pipeline construction, sliding-window splits, training, and metrics.
  - `f1pred/fetching.py`: FastF1 data loaders to assemble session/season datasets.

### Setup

- Python 3.12+ recommended
- Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Quickstart

1. Build processed training data from included session CSVs:

```bash
python fetch_training.py
```

2. Train the model and persist artifacts (`model_pipeline.pkl`, `label_encoder.pkl`):

```bash
python model_training.py
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Optionally, fetch a fresh snapshot for the upcoming race and predict via script:

```bash
python get_latest_race_data.py   # writes data/latest_race_data.csv
python predict.py                # prints predictions and prop odds
```

### CLI (no packaging required)

- Clean raw session data into processed training CSV:

```bash
python -m f1pred.cli clean --input auto     # or --input path/to/raw.csv
```

- Train on processed data:

```bash
python -m f1pred.cli train                  # uses data/processed_data.csv by default
```

### Data notes

- This repo includes `data/session_data_*.csv` to enable immediate training without external calls.
- `get_latest_race_data.py` uses FastF1 to fetch the next event; network access required. For faster repeated fetches, consider enabling a FastF1 cache (see FastF1 docs) and ensure `fastf1_cache/` directory exists.

### Modeling details

- Features include historical session metrics (lap times, gaps to pole, weather), categorical encodings for drivers/teams/circuits, and session-type indicators.
- Pipeline: `ColumnTransformer(StandardScaler + OneHotEncoder)` → `XGBClassifier(objective="multi:softprob")`.
- Evaluation: sliding-window by season year with optional recency weighting and class balancing; reports mean accuracy and OVR AUC.
