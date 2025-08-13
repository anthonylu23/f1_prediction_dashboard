from f1pred.preprocess import clean_training_dataframe as _clean_training_dataframe

import pandas as pd


def data_clean(session_data: pd.DataFrame, use_existing_dicts: bool = False) -> pd.DataFrame:
    """Backward-compatible wrapper for the new preprocessing function.

    Delegates to f1pred.preprocess.clean_training_dataframe.
    """
    return _clean_training_dataframe(session_data, use_existing_dicts=use_existing_dicts)


