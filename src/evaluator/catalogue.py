import logging

import pandas as pd

from .logging_util import log_err_with_raise

logger = logging.getLogger(__name__)

ID_COL = "ID"
VL_COL = "VL / Bereich"
QUERY_COL = "Query / Nutzerfrage"
QUERY_TYPE = "Fragetyp"
QUERY_TOPIC = "Thema"
ANSWER_COL = "Goldstandard-Antwort"

REQUIRED_COLS = (ID_COL, VL_COL, QUERY_COL, QUERY_TYPE, QUERY_TOPIC, ANSWER_COL)

_DTYPES = {col: str for col in REQUIRED_COLS}


def load_test_cases(path: str) -> list[dict]:
    df = pd.read_excel(path, dtype=_DTYPES)

    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        log_err_with_raise(logger, f"Catalogue '{path}' is missing column(s): {', '.join(missing)}")

    df = df.dropna(subset=[QUERY_COL, ANSWER_COL])
    if df.empty:
        log_err_with_raise(logger, f"Catalogue '{path}' contains no usable rows.")

    logger.info("Loaded [%s] test case(s)", len(df))
    return df.to_dict(orient="records")
