import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

logger = logging.getLogger(__name__)

ID_COL = "ID"
VL_COL = "VL / Bereich"
QUERY_COL = "Query / Nutzerfrage"
QUERY_TYPE = "Fragetyp"
QUERY_TOPIC = "Thema"
ANSWER_COL = "Goldstandard-Antwort"

REQUIRED_COLS = (ID_COL, VL_COL, QUERY_COL, QUERY_TYPE, QUERY_TOPIC, ANSWER_COL)

_MAX_ITERABLE_EMPTY_ROWS = 10


@dataclass(frozen=True)
class TestCase:
    id: str
    vl: str
    query: str
    query_type: str
    query_topic: str
    exp_answer: str

    @property
    def test_case_name(self) -> str:
        return f"ID={self.id} | VL={self.vl}"


def load_test_cases(path: Path) -> list[TestCase]:
    wb = load_workbook(path, read_only=True, data_only=True)
    logger.debug("Loaded workbook at '%s'", str(path))

    try:
        ws = wb.active
        assert ws is not None

        rows = ws.iter_rows(values_only=True)
        try:
            headers = next(rows)
        except StopIteration:
            raise ValueError(f"Catalogue with test cases '{path}' is empty.")

        if headers is None:
            raise ValueError(f"Catalogue '{path}' has no header row")

        headers = [str(header).strip() if header is not None else "" for header in headers]

        missing = [col for col in REQUIRED_COLS if col not in headers]
        if missing:
            raise ValueError(
                f"Expected columns are missing in catalogue '{path}': {', '.join(missing)}"
            )

        column_indices = {column: headers.index(column) for column in REQUIRED_COLS}

        empty_row_cnt = 0
        test_cases: list[TestCase] = []
        for row_idx, row in enumerate(rows):
            if not any(value is not None for value in row):
                if empty_row_cnt < _MAX_ITERABLE_EMPTY_ROWS:
                    logger.debug(
                        "Consecutivelly iterated over '%d' empty rows. Aborting iteration",
                        _MAX_ITERABLE_EMPTY_ROWS,
                    )
                    break
                empty_row_cnt += 1
                continue
            empty_row_cnt = 0

            query = row[column_indices[QUERY_COL]]
            answer = row[column_indices[ANSWER_COL]]

            if query is None or answer is None:
                logger.warning(
                    "Either query or golden answer is missing in catalogue '%s' at row '%d'",
                    path,
                    row_idx,
                )
                continue

            test_cases.append(
                TestCase(
                    id=_cell_to_str(row[column_indices[ID_COL]]),
                    vl=_cell_to_str(row[column_indices[VL_COL]], is_optional=True),
                    query=_cell_to_str(row[column_indices[QUERY_COL]]),
                    query_type=_cell_to_str(row[column_indices[QUERY_TYPE]], is_optional=True),
                    query_topic=_cell_to_str(row[column_indices[QUERY_TOPIC]], is_optional=True),
                    exp_answer=_cell_to_str(row[column_indices[ANSWER_COL]]),
                )
            )

    finally:
        wb.close()

    logger.info("Parsed '%d' test case(s)", len(test_cases))
    return test_cases


def _cell_to_str(value: Any, is_optional: bool = False) -> str:
    if value is None:
        if is_optional:
            return ""
        raise ValueError("Required column value is missing")
    return str(value)
