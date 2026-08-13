import sqlite3
import pandas as pd
from src.name_cleanup import clean_name_cols_db


def parish_events_to_df(cursor: sqlite3.Cursor, table: str, parish_id: int):
    parish_events = cursor.execute(
        f"SELECT * FROM {table} WHERE parish_id={parish_id};"
    ).fetchall()

    table_cols = cursor.execute(
        f"SELECT name FROM PRAGMA_TABLE_INFO('{table}');"
    ).fetchall()
    table_cols = [col[0] for col in table_cols]

    parish_events_as_dicts = [
        {col: value for col, value in zip(table_cols, row)}
        for row in parish_events
    ]
    for event in parish_events_as_dicts:
        event = clean_name_cols_db(event)

    events_df = pd.DataFrame.from_dict(parish_events_as_dicts)
    events_df = events_df.fillna("")

    return events_df
