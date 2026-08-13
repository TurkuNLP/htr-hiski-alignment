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

def db_query_and_params_from_direction(moving_direction, event_ids):
    placeholders = ", ".join(["?"] * len(event_ids))
    query_params = list(event_ids)

    query = ""
    if moving_direction in ["", "both", "unknown"]:
        query_params += list(event_ids)

        query = f"""
            SELECT * FROM immigrated WHERE event_id IN ({placeholders})
            UNION ALL
            SELECT * FROM emigrated WHERE event_id IN ({placeholders})
        """
    elif moving_direction.startswith("in"):
        query = f"""
            SELECT * FROM immigrated
            WHERE event_id IN ({placeholders});
        """
    elif moving_direction.startswith("out"):
        query = f"""
            SELECT * FROM emigrated
            WHERE event_id IN ({placeholders});
        """

    return query, query_params
