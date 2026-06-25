import sqlite3
import pandas as pd


def events_to_df(cursor: sqlite3.Cursor, table: str, parish_id: int):
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
        event["first_name"] = event["first_name"].split("\\K")[0]
        event["first_name"] = event["first_name"].split("föd")[0]
        event["patronym"] = event["patronym"].split("\\K")[0]
        event["last_name"] = event["last_name"].split("\\K")[0]

    events_df = pd.DataFrame.from_dict(parish_events_as_dicts)
    events_df = events_df.fillna("")

    return events_df
