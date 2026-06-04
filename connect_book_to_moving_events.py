import csv
import sqlite3
from pathlib import Path

def moving_records_to_list(moving_records: Path) -> tuple[list, dict]:
    with open(moving_records, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        cols = reader.fieldnames[:19]

    col_to_idx = {cols[i]: i for i in range(len(cols))}

    books = []
    with open(moving_records, newline='') as csvfile:
        reader = csv.reader(csvfile)
    
        for row in reader:
            if (
                row[0] == 'parish_normalized'
                or row[col_to_idx["Start_year"]] == ""
                or row[col_to_idx["End_year"]] == ""
            ):
                continue

            parish_id = int(row[col_to_idx["parish_id"]])
            parish_normalized = row[col_to_idx["parish_normalized"]]

            year_1 = int(row[col_to_idx["Start_year"]])
            year_2 = int(row[col_to_idx["End_year"]])

            source = row[col_to_idx["source"]]
            archive_id = row[col_to_idx["archive_id"]]
            doc = row[col_to_idx["doc"]]
            luokittelu = row[col_to_idx["Luokittelu"]]

            books.append(
                {
                    "parish_id": parish_id, 
                    "parish_normalized": parish_normalized, 
                    "year_1": year_1, 
                    "year_2": year_2, 
                    "source": source,
                    "archive_id": archive_id,
                    "doc": doc,
                    "luokittelu": luokittelu
                }
            )

    return (books, col_to_idx)

def db_to_moving_event_dict(db_path: Path, table:str) -> dict:
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute(f"SELECT event_id, parish_id, arr_year, dep_year FROM {table}")
    events = cursor.fetchall()
    events_dict = {
        event[0]: {
            "parish_id": event[1],
            "arr_year": event[2],
            "dep_year": event[3],
            "books": []
        } for event in events
    }

    return events_dict

def connect_books_to_moving_events(books: list[dict], emigration_events: dict, immigration_events: dict) -> list[dict]:

    for book in books:
        book["events"] = [
            event_id for event_id, event in emigration_events.items()
            if (
                event["parish_id"] == book["parish_id"]
                and ((
                    event["dep_year"] != 0
                    and event["dep_year"] >= book["year_1"]
                    and event["dep_year"] <= book["year_2"]
                )
                or (
                    event["arr_year"] != 0
                    and event["arr_year"] >= book["year_1"]
                    and event["arr_year"] <= book["year_2"]
                ))
            )
        ] + [
            event_id for event_id, event in immigration_events.items()
            if (
                event["parish_id"] == book["parish_id"]
                and ((
                    event["dep_year"] != 0
                    and event["dep_year"] >= book["year_1"]
                    and event["dep_year"] <= book["year_2"]
                )
                or (
                    event["arr_year"] != 0
                    and event["arr_year"] >= book["year_1"]
                    and event["arr_year"] <= book["year_2"]
                ))
            )
        ]
    
    return books

def connect_events_to_books(record: Path, db: Path) -> tuple[dict, dict]:
    books, col_to_idx = moving_records_to_list(
        record
    )

    emigration_events = db_to_moving_event_dict(
        db, "emigrated"
    )
    immigration_events = db_to_moving_event_dict(
        db, "immigrated"
    )

    books = connect_books_to_moving_events(
        books,
        emigration_events,
        immigration_events
    )

    for book in books:
        for event_id in book["events"]:
            if event_id in emigration_events:
                emigration_events[event_id]["books"].append(
                    f"{book["parish_normalized"]}/muuttaneet_{book["year_1"]}-{book["year_2"]}_{book["source"]}"
                )
            if event_id in immigration_events:
                immigration_events[event_id]["books"].append(
                    f"{book["parish_normalized"]}/muuttaneet_{book["year_1"]}-{book["year_2"]}_{book["source"]}"
                )

    return emigration_events, immigration_events

print(
    connect_events_to_books(
        Path("Moving_record_parishes_with_formats_v2.csv"),
        Path("all_hiski_records.sqlite3")
    )[0][10602]
)