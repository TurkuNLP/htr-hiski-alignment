from pathlib import Path
from tqdm.cli import tqdm
import sqlite3
import pandas as pd

INPUT_DIR = Path("more_test_outputs/json/migration_jsonls/2-final")

jsonl_files = list(Path(INPUT_DIR).glob("*.jsonl"))
jsonl_files = sorted(
    INPUT_DIR.glob("*.jsonl"),
    key=lambda x: x.stat().st_size,
    reverse=True
)

con = sqlite3.connect('more_test_outputs/test.sqlite')
for jsonl_path in tqdm(jsonl_files):
    df = pd.read_json(jsonl_path, lines=True)
    df[
        [
            "row_data",
            "row_data_filled",
            "possibly_related_toponyms",
            "row_columns",
            "row_columns_categorized",
            "duplicate_of"
        ]
    ] = df[
        [
            "row_data",
            "row_data_filled",
            "possibly_related_toponyms",
            "row_columns",
            "row_columns_categorized",
            "duplicate_of"
        ]
    ].astype(str)
    df.to_sql("migrations", con, if_exists='append')
con.close()