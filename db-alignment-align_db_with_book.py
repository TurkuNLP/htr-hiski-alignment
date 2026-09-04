import argparse
from collections import defaultdict
import csv
import json
import sqlite3
from natsort import natsorted
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

from src.csv_pages_to_names import flatten_page_names, read_files
from src.local_sequence_alignment import smith_waterman
from src.name_cleanup import get_cleaned_full_name_db
from src.post_alignment_fixes import (
    extract_alignment_block_centers,
    extract_unique,
)
from src.similarity_matrix import (
    create_sim_m,
    create_slices,
    reciprocal_best_matches,
)
from src.sqlite_events_to_names import db_query_and_params_from_direction

argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--book_data_base_dir")
argparser.add_argument("-db", "--database_path")
argparser.add_argument("-b", "--book_info")
argparser.add_argument("-p", "--print_metadata")
argparser.add_argument("-o", "--output_dir", default=None)
argparser.add_argument("--block_size", default=3, type=int)
argparser.add_argument("--omit_center_name", action="store_true")
argparser.add_argument("--anchor_emphasis", default=1e6, type=int)
argparser.add_argument("--comparison_join_chars", default=None)
argparser.add_argument("--verbose", action="store_true")
args = argparser.parse_args()

book_data_base_dir = Path(args.book_data_base_dir)
database_path = Path(args.database_path)
book_info_path = Path(args.book_info)
print_metadata_path = Path(args.print_metadata)
output_dir = Path(args.output_dir) if args.output_dir else None
block_size = args.block_size
omit_center_name = args.omit_center_name
comparison_join_chars = args.comparison_join_chars
anchor_emphasis = args.anchor_emphasis
print_matches = args.verbose

connection = sqlite3.connect(database_path)
cursor = connection.cursor()

table_cols = (
    "event_id",
    "dep_day",
    "dep_month",
    "dep_year",
    "arr_day",
    "arr_month",
    "arr_year",
    "destination/source",
    "village",
    "house",
    "profession",
    "first_name",
    "patronym",
    "last_name",
    "etc",
    "village_id",
    "parish_id",
    "destinationN",
    "sourceN",
    "first_nameN",
    "patronymN",
    "last_nameN",
)

with open(book_info_path, "r") as fp:
    book_info = json.load(fp)

print_metadata_dict = defaultdict(list)
with open(print_metadata_path, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row["print_id"]
        direction = row["direction"]
        side = row["side"]
        mapped_side = "both" if side == "single" else side
        print_metadata_dict[pid].append((direction, mapped_side))


aligned_books = []
skipped_books = []

for book in book_info:
    event_ids = book["events"]
    if len(event_ids) == 0:
        skipped_books.append((book, "no event's associated in book info"))
        continue

    parish = book["parish_normalized"].lower()
    parish_id = book["parish_id"]
    year_start = book["year_1"]
    year_end = book["year_2"]
    source_type = book["source"].lower()
    classification = book["luokittelu"].lower()

    ocr_files_path = (
        book_data_base_dir
        / f"{parish}/muuttaneet_{year_start}-{year_end}_{source_type}"
    )

    direction_info = []

    if classification.startswith("print"):
        print_id = classification.split(" ")[-1]

        direction_info.extend(print_metadata_dict[print_id])
    else:
        match classification:
            case "handrawn in/out":
                direction_info += [("in", "left"), ("out", "right")]
            case "handrawn out/in":
                direction_info += [("out", "left"), ("in", "right")]
            case "handrawn in":
                direction_info.append(("in", ""))
            case "handrawn out":
                direction_info.append(("out", ""))
            case _:
                direction_info.append(("", ""))

    for d_i in direction_info:
        left_or_right_filter = d_i[1]
        csv_files = tuple(
            str(path)
            for path in ocr_files_path.glob(f"*{left_or_right_filter}*.csv")
        )
        if len(csv_files) == 0:
            skipped_books.append((book, "no csv files found"))
            continue

        moving_direction = d_i[0]

        query, query_params = db_query_and_params_from_direction(
            moving_direction, event_ids
        )
        cursor.execute(query, query_params)

        parish_events = cursor.fetchall()

        db_names = []
        db_names_ids = []

        for row in parish_events:
            event = {
                col: ("" if value is None else value)
                for col, value in zip(table_cols, row)
            }

            full_name = get_cleaned_full_name_db(event)
            db_names.append(full_name)
            db_names_ids.append(event["event_id"])

        files_sorted = natsorted(csv_files)

        ocr_pages, ocr_names = read_files(files_sorted)

        ocr_names_flat = flatten_page_names(ocr_names)

        db_slices, db_ranges = create_slices(
            db_names,
            block_size=block_size,
            step_size=1,
            pad=True,
            pad_value="",
        )
        ocr_slices, ocr_ranges = create_slices(
            ocr_names_flat,
            block_size=block_size,
            step_size=1,
            pad=True,
            pad_value=("",),
        )

        if len(db_slices) == 0 or len(ocr_slices) == 0:
            skipped_books.append(
                (book, "creating blocks resulted in list with 0 blocks")
            )
            continue

        if omit_center_name:
            sim_m_db_slices = tuple(
                map(
                    lambda x: x[: len(x) // 2] + x[len(x) // 2 + 1 :],
                    db_slices,
                )
            )
            sim_m_ocr_slices = tuple(
                map(
                    lambda x: x[: len(x) // 2] + x[len(x) // 2 + 1 :],
                    ocr_slices,
                )
            )
        else:
            sim_m_db_slices = db_slices
            sim_m_ocr_slices = ocr_slices

        sim_m_ocr_slices = tuple(
            tuple(name[0] for name in ocr_slice)
            for ocr_slice in sim_m_ocr_slices
        )

        sim_m_fuzz_ratio, scores_fuzz_ratio = create_sim_m(
            a_slices=sim_m_db_slices,
            b_slices=sim_m_ocr_slices,
            scorer=fuzz.ratio,
            n_workers=-1,
            join_char=comparison_join_chars,
            progressbar=False,
        )

        sim_m_fuzz_ratio_rbm = reciprocal_best_matches(sim_m_fuzz_ratio)

        anchor_threshold = np.quantile(
            sim_m_fuzz_ratio,
            (
                1
                - 0.25
                * np.hypot(*sim_m_fuzz_ratio.shape)
                / sim_m_fuzz_ratio.size
            ).round(4),
        ).round(2)

        anchor_m = sim_m_fuzz_ratio_rbm >= anchor_threshold

        score_m = anchor_m * sim_m_fuzz_ratio_rbm + sim_m_fuzz_ratio
        score_m -= score_m.max() - 1
        score_m[anchor_m.nonzero()] = anchor_emphasis

        aln2 = smith_waterman(score_m, return_matrices=True)

        slice_pairs = np.array(aln2.aligned_pairs)
        flattened_alignment = extract_alignment_block_centers(
            slice_pairs,
            np.array(db_ranges),
            np.array(ocr_ranges),
            (db_ranges[0][1] - 1) // 2,
        )

        ocr_alignments_dict = extract_unique(
            flattened_alignment, sim_m_fuzz_ratio, True, True
        )

        alignment_with_unique_names = tuple(
            (db[0], ocr_i, db[1])
            for ocr_i, db in sorted(ocr_alignments_dict.items())
        )

        if print_matches:
            print(
                "\n".join(
                    f"{ocr_names_flat[ocr_i]} ~ {db_names[db_i]} | {block_similarity}"
                    for db_i, ocr_i, block_similarity in alignment_with_unique_names
                )
                + f"\nocr names: {len(ocr_names_flat)}\nmatches: {len(alignment_with_unique_names)}\n"
            )

        matches_json_data = tuple(
            {
                "ocr": ocr_names_flat[ocr_i][0],
                "db": db_names[db_i],
                "ocr_page": files_sorted[ocr_names_flat[ocr_i][1]],
                "ocr_page_row": ocr_names_flat[ocr_i][2],
                "block_score": float(block_similarity),
                "ocr_block": tuple(
                    tuple(name[0] for name in ocr_slice)
                    for ocr_slice in ocr_slices
                )[ocr_i],
                "db_block": db_slices[db_i],
            }
            for db_i, ocr_i, block_similarity in alignment_with_unique_names
        )

        if output_dir:
            (output_dir / parish).mkdir(exist_ok=True)
            with open(
                output_dir
                / f"{parish}/muuttaneet_{year_start}-{year_end}_{source_type}_{d_i[0]}.json",
                "w",
            ) as fp:
                json.dump(matches_json_data, fp, ensure_ascii=False, indent=4)

            with open(output_dir / "skipped_books.json", "w") as fp:
                json.dump(skipped_books, fp, indent=4)

        print(ocr_files_path)

        aligned_books.append(book)
