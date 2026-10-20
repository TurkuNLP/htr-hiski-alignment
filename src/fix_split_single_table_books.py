import csv
import re
from collections import defaultdict
from pathlib import Path
import pandas as pd
from natsort import natsorted
from tqdm.cli import tqdm
from src.csv_pages_to_names import read_files_to_pages

DOCTYPES = ("muuttaneet", "sekalaiset")
PAGE_SIDE_PATTERN = re.compile(r".*_(.*?)_(right|left)_.*?.csv")

BOOK_COLUMNS = [
    "source",
    "book",
    "side",
    "in_out",
    "img_range_start",
    "img_range_end",
]

## Metadata


def load_single_table_books(metadata_path):
    metadata = pd.read_csv(metadata_path)
    metadata["book"] = (
        metadata["parish_normalized"].astype(str)
        + "_"
        + metadata["doctype"].astype(str)
        + "_"
        + metadata["year_start"].astype(str)
        + "-"
        + metadata["year_end"].astype(str)
        + "_"
        + metadata["source"].astype(str)
    )
    return metadata[metadata["side"] == "both"]


## Locating files


def parse_book_name(book):
    doctype = next((d for d in DOCTYPES if d in book), None)
    if doctype is None:
        raise ValueError(f"Unknown doctype in book name: {book}")

    book_split = book.split(doctype)
    parish = book_split[0][:-1]  # drop trailing underscore
    years = book_split[1].split("_")[1]
    year_start, year_end = years.split("-")
    return parish, doctype, year_start, year_end


def find_book_files(
    input_dir, book, source, side, img_range_start, img_range_end
):
    parish, doctype, year_start, year_end = parse_book_name(book)
    book_directory = (
        f"{input_dir}{parish}/{doctype}_{year_start}-{year_end}_{source}"
    )

    side_filter = "" if side == "both" else side
    image_range = range(img_range_start, img_range_end + 1)

    found_files = natsorted(
        map(str, Path(book_directory).glob(f"*{side_filter}*.csv"))
    )
    return [f for f in found_files if int(f.split("_")[-3]) in image_range]


def collect_files_to_fix(books, input_dir):
    files_to_fix = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    rows = books[BOOK_COLUMNS].to_records(index=False)
    for source, book, side, in_out, range_start, range_end in tqdm(rows):
        found_files = find_book_files(
            input_dir, book, source, side, range_start, range_end
        )
        files_to_fix[book][f"{range_start}-{range_end}"][in_out] += found_files

    return files_to_fix


## Merging pages


def read_pages(file_paths):
    return dict(zip(file_paths, read_files_to_pages(file_paths)))


def group_paths_by_page(file_paths):
    pages = defaultdict(dict)
    for path in file_paths:
        match = PAGE_SIDE_PATTERN.search(path)
        page_number = int(match.group(1))
        side = match.group(2)
        pages[page_number][side] = path
    return pages


def merge_rows(left_page, right_page):
    return [
        (left_row[0] + right_row[0], left_row[1], left_row[2])
        for left_row, right_row in zip(left_page, right_page)
    ]


def merge_sides(file_paths):
    left_files = [path for path in file_paths if "left" in path]
    right_files = [path for path in file_paths if "right" in path]

    left_pages = read_pages(left_files)
    right_pages = read_pages(right_files)

    merged_pages = {}
    for sides in group_paths_by_page(file_paths).values():
        left_file = sides.get("left")
        right_file = sides.get("right")

        if left_file and right_file:
            rows = merge_rows(left_pages[left_file], right_pages[right_file])
            merged_pages[left_file.replace("_left_", "_both_")] = rows
        elif left_file:
            merged_pages[left_file.replace("_left_", "_both_")] = left_pages[
                left_file
            ]
        elif right_file:
            merged_pages[right_file.replace("_right_", "_both_")] = (
                right_pages[right_file]
            )

    return merged_pages


## Output


def to_output_path(path, input_dir, output_dir):
    return Path(path.replace(input_dir, output_dir))


def write_page(path, rows, delimiter=","):
    with open(path, "w") as fp:
        csv.writer(fp, delimiter=delimiter, lineterminator="\n").writerows(
            [row[0] for row in rows]
        )


def replace_with_merged(original_files, input_dir, output_dir):
    merged_pages = merge_sides(original_files)

    for original_file in original_files:
        to_output_path(original_file, input_dir, output_dir).unlink()

    for path, rows in merged_pages.items():
        write_page(to_output_path(path, input_dir, output_dir), rows)
