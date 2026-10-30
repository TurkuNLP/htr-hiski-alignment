from collections import defaultdict
from natsort import natsorted
from pathlib import Path
from tqdm.cli import tqdm
from src.csv_pages_to_names import read_files

BOOK_TYPE_SEPARATORS = ("muuttaneet", "sekalaiset")
GATHER_COLUMNS = [
    "source",
    "book",
    "side",
    "in_out",
    "img_range_start",
    "img_range_end",
]


def parse_book_name(book, book_type_separators=BOOK_TYPE_SEPARATORS):
    doctype = next((s for s in book_type_separators if s in book), None)
    if doctype is None:
        raise ValueError(f"Unrecognized book name: {book}")
    prefix, suffix = book.split(doctype)
    parish = prefix[:-1]
    years, source = suffix.split("_")[1:3]
    year_start, year_end = years.split("-")
    return doctype, parish, source.lower(), year_start, year_end


def find_book_files(
    csv_data_directory,
    book,
    source,
    side,
    img_range_start,
    img_range_end,
    book_type_separators=BOOK_TYPE_SEPARATORS,
):
    doctype, parish, _, year_start, year_end = parse_book_name(
        book, book_type_separators
    )
    book_dir = (
        Path(csv_data_directory)
        / parish
        / f"{doctype}_{year_start}-{year_end}_{source}"
    )
    image_range = range(img_range_start, img_range_end + 1)
    found_files = natsorted(map(str, book_dir.glob(f"*{side}*.csv")))
    return [
        path for path in found_files if int(path.split("_")[-3]) in image_range
    ]


def gather_file_paths(
    csv_data_directory,
    book_migration_metadata,
    gather_columns=GATHER_COLUMNS,
    book_type_separators=BOOK_TYPE_SEPARATORS,
):
    all_file_paths = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    rows = book_migration_metadata[gather_columns].to_records(index=False)
    for source, book, side, in_out, start, end in tqdm(rows):
        all_file_paths[book][f"{start}-{end}"][in_out] += find_book_files(
            csv_data_directory,
            book,
            source,
            side,
            start,
            end,
            book_type_separators,
        )
    return all_file_paths

def build_row(row_idx, row, cols, cols_categorized, metadata):
    row_writing = row[0]
    return {
        "row_data": row_writing,
        "row_columns": cols,
        "row_columns_categorized": cols_categorized,
        "possible_name": get_possible_name(row_writing, cols_categorized),
        "possibly_related_toponyms": get_possibly_related_toponyms(
            row_writing, cols_categorized
        ),
        "file_row": row_idx,
        "has_information": get_has_information(row_writing, threshold=4),
        **metadata,
    }

def get_possible_name(row_columns, cols_categorized) -> str | None:
    possible_name = ""
    if len(cols_categorized) == 0:
        return possible_name
    if cols_categorized != []:
        if "name" in cols_categorized and (
            name_index := cols_categorized.index("name")
        ) < len(row_columns):
            possible_name = row_columns[name_index]

    return possible_name


def get_possibly_related_toponyms(
    row_columns, cols_categorized
) -> list[tuple[int, str, str]] | list:
    if len(cols_categorized) == 0:
        return []

    row_len = len(row_columns)

    toponym_indexes = [
        (i, category)
        for i, category in enumerate(cols_categorized)
        if category in ("place", "source", "destination")
    ]

    if len(toponym_indexes) == 0:
        return []

    possibly_related_toponyms = [
        (i, row_columns[i], category)
        for i, category in toponym_indexes
        if i < row_len
    ]

    return possibly_related_toponyms


def get_has_information(row_columns, threshold) -> bool:
    joined_cols = "".join(row_columns)
    if len(tuple(c for c in joined_cols if c.isalpha())) > threshold:
        return True
    else:
        return False


def iter_book_rows(
    csv_data_directory,
    book,
    image_ranges,
    parish_to_parish_id,
    book_to_print_type,
    book_page_side_to_year,
    print_type_and_side_to_cols,
    print_type_and_side_to_categorized_cols,
):
    doctype, parish, book_source, year_start, year_end = parse_book_name(book)
    parish_id = parish_to_parish_id[parish]

    for image_range, directions in image_ranges.items():
        range_start, range_end = image_range.split("-")
        for in_out, file_paths in directions.items():
            pages, _ = read_files(file_paths)
            print_type = book_to_print_type[(book, in_out)]

            for file_path, page in zip(file_paths, pages):
                split_path = file_path.split("_")
                page_n = int(split_path[-3])
                side = split_path[-2]
                year = book_page_side_to_year.get(f"{book}/{page_n}/{side}", 0)
                cols = print_type_and_side_to_cols.get(print_type, {}).get(
                    side, []
                )
                cols_categorized = print_type_and_side_to_categorized_cols.get(
                    print_type, {}
                ).get(side, [])
                metadata = {
                    "year_prediction": int(year),
                    "file_path": str(file_path).replace(str(csv_data_directory), ""),
                    "parish_id": parish_id,
                    "book": book,
                    "print_type": print_type,
                    "book_source": book_source,
                    "book_year_start": year_start,
                    "book_year_end": year_end,
                    "image_range_start": range_start,
                    "image_range_end": range_end,
                    "page_n": page_n,
                    "in/out": in_out,
                    "side": side,
                }
                for row_idx, row in enumerate(page):
                    yield parish_id, build_row(
                        row_idx, row, cols, cols_categorized, metadata
                    )

def parish_output_path(parish_id, output_dir):
    return output_dir / f"{parish_id}_migration_events.jsonl"
