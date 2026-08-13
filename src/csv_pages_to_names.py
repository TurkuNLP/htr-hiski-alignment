import csv
from pathlib import Path
from natsort import natsorted
import numpy as np
import re


def book_to_directory(book: dict) -> Path:
    doctype = book["doctype"]
    years = book["years"]
    source = book["source"].lower()

    return Path(f"{doctype}_{years}_{source}/")


def file_paths_to_names(file_paths):
    files_sorted = natsorted(file_paths)
    pages, names = read_files(files_sorted)

    return pages, names

def count_col_upper_cases(row):
    col_upper_cases = [len(re.sub("[^A-Z]", "", col)) for col in row]
    return col_upper_cases

def read_files(files_sorted: list[str], known_name_col: int = None):
    pages = []
    names = []

    for file_idx, file_name in enumerate(files_sorted):
        with open(file_name, "r") as file:
            reader = csv.reader(file, delimiter=",")

            file_rows = []
            cols_with_most_upper_cases = []

            for row_idx, row in enumerate(reader):
                file_rows.append((tuple(word.strip(' "') for word in row), file_idx, row_idx))

                if not known_name_col:
                    upper_case_counts = count_col_upper_cases(row) or [0]
                    cols_with_most_upper_cases.append(
                        np.argmax(upper_case_counts)
                    )

            pages.append(file_rows)

        if known_name_col:
            name_col = known_name_col
        else:
            name_col = np.argmax(np.bincount(cols_with_most_upper_cases))

        names_in_file = []
        for row_idx, row in enumerate(file_rows):
            if len(row[0]) <= name_col:
                names_in_file.append(("", file_idx, row_idx))
            else:
                names_in_file.append((row[0][name_col], file_idx, row_idx))
        names.append(names_in_file)
    return pages, names


def flatten_page_names(page_names: list[list[str]]):
    names_flat = []
    for names in page_names:
        names_flat += names

    return names_flat
