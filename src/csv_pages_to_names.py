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

def read_files(files_sorted: list[str]):
    pages = []
    names = []
    for file_name in files_sorted:
        with open(file_name, "r") as file:
            reader = csv.reader(file, delimiter=",")

            page_rows = []
            cols_with_most_upper_cases = []
            for row in reader:
                page_rows.append([word.strip(' "') for word in row])

                upper_case_counts = count_col_upper_cases(row) or [0]
                cols_with_most_upper_cases.append(
                    np.argmax(upper_case_counts)
                )
            
            pages.append(page_rows)

        name_col = np.argmax(
            np.bincount(cols_with_most_upper_cases)
        )

        page_names = []
        for row in page_rows:
            if len(row) <= name_col:
                page_names.append("")
            else:
                page_names.append(row[name_col])
        names.append(page_names)

    return pages, names


def flatten_page_names(page_names: list[list[str]]):
    names_flat = []
    for names in page_names:
        names_flat += names

    return names_flat
