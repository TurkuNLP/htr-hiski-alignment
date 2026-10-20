#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pathlib import Path
from natsort import natsorted
import re
from src.csv_pages_to_names import read_files
from collections import defaultdict
from tqdm.notebook import tqdm
import json

def default_d():
    return defaultdict(dict)

# In[ ]:


csv_data_directory = "data/migration-data-csv-release-v1/"
output_dir = "outputs/csv/migration-data-table_fix/"

# ## Metadata

# In[12]:


original_book_metadata = pd.read_csv(
    "data/csv/Moving_record_parishes_with_formats_v2.csv"
)
original_book_metadata = original_book_metadata.dropna(subset="url")
original_book_metadata["book_id"] = original_book_metadata["url"].apply(
    lambda x: int(x.split("=")[-1])
)
original_book_metadata = original_book_metadata.drop("url", axis="columns")

# In[13]:


event_year_df = pd.read_csv("outputs/csv/year_predictions_extended.csv")
book_migration_metadata = pd.read_csv("outputs/csv/migration_books_metadata.csv")
parish_df = pd.read_csv("data/csv/parish.csv")

# In[14]:


book_migration_metadata["book"] = (
    book_migration_metadata["parish_normalized"].astype(str)
    + "_"
    + book_migration_metadata["doctype"].astype(str)
    + "_"
    + book_migration_metadata["year_start"].astype(str)
    + "-"
    + book_migration_metadata["year_end"].astype(str)
    + "_"
    + book_migration_metadata["source"].astype(str)
)

# ## Find mistakes

# In[15]:


books_with_one_table = book_migration_metadata[book_migration_metadata["side"] == "both"]

# In[ ]:


files_to_fix = defaultdict(default_d)
for (
    parish_id,
    parish_normalized,
    source,
    book,
    side,
    in_out,
    img_range_start,
    img_range_end,
) in tqdm(
    books_with_one_table[
        [
            "parish_id",
            "parish_normalized",
            "source",
            "book",
            "side",
            "in_out",
            "img_range_start",
            "img_range_end",
        ]
    ].to_records(index=False)
):

    if "muuttaneet" in book:
        doctype = "muuttaneet"
    elif "sekalaiset" in book:
        doctype = "sekalaiset"
    book_split = book.split(doctype)

    parish = book_split[0][:-1]
    book_info = book_split[1].split("_")[1:]

    book_years = book_info[0].split("-")
    book_year_start = book_years[0]
    book_year_end = book_years[1]

    p = (
        csv_data_directory
        + parish
        + "/"
        + f"{doctype}_{book_year_start}-{book_year_end}_{source}"
    )
    side = side if side != "both" else ""
    image_range = range(img_range_start, img_range_end + 1)

    found_files = natsorted(map(str, Path(p).glob(f"*{side}*.csv")))
    found_files = list(
        filter(lambda x: int(x.split("_")[-3]) in image_range, found_files)
    )
    if not files_to_fix[book][f"{img_range_start}-{img_range_end}"].get(
        in_out
    ):
        files_to_fix[book][f"{img_range_start}-{img_range_end}"][in_out] = []
    files_to_fix[book][f"{img_range_start}-{img_range_end}"][in_out] += (
        found_files
    )

# In[17]:


pages = defaultdict(dict)
for p in files_to_fix["alavieska_muuttaneet_1890-1899_mko5"]["1-17"]["in"]:
    match = re.search(
        ".*_(.*?)_(right|left)_.*?.csv", p
    )
    page_number = int(match.group(1))
    side = match.group(2)

    pages[page_number][side] = p

# In[18]:


def handle_single_table(file_paths):
    left_files = list(filter(lambda x: "left" in x, file_paths))
    right_files = list(filter(lambda x: "right" in x, file_paths))

    left_pages = {
        left_file: left_page
        for left_file, left_page in zip(left_files, read_files(left_files)[0])
    }
    right_pages = {
        right_file: right_page
        for right_file, right_page in zip(right_files, read_files(right_files)[0])
    }

    pages = defaultdict(dict)
    for path in file_paths:
        match = re.search(".*_(.*?)_(right|left)_.*?.csv", path)
        page_number = int(match.group(1))
        side = match.group(2)

        pages[page_number][side] = path

    new_pages = {}
    for page, sides in pages.items():
        left_file = sides.get("left")
        right_file = sides.get("right")

        left_page = left_pages.get(left_file)
        right_page = right_pages.get(right_file)

        if left_file and right_file:
            new_rows = [
                (left_row[0] + right_row[0], left_row[1], left_row[2])
                for left_row, right_row in zip(left_page, right_page)
            ]
            new_pages[left_file.replace("_left_", "_both_")] = new_rows
        elif left_file:
            new_rows = left_pages[left_file]
            new_pages[left_file.replace("_left_", "_both_")] = new_rows
        elif right_file:
            new_rows = right_pages[right_file]
            new_pages[right_file.replace("_right_", "_both_")] = new_rows
    return new_pages

# In[ ]:


for book, page_ranges in tqdm(files_to_fix.items()):
    for page_range, in_out in page_ranges.items():
        for direction, files in in_out.items():
            fixed_tables = handle_single_table(
                files
            )
            for original_file in files:
                original_file = Path(
                    original_file.replace(csv_data_directory, output_dir)
                )
                original_file.unlink()
            for path, page in fixed_tables.items():
                path = Path(path.replace(csv_data_directory, output_dir))
                with open(path, "w") as fp:
                    fp.writelines(
                        [
                            json.dumps(row[0], ensure_ascii=False) + "\n"
                            for row in page
                        ]
                    )
