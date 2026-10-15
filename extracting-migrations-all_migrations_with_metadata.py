#!/usr/bin/env python
# coding: utf-8

# In[170]:


import pandas as pd
from pathlib import Path
from natsort import natsorted
from src.csv_pages_to_names import read_files
from collections import defaultdict
from tqdm.notebook import tqdm
from src.extract_migrations_with_metadata.extract_from_row_columns import(
    get_possibly_related_toponym,
    get_possible_name,
    get_has_information
)
import json

# ## Metadata

# In[171]:


def book_id_to_path(id, book_id_to_book, csv_data_directory):
    book_info = book_id_to_book[id]
    start_year = book_info["year_start"]
    end_year = book_info["year_end"]
    source = book_info["source"].lower()

    return Path(csv_data_directory).glob(
        f"*{book_info["parish_normalized"]}/*{start_year}-{end_year}_{source}"
    )

# In[172]:


csv_data_directory = "data/migration-data-csv-release-v1/"
book_ids_df = pd.read_json("outputs/json/book_ids.json", orient="index")
book_ids_df.columns = ["parish_normalized", "year_start", "year_end", "source"]
book_ids_df = book_ids_df.rename_axis("book_id")
book_ids_to_info = book_ids_df.to_dict(orient="index")

id_to_book = book_ids_df.to_dict(orient="index")
book_path_stems = tuple(
    tuple(i)
    for i in book_ids_df.index.map(
        lambda x: book_id_to_path(x, id_to_book, csv_data_directory)
    ).to_list()
)
book_path_stems = pd.DataFrame(
    {
        "stem": tuple(
            i[0].stem if len(i) > 0 else None for i in book_path_stems
        )
    },
    index=book_ids_df.index,
)

book_to_id = {
    v: k
    for k, v in (
        book_ids_df["parish_normalized"] + "_" + book_path_stems["stem"]
    )
    .to_dict()
    .items()
}

# In[173]:


original_book_metadata = pd.read_csv(
    "data/csv/Moving_record_parishes_with_formats_v2.csv"
)
original_book_metadata = original_book_metadata.dropna(subset="url")
original_book_metadata["book_id"] = original_book_metadata["url"].apply(
    lambda x: int(x.split("=")[-1])
)
original_book_metadata = original_book_metadata.drop("url", axis="columns")

# In[ ]:


event_year_df = pd.read_csv("outputs/csv/year_predictions_extended.csv")
book_migration_metadata = pd.read_csv("outputs/csv/migration_books_metadata.csv")
parish_df = pd.read_csv("data/csv/parish.csv")

# In[ ]:


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

# In[ ]:


book_migration_metadata

# In[179]:


print_type_columns = pd.read_csv("outputs/csv/print_type_columns.csv")
with open("outputs/json/print_type_column_names_categorized.json", "r") as fp:
    categorize_column_name = json.load(fp)

# ## Gather migrations

# In[ ]:


def default_d():
    return defaultdict(dict)


all_file_paths = defaultdict(default_d)
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
    book_migration_metadata[
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
    if not all_file_paths[book][f"{img_range_start}-{img_range_end}"].get(
        in_out
    ):
        all_file_paths[book][f"{img_range_start}-{img_range_end}"][in_out] = []
    all_file_paths[book][f"{img_range_start}-{img_range_end}"][in_out] += (
        # natsorted(map(str, Path(p).glob(f"*{side}*.csv")))
        found_files
    )

# In[199]:


book_page_side_to_year = event_year_df[~(event_year_df["book_id"] == 0)]
book_page_side_to_year["key"] = (
    book_page_side_to_year["book"].astype(str)
    + "/" + book_page_side_to_year["page"].astype(str)
    + "/" + book_page_side_to_year["side"]
)
book_page_side_to_year = book_page_side_to_year.set_index("key")
book_page_side_to_year = book_page_side_to_year["year"].to_dict()

# In[200]:


print_type_columns["print type"] = print_type_columns["print type"].apply(
    lambda x: x.split(" ")[1]
)
print_type_and_side_to_cols = print_type_columns.set_index(
    ["print type", "side"]
)["sarakkeet"].to_dict()

for print_type, cols in print_type_and_side_to_cols.items():
    if cols == "[]":
        print_type_and_side_to_cols[print_type] = []
    else:
        print_type_and_side_to_cols[print_type] = cols[2:-2].split("', '")

print_type_and_side_to_categorized_cols = {
    print_type: tuple(
        categorize_column_name[column_name] for column_name in cols
    )
    for print_type, cols in print_type_and_side_to_cols.items()
}

# In[ ]:


book_to_print_type = (
    book_migration_metadata[
        [
            "book", "in_out", "print"
        ]
    ].set_index(["book", "in_out"])["print"].to_dict()
)

# In[ ]:


parish_to_parish_id = (
    book_migration_metadata[["parish_id", "parish_normalized"]]
    .drop_duplicates()
    .set_index("parish_normalized")["parish_id"]
    .to_dict()
)

# In[ ]:


for b_id in parish_to_parish_id.values():
    with open(
        "outputs/json/all_migration_events_parish_splits/"
        f"{parish_id}_migration_events.jsonl",
        "w",
    ) as fp:
        fp.seek(0, 0)
        fp.truncate()

for book, image_ranges in tqdm(list(all_file_paths.items())):
    migration_rows_to_write = []
    if "muuttaneet" in book:
        book_split = book.split("muuttaneet")
    elif "sekalaiset" in book:
        book_split = book.split("sekalaiset")
    parish = book_split[0][:-1]
    book_info = book_split[1].split("_")[1:]

    book_source = book_info[1].lower()
    book_years = book_info[0].split("-")
    book_year_start = book_years[0]
    book_year_end = book_years[1]
    parish_id = parish_to_parish_id[parish]

    for image_range, directions in image_ranges.items():
        image_range_split = image_range.split("-")
        for in_out, file_paths in directions.items():
            pages, names = read_files(file_paths)
            print_type = book_to_print_type[(book, in_out)]

            for i, page in enumerate(pages):
                file_idx = i
                file_path = file_paths[file_idx]
                split_path = file_path.split("_")
                page_n = int(split_path[-3])
                side = split_path[-2]
                year = book_page_side_to_year.get(
                   book + "/" + str(page_n) + "/" + side, 0
                )
                cols = print_type_and_side_to_cols.get(
                    (print_type, side), []
                )
                cols_categorized = (
                    print_type_and_side_to_categorized_cols.get(
                        (print_type, side), []
                    )
                )

                for row_idx, row in enumerate(page):
                    row_writing = row[0]
                    possible_related_toponym = (
                        get_possibly_related_toponym(
                            row_writing, cols_categorized, in_out
                        )
                    )
                    possible_name = get_possible_name(
                        row_writing, cols_categorized
                    )

                    r = {
                        "row_data": row[0],
                        "row_columns": cols,
                        "row_columns_categorized": cols_categorized,
                        "year_prediction": int(year),
                        "possible_name": possible_name,
                        "possibly_related_toponym": possible_related_toponym,
                        "file_path": file_path,
                        "parish_id": parish_id,
                        "book": book,
                        "print_type": print_type,
                        "book_source": book_source,
                        "book_year_start": book_year_start,
                        "book_year_end": book_year_end,
                        "image_range_start": image_range_split[0],
                        "image_range_end": image_range_split[1],
                        "page_n": page_n,
                        "file_row": row_idx,
                        "in/out": in_out,
                        "side": side,
                        "has_information": get_has_information(
                            row_writing, threshold=4
                        )
                    }
                    migration_rows_to_write.append(
                        json.dumps(r, ensure_ascii=False) + "\n"
                    )
    with open(
        "outputs/json/all_migration_events_parish_splits/"
        f"{parish_id}_migration_events.jsonl",
        "a",
    ) as fp:
        fp.writelines(migration_rows_to_write)
