#!/usr/bin/env python

import pandas as pd


# Print types
# -----------

# Load original print types data

print_types = pd.read_csv(
    "data/csv/print_types.csv",
)

# Clean up columns

print_types["print_type"] = (
    print_types["print type"].str.split(" ").str[-1].str.strip()
)
print_types.drop("print type", inplace=True, axis="columns")
print_types.drop("l_r_and_n_columns", inplace=True, axis="columns")

print_types["direction_extra_info"] = print_types[
    ["direction", "direction_simple"]
].apply(
    lambda x: x["direction"].replace(x["direction_simple"], "").strip(),
    axis="columns",
)
print_types = print_types[
    ["print_type", "side", "direction_simple", "direction_extra_info"]
]

# Function for looking up print type information

print_types_dict = {
    print_type: group.set_index("side", drop=False).to_dict("index")
    for print_type, group in print_types.groupby("print_type")
}


def get_in_out(print_type):
    print_type_sides_to_directions = print_types_dict[print_type]
    sides = print_types_dict[print_type].keys()

    in_out = ""
    for side in sides:
        if len(in_out) != 0:
            in_out += "/"
        extra_info = print_type_sides_to_directions[side][
            "direction_extra_info"
        ]
        in_out += print_type_sides_to_directions[side]["direction_simple"]
        if extra_info:
            in_out += ":" + extra_info

    return in_out


# Books metadata
# --------------

# Load original books metadata

books = pd.read_csv(
    "data/csv/parish_books_with_metadata.csv",
)

# Use only lower case classification
books["classification"] = books["classification_normalized"]
books.drop("classification_normalized", inplace=True, axis="columns")
books["source"] = books["source"].str.lower()

# Books with multiple classifications get more than one row
books["classification"] = books["classification"].str.split(",")
books = books.explode("classification").reset_index().drop("index", axis="columns")
books["classification"] = books["classification"].str.strip()

books["original_classification"] = books["classification"]

# Tags for classifications

books["digital"] = books["classification"].str.contains("digital")
books["classification"] = (
    books["classification"].str.replace("digital", "").str.strip()
)

books["free_text"] = books["classification"].str.contains("free text")
books["classification"] = (
    books["classification"].str.replace("free text", "").str.strip()
)

books["halfbook"] = books["classification"].str.contains("halfbook")
books["classification"] = (
    books["classification"].str.replace("halfbook", "").str.strip()
)

books["hand-drawn"] = books["classification"].str.contains("handrawn")
books["classification"] = (
    books["classification"].str.replace("handrawn", "").str.strip()
)

books["wrong"] = books["classification"].str.contains("wrong")
books["classification"] = (
    books["classification"].str.replace("wrong", "").str.strip()
)

books["misc"] = books["classification"].str.contains("misc")
books["classification"] = (
    books["classification"].str.replace("misc", "").str.strip()
)

books["empty"] = books["classification"].str.contains("empty")
books["classification"] = (
    books["classification"].str.replace("empty", "").str.strip()
)

# Image range starts and ends

range_re = r":\s*(\d+)\s*(?:-\s*(\d+))?\s*$"

rng = books["classification"].str.extract(range_re)
start = pd.to_numeric(rng[0])
end = pd.to_numeric(rng[1])

books["img_range_start"] = start.fillna(1).astype("Int64")
books["img_range_end"] = (
    end.fillna(start).fillna(books["images"]).astype("Int64")
)

books["classification"] = (
    books["classification"].str.replace(range_re, "", regex=True).str.strip()
)


# Handle print types and directions

books.loc[books["classification"].str.startswith("print"), "print"] = (
    books.loc[
        books["classification"].str.startswith("print"), "classification"
    ]
    .str.split(" ")
    .str[-1]
)

books.loc[~books["print"].isna(), "in_out"] = books.loc[
    ~books["print"].isna(), "print"
].apply(get_in_out)
books.loc[books["print"].isna(), "in_out"] = books.loc[
    books["print"].isna(), "classification"
]

books.loc[books["in_out"] == "", "in_out"] = "?"

books["in_out"] = books["in_out"].str.split("/")

# Separate migration directions of different pages to separate rows
# Sides are in the same order as the directions produced by get_in_out.
# 
# Without a print type: "a/b" means a is left and b is right and single
# value applies to both pages.
sides_by_direction_count = {1: ["both"], 2: ["left", "right"]}
books["side"] = [
    list(print_types_dict[print_type].keys())
    if isinstance(print_type, str)
    else sides_by_direction_count.get(len(in_out), [None] * len(in_out))
    for print_type, in_out in zip(books["print"], books["in_out"])
]
books = (
    books.explode(["in_out", "side"]).reset_index().drop("index", axis="columns")
)

# Pick final columns

books = books[[
    "parish_id",
    "parish",
    "parish_normalized",
    "source",
    "year_start",
    "year_end",
    "images",
    "img_range_start",
    "img_range_end",
    "doctype",
    "date",
    "original_classification",
    "print",
    "side",
    "in_out",
    "digital",
    "free_text",
    "halfbook",
    "hand-drawn",
    "wrong",
    "empty",
    "misc",
]]

books.to_csv("outputs/csv/migration_books_metadata.csv", index=False)
print_types.to_csv("outputs/csv/print_types_metadata.csv", index=False)
