import pandas as pd

def clean_up_corrected_year(year:str):
    parts = year.split(" ")
    parts = ["".join(filter(str.isnumeric, part)) for part in parts]
    parts = filter(lambda part: len(part) == 4, parts)
    output = next(parts, 0)
    return int(output)

def check_is_valid(year:int, valid_years_range:tuple=(1000, 2000)):
    return year in range(*valid_years_range)


def pick_year(year_predictions_df:pd.DataFrame):
    return (
        year_predictions_df["corrected"]
        .fillna("")
        .astype(str)
        .str.split("#")
        .apply(
            lambda x: next(
                (int(cleaned_up) for year in x if (
                    check_is_valid(
                        (cleaned_up := clean_up_corrected_year(year))
                    )
                )), 0,
            )
        )
    )


def fill_years(year_predictions_df:pd.DataFrame, limit=2):
    known = year_predictions_df["year_prediction"].where(
        year_predictions_df["year_prediction"] != 0
    )
    by_book = known.groupby(year_predictions_df["book"])
    prev_val, next_val = by_book.ffill(), by_book.bfill()
    fillable = known.isna() & (prev_val == next_val)
    filled = known.mask(fillable, prev_val)
    by_book_filled = filled.groupby(year_predictions_df["book"])
    year_predictions_df["year_prediction_filled"] = (
        by_book_filled.bfill(limit=limit)
        .groupby(year_predictions_df["book"])
        .ffill(limit=limit)
        .fillna(0)
        .astype(int)
    )
 
    return year_predictions_df