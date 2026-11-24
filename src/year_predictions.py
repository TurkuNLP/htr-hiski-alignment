from typing import Iterable, Optional, Sequence

import pandas as pd


def clean_up_corrected_year(year: str) -> int:
    """
    Extract a four-digit year from a free-form string.

    Year string is split on spaces. Parts containing ``/`` (day/month
    dates) are discarded. Non-decimal characters are removed from the
    remaining parts. First part is picked and returned.

    Parameters
    ----------
    year : str
        Raw year text, e.g. ``"1907."`` or ``"3/4 1907"``.

    Returns
    -------
    int
        The first four-digit part or 0 if none exists.
    """
    parts = year.split(" ")
    parts = [
        "".join(filter(str.isdecimal, part))
        for part in parts
        if "/" not in part # filter out day/month
    ]
    parts = filter(lambda part: len(part) == 4, parts)
    output = next(parts, 0)
    return int(output)


def check_is_valid(year: int, valid_years_range: Sequence[int]) -> bool:
    return valid_years_range[0] <= year and year <= valid_years_range[1]


def first_valid_year_in(
    value: str, year_range: Sequence[int] = (1000, 2000), leeway: int = 0
) -> int:
    """
    Return the first valid year in a ``#``-separated string of candidates.

    Parameters
    ----------
    value : str
        Candidate year string.
    year_range : Sequence[int], default (1000, 2000)
        Inclusive ``(start, end)`` bounds for a year.
    leeway : int, default 0
        Number of years by which the bounds are widened on both sides.

    Returns
    -------
    int
        The first candidate within bounds or 0 if none exists.
    """
    min_year = year_range[0] - leeway
    max_year = year_range[1] + leeway
    for year in value.split("#"):
        cleaned_up = clean_up_corrected_year(year)
        if check_is_valid(cleaned_up, valid_years_range=(min_year, max_year)):
            return int(cleaned_up)

    return 0


def first_valid_year(
    values: pd.Series,
    year_ranges: Iterable[Sequence[int]],
    leeway: int = 0,
) -> pd.Series:
    """
    Apply `first_valid_year_in` to each value using its own year range.

    Parameters
    ----------
    values : pandas.Series
        ``#``-separated candidate years. Missing values are treated as
        empty.
    year_ranges : Iterable[Sequence[int]]
        ``(start, end)`` bounds, one per value, in the same order as
        `values`.
    leeway : int, default 0
        Number of years by which each range is widened on both sides.

    Returns
    -------
    pandas.Series
        Picked year per value, 0 where none is valid. Indexed like
        `values`.
    """
    return pd.Series(
        [
            first_valid_year_in(value, year_range, leeway=leeway)
            for value, year_range in zip(
                values.fillna("").astype(str), year_ranges
            )
        ],
        index=values.index,
    )


def pick_year(
    year_predictions: pd.DataFrame,
    column_to_pick_from: str,
    fallback_column: Optional[str] = None,
    year_range_column: str = "year_range",
) -> pd.Series:
    """
    Pick a year for each row, optionally falling back to a second column.

    Years are first taken from `column_to_pick_from`, accepting values up
    to 100 years outside the row's year range. Rows still at 0 are then
    taken from `fallback_column` with a stricter leeway of 20 years.

    Parameters
    ----------
    year_predictions : pandas.DataFrame
        Frame containing the columns below.
    column_to_pick_from : str
        Column of ``#``-separated candidate years.
    fallback_column : str, optional
        Column used for rows where the main column yields no year.
    year_range_column : str, default "year_range"
        Column holding each row's range as ``"start-end"``.

    Returns
    -------
    pandas.Series
        Picked year per row, 0 where none was found.
    """
    picked = first_valid_year(
        year_predictions[column_to_pick_from],
        year_ranges=year_predictions[year_range_column].map(
            lambda year_range: [int(year) for year in year_range.split("-")]
        ),
        # pick year from the main col almost always when it exists,
        # even if not exactly in range
        leeway=100
    )
    if fallback_column is None:
        return picked

    missing = picked == 0
    picked.loc[missing] = first_valid_year(
        year_predictions.loc[missing, fallback_column],
        year_ranges=year_predictions.loc[missing, year_range_column].map(
            lambda year_range: [int(year) for year in year_range.split("-")]
        ),
        # be stricter when picking year from fallback
        leeway=20
    )
    return picked


def fill_years(
    year_predictions_df: pd.DataFrame,
    column_to_fill_from: str = "year",
    column_to_group_by: str = "book",
    column_to_read_range_from: str = "year_range",
    limit: int = 2,
) -> pd.Series:
    """
    Fill unknown (0) years using neighboring rows of the same book.

    Rows must be in reading order within each book. The filling is done
    in three steps:

    1. An unknown year on the first or last row of a book is set to the
       start or end of the book's year range.
    2. A run of unknown years between two equal known years is filled
       with that year.
    3. Remaining unknown years are filled from the nearest known year at
       most `limit` rows away in either direction.

    Fills never cross book boundaries.

    Parameters
    ----------
    year_predictions_df : pandas.DataFrame
        Frame with a unique index.
    column_to_fill_from : str, default "year"
        Column of integer years where 0 means unknown.
    column_to_group_by : str, default "book"
        Column identifying the book of each row.
    column_to_read_range_from : str, default "year_range"
        Column holding each book's range as ``"start-end"``.
    limit : int, default 2
        Maximum number of consecutive rows filled in step 3.

    Returns
    -------
    pandas.Series
        Filled years as integers, 0 where still unknown. Indexed like
        `year_predictions_df`.
    """
    year = year_predictions_df[column_to_fill_from].copy()
    book_rows = year_predictions_df.groupby(column_to_group_by)
    for idx, part in (
        (book_rows.head(1).index, 0),
        (book_rows.tail(1).index, 1),
    ):
        bounds = (
            year_predictions_df.loc[idx, column_to_read_range_from]
            .str.split("-")
            .str[part]
            .astype(int)
        )
        year.loc[idx] = year.loc[idx].where(year.loc[idx] != 0, bounds)
    known = year.where(year != 0)
    by_book = known.groupby(year_predictions_df[column_to_group_by])
    prev_val, next_val = by_book.ffill(), by_book.bfill()
    fillable = known.isna() & (prev_val == next_val)
    filled = known.mask(fillable, prev_val)
    by_book_filled = filled.groupby(year_predictions_df[column_to_group_by])
    return (
        by_book_filled
        .ffill(limit=limit).groupby(year_predictions_df[column_to_group_by])
        .bfill(limit=limit)
        .fillna(0)
        .astype(int)
    )
