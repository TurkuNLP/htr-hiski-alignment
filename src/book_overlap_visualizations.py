from matplotlib import pyplot as plt
import numpy as np
from rapidfuzz import fuzz
import seaborn as sns

from src.similarity_matrix import create_sim_m, create_sim_m_heatmap


def migration_duplicate_filtering_before_after_counts(
    df, years, in_out_filter: list = None
):
    year_mask = df["year_prediction"].isin(range(*years))
    if in_out_filter is None:
        df_years = df[year_mask]
    else:
        df_years = df[year_mask & (df["in/out"].isin(in_out_filter))]

    counts_before = df_years.groupby("book").size()
    counts_after = df_years[~df_years["duplicate"]].groupby("book").size()

    counts = {}
    for book in counts_before.index:
        before = counts_before[book]
        after = counts_after.get(book, 0)
        counts[book] = {"before": int(before), "after": int(after)}

    return counts


def plot_migration_similarity_heatmaps(
    df, parish, years, scorer=fuzz.ratio, vmax=100, in_out_filter: list = None
):
    year_mask = df["year_prediction"].isin(range(*years))
    if in_out_filter is None:
        df_years = df[year_mask]
    else:
        df_years = df[year_mask & (df["in/out"].isin(in_out_filter))]

    pages = df_years.groupby(["book", "page_n"], sort=False)

    page_keys = list(pages.groups.keys())
    page_sort_key = [
        (pages.get_group(k)["year_prediction"].min(), k[1]) for k in page_keys
    ]
    page_keys = [k for _, k in sorted(zip(page_sort_key, page_keys))]

    row_data_per_page = [
        ",".join(pages.get_group(k)["row_data"].sum()) for k in page_keys
    ]
    duplicate_per_page = [
        pages.get_group(k)["duplicate"].any() for k in page_keys
    ]

    sim_m = create_sim_m(
        row_data_per_page,
        row_data_per_page,
        scorer,
        join_char="",
        n_workers=-1,
    )[0]

    sim_m_zeroed = sim_m.copy()
    dup_idx = [i for i, is_dup in enumerate(duplicate_per_page) if is_dup]
    sim_m_zeroed[dup_idx, :] = 0
    sim_m_zeroed[:, dup_idx] = 0

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle(f"{parish}, year range: {years}")

    create_sim_m_heatmap(
        sim_m,
        xlabel="",
        ylabel="",
        ax=axs[0],
        cbar=False,
        vmin=0,
        vmax=vmax,
    )
    axs[0].set_title("No duplicate filtering")

    create_sim_m_heatmap(
        sim_m,
        xlabel="",
        ylabel="",
        ax=axs[1],
        cbar=False,
        vmin=0,
        vmax=vmax,
        cmap=sns.cm.crest_r,
    )
    axs[1].set_title("Duplicate filtering")

    for i, is_dup in enumerate(duplicate_per_page):
        if is_dup:
            axs[1].axhspan(i, i + 1, color="red", alpha=0.3, zorder=2, lw=0)
            axs[1].axvspan(i, i + 1, color="red", alpha=0.3, zorder=2, lw=0)

    create_sim_m_heatmap(
        sim_m_zeroed,
        xlabel="",
        ylabel="",
        ax=axs[2],
        cbar=False,
        vmin=0,
        vmax=vmax,
    )
    axs[2].set_title("Duplicate areas zeroed")

    return fig, axs


def plot_book_year_event_heatmap(df, parish, years):
    year_mask = df["year_prediction"].isin(range(*years))
    df_years = df[year_mask]

    year_range = list(range(*years))
    row_keys = sorted(
        df_years[["book", "in/out"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    row_idx = {k: i for i, k in enumerate(row_keys)}
    year_idx = {y: i for i, y in enumerate(year_range)}

    count_matrix = np.full((len(row_keys), len(year_range)), np.nan)
    dup_matrix = np.zeros((len(row_keys), len(year_range)), dtype=bool)

    grouped = df_years.groupby(["book", "in/out", "year_prediction"])
    counts = grouped.size()
    dup_flags = grouped["duplicate"].any()

    for (book, in_out, year), count in counts.items():
        if year in year_idx:
            count_matrix[row_idx[(book, in_out)], year_idx[year]] = count

    for (book, in_out, year), is_dup in dup_flags.items():
        if year in year_idx and is_dup:
            dup_matrix[row_idx[(book, in_out)], year_idx[year]] = True

    fig, ax = plt.subplots(figsize=(30, 5))
    fig.suptitle(f"{parish}, year range: {years}")

    masked = np.ma.masked_invalid(count_matrix)
    mesh = ax.pcolormesh(
        masked, cmap="viridis", edgecolors="grey", linewidth=0.5
    )
    fig.colorbar(mesh, ax=ax, label="Event count")

    for i in range(len(row_keys)):
        for j in range(len(year_range)):
            if dup_matrix[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j, i),
                        1,
                        1,
                        fill=False,
                        hatch="//",
                        edgecolor="red",
                        linewidth=1.5,
                    )
                )

    ax.set_xticks(np.arange(len(year_range)) + 0.5)
    ax.set_xticklabels(year_range, rotation=90)
    ax.set_yticks(np.arange(len(row_keys)) + 0.5)
    ax.set_yticklabels([f"{book} ({in_out})" for book, in_out in row_keys])
    ax.set_xlabel("Year")
    ax.set_ylabel("book (in/out)")
    ax.set_aspect("equal")
    ax.invert_yaxis()

    return fig, ax


def plot_non_duplicate_migration_histogram(df, in_out_filter: list = None):
    year_non_zero_mask = df["year_prediction"] != 0
    if in_out_filter is None:
        df_filtered = df[~df["duplicate"] & year_non_zero_mask][
            "year_prediction"
        ]
    else:
        df_filtered = df[
            ~df["duplicate"]
            & (df["in/out"].isin(in_out_filter))
            & year_non_zero_mask
        ]["year_prediction"]

    year_range = range(df_filtered.min(), df_filtered.max() + 1)

    fig, ax = plt.subplots(figsize=(30, 5))
    sns.histplot(df_filtered, bins=year_range)
    ax.set_xticks(year_range)
    ax.set_xticklabels(year_range, rotation=90)
