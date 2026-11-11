import json
from orjson import loads as parse_json_line
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from plotly.colors import sample_colorscale
from pyproj import Geod, Transformer
from rapidfuzz import fuzz, process
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COUNTERPART_ROLE_BY_DIRECTION = {"out": "destination", "in": "source"}
UNSPECIFIED_ROLE = "place"

FUZZY_SCORE_CUTOFF = 100
FINLAND_CENTRE_LATITUDE = 62

# All source coordinates are EPSG:3067; plots use WGS84.
EPSG3067_TO_WGS84 = Transformer.from_crs(
    "EPSG:3067", "EPSG:4326", always_xy=True
)
WGS84_GEOD = Geod(ellps="WGS84")

ROUTE_COORDINATE_COLUMNS = [
    "origin_longitude",
    "origin_latitude",
    "target_longitude",
    "target_latitude",
]

YEAR_BINS = list(range(1820, 1926, 5))
YEAR_BIN_COLUMN = "year_bin"

# Counts are binned so each bin is one trace (a trace can only have one line colour)
COUNT_BIN_EDGES = [0, 5, 20, 50, 100, 200, np.inf]
COUNT_BIN_LABELS = [
    "(0, 5]",
    "(5, 20]",
    "(20, 50]",
    "(50, 100]",
    "(100, 200]",
    "> 200",
]
COUNT_BIN_COUNT = len(COUNT_BIN_LABELS)
COUNT_BIN_COLORS = sample_colorscale(
    "Viridis", np.linspace(0, 1, COUNT_BIN_COUNT)
)
COUNT_BIN_LINE_WIDTHS = np.linspace(0.8, 4, COUNT_BIN_COUNT)

POINTS_PER_ARC = 16
ARC_BEND = 0.15  # sideways offset as a fraction of route length; 0 gives straight lines
ARROW_POSITION = (
    0.6  # how far along the arc the arrowhead sits (0 = origin, 1 = target)
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_toponym_lookup(toponym_json_path: Path) -> dict:
    with open(toponym_json_path, "r") as toponym_file:
        return json.load(toponym_file)


def load_database_migration_tables(
    database_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (emigrated_records, immigrated_records) straight from the database."""
    with closing(sqlite3.connect(database_path)) as connection:
        emigrated_records = pd.read_sql_query(
            "SELECT * FROM emigrated", connection
        )
        immigrated_records = pd.read_sql_query(
            "SELECT * FROM immigrated", connection
        )
    return emigrated_records, immigrated_records


def load_parish_records(database_path: Path) -> pd.DataFrame:
    with closing(sqlite3.connect(database_path)) as connection:
        parish_records = pd.read_sql_query("SELECT * FROM parish", connection)

    # The database columns "lon" and "lat" actually
    # hold EPSG:3067 easting and northing (metres)
    return parish_records.rename(columns={"lon": "easting", "lat": "northing"})


def score_or_zero(toponym_with_score: tuple) -> float:
    score = toponym_with_score[1]
    return score if score is not None else 0.0


def select_counterpart_toponym(
    related_toponyms: list, direction: str
) -> tuple[str | None, str]:
    """Pick the toponym at the other end of the event from the recording parish.

    Each related toponym looks like [column_index, name, role, score], where role is
    "source", "destination" or "place".
    Returns (counterpart name or None, reason), where the reason is used for diagnostics.
    """
    counterpart_role = COUNTERPART_ROLE_BY_DIRECTION[direction]

    toponyms_by_role = defaultdict(list)
    for related_toponym in related_toponyms:
        toponym_name, role = related_toponym[1], related_toponym[2]
        score = related_toponym[3] if len(related_toponym) > 3 else None
        toponyms_by_role[role].append((toponym_name, score))

    # "out" -> destination, "in" -> source
    explicit_candidates = toponyms_by_role.get(counterpart_role)
    if explicit_candidates:
        return max(explicit_candidates, key=score_or_zero)[0], "explicit_role"

    # Otherwise an unspecified "place" is the counterpart, e.g. "out" with ["source", "place"]
    place_candidates = toponyms_by_role.get(UNSPECIFIED_ROLE, [])
    if len(place_candidates) == 1:
        return place_candidates[0][0], "single_place"
    if len(place_candidates) > 1:
        return None, "multiple_places"
    return None, "only_own_place_role"


def read_migration_events(jsonl_path: Path, skip_counts: Counter):
    """Yield one slim dict per usable migration row; the full rows are never kept in memory."""
    filename_parish_id = int(jsonl_path.name.split("_")[0])

    with open(jsonl_path, "rb") as jsonl_file:
        for jsonl_line in jsonl_file:
            skip_counts["rows_read"] += 1
            migration_record = parse_json_line(jsonl_line)

            if not migration_record.get("has_information"):
                skip_counts["skipped: no information"] += 1
                continue
            if migration_record.get("duplicate"):
                skip_counts["skipped: duplicate"] += 1
                continue

            related_toponyms = migration_record.get(
                "possibly_related_toponyms"
            )
            if isinstance(related_toponyms, str):
                related_toponyms = json.loads(related_toponyms)
            if not related_toponyms:
                skip_counts["skipped: no related toponyms"] += 1
                continue

            direction = migration_record.get("in/out")
            if direction not in COUNTERPART_ROLE_BY_DIRECTION:
                skip_counts["skipped: unknown in/out"] += 1
                continue

            counterpart_name, reason = select_counterpart_toponym(
                related_toponyms, direction
            )
            skip_counts[f"counterpart: {reason}"] += 1
            if counterpart_name is None or not str(counterpart_name).strip():
                continue

            parish_id = migration_record.get("parish_id")
            yield {
                "parish_id": (
                    parish_id if parish_id is not None else filename_parish_id
                ),
                "direction": direction,
                "year": migration_record.get("year_prediction"),
                "counterpart_name": str(counterpart_name).strip(),
                "book": migration_record.get("book"),
                "page_n": migration_record.get("page_n"),
                "file_row": migration_record.get("file_row"),
            }


def load_migration_events(
    migration_jsonl_dir: Path,
) -> tuple[pd.DataFrame, Counter]:
    """Read every *_migration_events.jsonl file. Returns (events, skip_counts)."""
    migration_jsonl_paths = sorted(
        migration_jsonl_dir.glob("*_migration_events.jsonl")
    )
    print(len(migration_jsonl_paths), "jsonl files")

    skip_counts = Counter()
    migration_event_rows = []
    for migration_jsonl_path in tqdm(migration_jsonl_paths):
        migration_event_rows.extend(
            read_migration_events(migration_jsonl_path, skip_counts)
        )

    migration_events = pd.DataFrame(migration_event_rows)
    migration_events["parish_id"] = pd.to_numeric(
        migration_events["parish_id"], errors="coerce"
    )
    migration_events["year"] = pd.to_numeric(
        migration_events["year"], errors="coerce"
    )
    return migration_events, skip_counts


def split_by_direction(
    migration_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (emigrated_records, immigrated_records)."""
    emigrated_records = (
        migration_events.loc[migration_events["direction"] == "out"]
        .drop(columns="direction")
        .rename(
            columns={"year": "dep_year", "counterpart_name": "destination"}
        )
        .reset_index(drop=True)
    )
    immigrated_records = (
        migration_events.loc[migration_events["direction"] == "in"]
        .drop(columns="direction")
        .rename(columns={"year": "arr_year", "counterpart_name": "source"})
        .reset_index(drop=True)
    )
    return emigrated_records, immigrated_records


# ---------------------------------------------------------------------------
# Toponym resolution
# ---------------------------------------------------------------------------


# The "counterpart" is the place at the other end of the event from the parish
# that recorded it: the destination for emigrations, the source for immigrations.
def get_toponym_field(toponym_entry: dict | None, field_name: str):
    return toponym_entry[field_name] if toponym_entry else None


def find_closest_toponym_keys(
    unmatched_names,
    toponym_lookup: dict,
    score_cutoff: float = FUZZY_SCORE_CUTOFF,
) -> tuple[dict, dict]:
    toponym_keys = list(toponym_lookup)
    casefolded_toponym_keys = [
        toponym_key.casefold() for toponym_key in toponym_keys
    ]

    closest_key_by_name = {}
    score_by_name = {}
    for place_name in tqdm(unmatched_names):
        best_match = process.extractOne(
            place_name.casefold(),
            casefolded_toponym_keys,
            scorer=fuzz.ratio,
            score_cutoff=score_cutoff,
        )
        if best_match is not None:
            _, match_score, key_index = best_match
            closest_key_by_name[place_name] = toponym_keys[key_index]
            score_by_name[place_name] = match_score
    return closest_key_by_name, score_by_name


def resolve_toponyms(
    place_names: pd.Series,
    toponym_lookup: dict,
    score_cutoff: float = FUZZY_SCORE_CUTOFF,
) -> pd.DataFrame:
    stripped_names = place_names.fillna("").astype(str).str.strip()

    is_exact_match = stripped_names.isin(toponym_lookup.keys())
    names_needing_fuzzy_match = stripped_names[
        ~is_exact_match & (stripped_names != "")
    ].unique()
    closest_key_by_name, score_by_name = find_closest_toponym_keys(
        names_needing_fuzzy_match, toponym_lookup, score_cutoff
    )

    matched_keys = stripped_names.where(
        is_exact_match, stripped_names.map(closest_key_by_name)
    )
    match_scores = stripped_names.map(score_by_name).mask(
        is_exact_match, 100.0
    )
    toponym_entries = matched_keys.map(toponym_lookup.get)

    return pd.DataFrame(
        {
            "counterpart_raw_name": stripped_names,
            "counterpart_is_in_dict": is_exact_match,
            "counterpart_is_fuzzy_match": matched_keys.notna()
            & ~is_exact_match,
            "counterpart_matched_key": matched_keys,
            "counterpart_match_score": match_scores,
            "counterpart_normalized_name": toponym_entries.map(
                lambda toponym_entry: get_toponym_field(
                    toponym_entry, "normalized"
                )
            ),
            "counterpart_easting": pd.to_numeric(
                toponym_entries.map(
                    lambda toponym_entry: get_toponym_field(toponym_entry, "e")
                ),
                errors="coerce",
            ),
            "counterpart_northing": pd.to_numeric(
                toponym_entries.map(
                    lambda toponym_entry: get_toponym_field(toponym_entry, "n")
                ),
                errors="coerce",
            ),
        },
        index=place_names.index,
    )


def attach_counterpart_toponyms(
    records: pd.DataFrame,
    place_column: str,
    toponym_lookup: dict,
    score_cutoff: float = FUZZY_SCORE_CUTOFF,
) -> pd.DataFrame:
    """Append the resolved counterpart columns to records ("destination" or "source")."""
    return pd.concat(
        [
            records,
            resolve_toponyms(
                records[place_column], toponym_lookup, score_cutoff
            ),
        ],
        axis=1,
    )


def normalize_toponym(
    place_name: str | None, toponym_lookup: dict
) -> str | None:
    if pd.isna(place_name):
        return None
    stripped_name = str(place_name).strip()
    toponym_entry = toponym_lookup.get(stripped_name)
    return get_toponym_field(toponym_entry, "normalized") or stripped_name


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------


def convert_to_longitude_latitude(
    eastings: pd.Series, northings: pd.Series
) -> tuple[pd.Series, pd.Series]:
    longitudes = pd.Series(np.nan, index=eastings.index)
    latitudes = pd.Series(np.nan, index=northings.index)
    has_coordinates = eastings.notna() & northings.notna()
    longitudes[has_coordinates], latitudes[has_coordinates] = (
        EPSG3067_TO_WGS84.transform(
            eastings[has_coordinates].to_numpy(),
            northings[has_coordinates].to_numpy(),
        )
    )
    return longitudes, latitudes


def build_parish_coordinates(
    parish_records: pd.DataFrame, toponym_lookup: dict
) -> pd.DataFrame:
    parish_coordinates = parish_records[
        ["id", "name", "easting", "northing"]
    ].rename(
        columns={
            "id": "parish_id",
            "name": "parish_name",
            "easting": "parish_easting",
            "northing": "parish_northing",
        }
    )
    (
        parish_coordinates["parish_longitude"],
        parish_coordinates["parish_latitude"],
    ) = convert_to_longitude_latitude(
        parish_coordinates["parish_easting"],
        parish_coordinates["parish_northing"],
    )
    parish_coordinates["parish_normalized_name"] = parish_coordinates[
        "parish_name"
    ].map(lambda parish_name: normalize_toponym(parish_name, toponym_lookup))
    return parish_coordinates


def attach_coordinates(
    records_with_counterpart: pd.DataFrame, parish_coordinates: pd.DataFrame
) -> pd.DataFrame:
    located_records = records_with_counterpart.merge(
        parish_coordinates, on="parish_id", how="left"
    )
    (
        located_records["counterpart_longitude"],
        located_records["counterpart_latitude"],
    ) = convert_to_longitude_latitude(
        located_records["counterpart_easting"],
        located_records["counterpart_northing"],
    )
    return located_records


def drop_records_missing_coordinates(
    located_records: pd.DataFrame,
) -> pd.DataFrame:
    return located_records.dropna(
        subset=[
            "counterpart_longitude",
            "counterpart_latitude",
            "parish_longitude",
            "parish_latitude",
        ]
    )


def build_migrations(
    plottable_emigrations: pd.DataFrame, plottable_immigrations: pd.DataFrame
) -> pd.DataFrame:
    """Combine both directions into origin -> target rows."""
    emigration_events = pd.DataFrame(
        {
            "kind": "emigration",
            "year": plottable_emigrations["dep_year"],
            "origin_name": plottable_emigrations["parish_normalized_name"],
            "target_name": plottable_emigrations[
                "counterpart_normalized_name"
            ],
            "origin_raw_name": plottable_emigrations["parish_name"],
            "target_raw_name": plottable_emigrations["counterpart_raw_name"],
            "origin_longitude": plottable_emigrations["parish_longitude"],
            "origin_latitude": plottable_emigrations["parish_latitude"],
            "target_longitude": plottable_emigrations["counterpart_longitude"],
            "target_latitude": plottable_emigrations["counterpart_latitude"],
        }
    )
    immigration_events = pd.DataFrame(
        {
            "kind": "immigration",
            "year": plottable_immigrations["arr_year"],
            "origin_name": plottable_immigrations[
                "counterpart_normalized_name"
            ],
            "target_name": plottable_immigrations["parish_normalized_name"],
            "origin_raw_name": plottable_immigrations["counterpart_raw_name"],
            "target_raw_name": plottable_immigrations["parish_name"],
            "origin_longitude": plottable_immigrations[
                "counterpart_longitude"
            ],
            "origin_latitude": plottable_immigrations["counterpart_latitude"],
            "target_longitude": plottable_immigrations["parish_longitude"],
            "target_latitude": plottable_immigrations["parish_latitude"],
        }
    )
    return pd.concat(
        [emigration_events, immigration_events], ignore_index=True
    )


# ---------------------------------------------------------------------------
# Static map
# ---------------------------------------------------------------------------


def plot_gradient_routes(
    map_axes,
    migration_events: pd.DataFrame,
    colormap,
    segments_per_route=30,
    min_line_width=0.3,
    max_line_width=2.5,
    alpha=0.5,
):
    routes = (
        migration_events.groupby(ROUTE_COORDINATE_COLUMNS)
        .size()
        .rename("count")
        .reset_index()
    )

    is_zero_length_route = (
        routes["origin_longitude"] == routes["target_longitude"]
    ) & (routes["origin_latitude"] == routes["target_latitude"])

    # Busy routes are drawn on top
    routes = (
        routes[~is_zero_length_route]
        .sort_values("count")
        .reset_index(drop=True)
    )

    route_origins = routes[["origin_longitude", "origin_latitude"]].to_numpy()
    route_vectors = (
        routes[["target_longitude", "target_latitude"]].to_numpy()
        - route_origins
    )

    fractions_along_route = np.linspace(0, 1, segments_per_route + 1)
    route_points = (
        route_origins[:, None, :]
        + fractions_along_route[None, :, None] * route_vectors[:, None, :]
    )
    route_segments = np.stack(
        [route_points[:, :-1], route_points[:, 1:]], axis=2
    ).reshape(-1, 2, 2)

    segment_midpoints = (
        fractions_along_route[:-1] + fractions_along_route[1:]
    ) / 2
    segment_colors = colormap(np.tile(segment_midpoints, len(routes)))
    segment_line_widths = np.repeat(
        min_line_width
        + (max_line_width - min_line_width)
        * np.sqrt(routes["count"] / routes["count"].max()),
        segments_per_route,
    )

    # capstyle="butt" prevents overlapping caps from showing up as dots
    # when alpha < 1
    map_axes.add_collection(
        LineCollection(
            route_segments,
            colors=segment_colors,
            linewidths=segment_line_widths,
            alpha=alpha,
            capstyle="butt",
        )
    )

    route_locations = pd.concat(
        [
            routes[["origin_longitude", "origin_latitude"]].set_axis(
                ["longitude", "latitude"], axis=1
            ),
            routes[["target_longitude", "target_latitude"]].set_axis(
                ["longitude", "latitude"], axis=1
            ),
        ]
    ).drop_duplicates()
    map_axes.scatter(
        route_locations["longitude"],
        route_locations["latitude"],
        s=4,
        color="0.25",
        alpha=0.6,
        zorder=3,
    )

    map_axes.set_xlabel("Longitude")
    map_axes.set_ylabel("Latitude")
    # Fix aspect ratio
    map_axes.set_aspect(1 / np.cos(np.radians(FINLAND_CENTRE_LATITUDE)))
    map_axes.autoscale()

    return routes


def draw_static_migration_map(
    migrations: pd.DataFrame, min_line_width=1, max_line_width=5
):
    """Returns (figure, map_axes, unique_routes)."""
    origin_to_target_colormap = LinearSegmentedColormap.from_list(
        "origin_target", ["tab:blue", "tab:red"]
    )

    figure, map_axes = plt.subplots(figsize=(10, 12))
    unique_routes = plot_gradient_routes(
        map_axes,
        migrations,
        origin_to_target_colormap,
        min_line_width=min_line_width,
        max_line_width=max_line_width,
    )
    map_axes.set_title(
        f"Migrations ({len(migrations)} events, {len(unique_routes)} unique routes)"
    )

    origin_target_colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=origin_to_target_colormap),
        ax=map_axes,
        ticks=[0, 1],
        shrink=0.4,
        pad=0.02,
    )
    origin_target_colorbar.ax.set_yticklabels(["origin", "target"])

    figure.tight_layout()
    return figure, map_axes, unique_routes


# ---------------------------------------------------------------------------
# Interactive map
# ---------------------------------------------------------------------------


def prepare_plot_events(
    migrations: pd.DataFrame,
    migration_kind: str | None = None,
    year_bins: list | None = YEAR_BINS,
) -> pd.DataFrame:
    """Keep one kind of migration with valid years and coordinates.

    With year_bins=None every year is its own frame; otherwise years are binned.
    """
    if migration_kind != None:
        kind_migrations = migrations[migrations["kind"] == migration_kind]
    else:
        kind_migrations = migrations

    plot_events = kind_migrations.dropna(
        subset=["year"] + ROUTE_COORDINATE_COLUMNS
    ).copy()
    plot_events = plot_events[
        np.isfinite(plot_events[ROUTE_COORDINATE_COLUMNS]).all(axis=1)
    ]
    plot_events["year"] = plot_events["year"].astype(int)
    plot_events = plot_events[plot_events["year"] > 0].copy()

    if year_bins is None:
        plot_events[YEAR_BIN_COLUMN] = plot_events["year"]
    else:
        plot_events[YEAR_BIN_COLUMN] = pd.cut(
            plot_events["year"],
            bins=year_bins,
            right=False,
            labels=year_bins[:-1],
        )
    return plot_events


def build_yearly_routes(plot_events: pd.DataFrame) -> pd.DataFrame:
    """Count events per (year bin, route) and assign each route to a count bin."""
    yearly_routes = (
        plot_events.groupby(
            [YEAR_BIN_COLUMN] + ROUTE_COORDINATE_COLUMNS, observed=True
        )
        .size()
        .reset_index(name="count")
    )
    is_zero_length_route = (
        yearly_routes["origin_longitude"] == yearly_routes["target_longitude"]
    ) & (yearly_routes["origin_latitude"] == yearly_routes["target_latitude"])
    yearly_routes = yearly_routes[~is_zero_length_route].copy()

    yearly_routes["count_bin"] = pd.cut(
        yearly_routes["count"], bins=COUNT_BIN_EDGES, labels=False
    )
    return yearly_routes


def compute_arcs(routes: pd.DataFrame) -> np.ndarray:
    """
    Quadratic Bezier arcs that bend to the right of the direction of travel.
    Shape: (routes, points, 2).
    """
    # Work in a space where one unit of longitude is as long as one unit of latitude
    longitude_scale = np.cos(np.radians(FINLAND_CENTRE_LATITUDE))
    origins = np.column_stack(
        [
            routes["origin_longitude"] * longitude_scale,
            routes["origin_latitude"],
        ]
    )
    targets = np.column_stack(
        [
            routes["target_longitude"] * longitude_scale,
            routes["target_latitude"],
        ]
    )
    route_vectors = targets - origins
    right_hand_normals = np.column_stack(
        [route_vectors[:, 1], -route_vectors[:, 0]]
    )
    control_points = (origins + targets) / 2 + ARC_BEND * right_hand_normals

    fractions = np.linspace(0, 1, POINTS_PER_ARC)[None, :, None]
    arcs = (
        (1 - fractions) ** 2 * origins[:, None, :]
        + 2 * (1 - fractions) * fractions * control_points[:, None, :]
        + fractions**2 * targets[:, None, :]
    )
    arcs[:, :, 0] /= longitude_scale
    return arcs


def build_route_traces(
    yearly_routes: pd.DataFrame,
    year: int,
    count_bin_labels: list = COUNT_BIN_LABELS,
) -> list:
    routes_in_year = yearly_routes[yearly_routes[YEAR_BIN_COLUMN] == year]
    arrow_point_index = int(ARROW_POSITION * (POINTS_PER_ARC - 1))
    points_with_gap = (
        POINTS_PER_ARC + 1
    )  # one None per route separates the lines

    route_traces = []
    for bin_index in range(COUNT_BIN_COUNT):
        bin_routes = routes_in_year[routes_in_year["count_bin"] == bin_index]
        arcs = compute_arcs(bin_routes)
        route_count = len(bin_routes)

        longitude_path = np.full(
            (route_count, points_with_gap), None, dtype=object
        )
        latitude_path = np.full(
            (route_count, points_with_gap), None, dtype=object
        )
        longitude_path[:, :POINTS_PER_ARC] = arcs[:, :, 0]
        latitude_path[:, :POINTS_PER_ARC] = arcs[:, :, 1]

        # Only one point per route gets a visible marker: the arrowhead
        arrow_sizes = np.zeros((route_count, points_with_gap))
        arrow_sizes[:, arrow_point_index] = 6 + 2 * bin_index

        route_traces.append(
            go.Scattergeo(
                lon=longitude_path.ravel(),
                lat=latitude_path.ravel(),
                mode="lines+markers",
                line=dict(
                    width=COUNT_BIN_LINE_WIDTHS[bin_index],
                    color=COUNT_BIN_COLORS[bin_index],
                ),
                marker=dict(
                    symbol="arrow",
                    size=arrow_sizes.ravel(),
                    angleref="previous",
                    color=COUNT_BIN_COLORS[bin_index],
                ),
                opacity=0.6,
                name=count_bin_labels[bin_index],
                legendgroup=count_bin_labels[bin_index],
                hoverinfo="skip",
            )
        )
    return route_traces


def build_colorbar_trace(
    count_bin_labels: list = COUNT_BIN_LABELS,
) -> go.Scattergeo:
    """Dummy trace that provides the discrete colorbar."""
    discrete_colorscale = []
    for bin_index, bin_color in enumerate(COUNT_BIN_COLORS):
        discrete_colorscale += [
            [bin_index / COUNT_BIN_COUNT, bin_color],
            [(bin_index + 1) / COUNT_BIN_COUNT, bin_color],
        ]

    return go.Scattergeo(
        lon=[None],
        lat=[None],
        mode="markers",
        marker=dict(
            color=[0],
            colorscale=discrete_colorscale,
            cmin=0,
            cmax=COUNT_BIN_COUNT,
            showscale=True,
            opacity=0,
            size=0.001,
            colorbar=dict(
                title="Migrations per pair\n",
                tickvals=[
                    bin_index + 0.5 for bin_index in range(COUNT_BIN_COUNT)
                ],
                ticktext=count_bin_labels,
            ),
        ),
        hoverinfo="skip",
        showlegend=False,
    )


def build_migration_figure(
    yearly_routes: pd.DataFrame,
    migration_kind: str,
    source_description: str = "",
    count_bin_labels: list = COUNT_BIN_LABELS,
) -> go.Figure:
    if len(count_bin_labels) != COUNT_BIN_COUNT:
        raise ValueError(f"count_bin_labels needs {COUNT_BIN_COUNT} entries")

    years = sorted(yearly_routes[YEAR_BIN_COLUMN].unique())

    animation_frames = [
        go.Frame(
            data=build_route_traces(yearly_routes, year, count_bin_labels),
            name=str(year),
            traces=list(range(COUNT_BIN_COUNT)),
        )
        for year in years
    ]

    migration_figure = go.Figure(
        data=build_route_traces(yearly_routes, years[0], count_bin_labels)
        + [build_colorbar_trace(count_bin_labels)],
        frames=animation_frames,
    )

    migration_figure.update_geos(
        projection_type="transverse mercator",
        projection_rotation_lon=27,
        lonaxis_range=[19, 32],
        lataxis_range=[59.5, 70.2],
        resolution=50,
        showland=True,
        landcolor="#f5f2e9",
        showocean=True,
        oceancolor="#dfeaf2",
        showlakes=True,
        lakecolor="#dfeaf2",
        showcountries=True,
        countrycolor="gray",
        showcoastlines=True,
        coastlinecolor="gray",
        showframe=True,
        framecolor="black",
    )

    migration_dir_in_title = (
        "migration" if migration_kind is None else migration_kind
    )
    title = (
        f"{migration_dir_in_title} between parishes in Finland "
        + "({source_description})"
    )
    migration_figure.update_layout(
        width=900,
        height=1000,
        title=title,
        title_x=0.5,
        showlegend=False,
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [frame.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": frame.name,
                        "method": "animate",
                    }
                    for frame in animation_frames
                ],
                "currentvalue": {"prefix": "Year: "},
            }
        ],
    )
    return migration_figure


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_distances_in_metres(migrations: pd.DataFrame) -> np.ndarray:
    # Geod.inv expects (lons1, lats1, lons2, lats2)
    _, _, distances_in_metres = WGS84_GEOD.inv(
        migrations["origin_longitude"].to_numpy(),
        migrations["origin_latitude"].to_numpy(),
        migrations["target_longitude"].to_numpy(),
        migrations["target_latitude"].to_numpy(),
    )
    return distances_in_metres


def summarize_toponym_coverage(located_records: pd.DataFrame) -> pd.DataFrame:
    has_place_name = located_records["counterpart_raw_name"] != ""
    is_resolved = located_records["counterpart_matched_key"].notna()
    has_coordinates = (
        located_records["counterpart_easting"].notna()
        & located_records["counterpart_northing"].notna()
    )

    return pd.DataFrame.from_dict(
        {
            "count": {
                "rows": len(located_records),
                "empty place name": (~has_place_name).sum(),
                "exact match": located_records["counterpart_is_in_dict"].sum(),
                "fuzzy match": located_records[
                    "counterpart_is_fuzzy_match"
                ].sum(),
                "no match": (has_place_name & ~is_resolved).sum(),
                "matched, but no coordinates": (
                    is_resolved & ~has_coordinates
                ).sum(),
                "matched, with coordinates": has_coordinates.sum(),
            }
        }
    )


def count_fuzzy_matches(located_records: pd.DataFrame) -> pd.Series:
    return (
        located_records.loc[located_records["counterpart_is_fuzzy_match"]]
        .groupby(
            [
                "counterpart_raw_name",
                "counterpart_matched_key",
                "counterpart_normalized_name",
                "counterpart_match_score",
            ]
        )
        .size()
        .sort_values(ascending=False)
    )


def count_unmatched_names(located_records: pd.DataFrame) -> pd.Series:
    is_unmatched = (
        located_records["counterpart_raw_name"] != ""
    ) & located_records["counterpart_matched_key"].isna()
    return located_records.loc[
        is_unmatched, "counterpart_raw_name"
    ].value_counts()
