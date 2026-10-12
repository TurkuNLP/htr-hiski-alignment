from pathlib import Path
from bs4 import BeautifulSoup
from natsort import natsorted

def get_xml_paths(book: str, xml_data_directory: Path) -> tuple:
    """Get a sorted tuple of xml paths for book.

    Args:
        book (str): `<parish>_<doctype>_<year_start>-<year_end>_<source>`
        xml_data_directory (Path): Path to directory with XML files for
                                   all books.
    Returns:
        tuple: Sorted tuple of paths to images for book.
    """
    book_split = book.split("_")
    parish = book_split[0]
    doctype = book_split[1]
    years = book_split[2]
    source = book_split[3]

    return natsorted(
        tuple(
            xml_data_directory.rglob(
                f"*/{doctype}_{years}_{source}/pageTextClassified/*.xml"
            )
        )
    )


def get_img_paths(book: str, img_data_directory: Path) -> tuple:
    """Get a sorted tuple of jpg paths for book.

    Args:
        book (str): `<parish>_<doctype>_<year_start>-<year_end>_<source>`
        img_data_directory (Path): Path to directory with JPG files for
                                   all books.
    Returns:
        tuple: Sorted tuple of paths to images for book.
    """
    book_split = book.split("_")
    parish = book_split[0]
    doctype = book_split[1]
    years = book_split[2]
    source = book_split[3]

    return natsorted(
        tuple(img_data_directory.rglob(f"*/{doctype}_{years}_{source}/*.jpg"))
    )


def get_single_xml_path(file_stem: str, xml_data_directory: Path) -> Path:
    """Find xml file for specific page of book.

    Args:
        file_stem (str): stem of file to find.
        xml_data_directory (Path): Path to directory with XML files for
                                   all books.
    Returns:
        Path|None: `Path` to found XML file or `None` if not found
    """
    file_stem_split = file_stem.split("_")
    parish = file_stem_split[1]
    doctype = file_stem_split[2]
    years = file_stem_split[3]
    source = file_stem_split[4]
    page_number = file_stem_split[5]
    side = file_stem_split[6]
    num = file_stem_split[7]
    return next(
        xml_data_directory.rglob(
            "_".join(("autods", parish, doctype, years, source, page_number))
            + ".xml"
        ),
        None,
    )


def get_single_img_path(file_stem: str, xml_data_directory: Path) -> Path:
    """Find xml file for specific page of book.

    Args:
        file_stem (str): stem of file to find.
        xml_data_directory (Path): Path to directory with XML files for
                                   all books.
    Returns:
        Path|None: `Path` to found JPG file or `None` if not found
    """
    file_stem_split = file_stem.split("_")
    parish = file_stem_split[1]
    doctype = file_stem_split[2]
    years = file_stem_split[3]
    source = file_stem_split[4]
    page_number = file_stem_split[5]
    side = file_stem_split[6]
    num = file_stem_split[7]
    return next(
        xml_data_directory.rglob(
            "_".join(("autods", parish, doctype, years, source, page_number))
            + ".jpg"
        ),
        None,
    )


def parse_coordinate_points(coords: str):
    coords = [point.split(",") for point in coords]
    coords = [(int(point[0]), int(point[1])) for point in coords]

    return coords


def get_first_coords(xml: BeautifulSoup):
    coords = xml.find("Coords")["points"].split(" ")

    return parse_coordinate_points(coords)


def get_cell_coords(xml: BeautifulSoup):
    cell_coords = xml.find("Coords")
    coords_points = cell_coords["points"].split(" ")

    return parse_coordinate_points(coords_points)


def get_all_coords(xml: BeautifulSoup):
    coords_points = [
        coords["points"].split(" ") for coords in xml.find_all("Coords")
    ]
    coords = []
    for current_coords in coords_points:
        points = parse_coordinate_points(current_coords)
        coords.append(points)

    return coords


def get_left_right_xml_tables(xml: BeautifulSoup) -> tuple:
    """Find left and right tables from XML beautifulsoup.

    Args:
        xml (BeautifulSoup): XML to get tables from

    Returns:
        tuple: (left table, right table)
    """
    most_likely_tables = sorted(
        xml.find_all("TableRegion"), key=len, reverse=True
    )[:2]

    left_table, right_table = sorted(
        most_likely_tables,
        key=lambda x: get_first_coords(x)[0][0]
    )

    return (left_table, right_table)
