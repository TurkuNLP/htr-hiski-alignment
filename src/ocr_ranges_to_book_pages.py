def ocr_ranges_to_book_pages(
    ocr_page_lens, ocr_ranges, files
):
    page_row_ranges = []
    first_row_of_page = 0
    for i in range(len(ocr_page_lens)):
        page_row_ranges.append(
            (first_row_of_page, first_row_of_page + ocr_page_lens[i])
        )
        first_row_of_page += ocr_page_lens[i]

    ocr_range_to_book_page = {}
    for ocr_range in ocr_ranges:
        ocr_range_to_book_page[ocr_range] = []
        for i in range(len(page_row_ranges)):
            if (
                ocr_range[1] >= page_row_ranges[i][0]
                and ocr_range[0] <= page_row_ranges[i][1]
            ):
                ocr_range_to_book_page[ocr_range].append(files[i])

    return ocr_range_to_book_page
