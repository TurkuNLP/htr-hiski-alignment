import csv
import numpy as np

def read_files(files_sorted: list[str]):
    ocr_output_pages = []
    ocr_output_names = []
    for file_name in files_sorted:
        with open(file_name, "r") as file:
            reader = csv.reader(file, delimiter=",")

            page_rows = []
            col_total_lens = []
            for row in reader:
                page_rows.append([word.strip(' "') for word in row[:-1]])

                col_total_lens.append([len(col) for col in row[:-1]])

            ocr_output_pages.append(page_rows)

        name_col = np.argmax(
            np.bincount([np.argmax(lens) for lens in col_total_lens])
        )
        ocr_output_names.append([r[name_col] for r in page_rows])

    return ocr_output_pages, ocr_output_names
