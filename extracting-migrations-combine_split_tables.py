#!/usr/bin/env python
# coding: utf-8

# In[7]:


from src.fix_split_single_table_books import (
    collect_files_to_fix,
    load_single_table_books,
    replace_with_merged
)
from tqdm.cli import tqdm

# In[8]:


INPUT_DIR = "data/migration-data-csv-release-v1/"
OUTPUT_DIR = "outputs/csv/migration-data-table_fix/"
BOOK_METADATA_PATH = "outputs/csv/migration_books_metadata.csv"

# Copy contents of `INPUT_DIR` to `OUTPUT_DIR`, then run this script to fix
# tables in `OUTPUT_DIR`.

# In[ ]:


#!rm -r $OUTPUT_DIR
#!cp -r $INPUT_DIR $OUTPUT_DIR

# In[10]:


books = load_single_table_books(BOOK_METADATA_PATH)
files_to_fix = collect_files_to_fix(books, INPUT_DIR)

# In[11]:


for page_ranges in tqdm(files_to_fix.values()):
    for in_out_files in page_ranges.values():
        for files in in_out_files.values():
            replace_with_merged(files, INPUT_DIR, OUTPUT_DIR)

# In[6]:


#!jupyter \
    #nbconvert \
    #--to script \
    #extracting-migrations-combine_split_tables.ipynb
