import re
import xlrd
import csv
import logging

from io import TextIOWrapper, BytesIO
from itertools import chain, islice

ROWS_TO_CHECK = 10
FILL_THRESH = .9

def schema_extract(file_id, f_bytes, ext):

    pattern = re.compile('[\W_]+', re.UNICODE)

    try:
        if ext == "csv":
            schema = csv_extract(f_bytes)
        elif ext == "tsv":
            schema = csv_extract(f_bytes, delim = "\t")
        else:
            schema = gen_spread_extract(f_bytes)

        schema = [pattern.sub('', s).lower() for s in schema]
        #print("Extracted schema: {0}".format(schema))

        return schema

    except Exception as e:
        logging.error("Exception in schema processing with file {}: {}".format(file_id, e))
        return []

def core_spread_extract(row_iter):

    for i, row in enumerate(row_iter):
        true_vals = [val for val in row if val != '']
        per_fill = len(true_vals) / len(row)
        if per_fill > 0:
            if per_fill > FILL_THRESH:
                return true_vals
            else:
                return []

def gen_spread_extract(fb):

    wb = xlrd.open_workbook(file_contents = fb)
    headers = []
    for sheet in wb.sheets():
        raw_rows = islice(sheet.get_rows(), 0, ROWS_TO_CHECK, 1)
        rows = [[c.value for c in r] for r in raw_rows]
        headers += core_spread_extract(rows)
    return headers

def csv_extract(fb, delim=","):
    reader = csv.reader(TextIOWrapper(BytesIO(fb)), delimiter=delim)
    return core_spread_extract(reader)
