import pandas as pd


def read_csv_file(file_path):
    return pd.read_csv(file_path, encoding="utf-8")