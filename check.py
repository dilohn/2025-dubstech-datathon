import pandas as pd
from config_utils import load_config


config = load_config()
COLUMNS_OF_INTEREST = config['COLUMNS_OF_INTEREST']
DEFAULT_CSV_PATH = config['DEFAULT_CSV_PATH']
MISSING_VALUE_METHOD = config['MISSING_VALUE_METHOD']


def print_unique_and_missing(csv_file_path: str):
    try:
        df = pd.read_csv(csv_file_path)

        for column in COLUMNS_OF_INTEREST:
            print(f'\nUnique items in "{column}":')
            if column in df.columns:
                unique_vals = df[column].dropna().unique()
                for item in unique_vals:
                    print(f'- {item}')
            else:
                print(f'Warning: Column "{column}" not found in the CSV.')

        total_rows = len(df)
        if MISSING_VALUE_METHOD == 'isna':
            rows_with_missing = df.isna().any(axis=1).sum()
        else:
            rows_with_missing = df.isnull().any(axis=1).sum()
        print(f'\nTotal rows: {total_rows}')
        print(f'Rows with at least one missing value: {rows_with_missing}')

    except Exception as e:
        print(f'An error occurred: {e}')


if __name__ == '__main__':
    print_unique_and_missing(DEFAULT_CSV_PATH)
