import pandas as pd
import re
from pandas.tseries.holiday import USFederalHolidayCalendar
from darts import TimeSeries
from config_utils import load_config

config = load_config()
INPUT_DATE_COL = config['INPUT_DATE_COL']
NORMALIZED_DATE_COL = config['NORMALIZED_DATE_COL']
NEIGHBORHOOD_COLUMN = config['NEIGHBORHOOD_COLUMN']
NEIGHBORHOOD_EXCLUDE = config['NEIGHBORHOOD_EXCLUDE']
TEST_SERVICE_REQUEST_TYPES = config['TEST_SERVICE_REQUEST_TYPES']
MISSING_VALUE_TOKEN = config['MISSING_VALUE_TOKEN']
ZIP_CODE_COLUMN = config['ZIP_CODE_COLUMN']
ZIP_CODE_PATTERN = config['ZIP_CODE_PATTERN']
GROUP_COLUMNS = config['GROUP_COLUMNS']

ZIP_CODE_RE = re.compile(ZIP_CODE_PATTERN)

DYNAMIC_FEATURES = {
    'day_of_week': lambda idx: idx.dayofweek,
    'month': lambda idx: idx.month,
    'is_weekend': lambda idx: idx.dayofweek >= 5,
    'is_holiday': None
}

def load_series(input_file: str, min_series_length: int):
    df = pd.read_csv(input_file, parse_dates=[INPUT_DATE_COL])
    dup_mask = df.duplicated(subset=['Service Request Number'])
    print(f'Removed {dup_mask.sum()} rows with duplicate Service Request Numbers.')
    df = df.loc[~dup_mask]

    df[NORMALIZED_DATE_COL] = df[INPUT_DATE_COL].dt.normalize()

    mask_test = df['Service Request Type'].isin(TEST_SERVICE_REQUEST_TYPES)
    print(f'Removed {mask_test.sum()} rows that are for testing')
    df = df.loc[~mask_test]

    mask_ooo = df[NEIGHBORHOOD_COLUMN] == NEIGHBORHOOD_EXCLUDE
    print(f'Removed {mask_ooo.sum()} rows with Neighborhood == "{NEIGHBORHOOD_EXCLUDE}".')
    df = df.loc[~mask_ooo]

    rows_with_na = df.isna().any(axis=1).sum()
    df = df.fillna(MISSING_VALUE_TOKEN)
    print(f'Filled "{MISSING_VALUE_TOKEN}" in {rows_with_na} rows that had at least one missing value')

    for col in GROUP_COLUMNS:
        df[col] = df[col].astype('category')

    # Normalize and filter ZIP codes
    df[ZIP_CODE_COLUMN] = (
        df[ZIP_CODE_COLUMN]
          .astype(str)
          .str.extract(ZIP_CODE_RE)[0]
          .fillna('')
          .str.zfill(5)
    )
    zip_mask = df[ZIP_CODE_COLUMN].str.startswith(('981', '980'))
    removed_zip = (~zip_mask).sum()
    print(f'Removed {removed_zip} rows with ZIP codes not starting with 981 or 980.')
    df = df.loc[zip_mask]

    # Aggregate counts
    daily = (
        df.groupby(GROUP_COLUMNS + [NORMALIZED_DATE_COL], observed=True)
          .size()
          .reset_index(name='count')
    )

    # Features
    start, end = daily[NORMALIZED_DATE_COL].min(), daily[NORMALIZED_DATE_COL].max()
    full_range = pd.date_range(start, end, freq='D')
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start, end=end)

    features = pd.DataFrame(index=full_range)
    for name, func in DYNAMIC_FEATURES.items():
        if name == 'is_holiday':
            features[name] = features.index.isin(holidays)
        else:
            features[name] = func(features.index)

    # One-hot covariates
    service_types = df['Service Request Type'].cat.categories
    departments = df['City Department'].cat.categories
    zip_codes = df[ZIP_CODE_COLUMN].unique()
    static_cols = (
        [f'SRT_{s}' for s in service_types] +
        [f'Dept_{d}' for d in departments] +
        [f'ZIP_{z}' for z in zip_codes]
    )

    series_list, cov_list = [], []
    for (srt, dept, zipc), grp in daily.groupby(GROUP_COLUMNS, observed=True):
        ts_df = grp.set_index(NORMALIZED_DATE_COL)['count'].reindex(full_range, fill_value=0)
        ts = TimeSeries.from_series(ts_df)

        static_df = pd.DataFrame(0, index=full_range, columns=static_cols)
        static_df[f'SRT_{srt}'] = 1
        static_df[f'Dept_{dept}'] = 1
        static_df[f'ZIP_{zipc}'] = 1

        cov_df = pd.concat([features, static_df], axis=1)
        cov_ts = TimeSeries.from_dataframe(cov_df)

        if len(ts) >= min_series_length:
            series_list.append(ts)
            cov_list.append(cov_ts)

    return series_list, cov_list

if __name__ == '__main__':
    series, covariates = load_series('small.csv', min_series_length=120)
    print(f'Generated {len(series)} series with covariates.')
