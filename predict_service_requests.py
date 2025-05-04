import os
import pickle
import numpy as np
import torch
from darts import TimeSeries
from darts.models import NBEATSModel
from data_preprocessing import load_series
from config_utils import load_config
import pytorch_lightning as pl

class MetricsLogger(pl.Callback):
    pass

config = load_config()
MODEL_DIR         = config['MODEL_DIR']
SCALER_FILE       = config['SCALER_FILE']
DATA_FILE         = config['DATA_FILE']
FORECAST_HORIZON  = config['FORECAST_HORIZON']
MIN_SERIES_LENGTH = config['MIN_SERIES_LENGTH']
INPUT_CHUNK_LENGTH= config['INPUT_CHUNK_LENGTH']
TENSOR_CORE       = config['TENSOR_CORE']
CPU_CORES         = config['CPU_CORES']

torch.set_float32_matmul_precision(TENSOR_CORE)
torch.set_num_threads(CPU_CORES)
torch.set_num_interop_threads(CPU_CORES)

def load_model(model_dir: str, scaler_file: str):
    model = NBEATSModel.load(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.eval()
    model.model.to(device)
    model.pl_trainer_kwargs = {}
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def predict(model, scaler, series: TimeSeries, covariates: TimeSeries = None,
            horizon: int = FORECAST_HORIZON) -> TimeSeries:
    series = series.astype(np.float32)
    scaled_series = scaler.transform(series).astype(np.float32)

    past_cov = None
    if covariates is not None:
        cov = covariates.astype(np.float32)
        past_cov = cov.slice(series.start_time(), series.end_time())

    scaled_pred = model.predict(
        n=horizon,
        series=scaled_series,
        past_covariates=past_cov,
        verbose=False
    )
    pred = scaler.inverse_transform(scaled_pred)
    pred = pred.with_values(np.maximum(pred.values(), 0.0))
    return pred


def summarize_and_print(forecast: TimeSeries, label: str):
    vals = forecast.values().flatten()
    total = vals.sum()
    print(f"\n{label} | Next {FORECAST_HORIZON} days total: {total:.0f}\n")
    print(forecast.to_dataframe())


def get_unique_keys(cov_list, prefix: str):
    keys = set()
    for cov_ts in cov_list:
        df = cov_ts.to_dataframe()
        for col in df.columns:
            if col.startswith(prefix) and df[col].iloc[0] == 1:
                keys.add(col[len(prefix):])
    return sorted(keys)


def select_option(options, prompt: str):
    if not options:
        print(f'No options available for {prompt}.')
        return None
    print(f"\n{prompt}:")
    for i, opt in enumerate(options, 1):
        print(f'  {i}) {opt}')
    while True:
        choice = input(f'Select an option (1-{len(options)}): ').strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print('Invalid choice')


def main():
    model, scaler = load_model(MODEL_DIR, SCALER_FILE)
    series_list, cov_list = load_series(DATA_FILE, MIN_SERIES_LENGTH)

    srt_options  = get_unique_keys(cov_list, 'SRT_')
    dept_options = get_unique_keys(cov_list, 'Dept_')

    print('\nWhat type of prediction would you like?')
    print('  1) Service Request Type')
    print('  2) Department')
    print('  3) ZIP and Service Request Type')
    choice = input('Enter 1, 2, or 3: ').strip()

    if choice == '1':
        sel = select_option(srt_options, 'Service Request Types')
        if sel:
            key = f'SRT_{sel}'
            idxs = [i for i, cov in enumerate(cov_list)
                    if cov.to_dataframe()[key].iloc[0] == 1]
            if not idxs:
                print(f'No series for Service Request Type="{sel}"')
            else:
                ts, cov_ts = series_list[idxs[0]], cov_list[idxs[0]]
                fc = predict(model, scaler, ts, cov_ts)
                summarize_and_print(fc, f'Service Request Type="{sel}"')

    elif choice == '2':
        sel = select_option(dept_options, 'Departments')
        if sel:
            key = f'Dept_{sel}'
            idxs = [i for i, cov in enumerate(cov_list)
                    if cov.to_dataframe()[key].iloc[0] == 1]
            if not idxs:
                print(f'No series for Department="{sel}"')
            else:
                ts, cov_ts = series_list[idxs[0]], cov_list[idxs[0]]
                fc = predict(model, scaler, ts, cov_ts)
                summarize_and_print(fc, f'Department="{sel}"')

    elif choice == '3':
        sel_srt = select_option(srt_options, 'Service Request Types')
        if sel_srt:
            key_srt = f'SRT_{sel_srt}'
            zips = {col[len('ZIP_'):]
                    for cov in cov_list
                    for col, v in cov.to_dataframe().iloc[0].items()
                    if col.startswith('ZIP_') and cov.to_dataframe()[key_srt].iloc[0] == 1 and v == 1}
            sel_zip = select_option(sorted(zips), 'ZIP Codes')
            if sel_zip:
                key_zip = f'ZIP_{sel_zip}'
                idxs = [i for i, cov in enumerate(cov_list)
                        if cov.to_dataframe().iloc[0].get(key_srt, 0) == 1
                        and cov.to_dataframe().iloc[0].get(key_zip, 0) == 1]
                if not idxs:
                    print(f'No series for SRT="{sel_srt}" & ZIP="{sel_zip}"')
                else:
                    ts, cov_ts = series_list[idxs[0]], cov_list[idxs[0]]
                    fc = predict(model, scaler, ts, cov_ts)
                    summarize_and_print(fc, f'SRT="{sel_srt}", ZIP="{sel_zip}"')

    else:
        print('Invalid selection. Please run again and choose 1, 2, or 3.')


if __name__ == '__main__':
    main()
