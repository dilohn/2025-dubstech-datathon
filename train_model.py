import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
from darts.metrics import mase
from data_preprocessing import load_series
from config_utils import load_config
import pytorch_lightning as pl
                           
config = load_config()
INPUT_FILE = config['INPUT_FILE']
MODEL_DIR = config['MODEL_DIR']
SCALER_FILE = config['SCALER_FILE']
RANDOM_STATE = config['RANDOM_STATE']
FORECAST_HORIZON = config['FORECAST_HORIZON']
EPOCHS = config['EPOCHS']
INPUT_CHUNK_LENGTH = config['INPUT_CHUNK_LENGTH']
MIN_SERIES_LENGTH = config['MIN_SERIES_LENGTH']
TENSOR_CORE = config['TENSOR_CORE']
CPU_CORES = config['CPU_CORES']
BATCH_SIZE = config['BATCH_SIZE']
PRECISION = config['PRECISION']
  
torch.set_float32_matmul_precision(TENSOR_CORE)
torch.set_num_threads(CPU_CORES)
torch.set_num_interop_threads(CPU_CORES)

class MetricsLogger(pl.Callback):

    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get('train_loss')
        if loss is not None:
            self.train_losses.append(loss.cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            self.val_losses.append(val_loss.cpu().item())


def train_global_model(
    train_series_list,
    train_covariate_list,
    val_series_list=None,
    val_covariate_list=None
):
    # Filter training series
    filtered_train = [
        (ts, cov) for ts, cov in zip(train_series_list, train_covariate_list)
        if len(ts) >= MIN_SERIES_LENGTH
    ]
    print(f'Using {len(filtered_train)} time series for training.')
    if not filtered_train:
        raise ValueError('Not enough data for training')

    train_ts, train_covs = zip(*filtered_train)
    scaler = Scaler()
    scaled_train = [scaler.fit_transform(ts).astype(np.float32) for ts in train_ts]
    scaled_train_covs = [cov.astype(np.float32) for cov in train_covs]

    metrics_logger = MetricsLogger()
    pl_kwargs = {'callbacks': [metrics_logger]}
    if torch.cuda.is_available():
        pl_kwargs.update({'accelerator': 'gpu', 'devices': 1, 'precision': PRECISION})

    model = NBEATSModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=FORECAST_HORIZON,
        n_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
        pl_trainer_kwargs=pl_kwargs
    )

    fit_kwargs = {
        'series': scaled_train,
        'past_covariates': scaled_train_covs,
        'verbose': True
    }
    if val_series_list and val_covariate_list:
        # Val series long enough for one window
        scaled_val = [scaler.transform(ts).astype(np.float32) for ts in val_series_list]
        scaled_val_covs = [cov.astype(np.float32) for cov in val_covariate_list]
        fit_kwargs['val_series'] = scaled_val
        fit_kwargs['val_past_covariates'] = scaled_val_covs

    model.fit(**fit_kwargs)
    return model, scaler, metrics_logger


def save_model(model, scaler):
    model.save(MODEL_DIR)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)


def evaluate_model(model, scaler, targets, covariates,
                   n_series=5, input_len=INPUT_CHUNK_LENGTH):
    print('\nEvaluating model on a subset of series\n')
    for idx, (ts, cov) in enumerate(zip(targets[:n_series], covariates[:n_series])):
        if len(ts) <= input_len:
            print(f'Series #{idx} too short')
            continue

        train_ts = ts[:-input_len]
        cov_train = cov.slice(train_ts.start_time(), train_ts.end_time())
        actual = ts[-input_len:]

        scaled_train = scaler.fit_transform(train_ts).astype(np.float32)
        pred_scaled = model.predict(
            n=input_len,
            series=scaled_train,
            past_covariates=cov_train.astype(np.float32)
        )
        # Force pos
        pred = scaler.inverse_transform(pred_scaled)
        pred = pred.with_values(np.maximum(pred.values(), 0.0))

        y_true = actual.values().flatten()
        y_pred = pred.values().flatten()
        mae_val  = np.mean(np.abs(y_true - y_pred))
        rmse_val = np.sqrt(np.mean((y_true - y_pred)**2))
        try:
            mase_val = mase(actual, pred, train_ts)
        except:
            mase_val = np.nan

        print(
            f'Series #{idx}: MAE={mae_val:.2f}, RMSE={rmse_val:.2f}, MASE={mase_val:.2f}'
        )
    print()


def plot_metrics(metrics_logger):
    if not metrics_logger.train_losses:
        print("No training-loss history to plot.")
        return

    epochs_train = np.arange(1, len(metrics_logger.train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_train, metrics_logger.train_losses, marker='o', label='Train Loss')

    if metrics_logger.val_losses:
        epochs_val = np.arange(1, len(metrics_logger.val_losses) + 1)
        plt.plot(epochs_val, metrics_logger.val_losses, marker='s', label='Validation Loss')

    plt.title('Training and Validation Loss by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    print(f'CUDA available? {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'CPU count: {os.cpu_count()}')

    if os.path.exists(MODEL_DIR) or os.path.exists(SCALER_FILE):
        resp = input('A saved model or scaler exists. Overwrite? (Y/N): ')
        if resp.strip().lower() != 'y':
            print('Aborting.')
            return

    print('!!! Loading and preprocessing data')
    target_list, cov_list = load_series(INPUT_FILE, MIN_SERIES_LENGTH)

    # Temporal hold-out split
    val_series_list, val_cov_list = [], []
    for ts, cov in zip(target_list, cov_list):
        if len(ts) < INPUT_CHUNK_LENGTH + FORECAST_HORIZON:
            continue
        val_ts = ts[-(INPUT_CHUNK_LENGTH + FORECAST_HORIZON):]
        val_cov = cov.slice(val_ts.start_time(), val_ts.end_time())
        val_series_list.append(val_ts)
        val_cov_list.append(val_cov)
    print(f'Using {len(target_list)} train series and {len(val_series_list)} validation series')

    print('!!! Training global model')
    model, scaler, metrics_logger = train_global_model(
        target_list,
        cov_list,
        val_series_list=val_series_list,
        val_covariate_list=val_cov_list
    )

    print('!!! Saving model and scaler')
    save_model(model, scaler)

    print('!!! Evaluating on held-out slices')
    evaluate_model(model, scaler, target_list, cov_list)

    print('!!! Plotting training and validation loss curve')
    plot_metrics(metrics_logger)

if __name__ == '__main__':
    main()
