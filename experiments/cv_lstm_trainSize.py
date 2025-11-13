import os
import gc
import pickle
import argparse
from datetime import datetime

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, AdamW
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from scipy.stats import pearsonr
import optuna

from models.lstm import LSTMForecaster
from experiments.train_utils import train_model, objective_generic

print('Start running script')
print('Starting time: ' + str(datetime.now().time()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'The current device is {device}.')

parser = argparse.ArgumentParser(description='Arguments for running script')
parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
parser.add_argument('--train_size', type=str, required=False, help='Comma-separated training sizes (overrides config)')
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)['cv_lstm_trainSize']

# IO / data
data_path = cfg['data']
target_path = cfg['target']
time_seq_path = cfg['time_seq']
optim_size = int(cfg['optim_size'])
val_size = int(cfg['val_size'])
res_dir = cfg['res_dir']
tmp_dir = cfg['tmp_dir']
suffix = cfg['suffix']

# training / BO
batch_size = int(cfg.get('batch_size', 16))
n_trials = int(cfg.get('n_trials', 50))
n_resamples = int(cfg.get('n_resamples', 5))
epochs = int(cfg.get('epochs', 50))
patience = int(cfg.get('patience', 5))
best_params_path = cfg.get('best_params', None)
train_sizes = list(map(int, (args.train_size or cfg.get('train_size', '50')).split(',')))
num_classes = cfg.get('num_classes', None)

def get_criterion(num_classes):
    if num_classes is None:
        return MSELoss()
    elif num_classes == 2:
        return BCEWithLogitsLoss()
    else:
        return CrossEntropyLoss()

print("Load data")
X = torch.load(data_path)
y = torch.load(target_path)
time_sequence = torch.load(time_seq_path) - cfg.get('time_seq_offset', 1)

# reproducible index selection
torch.manual_seed(int(cfg.get('seed', 93)))
indices_all = torch.randperm(X.size(0))
optim_indices = indices_all[:optim_size]
main_indices = indices_all[optim_size:]

# split optimization subset 80/20
optim_train_N = int(0.8 * len(optim_indices))
optim_train_idx = optim_indices[:optim_train_N]
optim_val_idx = optim_indices[optim_train_N:]

optim_train_loader = DataLoader(
    TensorDataset(X[optim_train_idx], time_sequence[optim_train_idx], y[optim_train_idx]),
    batch_size=batch_size, shuffle=True
)
optim_val_loader = DataLoader(
    TensorDataset(X[optim_val_idx], time_sequence[optim_val_idx], y[optim_val_idx]),
    batch_size=batch_size, shuffle=True
)

input_dim = X.size(2)
cleaned_suffix = suffix.replace(" ", "_")
os.makedirs(tmp_dir, exist_ok=True)
best_loss_path = os.path.join(tmp_dir, f"{cleaned_suffix}_lstm_best_loss_{len(optim_indices)}.pkl")
best_model_path = os.path.join(tmp_dir, f"{cleaned_suffix}_lstm_best_model_{len(optim_indices)}.pth")

# model_builder for Optuna (keeps same construction & param groups as CV)
def model_builder(trial):
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    hidden_dim = trial.suggest_int('hidden_dim', 50, 512)
    dropout = trial.suggest_float('dropout', 0.0, 0.6)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    model = LSTMForecaster(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes
    )
    model.to(device)

    # single param group for LSTM
    param_groups = [
        {"params": model.parameters(), "lr": lr, "weight_decay": weight_decay}
    ]

    criterion = get_criterion(num_classes)
    return model, param_groups, criterion

# Run Bayesian optimization if no best params provided
if best_params_path is None:
    print("Running Bayesian Optimization to find best hyperparameters")
    # remove previous artifacts to restart search if present
    if os.path.exists(best_loss_path):
        os.remove(best_loss_path)
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective_generic(
        t,
        train_loader=optim_train_loader,
        val_loader=optim_val_loader,
        device=device,
        model_builder=model_builder,
        best_loss_path=best_loss_path,
        best_model_path=best_model_path,
        epochs=epochs,
        patience=patience,
    ), n_trials=n_trials)
    best_params = study.best_params
    with open(os.path.join(tmp_dir, f"{cleaned_suffix}_lstm_best_params_{len(optim_indices)}.pkl"), 'wb') as f:
        pickle.dump(best_params, f)
else:
    with open(best_params_path, 'rb') as f:
        best_params = pickle.load(f)

print("Using best params:", best_params)

# load best model checkpoint for re-use in CV
model_template = LSTMForecaster(
    input_dim=input_dim,
    hidden_dim=best_params['hidden_dim'],
    num_layers=best_params['num_layers'],
    dropout=best_params['dropout'],
    num_classes=num_classes
)
model_template.to(device)
if os.path.exists(best_model_path):
    model_template.load_state_dict(torch.load(best_model_path, map_location=device))

result_loss = []
result_corr = []

print("Start cross-validation")
for size in train_sizes:
    print('Train size:', size)
    losses_for_size = []
    corrs_for_size = []

    for seed in range(n_resamples):
        print('...Iteration:', seed)
        torch.manual_seed(seed)
        perm = torch.randperm(len(main_indices))
        train_idx = main_indices[perm[:size]]
        val_idx = main_indices[perm[size:size+val_size]]
        test_idx = main_indices[perm[size+val_size:]]

        train_loader = DataLoader(TensorDataset(X[train_idx], time_sequence[train_idx], y[train_idx]),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X[val_idx], time_sequence[val_idx], y[val_idx]),
                                batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X[test_idx], time_sequence[test_idx], y[test_idx]),
                                 batch_size=batch_size, shuffle=False)

        # instantiate fresh model for each run
        model = LSTMForecaster(
            input_dim=input_dim,
            hidden_dim=best_params['hidden_dim'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout'],
            num_classes=num_classes
        )
        model.to(device)

        criterion = get_criterion(num_classes)
        optimizer = Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

        model_trained, _ = train_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            patience=patience,
            use_amp=torch.cuda.is_available()
        )

        # evaluate
        model_trained.eval()
        total_loss = 0.0
        all_preds = []
        all_targs = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0].to(device).float()
                seq_len = batch[1]  # forwarded but might be unused by model
                targets = batch[2].to(device).float()

                outputs = model_trained(inputs, seq_len)
                outputs = outputs.squeeze(-1)

                all_preds.append(outputs.cpu())
                all_targs.append(targets.cpu())
                total_loss += criterion(outputs, targets).item()

        tloss = total_loss / max(1, len(test_loader))
        all_preds = torch.cat(all_preds).numpy()
        all_targs = torch.cat(all_targs).numpy()

        if np.isnan(all_preds).any() or np.isinf(all_preds).any():
            corr = float('nan')
        else:
            corr, _ = pearsonr(all_preds.squeeze(), all_targs)

        losses_for_size.append(tloss)
        corrs_for_size.append(float(corr))

        del train_loader, val_loader, test_loader, model_trained
        gc.collect()
        torch.cuda.empty_cache()

    result_loss.append(losses_for_size)
    result_corr.append(corrs_for_size)

os.makedirs(res_dir, exist_ok=True)
np.save(os.path.join(res_dir, f"mse_{suffix}_lstm.npy"), np.array(result_loss, dtype=object))
np.save(os.path.join(res_dir, f"pearsonr_{suffix}_lstm.npy"), np.array(result_corr, dtype=object))
print("Finished.")