import os
import gc
import pickle
import argparse
from datetime import datetime

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from scipy.stats import pearsonr
import optuna

from models.transformer import Transformer
from experiments.train_utils import train_model, objective_generic

print('Start running script')
print('Starting time: ' + str(datetime.now().time()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'The current device is {device}.')

parser = argparse.ArgumentParser(description='Arguments for running script')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
args = parser.parse_args()

# Load configuration
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# Load main options (allow CLI overrides)
data_path = cfg['data']
target_path = cfg['target']
time_seq_path = cfg['time_seq']
optim_size = int(cfg['optim_size'])
val_size = int(cfg['val_size'])
res_dir = cfg['res_dir']
tmp_dir = cfg['tmp_dir']
suffix = cfg['suffix']
batch_size = int(cfg.get('batch_size', 16))

n_trials = int(cfg.get('n_trials', 50))
n_resamples = int(cfg.get('n_resamples', 5))
epochs = int(cfg.get('epochs', 50))
patience = int(cfg.get('patience', 5))
best_params_path = cfg.get('best_params', None)
train_sizes = list(map(int, cfg.get('train_size', '50').split(',')))
num_classes = cfg.get('num_classes', None)

# helper: choose loss
def get_criterion(num_classes):
    if num_classes is None:
        return MSELoss()
    elif num_classes == 2:
        return BCEWithLogitsLoss()
    else:
        return CrossEntropyLoss()

print("Load data")
X = torch.load(data_path)
X = X[:, 4:, :] # remove first 4 timepoints
y = torch.load(target_path)
time_sequence = torch.load(time_seq_path) - cfg.get('time_seq_offset', 5)

# timestamp feature (first channel)
time_stamp = X[:, :, :1]

# attention mask
max_seq = cfg.get('max_seq_len', 45)
range_tensor = torch.arange(max_seq).unsqueeze(0).expand(len(time_sequence), max_seq)
attention_mask_np = (range_tensor < time_sequence.unsqueeze(1)).int().numpy()
attention_mask_np = np.fliplr(attention_mask_np)
attention_mask = torch.from_numpy(attention_mask_np.copy())

# reproducible index selection
torch.manual_seed(93)
optim_indices = torch.randperm(X.size(0))[:optim_size]
optim_indices = optim_indices[:min(len(optim_indices), 1000)]
torch.manual_seed(93)
main_indices = torch.randperm(X.size(0))[optim_size:]

# split optim subset 80/20
optim_train_N = int(0.8 * len(optim_indices))
optim_train_idx = optim_indices[:optim_train_N]
optim_val_idx = optim_indices[optim_train_N:]

optim_train_loader = DataLoader(TensorDataset(X[optim_train_idx], time_sequence[optim_train_idx], time_stamp[optim_train_idx], attention_mask[optim_train_idx], y[optim_train_idx]), batch_size=batch_size, shuffle=True)
optim_val_loader = DataLoader(TensorDataset(X[optim_val_idx], time_sequence[optim_val_idx], time_stamp[optim_val_idx], attention_mask[optim_val_idx], y[optim_val_idx]), batch_size=batch_size, shuffle=True)

input_dim = X.size(2)
best_loss_path = os.path.join(tmp_dir, f"{suffix}_transformer_best_loss_{len(optim_indices)}.pkl")
best_model_path = os.path.join(tmp_dir, f"{suffix}_transformer_best_model_{len(optim_indices)}.pth")
os.makedirs(tmp_dir, exist_ok=True)

# model_builder for Optuna 
def model_builder(trial):
    lr_embedding = trial.suggest_float('lr_embedding', 1e-6, 1e-2, log=True)
    lr_regressor = trial.suggest_float('lr_regressor', 1e-6, 1e-2, log=True)
    lr_transformer = trial.suggest_float('lr_transformer', 1e-6, 1e-2, log=True)
    lr_timestamp_encoder = trial.suggest_float('lr_timestamp_encoder', 1e-6, 1e-2, log=True)

    num_layers = trial.suggest_int('num_layers', 1, 3)
    hidden_dim = trial.suggest_int('hidden_dim', 50, 300)

    valid_pairs = [(e, h) for e in range(64, 257, 8) for h in range(2, 10, 2) if e % h == 0]
    valid_pair_strings = [f"{e}_{h}" for e, h in valid_pairs]
    pair_str = trial.suggest_categorical('embed_heads_pair', valid_pair_strings)
    embedding_dim, num_heads = map(int, pair_str.split("_"))

    dropout = trial.suggest_float('dropout', 0.1, 0.8)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    model = Transformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes
    )
    model.to(device)

    param_groups = [
        {'params': model.embedding.parameters(), 'lr': lr_embedding},
        {'params': model.timestamp_encoder.parameters(), 'lr': lr_timestamp_encoder},
        {'params': model.transformer_encoder.parameters(), 'lr': lr_transformer},
        {'params': model.regressor.parameters(), 'lr': lr_regressor},
    ]

    criterion = get_criterion(num_classes)
    return model, param_groups, criterion

# Run BO if needed
if best_params_path is None:
    print("Running Bayesian Optimization to find best hyperparameters")
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
    with open(os.path.join(tmp_dir, f"{suffix}_transformer_best_params_{len(optim_indices)}.pkl"), 'wb') as f:
        pickle.dump(best_params, f)
else:
    with open(best_params_path, 'rb') as f:
        best_params = pickle.load(f)
print("Using best params:", best_params)

# parse embed/head pair
best_embedding_dim, best_num_heads = map(int, best_params['embed_heads_pair'].split("_"))

result_loss = []
result_corr = []

for size in train_sizes:
    print('Train size:', size)
    losses_for_size = []
    corrs_for_size = []

    for seed in range(n_resamples):
        torch.manual_seed(seed)
        train_idx = main_indices[torch.randperm(len(main_indices))[:size]]
        val_idx = main_indices[torch.randperm(len(main_indices))[size:(size + val_size)]]
        test_idx = main_indices[torch.randperm(len(main_indices))[(size + val_size):]]

        train_loader = DataLoader(TensorDataset(X[train_idx], time_sequence[train_idx], time_stamp[train_idx], attention_mask[train_idx], y[train_idx]), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X[val_idx], time_sequence[val_idx], time_stamp[val_idx], attention_mask[val_idx], y[val_idx]), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X[test_idx], time_sequence[test_idx], time_stamp[test_idx], attention_mask[test_idx], y[test_idx]), batch_size=batch_size, shuffle=False)

        model = Transformer(
            input_dim=input_dim,
            hidden_dim=best_params['hidden_dim'],
            embedding_dim=best_embedding_dim,
            num_heads=best_num_heads,
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout'],
            num_classes=num_classes
        )
        model.to(device)

        criterion = get_criterion(num_classes)

        optimizer = AdamW([
            {'params': model.embedding.parameters(), 'lr': best_params['lr_embedding']},
            {'params': model.timestamp_encoder.parameters(), 'lr': best_params['lr_timestamp_encoder']},
            {'params': model.transformer_encoder.parameters(), 'lr': best_params['lr_transformer']},
            {'params': model.regressor.parameters(), 'lr': best_params['lr_regressor']}
        ], weight_decay=best_params['weight_decay'])

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

        model_trained.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, seq_length, time, mask, targets in test_loader:
                inputs = inputs.to(device).float()
                time = time.to(device).float()
                mask = mask.to(device).float()
                targets = targets.to(device).float()

                outputs = model_trained(inputs, timestamps=time, attention_mask=mask)
                if num_classes is not None and num_classes > 2:
                    outputs = outputs.squeeze(-1)

                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())

                total_loss += criterion(outputs.squeeze(), targets).item()

        tloss = total_loss / max(1, len(test_loader))
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        if torch.isnan(all_predictions).any() or torch.isinf(all_predictions).any():
            corr = float('nan')
        else:
            corr, _ = pearsonr(all_predictions.numpy().squeeze(), all_targets.numpy())

        losses_for_size.append(tloss)
        corrs_for_size.append(float(corr))

        del train_loader, val_loader, test_loader, model_trained
        gc.collect()
        torch.cuda.empty_cache()

    result_loss.append(losses_for_size)
    result_corr.append(corrs_for_size)

os.makedirs(res_dir, exist_ok=True)
np.save(os.path.join(res_dir, f"mse_{suffix}_transformer.npy"), np.array(result_loss, dtype=object))
np.save(os.path.join(res_dir, f"pearsonr_{suffix}_transformer.npy"), np.array(result_corr, dtype=object))
np.savetxt(os.path.join(res_dir, f"mse_{suffix}_transformer.csv"), np.array(result_loss, dtype=object), delimiter=",", fmt='%s')
np.savetxt(os.path.join(res_dir, f"pearsonr_{suffix}_transformer.csv"), np.array(result_corr, dtype=object), delimiter=",", fmt='%s')

print("Finished.")