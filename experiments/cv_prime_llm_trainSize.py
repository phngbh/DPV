import argparse
import yaml
from datetime import datetime
import numpy as np
import os
import pickle
import gc
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW
from scipy.stats import pearsonr

from transformers import AutoConfig

from models.prime_llm import PRIME_LLM
from experiments.train_utils import train_model, objective_generic

print('Start running script')
print('Starting time: ' + str(datetime.now().time()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'The current device is {device}.')

# Parse arguments
parser = argparse.ArgumentParser(description='Arguments for running script')
parser.add_argument('--config', type=str, help='Path to the configuration file')
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
use_lora = bool(cfg.get('use_lora', False))
n_trials = int(cfg.get('n_trials', 50))
n_resamples = int(cfg.get('n_resamples', 5))
epochs = int(cfg.get('epochs', 50))
patience = int(cfg.get('patience', 5))
best_params_path = cfg.get('best_params', None)
train_sizes = list(map(int, cfg.get('train_size', '50').split(',')))
pretrained_model = cfg['pretrained_model']

print(f"The LLM used is {pretrained_model}")

# Retrieve the hidden size from the model's configuration
hf_config = AutoConfig.from_pretrained(pretrained_model, trust_remote_code=True)
embedding_dim = getattr(hf_config, "embedding_size", hf_config.hidden_size)

# load data
X = torch.load(data_path)
X = X[:, 4:, :] # remove first 4 timepoints
y = torch.load(target_path)
time_sequence = torch.load(time_seq_path) - cfg.get('time_seq_offset', 5)

# build attention mask
max_seq = cfg.get('max_seq_len', X.shape[1])
range_tensor = torch.arange(max_seq).unsqueeze(0).expand(len(time_sequence), max_seq)
attention_mask_np = (range_tensor < time_sequence.unsqueeze(1)).int().numpy()
attention_mask_np = np.fliplr(attention_mask_np)
attention_mask = torch.from_numpy(attention_mask_np.copy())

# prepare optimization subset
torch.manual_seed(93)
indices = torch.randperm(X.size(0))
optim_indices = indices[:optim_size]
main_indices = indices[optim_size:]

# small optimization set split (80/20)
optim_train_N = int(0.8 * len(optim_indices))
optim_train_idx = optim_indices[:optim_train_N]
optim_val_idx = optim_indices[optim_train_N:]

optim_train_loader = DataLoader(TensorDataset(X[optim_train_idx], time_sequence[optim_train_idx], attention_mask[optim_train_idx], y[optim_train_idx]), batch_size=cfg.get('batch_size',16), shuffle=True)
optim_val_loader = DataLoader(TensorDataset(X[optim_val_idx], time_sequence[optim_val_idx], attention_mask[optim_val_idx], y[optim_val_idx]), batch_size=cfg.get('batch_size',16), shuffle=True)

input_dim = X.size(2)
cleaned_pretrained_model = pretrained_model.split('/')[-1] if '/' in pretrained_model else pretrained_model

best_loss_path = os.path.join(tmp_dir, f"{suffix}_{cleaned_pretrained_model}_best_loss_{optim_size}.pth")
best_model_path = os.path.join(tmp_dir, f"{suffix}_{cleaned_pretrained_model}_best_model_{optim_size}.pth")

# model builder closure for Optuna: this is model-specific -> keep here so objective_generic is reusable
def model_builder(trial):
    # hyperparameter suggestions (example for PRIME_LLM)
    if use_lora:
        lora_r = trial.suggest_int("lora_r", 4, 64)
        lora_alpha = trial.suggest_int("lora_alpha", 4, 32)
        lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.3)
    else:
        lora_r = lora_alpha = lora_dropout = None

    lr_embedding = trial.suggest_float('lr_embedding', 1e-6, 1e-2, log=True)
    lr_regressor = trial.suggest_float('lr_regressor', 1e-6, 1e-2, log=True)
    lr_adapter = trial.suggest_float("lr_adapter", 1e-6, 1e-3, log=True)
    hidden_dim = trial.suggest_int('hidden_dim', 50, 300)
    dropout = trial.suggest_float('dropout', 0.0, 0.6)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    model = PRIME_LLM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        dropout=dropout,
        pretrained_model=pretrained_model,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        num_classes=cfg.get('num_classes', None),
    )

    param_groups = [
        {"params": model.embedding.parameters(), "lr": lr_embedding, "weight_decay": weight_decay},
        {"params": model.regressor.parameters(), "lr": lr_regressor, "weight_decay": weight_decay},
    ]
    if use_lora:
        adapter_params = [p for p in model.transformer.parameters() if p.requires_grad]
        param_groups.append({"params": adapter_params, "lr": lr_adapter, "weight_decay": 0.0})
    else:
        param_groups.append({"params": model.transformer.parameters(), "lr": lr_adapter, "weight_decay": weight_decay})

    # choose loss by task
    if cfg.get('num_classes', None) is None:
        criterion = MSELoss()
    elif cfg.get('num_classes') == 2:
        criterion = BCEWithLogitsLoss()
    else:
        criterion = CrossEntropyLoss()

    return model, param_groups, criterion

# Run Bayesian optimization if no best params provided
if best_params_path is None:
    import optuna
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
    with open(os.path.join(tmp_dir, f"{suffix}_{cleaned_pretrained_model}_best_params_{optim_size}.pkl"), 'wb') as f:
        pickle.dump(best_params, f)
else:
    with open(best_params_path, 'rb') as f:
        best_params = pickle.load(f)

print('Start cross validation')
result_loss = []
result_corr = []

for size in train_sizes:
    print('Train size: ', size)
    losses_for_size = []
    corrs_for_size = []
    for seed in range(n_resamples):
        torch.manual_seed(seed)
        perm = torch.randperm(len(main_indices))
        train_idx = main_indices[perm[:size]]
        val_idx = main_indices[perm[size:size+val_size]]
        test_idx = main_indices[perm[size+val_size:]]

        train_loader = DataLoader(TensorDataset(X[train_idx], time_sequence[train_idx], attention_mask[train_idx], y[train_idx]), batch_size=cfg.get('batch_size',16), shuffle=True)
        val_loader = DataLoader(TensorDataset(X[val_idx], time_sequence[val_idx], attention_mask[val_idx], y[val_idx]), batch_size=cfg.get('batch_size',16), shuffle=True)
        test_loader = DataLoader(TensorDataset(X[test_idx], time_sequence[test_idx], attention_mask[test_idx], y[test_idx]), batch_size=cfg.get('batch_size',16), shuffle=False)

        # Build model with best_params (must match names suggested above)
        if use_lora:
            lora_r = best_params.get('lora_r')
            lora_alpha = best_params.get('lora_alpha')
            lora_dropout = best_params.get('lora_dropout')
        else:
            lora_r = lora_alpha = lora_dropout = None

        model = PRIME_LLM(
            input_dim=input_dim,
            hidden_dim=best_params['hidden_dim'],
            embedding_dim=embedding_dim,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            dropout=best_params['dropout'],
            pretrained_model=pretrained_model,
            num_classes=cfg.get('num_classes', None),
        )
        model.to(device)

        if cfg.get('num_classes', None) is None:
            criterion = MSELoss()
        elif cfg.get('num_classes') == 2:
            criterion = BCEWithLogitsLoss()
        else:
            criterion = CrossEntropyLoss()

        param_groups = [
            {"params": model.embedding.parameters(), "lr": best_params['lr_embedding'], "weight_decay": best_params['weight_decay']},
            {"params": model.regressor.parameters(), "lr": best_params['lr_regressor'], "weight_decay": best_params['weight_decay']},
        ]
        if use_lora:
            adapter_params = [p for p in model.transformer.parameters() if p.requires_grad]
            param_groups.append({"params": adapter_params, "lr": best_params['lr_adapter'], "weight_decay": 0.0})
        else:
            param_groups.append({"params": model.transformer.parameters(), "lr": best_params['lr_adapter'], "weight_decay": best_params['weight_decay']})

        optimizer = AdamW(param_groups)

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
                mask = batch[2].to(device).float()
                targets = batch[-1].to(device).float()
                outputs = model_trained(inputs, attention_mask=mask).float().squeeze(-1)
                all_preds.append(outputs.cpu())
                all_targs.append(targets.cpu())
                total_loss += criterion(outputs, targets).item()

        tloss = total_loss / max(1, len(test_loader))
        all_preds = torch.cat(all_preds).numpy()
        all_targs = torch.cat(all_targs).numpy()

        if np.isnan(all_preds).any() or np.isinf(all_preds).any():
            corr = float('nan')
        else:
            corr, _ = pearsonr(all_preds, all_targs)

        losses_for_size.append(tloss)
        corrs_for_size.append(float(corr))

        del train_loader, val_loader, test_loader, model_trained
        gc.collect()
        torch.cuda.empty_cache()

    result_loss.append(losses_for_size)
    result_corr.append(corrs_for_size)

# save results
os.makedirs(res_dir, exist_ok=True)
np.save(os.path.join(res_dir, f"mse_{suffix}_{cleaned_pretrained_model}.npy"), np.array(result_loss, dtype=object))
np.save(os.path.join(res_dir, f"pearsonr_{suffix}_{cleaned_pretrained_model}.npy"), np.array(result_corr, dtype=object))
print("Finished.")