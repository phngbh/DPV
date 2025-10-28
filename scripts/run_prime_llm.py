import os
import gc
import argparse
import pickle
from datetime import datetime

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from transformers import AutoConfig
from scipy.stats import pearsonr

import optuna

from models.prime_llm import PRIME_LLM 
from scripts.train_utils import train_model, objective_generic

print('Start running script')
print('Starting time:', str(datetime.now().time()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

parser = argparse.ArgumentParser(description='Run single PRIME experiment (optional BO)')
parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
parser.add_argument('--pretrained_model', type=str, help='Override pretrained model name', required=False)
parser.add_argument('--best_params', type=str, help='Path to best params .pkl (skip BO)', required=False)
parser.add_argument('--run_bo', action='store_true', help='Run Optuna BO on optimization subset')
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg_all = yaml.safe_load(f)
cfg = cfg_all.get('cv_prime_llm_trainSize', cfg_all)

# IO / experiment settings
data_path = cfg['data']
target_path = cfg['target']
time_seq_path = cfg['time_seq']
optim_size = int(cfg.get('optim_size', 1000))
train_size = int(cfg.get('train_size', 200))
val_size = int(cfg.get('val_size', 200))
test_size = int(cfg.get('test_size', 200))
res_dir = cfg['res_dir']
tmp_dir = cfg.get('tmp_dir', './tmp/')
suffix = cfg.get('suffix', 'prime_experiment')
batch_size = int(cfg.get('batch_size', 16))
os.makedirs(res_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

# training options
use_lora = bool(cfg.get('use_lora', False))
n_trials = int(cfg.get('n_trials', 50))
epochs = int(cfg.get('epochs', 50))
patience = int(cfg.get('patience', 5))
seed = int(cfg.get('seed', 93))
num_classes = cfg.get('num_classes', None)

pretrained_model = args.pretrained_model or cfg.get('pretrained_model')
best_params_path = args.best_params or cfg.get('best_params', None)
run_bo_flag = args.run_bo or bool(cfg.get('run_bo', False))

print(f"Pretrained model: {pretrained_model}")

# embedding dimension from HF config
hf_config = AutoConfig.from_pretrained(pretrained_model, trust_remote_code=True)
embedding_dim = getattr(hf_config, "embedding_size", hf_config.hidden_size)

# load tensors
print("Loading data")
X = torch.load(data_path)
X = X[:, 4:, :]  # keep legacy trimming
y = torch.load(target_path)
time_sequence = torch.load(time_seq_path) - cfg.get('time_seq_offset', 0)

# filter invalid samples
keep_idx = torch.where(y > -4)[0]
X = X[keep_idx]
y = y[keep_idx]
time_sequence = time_sequence[keep_idx]

print(f"Samples remaining: {X.size(0)} | X shape: {X.shape}")

# attention mask
max_seq = cfg.get('max_seq_len', X.shape[1])
range_tensor = torch.arange(max_seq).unsqueeze(0).expand(len(time_sequence), max_seq)
attention_mask_np = (range_tensor < time_sequence.unsqueeze(1)).int().numpy()
attention_mask_np = np.fliplr(attention_mask_np)
attention_mask = torch.from_numpy(attention_mask_np.copy())

# reproducible split: create an optimization subset and a main pool
torch.manual_seed(seed)
indices = torch.randperm(X.size(0))
optim_indices = indices[:optim_size]
main_indices = indices[optim_size:]

# small optimization split (80/20) used for BO
optim_train_N = int(0.8 * len(optim_indices))
optim_train_idx = optim_indices[:optim_train_N]
optim_val_idx = optim_indices[optim_train_N:]

optim_train_loader = DataLoader(
    TensorDataset(X[optim_train_idx], time_sequence[optim_train_idx], attention_mask[optim_train_idx], y[optim_train_idx]),
    batch_size=batch_size, shuffle=True, num_workers=0
)
optim_val_loader = DataLoader(
    TensorDataset(X[optim_val_idx], time_sequence[optim_val_idx], attention_mask[optim_val_idx], y[optim_val_idx]),
    batch_size=batch_size, shuffle=True, num_workers=0
)

input_dim = X.size(2)
cleaned_pretrained_model = pretrained_model.split('/')[-1] if '/' in pretrained_model else pretrained_model

best_loss_path = os.path.join(tmp_dir, f"{suffix}_{cleaned_pretrained_model}_best_loss_{len(optim_indices)}.pkl")
best_model_path = os.path.join(tmp_dir, f"{suffix}_{cleaned_pretrained_model}_best_model_{len(optim_indices)}.pth")

# model_builder (same param groups as final training)
def model_builder(trial):
    if use_lora:
        lora_r = trial.suggest_int("lora_r", 4, 64)
        lora_alpha = trial.suggest_int("lora_alpha", 4, 32)
        lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.3)
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
        num_classes=num_classes
    )
    model.to(device)

    param_groups = [
        {"params": model.embedding.parameters(), "lr": lr_embedding, "weight_decay": weight_decay},
        {"params": model.regressor.parameters(), "lr": lr_regressor, "weight_decay": weight_decay},
    ]
    if use_lora:
        adapter_params = [p for p in model.transformer.parameters() if p.requires_grad]
        param_groups.append({"params": adapter_params, "lr": lr_adapter, "weight_decay": 0.0})
    else:
        param_groups.append({"params": model.transformer.parameters(), "lr": lr_adapter, "weight_decay": weight_decay})

    if num_classes is None:
        criterion = MSELoss()
    elif num_classes == 2:
        criterion = BCEWithLogitsLoss()
    else:
        criterion = CrossEntropyLoss()

    return model, param_groups, criterion

# run BO if requested and no best_params provided
if best_params_path:
    with open(best_params_path, 'rb') as f:
        best_params = pickle.load(f)
    print("Loaded best params from", best_params_path)
elif run_bo_flag:
    print("Running Optuna BO on optimization subset")
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
    with open(os.path.join(tmp_dir, f"{suffix}_{cleaned_pretrained_model}_best_params_{len(optim_indices)}.pkl"), 'wb') as f:
        pickle.dump(best_params, f)
    print("Saved BO best params.")
else:
    raise RuntimeError("No best params provided and BO disabled. Provide --best_params or use --run_bo.")

print("Using best params:", best_params)

# Build final single train / val / test split from main_indices
torch.manual_seed(seed)
perm = main_indices[torch.randperm(len(main_indices))]
if train_size + val_size + test_size > len(perm):
    raise ValueError("train+val+test exceed available samples in main_indices")

train_idx = perm[:train_size]
val_idx = perm[train_size:train_size + val_size]
test_idx = perm[train_size + val_size:train_size + val_size + test_size]

train_loader = DataLoader(TensorDataset(X[train_idx], time_sequence[train_idx], attention_mask[train_idx], y[train_idx]),
                          batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(TensorDataset(X[val_idx], time_sequence[val_idx], attention_mask[val_idx], y[val_idx]),
                        batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(TensorDataset(X[test_idx], time_sequence[test_idx], attention_mask[test_idx], y[test_idx]),
                         batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

# instantiate final model from best_params
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
    dropout=best_params['dropout'],
    pretrained_model=pretrained_model,
    use_lora=use_lora,
    lora_r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    num_classes=num_classes
)
model.to(device)

if num_classes is None:
    criterion = MSELoss()
elif num_classes == 2:
    criterion = BCEWithLogitsLoss()
else:
    criterion = CrossEntropyLoss()

param_groups = [
    {"params": model.embedding.parameters(), "lr": best_params['lr_embedding'], "weight_decay": best_params.get('weight_decay', 0.0)},
    {"params": model.regressor.parameters(), "lr": best_params['lr_regressor'], "weight_decay": best_params.get('weight_decay', 0.0)},
]
if use_lora:
    adapter_params = [p for p in model.transformer.parameters() if p.requires_grad]
    param_groups.append({"params": adapter_params, "lr": best_params['lr_adapter'], "weight_decay": 0.0})
else:
    param_groups.append({"params": model.transformer.parameters(), "lr": best_params['lr_adapter'], "weight_decay": best_params.get('weight_decay', 0.0)})

optimizer = AdamW(param_groups)

print("Start training final model")
model_trained, best_val_loss = train_model(
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

print("Evaluating on test set")
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

test_loss = total_loss / max(1, len(test_loader))
all_preds = torch.cat(all_preds).numpy()
all_targs = torch.cat(all_targs).numpy()
try:
    corr, _ = pearsonr(all_preds, all_targs)
except Exception:
    corr = float('nan')

print(f"Test loss: {test_loss:.6f} | Pearson: {corr}")

# save artifacts
model_file = os.path.join(res_dir, f"{suffix}_{cleaned_pretrained_model}_model.pth")
npz_file = os.path.join(res_dir, f"{suffix}_{cleaned_pretrained_model}_preds.npz")
params_file = os.path.join(res_dir, f"{suffix}_{cleaned_pretrained_model}_best_params.pkl")

torch.save(model_trained.state_dict(), model_file)
np.savez(npz_file, predictions=all_preds, targets=all_targs, test_idx=test_idx.cpu().numpy())
with open(params_file, "wb") as f:
    pickle.dump(best_params, f)

print("Saved:", model_file, npz_file, params_file)
print("Finished:", str(datetime.now().time()))