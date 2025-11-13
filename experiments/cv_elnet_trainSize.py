import os
import gc
import pickle
import warnings
import argparse
from datetime import datetime

import yaml
import numpy as np
import torch
import optuna
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectKBest, mutual_info_regression, VarianceThreshold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

print("Start running script")
print("Starting time:", str(datetime.now().time()))

parser = argparse.ArgumentParser(description="ElasticNet CV + BO (flattened time-series)")
parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

# config / io
data_path = cfg["data"]
target_path = cfg["target"]
optim_size = int(cfg.get("optim_size", 1000))
res_dir = cfg.get("res_dir", "./results/")
tmp_dir = cfg.get("tmp_dir", "./tmp/")
suffix = cfg.get("suffix", "elnet_experiment")
n_resamples = int(cfg.get("n_resamples", 5))
n_trials = int(cfg.get("n_trials", 100))
seed = int(cfg.get("seed", 93))
feature_k = int(cfg.get("feature_k", 2000))
var_thresh_val = float(cfg.get("var_thresh", 1e-5))
fs_sample = int(cfg.get("fs_sample", 1000))
best_params_path = cfg.get("best_params", None)
train_sizes = list(map(int, (args.train_size or cfg.get("train_size", "50")).split(",")))

os.makedirs(res_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Load data")
X = torch.load(data_path)          # expected (N, T, F)
X = X[:, 4:, :]                    # keep consistent trimming
y = torch.load(target_path)

# filter invalid samples (same filtering as other scripts)
valid_idx = torch.where(y > -4)[0]
X = X[valid_idx]
y = y[valid_idx]

print(f"Remaining samples: {X.size(0)}  | X shape: {X.shape}")

# deterministic sampling for optimization subset and main indices
torch.manual_seed(seed)
indices = torch.randperm(X.size(0))
optim_indices = indices[:optim_size]
main_indices = indices[optim_size:]

# split optim subset into train/val for BO (80/20)
optim_train_N = int(0.8 * len(optim_indices))
optim_train_idx = optim_indices[:optim_train_N]
optim_val_idx = optim_indices[optim_train_N:]

# helper: flatten torch tensor (N, T, F) -> (N, T*F) numpy
def flatten(t: torch.Tensor) -> np.ndarray:
    return t.reshape(t.size(0), -1).cpu().numpy()

# prepare flattened arrays for BO
X_opt_train = flatten(X[optim_train_idx])
y_opt_train = y[optim_train_idx].cpu().numpy()
X_opt_val = flatten(X[optim_val_idx])
y_opt_val = y[optim_val_idx].cpu().numpy()

# feature selection pipeline fit on optimization training subset
print("Variance thresholding")
var_selector = VarianceThreshold(threshold=var_thresh_val)
X_opt_train = var_selector.fit_transform(X_opt_train)
X_opt_val = var_selector.transform(X_opt_val)

print(f"Selecting top {feature_k} features by mutual information")
k_sel = min(feature_k, X_opt_train.shape[1])
mi_selector = SelectKBest(lambda X_, y_: mutual_info_regression(X_, y_, n_neighbors=10), k=k_sel)
X_opt_train = mi_selector.fit_transform(X_opt_train, y_opt_train)
X_opt_val = mi_selector.transform(X_opt_val)

print("Scaling")
scaler = StandardScaler()
X_opt_train = scaler.fit_transform(X_opt_train)
X_opt_val = scaler.transform(X_opt_val)

# Optuna objective
def objective(trial, X_tr, y_tr, X_v, y_v):
    alpha = trial.suggest_loguniform("alpha", 1e-4, 10.0)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=seed)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_v)
    return mean_squared_error(y_v, preds)

# run BO or load best params
best_params_file = os.path.join(tmp_dir, f"{suffix}_best_params.pkl")
if best_params_path:
    with open(best_params_path, "rb") as f:
        best_params = pickle.load(f)
    print("Loaded best params from", best_params_path)
else:
    print("Running Optuna Bayesian optimization for ElasticNet")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, X_opt_train, y_opt_train, X_opt_val, y_opt_val), n_trials=n_trials)
    best_params = study.best_params
    with open(best_params_file, "wb") as f:
        pickle.dump(best_params, f)
    print("Saved best params to", best_params_file)

print("Best params:", best_params)

# Prepare main flattened data for CV
X_main = flatten(X[main_indices])
y_main = y[main_indices].cpu().numpy()

# Apply feature pipeline fitted on optimization subset to main data
X_main = var_selector.transform(X_main)
X_main = mi_selector.transform(X_main)
X_main = scaler.transform(X_main)

result_loss = []
result_corr = []

print("Start cross-validation")
for size in train_sizes:
    print("Train size:", size)
    losses_for_size = []
    corrs_for_size = []

    for s in range(n_resamples):
        print("  Iteration:", s)
        torch.manual_seed(s)
        perm = torch.randperm(len(main_indices))
        train_idx = perm[:size].cpu().numpy()
        test_idx = perm[size:].cpu().numpy()

        X_train = X_main[train_idx]
        y_train = y_main[train_idx]
        X_test = X_main[test_idx]
        y_test = y_main[test_idx]

        # fit final model
        model = ElasticNet(alpha=best_params["alpha"], l1_ratio=best_params["l1_ratio"], max_iter=5000, random_state=seed)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        try:
            p_corr, _ = pearsonr(y_test, preds)
        except Exception:
            p_corr = float("nan")

        losses_for_size.append(float(mse))
        corrs_for_size.append(float(p_corr))

        gc.collect()

    result_loss.append(losses_for_size)
    result_corr.append(corrs_for_size)

# save results
os.makedirs(res_dir, exist_ok=True)
np.save(os.path.join(res_dir, f"mse_{suffix}_elnet.npy"), np.array(result_loss, dtype=object))
np.save(os.path.join(res_dir, f"pearsonr_{suffix}_elnet.npy"), np.array(result_corr, dtype=object))
np.savetxt(os.path.join(res_dir, f"mse_{suffix}_elnet.csv"), np.array(result_loss, dtype=object), delimiter=",", fmt="%s")
np.savetxt(os.path.join(res_dir, f"pearsonr_{suffix}_elnet.csv"), np.array(result_corr, dtype=object), delimiter=",", fmt="%s")

print("Finished:", str(datetime.now().time()))