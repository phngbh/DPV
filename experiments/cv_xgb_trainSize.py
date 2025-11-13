import os
import gc
import argparse
from datetime import datetime

import yaml
import numpy as np
import torch
import optuna
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, mutual_info_regression, VarianceThreshold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

print("Start running script")
print("Starting time:", str(datetime.now().time()))

parser = argparse.ArgumentParser(description="XGBoost CV + BO for time-series flattened features")
parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

# Config values
data_path = cfg["data"]
target_path = cfg["target"]
optim_size = int(cfg.get("optim_size", 1000))
res_dir = cfg.get("res_dir", "./results/")
tmp_dir = cfg.get("tmp_dir", "./tmp/")
suffix = cfg.get("suffix", "xgb_experiment")
n_resamples = int(cfg.get("n_resamples", 5))
n_trials = int(cfg.get("n_trials", 100))
random_seed = int(cfg.get("seed", 93))
feature_k = int(cfg.get("feature_k", 2000))
var_thresh_val = float(cfg.get("var_thresh", 1e-5))
n_jobs = int(cfg.get("n_jobs", 8))
optim_limit = int(cfg.get("optim_limit", 1000))  # number of samples to use for BO (cap)
train_sizes = list(map(int, cfg.get("train_size", "50").split(",")))

os.makedirs(res_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
print("Loading data")
X = torch.load(data_path)         # expected shape (N, T, F)
X = X[:, 4:, :]                   # drop first 4 timesteps if required
y = torch.load(target_path)

print(f"Remaining samples: {X.size(0)}  | X shape: {X.shape}")

# deterministic splits
torch.manual_seed(random_seed)
indices = torch.randperm(X.size(0))
optim_indices = indices[:optim_size]
optim_indices = optim_indices[:min(len(optim_indices), optim_limit)]
main_indices = indices[optim_size:]

# Further split optim subset into train/val for BO (80/20)
optim_train_N = int(0.8 * len(optim_indices))
optim_train_idx = optim_indices[:optim_train_N]
optim_val_idx = optim_indices[optim_train_N:]

# Flatten tensors to 2D numpy arrays for scikit/xgboost
def flatten_tensor(t):
    return t.view(t.size(0), -1).cpu().numpy()

X_optim_train = flatten_tensor(X[optim_train_idx])
y_optim_train = y[optim_train_idx].cpu().numpy()
X_optim_val = flatten_tensor(X[optim_val_idx])
y_optim_val = y[optim_val_idx].cpu().numpy()
X_main = flatten_tensor(X[main_indices])
y_main = y[main_indices].cpu().numpy()

# Preprocessing pipeline: variance threshold -> mutual info selection -> scaling
print("Variance thresholding")
var_selector = VarianceThreshold(threshold=var_thresh_val)
X_optim_train = var_selector.fit_transform(X_optim_train)
X_optim_val = var_selector.transform(X_optim_val)
X_main = var_selector.transform(X_main)

print(f"After variance threshold: features = {X_optim_train.shape[1]}")

print(f"Selecting top {feature_k} features by mutual information")
mi_selector = SelectKBest(lambda X_, y_: mutual_info_regression(X_, y_, n_neighbors=10), k=min(feature_k, X_optim_train.shape[1]))
X_optim_train = mi_selector.fit_transform(X_optim_train, y_optim_train)
X_optim_val = mi_selector.transform(X_optim_val)
X_main = mi_selector.transform(X_main)

print("Scaling features (StandardScaler)")
scaler = StandardScaler()
X_optim_train = scaler.fit_transform(X_optim_train)
X_optim_val = scaler.transform(X_optim_val)
X_main = scaler.transform(X_main)

# Objective for Optuna
def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "verbosity": 0,
        "random_state": random_seed,
        "n_jobs": n_jobs,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds)

# Run Bayesian optimization (unless best params provided)
best_params_path = cfg.get("best_params", None)
best_params_file = os.path.join(tmp_dir, f"{suffix}_xgb_best_params.pkl")
if best_params_path:
    with open(best_params_path, "rb") as f:
        import pickle
        best_params = pickle.load(f)
    print("Loaded best params from", best_params_path)
else:
    print("Running Optuna Bayesian optimization")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, X_optim_train, y_optim_train, X_optim_val, y_optim_val), n_trials=n_trials)
    best_params = study.best_params
    # add fixed params
    best_params.update({
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "verbosity": 0,
        "random_state": random_seed,
        "n_jobs": n_jobs,
    })
    with open(best_params_file, "wb") as f:
        import pickle
        pickle.dump(best_params, f)
    print("Saved best params to", best_params_file)

print("Best params:", best_params)

# Cross-validation loop: evaluate for different training set sizes
result_loss = []
result_corr = []

print("Start cross-validation")
for size in train_sizes:
    print("Train size:", size)
    losses_for_size = []
    corrs_for_size = []

    for seed in range(n_resamples):
        print("  Iteration:", seed)
        torch.manual_seed(seed)
        perm = torch.randperm(len(main_indices))
        train_idx = perm[:size].cpu().numpy()
        test_idx = perm[size:].cpu().numpy()

        X_train = X_main[train_idx]
        y_train = y_main[train_idx]
        X_test = X_main[test_idx]
        y_test = y_main[test_idx]

        # Fit final model on training subset
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train, verbose=False)

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        try:
            pearson_corr, _ = pearsonr(y_test, preds)
        except Exception:
            pearson_corr = float("nan")

        losses_for_size.append(float(mse))
        corrs_for_size.append(float(pearson_corr))

        print(f"    MSE: {mse:.6f} | Pearson: {pearson_corr:.4f}")

        gc.collect()

    result_loss.append(losses_for_size)
    result_corr.append(corrs_for_size)

# Save results
np.save(os.path.join(res_dir, f"mse_{suffix}_xgb.npy"), np.array(result_loss, dtype=object))
np.save(os.path.join(res_dir, f"pearsonr_{suffix}_xgb.npy"), np.array(result_corr, dtype=object))
np.savetxt(os.path.join(res_dir, f"mse_{suffix}_xgb.csv"), np.array(result_loss, dtype=object), delimiter=",", fmt="%s")
np.savetxt(os.path.join(res_dir, f"pearsonr_{suffix}_xgb.csv"), np.array(result_corr, dtype=object), delimiter=",", fmt="%s")

print("Finished.")