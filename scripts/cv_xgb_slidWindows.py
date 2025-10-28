import os
import gc
import argparse
from datetime import datetime
import warnings
import pickle

import yaml
import numpy as np
import torch
import optuna
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, mutual_info_regression, VarianceThreshold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

print("Start running script")
print("Starting time:", str(datetime.now().time()))

parser = argparse.ArgumentParser(description="XGBoost sliding-window CV (flattened time-series)")
parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

# config
data_prefix = cfg["data_prefix"]                 # prefix, script will append "0.pth", "1.pth", ...
target_path = cfg["target"]
subset_size = int(cfg.get("subset_size", 1000))
train_size = int(cfg.get("train_size", 200))
fs_sample = int(cfg.get("fs_sample", 1000))      # samples used for MI selection
feature_k = int(cfg.get("feature_k", 2000))
var_thresh_val = float(cfg.get("var_thresh", 1e-5))
n_resamples = int(cfg.get("n_resamples", 5))
seed0 = int(cfg.get("seed", 93))
windows = int(cfg.get("n_windows", 15))
res_dir = cfg.get("res_dir", "./results/")
tmp_dir = cfg.get("tmp_dir", "./tmp/")
suffix = cfg.get("suffix", "xgb_slidwin")
best_params_path = cfg.get("best_params", None)  # if provided, will be used; else run BO when run_bo true
run_bo = bool(cfg.get("run_bo", False))
n_trials = int(cfg.get("n_trials", 100))
random_state = int(cfg.get("random_state", seed0))
n_jobs = int(cfg.get("n_jobs", 8))

os.makedirs(res_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Helper: flatten torch tensor (N, T, F) -> (N, T*F) numpy
def flatten_tensor(t: torch.Tensor) -> np.ndarray:
    return t.reshape(t.size(0), -1).cpu().numpy()

# Optional: run BO on a small optimization subset to get best_params
def run_optuna_bo(X_train_fs, y_train_fs, X_val_fs, y_val_fs):
    def objective(trial):
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
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_fs, y_train_fs, eval_set=[(X_val_fs, y_val_fs)], early_stopping_rounds=20, verbose=False)
        preds = model.predict(X_val_fs)
        return mean_squared_error(y_val_fs, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    best.update({
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "verbosity": 0,
        "random_state": random_state,
        "n_jobs": n_jobs,
    })
    return best

print("Start cross validation (sliding windows)")
result_loss = []
result_corr = []
test_indices_list = []

for seed in range(seed0, seed0 + n_resamples):
    print("Resample:", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load base window 0
    data_0 = torch.load(os.path.join(data_prefix + "0.pth"))   # e.g. "/path/prefix_0.pth"
    # trim if specified by upstream pipeline (keeps same behaviour as prior version)
    if cfg.get("trim_first_timesteps", 14):
        data_0 = data_0[:, cfg.get("trim_first_timesteps", 14):, :]
    y = torch.load(target_path)

    # filter bad samples (same rule as original scripts)
    keep_idx = torch.where(y > -4)[0]
    data_0 = data_0[keep_idx]
    y = y[keep_idx]

    print(f"...Remaining samples: {data_0.size(0)}  | data shape: {data_0.shape}")

    # sample subset
    perm_all = torch.randperm(data_0.size(0))
    subset_indices = perm_all[:min(subset_size, data_0.size(0))]
    # deterministic shuffle for train/test split within subset
    rng = torch.Generator().manual_seed(93)
    shuffled = subset_indices[torch.randperm(len(subset_indices), generator=rng)]
    train_idx = shuffled[:min(train_size, len(shuffled))]
    test_idx = shuffled[min(train_size, len(shuffled)):]

    test_indices_list.append(test_idx.cpu().numpy())

    # flatten
    data_0_flat = flatten_tensor(data_0)

    data_train = data_0_flat[train_idx.cpu().numpy()]
    y_train = y[train_idx].cpu().numpy()
    data_test = data_0_flat[test_idx.cpu().numpy()]
    y_test = y[test_idx].cpu().numpy()

    # feature selection pipeline fit on small subset of training data
    fs_idx = np.random.RandomState(93).permutation(len(data_train))[:min(fs_sample, len(data_train))]
    data_fs = data_train[fs_idx]
    y_fs = y_train[fs_idx]

    # variance threshold
    var_selector = VarianceThreshold(threshold=var_thresh_val)
    data_fs = var_selector.fit_transform(data_fs)
    data_train_sel = var_selector.transform(data_train)
    data_test_sel = var_selector.transform(data_test)

    # mutual information selection
    k = min(feature_k, data_fs.shape[1])
    mi_selector = SelectKBest(lambda X, y: mutual_info_regression(X, y, n_neighbors=10), k=k)
    data_fs = mi_selector.fit_transform(data_fs, y_fs)
    data_train_sel = mi_selector.transform(data_train_sel)
    data_test_sel = mi_selector.transform(data_test_sel)

    # scaling
    scaler = StandardScaler()
    data_train_sel = scaler.fit_transform(data_train_sel)
    data_test_sel = scaler.transform(data_test_sel)

    # decide best_params: load or run BO on small train/val split
    if best_params_path:
        with open(best_params_path, "rb") as f:
            best_params = pickle.load(f)
        print("...Loaded best params from", best_params_path)
    elif run_bo:
        # split a small validation set from data_train_sel for BO
        X_bo_train, X_bo_val, y_bo_train, y_bo_val = train_test_split(data_fs, y_fs, test_size=0.2, random_state=93)
        print("...Running BO to find best XGBoost params")
        best_params = run_optuna_bo(X_bo_train, y_bo_train, X_bo_val, y_bo_val)
        # persist
        best_params_file = os.path.join(tmp_dir, f"{suffix}_best_params_seed{seed}.pkl")
        with open(best_params_file, "wb") as f:
            pickle.dump(best_params, f)
        print("...Saved BO params to", best_params_file)
    else:
        raise RuntimeError("No best_params provided and run_bo is False. Set 'best_params' path or enable 'run_bo' in config.")

    # train final model on selected features
    print("...Train final XGBoost model")
    model = xgb.XGBRegressor(**best_params)
    model.fit(data_train_sel, y_train, verbose=False)

    del data_train, data_fs
    gc.collect()

    # evaluate on sliding windows
    losses_w = []
    corrs_w = []
    for w in range(windows):
        print("......Evaluate window", w)
        if w == 0:
            data_w = data_0
        else:
            data_w = torch.load(os.path.join(data_prefix + f"{w}.pth"))
            if cfg.get("trim_first_timesteps", 14):
                data_w = data_w[:, cfg.get("trim_first_timesteps", 14):, :]
            data_w = data_w[keep_idx]   # apply same filtering

        data_w_flat = flatten_tensor(data_w)
        data_w_sel = var_selector.transform(data_w_flat[test_idx.cpu().numpy()])
        data_w_sel = mi_selector.transform(data_w_sel)
        data_w_sel = scaler.transform(data_w_sel)

        preds = model.predict(data_w_sel)
        mse = mean_squared_error(y_test, preds)
        try:
            p_corr, _ = pearsonr(y_test, preds)
        except Exception:
            p_corr = float("nan")

        print(f".........MSE: {mse:.6f} | Pearson: {p_corr:.4f}")
        losses_w.append(float(mse))
        corrs_w.append(float(p_corr))

        # free per-window memory
        del data_w, data_w_flat, data_w_sel
        gc.collect()

    result_loss.append(losses_w)
    result_corr.append(corrs_w)

    # free resample-level memory
    del model, data_0, data_test_sel, data_test, data_0_flat
    gc.collect()

# save results
os.makedirs(res_dir, exist_ok=True)
np.save(os.path.join(res_dir, f"mse_{suffix}.npy"), np.array(result_loss, dtype=object))
np.save(os.path.join(res_dir, f"pearsonr_{suffix}.npy"), np.array(result_corr, dtype=object))
np.save(os.path.join(res_dir, f"test_indices_{suffix}.npy"), np.array(test_indices_list, dtype=object))

print("Finished:", str(datetime.now().time()))