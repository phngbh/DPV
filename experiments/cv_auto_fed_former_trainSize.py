import os
import json
import argparse
import math
import random
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml
import optuna
from optuna.pruners import MedianPruner
import pandas as pd

from models.autoformer import AutoformerOfficialAdapter
from models.fedformer import FEDformerOfficialAdapter
from models.official_adapter_base import replace_left_padding
from experiments.train_utils import PaddedTSDataset, set_seed, make_padding_mask, train_one_epoch, evaluate

# --- model factory (keeps same behaviour as adapters) --------------------------------
def build_model(cfg_model: dict, arch: dict):
    model_type = cfg_model.get("type", "fedformer").lower()
    repo_path = cfg_model["repo_path"]
    common = dict(
        repo_path=repo_path,
        input_dim=arch["input_dim"],
        seq_len=arch["seq_len"],
        label_len=cfg_model.get("label_len", 3),
        pred_len=cfg_model.get("pred_len", 1),
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        e_layers=arch["e_layers"],
        d_ff=arch["d_ff"],
        dropout=arch["dropout"],
        embed=cfg_model.get("embed", "fixed"),
        freq=cfg_model.get("freq", "h"),
    )
    if model_type == "autoformer":
        return AutoformerOfficialAdapter(
            factor=arch.get("factor", 3),
            moving_avg=arch.get("moving_avg", 25),
            **common
        )
    else:
        return FEDformerOfficialAdapter(
            factor=arch.get("factor", 3),
            moving_avg=arch.get("moving_avg", 25),
            modes=cfg_model.get("modes", 8),
            mode_select=cfg_model.get("mode_select", "low"),
            **common
        )

# --- Stage 1: Bayesian optimization ---------------------------------------------------
def run_bo(cfg: Dict[str, Any], X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = cfg.get("seed", 42)
    cv_pool_n = cfg["stage2"]["cv_pool_n"]; bo_pool_n = cfg["stage1"]["bo_pool_n"]
    bo_train_n = cfg["stage1"]["bo_train_n"]; bo_train_ratio = cfg["stage1"]["bo_train_ratio"]
    assert X.size(0) >= cv_pool_n + bo_pool_n, "Not enough samples for requested pools"

    idx_all = np.arange(X.size(0))
    idx_bo_pool = idx_all[cv_pool_n:cv_pool_n + bo_pool_n]
    rng = np.random.default_rng(seed)
    idx_bo_subset = rng.choice(idx_bo_pool, size=bo_train_n, replace=False)
    n_train = int(round(bo_train_ratio * bo_train_n)); n_val = bo_train_n - n_train
    idx_bo_train = idx_bo_subset[:n_train]; idx_bo_val = idx_bo_subset[n_train:]

    X_bo_train, y_bo_train = X[idx_bo_train], y[idx_bo_train]
    X_bo_val, y_bo_val     = X[idx_bo_val],   y[idx_bo_val]

    def objective(trial: optuna.Trial):
        d_model = trial.suggest_categorical("d_model", cfg["bo"]["d_model_choices"])
        n_heads = trial.suggest_categorical("n_heads", cfg["bo"]["n_heads_choices"])
        e_layers= trial.suggest_int("e_layers", *cfg["bo"]["e_layers_range"])
        d_ff   = trial.suggest_categorical("d_ff", cfg["bo"]["d_ff_choices"])
        dropout= trial.suggest_float("dropout", *cfg["bo"]["dropout_range"])
        lr     = trial.suggest_float("lr", *cfg["bo"]["lr_range"])
        weight_decay = trial.suggest_float("weight_decay", *cfg["bo"]["weight_decay_range"])
        moving_avg = trial.suggest_categorical("moving_avg", cfg["bo"]["moving_avg_choices"])
        label_len = trial.suggest_categorical("label_len", cfg["bo"].get("label_len_choices", [1,2,3,4,5]))
        factor = trial.suggest_categorical("factor", cfg["bo"].get("factor_choices", [2,3,4]))

        pred_len = cfg["model"].get("pred_len", 1)
        L_dec = int(label_len) + int(pred_len)
        k = int(factor * math.log(max(L_dec, 2)))
        if k < 1 or k > L_dec:
            raise optuna.TrialPruned(f"Invalid top_k={k} for L_dec={L_dec}")

        arch = dict(
            input_dim=int(X.size(-1)), seq_len=int(X.size(1)),
            d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff,
            dropout=dropout, moving_avg=moving_avg, factor=factor, label_len=label_len
        )
        model = build_model(cfg["model"], arch).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        crit = nn.MSELoss()
        bs = cfg["train"]["batch_size"]; max_ep = cfg["train"]["max_epochs"]; pat = cfg["train"]["patience"]
        es = cfg["train"].get("early_stopping", True); min_delta = cfg["train"].get("min_delta", 0.0)

        tr_loader = DataLoader(PaddedTSDataset(X_bo_train, y_bo_train), batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
        va_loader = DataLoader(PaddedTSDataset(X_bo_val, y_bo_val), batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
        best_val = float("inf"); best_state = None; patience_ctr = 0
        for ep in range(max_ep):
            _ = train_one_epoch(model, tr_loader, optim, crit, device)
            val = evaluate(model, va_loader, crit, device)
            trial.report(val, step=ep)
            if trial.should_prune(): raise optuna.TrialPruned()
            if val < (best_val - min_delta):
                best_val = val; patience_ctr = 0
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            else:
                if es:
                    patience_ctr += 1
                    if patience_ctr >= pat: break

        if best_state is not None:
            trial.set_user_attr("best_state", best_state)
            trial.set_user_attr("arch", arch)
            trial.set_user_attr("optim", {"lr": lr, "weight_decay": weight_decay})
        return best_val

    study = optuna.create_study(direction="minimize", pruner=MedianPruner())
    study.optimize(objective, n_trials=cfg["bo"]["trials"], gc_after_trial=True)
    os.makedirs(cfg["paths"]["res_dir"], exist_ok=True); os.makedirs(cfg["paths"]["tmp_dir"], exist_ok=True)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(cfg["paths"]["res_dir"], f'{cfg["model"]["type"]}_bo_trials.csv'), index=False)
    best_trial = study.best_trial
    best_state = best_trial.user_attrs.get("best_state", None)
    arch = best_trial.user_attrs.get("arch", None)
    optim_params = best_trial.user_attrs.get("optim", None)
    if best_state is None or arch is None:
        raise RuntimeError("Best trial missing state/arch.")
    torch.save({"state_dict": best_state, "arch": arch, "optim": optim_params, "config": cfg},
               os.path.join(cfg["paths"]["tmp_dir"], f'{cfg["model"]["type"]}_bo_best.pt'))
    with open(os.path.join(cfg["paths"]["tmp_dir"], f'{cfg["model"]["type"]}_bo_best_params.json'), "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params, "best_trial_number": study.best_trial.number}, f, indent=2)
    return {"arch": arch, "optim": optim_params, "best_val": study.best_value}

# --- Stage 2: CV by train size -------------------------------------------------------
def run_cv_by_size(cfg: Dict[str,Any], X: torch.Tensor, y: torch.Tensor, bo_best: Dict[str,Any]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cv_pool_n = cfg["stage2"]["cv_pool_n"]; cv_val_n = cfg["stage2"]["cv_val_n"]
    repeats = cfg["stage2"]["repeats_per_size"]; sizes: List[int] = cfg["stage2"]["train_sizes"]
    assert cv_val_n < cv_pool_n, "cv_val_n must be smaller than cv_pool_n"
    N = X.size(0); assert N >= cv_pool_n + cfg["stage1"]["bo_pool_n"]
    idx_cv_pool = np.arange(0, cv_pool_n)

    arch = dict(bo_best["arch"]); arch["input_dim"] = int(X.size(-1)); arch["seq_len"] = int(X.size(1))
    opt_hp = bo_best.get("optim", {"lr": cfg["train"]["lr"], "weight_decay": cfg["train"]["weight_decay"]})
    bs = cfg["train"]["batch_size"]; max_ep = cfg["train"]["max_epochs"]; pat = cfg["train"]["patience"]
    es = cfg["train"].get("early_stopping", True); min_delta = cfg["train"].get("min_delta", 0.0)

    rows = []
    for S in sizes:
        if S + cv_val_n > cv_pool_n:
            print(f"[WARN] train_size {S} + val {cv_val_n} > pool {cv_pool_n}; skip.")
            continue
        for rep in range(repeats):
            rng = np.random.default_rng(cfg["seed"] + rep + S)
            perm = rng.permutation(idx_cv_pool)
            tr_idx = perm[:S]; va_idx = perm[S:S+cv_val_n]; te_idx = perm[S+cv_val_n:]
            tr_loader = DataLoader(PaddedTSDataset(X, y, tr_idx), batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
            va_loader = DataLoader(PaddedTSDataset(X, y, va_idx), batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
            te_loader = DataLoader(PaddedTSDataset(X, y, te_idx), batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

            model = build_model(cfg["model"], arch).to(device)
            ckpt = torch.load(os.path.join(cfg["paths"]["tmp_dir"], f'{cfg["model"]["type"]}_bo_best.pt'), map_location="cpu")
            model.load_state_dict(ckpt["state_dict"], strict=False)

            optim = torch.optim.AdamW(model.parameters(), lr=opt_hp.get("lr", cfg["train"]["lr"]),
                                      weight_decay=opt_hp.get("weight_decay", cfg["train"]["weight_decay"]))
            crit = nn.MSELoss()
            best_val = float("inf"); best_epoch = -1; patience_ctr = 0; best_state = None
            for ep in range(max_ep):
                _ = train_one_epoch(model, tr_loader, optim, crit, device)
                val = evaluate(model, va_loader, crit, device)
                if val < (best_val - min_delta):
                    best_val = val; best_epoch = ep; patience_ctr = 0
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                else:
                    if es:
                        patience_ctr += 1
                        if patience_ctr >= pat: break
            if best_state is not None:
                model.load_state_dict(best_state, strict=False)
            test_loss = evaluate(model, te_loader, crit, device)
            if cfg["stage2"].get("save_ft_checkpoints", False):
                torch.save({"state_dict": best_state or model.state_dict(), "size": S, "rep": rep, "config": cfg},
                           os.path.join(cfg["paths"]["tmp_dir"], f'{cfg["model"]["type"]}_size{S}_rep{rep}_best.pt'))
            rows.append({"train_size": int(S), "repeat": int(rep), "val_best": float(best_val), "best_epoch": int(best_epoch), "test_loss": float(test_loss)})
    df = pd.DataFrame(rows)
    os.makedirs(cfg["paths"]["res_dir"], exist_ok=True)
    csv_path = os.path.join(cfg["paths"]["res_dir"], f'{cfg["model"]["type"]}_cv_by_size.csv'); df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved CV-by-size results to {csv_path}")

# --- data loading / prep ----------------------------------------------------------------
def load_and_prepare(cfg: Dict[str,Any]):
    X = torch.load(cfg["paths"]["data_dir"])
    X = X[:, 4:, :]  # Remove the first 4 time steps (legacy)
    y = torch.load(cfg["paths"]["target_dir"])
    max_len = int(X.size(1))

    time_seq_path = cfg["paths"].get("time_seq_path", None)
    mask = None
    if time_seq_path is not None and os.path.exists(time_seq_path):
        if time_seq_path.endswith(".pth") or time_seq_path.endswith(".pt"):
            time_seq = torch.load(time_seq_path) - cfg.get("time_seq_offset", 5)
        elif time_seq_path.endswith(".npy"):
            time_seq = torch.from_numpy(np.load(time_seq_path)) - cfg.get("time_seq_offset", 5)
        else:
            raise ValueError("Unsupported time_seq_path extension. Use .pt or .npy")
        mask = make_padding_mask(time_seq, max_len=max_len)
    else:
        mask_path = os.path.join(os.path.dirname(cfg["paths"]["data_dir"]), "mask.pt")
        if os.path.exists(mask_path):
            mask = torch.load(mask_path).long()
        else:
            # fallback: non-zero rows
            mask = (X.abs().sum(dim=-1) != 0).long()

    total_needed = cfg["stage2"]["cv_pool_n"] + cfg["stage1"]["bo_pool_n"]
    if X.size(0) > total_needed:
        X = X[:total_needed].contiguous(); y = y[:total_needed].contiguous(); mask = mask[:total_needed].contiguous()

    if cfg["data"].get("replace_left_padding", True):
        X = replace_left_padding(X, mask, time_idx=cfg["data"].get("time_idx", None),
                                 backfill_time_linearly=cfg["data"].get("backfill_time_linearly", True))
    return X, y

# --- main -------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f: cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    os.makedirs(cfg["paths"]["res_dir"], exist_ok=True); os.makedirs(cfg["paths"]["tmp_dir"], exist_ok=True)
    X, y = load_and_prepare(cfg)
    bo_best = run_bo(cfg, X, y)
    print("[INFO] Stage 1 (BO) done. Best val:", bo_best["best_val"])
    run_cv_by_size(cfg, X, y, bo_best)

if __name__ == "__main__":
    main()