import os
import argparse
from datetime import datetime
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

from models.lstm_vae import VAE
from .utils_functions import TimeSeriesDataset, train_epoch, evaluate, reconstruct_and_save_batches

print("Start LSTM-VAE synthetic generation")
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)['run_lstm_vae']

# config
data_path = cfg["data"]
time_seq_path = cfg["time_seq"]
seq_len = int(cfg["seq_len"])
input_size = int(cfg["input_size"])
hidden_size = int(cfg["hidden_size"])
latent_size = int(cfg["latent_size"])
num_layers = int(cfg.get("num_layers", 1))
batch_size = int(cfg.get("batch_size", 64))
num_epochs = int(cfg.get("epoch", 50))
lr = float(cfg.get("learning_rate", 1e-3))
data_type = cfg.get("data_type", "numeric")
res_dir = cfg.get("res_dir", "./results/lstm_vae/")
os.makedirs(res_dir, exist_ok=True)

# feature index groups (relative to model input -- i.e. after removing ID column)
numeric_cols = cfg.get("numeric_cols", [])        # e.g. [0,1,2,3]
discrete_cols = cfg.get("discrete_cols", [])      # e.g. [4,5]
static_cols = cfg.get("static_cols", [])          # e.g. [0,1] indices among input features that are static per patient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# load data
data = np.load(data_path, allow_pickle=True)    # expected shape (N, T, F) or list->stacked
time_seq = np.load(time_seq_path, allow_pickle=True)
time_seq = np.asarray(time_seq).astype(int).ravel()

# add ID column if not present (same format as original)
if data.shape[-1] == input_size:  # no id col present
    new_data = []
    for i in range(len(data)):
        vec = np.full((data[i].shape[0], 1), i, dtype=np.int32)
        tmp = np.concatenate((vec, data[i]), axis=1)
        new_data.append(tmp)
    data = np.stack(new_data)
else:
    # assume already has id in first column
    pass

# split indices to also split time_seq correctly
indices = np.arange(len(data))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_data = data[train_idx]
test_data = data[test_idx]
train_time_seq = time_seq[train_idx]
test_time_seq = time_seq[test_idx]

# prepare inputs (without ID) for scaling
train_inputs = train_data[:, :, 1:]   # shape (N_train, T, F_in)
all_inputs = data[:, :, 1:]           # shape (N, T, F_in)

# fit scaler on numeric columns using only valid timesteps
numeric_cols = [int(x) for x in numeric_cols]
if len(numeric_cols) > 0:
    rows = []
    for i in range(train_inputs.shape[0]):
        t = int(train_time_seq[i])
        if t > 0:
            rows.append(train_inputs[i, :t, numeric_cols])
    if len(rows) > 0:
        stack_rows = np.concatenate(rows, axis=0)
    else:
        stack_rows = np.empty((0, len(numeric_cols)))
    scaler = StandardScaler().fit(stack_rows)
    # apply scaler to all_inputs only on valid timesteps, leave padding unchanged
    scaled_all_inputs = all_inputs.copy()
    for i in range(all_inputs.shape[0]):
        t = int(time_seq[i])
        if t > 0:
            scaled_all_inputs[i, :t, numeric_cols] = scaler.transform(all_inputs[i, :t, numeric_cols])
else:
    scaler = None
    scaled_all_inputs = all_inputs.copy()

# reassemble scaled data with ID column at front so dataset format remains (ID, features...)
scaled_full = np.concatenate([data[:, :, :1], scaled_all_inputs], axis=2)

# split scaled arrays into train/test datasets
train_scaled = scaled_full[train_idx]
test_scaled = scaled_full[test_idx]

train_ds = TimeSeriesDataset(train_scaled)
test_ds = TimeSeriesDataset(test_scaled)

# compute training prevalences for discrete cols (use original unscaled values)
discrete_cols = [int(x) for x in discrete_cols]
train_prevalences = None
if len(discrete_cols) > 0:
    # flatten valid timesteps only
    vals = []
    for i in range(train_data.shape[0]):
        t = int(train_time_seq[i])
        if t > 0:
            vals.append(train_data[i, :t, 1:][:, discrete_cols])
    if len(vals) > 0:
        vals_all = np.concatenate(vals, axis=0)
        # handle 2D when single column
        train_prevalences = np.mean(vals_all, axis=0)
    else:
        train_prevalences = np.zeros(len(discrete_cols))

# model
model = VAE(
    seq_len=seq_len,
    input_size=input_size,
    hidden_size=hidden_size,
    latent_size=latent_size,
    num_layers=num_layers,
    data_type=data_type,
    cond_size=len(static_cols)
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

# training
for epoch in range(num_epochs):
    tr_loss, tr_recon, tr_kl = train_epoch(model, DataLoader(train_ds, batch_size=batch_size, shuffle=True), optimizer, data_type, time_seq, device, static_cols=static_cols)
    val_loss, val_recon, val_kl = evaluate(model, DataLoader(test_ds, batch_size=batch_size, shuffle=False), data_type, time_seq, device, static_cols=static_cols)
    print(f"Epoch {epoch+1}/{num_epochs} | train loss {tr_loss:.4f} recon {tr_recon:.4f} kl {tr_kl:.4f} | val loss {val_loss:.4f}")

# save model and scaler and metadata
torch.save(model.state_dict(), os.path.join(res_dir, f"lstm_vae_{data_type}_final.pth"))
if scaler is not None:
    with open(os.path.join(res_dir, "scaler.pkl"), "wb") as fh:
        pickle.dump({"scaler": scaler, "numeric_cols": numeric_cols, "discrete_cols": discrete_cols, "static_cols": static_cols}, fh)

# reconstruct and save synthetic outputs (tmp parts + combined numpy in original units)
tmp_files, combined_file = reconstruct_and_save_batches(
    model,
    TimeSeriesDataset(scaled_full),
    time_seq,
    res_dir,
    batch_size=int(cfg.get("gen_batch_size", 200)),
    device=device,
    numeric_cols=numeric_cols,
    scaler=scaler,
    discrete_cols=None,    # do not threshold here; we'll calibrate below
    sample=False
)
print("Saved reconstructed (pre-calibration) parts:", tmp_files)
print("Saved reconstructed (pre-calibration) combined numpy:", combined_file)

# load reconstructed floats (numeric already inverse-transformed inside reconstruct if scaler provided)
recon = np.load(combined_file, allow_pickle=False)

# calibrate thresholds for discrete columns to match training prevalences (if provided)
if len(discrete_cols) > 0 and train_prevalences is not None and recon.size > 0:
    for i_col, col in enumerate(discrete_cols):
        probs = recon[:, col]
        p_target = float(train_prevalences[i_col])
        # edge cases
        if p_target <= 0:
            thresh = 1.0
        elif p_target >= 1:
            thresh = 0.0
        else:
            thresh = np.quantile(probs, 1.0 - p_target)
        recon[:, col] = (probs >= thresh).astype(recon.dtype)
    # save calibrated binarized output
    calibrated_file = os.path.join(res_dir, "reconstructed_combined_calibrated.npy")
    np.save(calibrated_file, recon)
    print("Saved calibrated & binarized reconstructed data to:", calibrated_file)
else:
    print("No discrete columns to calibrate or no training prevalence available.")

print("Finished:", datetime.now())