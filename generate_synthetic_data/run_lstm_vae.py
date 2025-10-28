import os
import argparse
from datetime import datetime
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# load data
data = np.load(data_path, allow_pickle=True)
time_seq = np.load(time_seq_path, allow_pickle=True)

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

# split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_ds = TimeSeriesDataset(train_data)
test_ds = TimeSeriesDataset(test_data)

# model
model = VAE(seq_len=seq_len, input_size=input_size, hidden_size=hidden_size,
            latent_size=latent_size, num_layers=num_layers, data_type=data_type).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

# training
for epoch in range(num_epochs):
    tr_loss, tr_recon, tr_kl = train_epoch(model, DataLoader(train_ds, batch_size=batch_size, shuffle=True), optimizer, data_type, time_seq, device)
    val_loss, val_recon, val_kl = evaluate(model, DataLoader(test_ds, batch_size=batch_size, shuffle=False), data_type, time_seq, device)
    print(f"Epoch {epoch+1}/{num_epochs} | train loss {tr_loss:.4f} recon {tr_recon:.4f} kl {tr_kl:.4f} | val loss {val_loss:.4f}")

# save model
torch.save(model.state_dict(), os.path.join(res_dir, f"lstm_vae_{data_type}_final.pth"))

# reconstruct and save synthetic outputs (tmp parts + combined numpy)
tmp_files, combined_file = reconstruct_and_save_batches(model, TimeSeriesDataset(data), time_seq, res_dir, batch_size=int(cfg.get("gen_batch_size",200)), device=device.type)
print("Saved reconstructed parts:", tmp_files)
print("Saved combined numpy:", combined_file)
print("Finished:", datetime.now())