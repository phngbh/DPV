import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """Dataset expects padded sequences with ID in column 0."""
    def __init__(self, data):
        # keep on cpu; move to device inside training loop
        assert isinstance(data, np.ndarray) or torch.is_tensor(data)
        if torch.is_tensor(data):
            data = data.cpu().numpy()
        self.data = data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

def make_padding_mask(batch_tensor: torch.Tensor, global_time_seq: np.ndarray, device=None):
    """
    batch_tensor: tensor shape (B, T, F) where first channel col 0 holds sample id index (int)
    global_time_seq: numpy array of lengths per original sample id
    returns: mask tensor (B, T, 1) float where valid positions are 1.0
    """
    if device is None:
        device = batch_tensor.device
    ids = batch_tensor[:, :, 0].long().cpu().numpy()[:, 0] if batch_tensor.dim() == 3 else batch_tensor[:,0].long().cpu().numpy()
    # If ids are not present, assume full length
    if ids.dtype == np.float32 or ids.dtype == np.float64:
        ids = ids.astype(int)
    lengths = np.array([global_time_seq[int(i)] for i in ids])
    max_len = batch_tensor.size(1)
    ranges = np.arange(max_len)[None, :]
    mask = (ranges < lengths[:, None]).astype(np.float32)
    mask_t = torch.from_numpy(mask).unsqueeze(-1).to(device)
    return mask_t

def custom_loss_discrete(output, target, mask):
    # BCE over masked positions
    bce = nn.BCELoss(reduction='none')
    loss_per_elem = bce(output, target)
    masked = loss_per_elem * mask
    return masked.sum() / max(1.0, mask.sum())

def custom_loss_numeric(output, target, mask):
    mse = (output - target) ** 2
    masked = mse * mask
    return masked.sum() / max(1.0, mask.sum())

def train_epoch(model, loader, optimizer, data_type, global_time_seq, device):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    for batch in loader:
        batch = batch.to(torch.device('cpu'))  # ensure tensor on CPU then move inputs to device
        inputs = batch[:, :, 1:].to(device).float()
        mask = make_padding_mask(batch, global_time_seq, device=device)
        optimizer.zero_grad()
        output, mean, logvar = model(inputs)
        if data_type == 'discrete':
            recon = custom_loss_discrete(output, inputs, mask)
        else:
            recon = custom_loss_numeric(output, inputs, mask)
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / inputs.size(0)
        loss = recon + kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        total_recon += recon.item() * inputs.size(0)
        total_kl += kl.item() * inputs.size(0)
    return total_loss / len(loader.dataset), total_recon / len(loader.dataset), total_kl / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, data_type, global_time_seq, device):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    for batch in loader:
        batch = batch.to(torch.device('cpu'))
        inputs = batch[:, :, 1:].to(device).float()
        mask = make_padding_mask(batch, global_time_seq, device=device)
        output, mean, logvar = model(inputs)
        if data_type == 'discrete':
            recon = custom_loss_discrete(output, inputs, mask)
        else:
            recon = custom_loss_numeric(output, inputs, mask)
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / inputs.size(0)
        total_loss += (recon + kl).item() * inputs.size(0)
        total_recon += recon.item() * inputs.size(0)
        total_kl += kl.item() * inputs.size(0)
    return total_loss / len(loader.dataset), total_recon / len(loader.dataset), total_kl / len(loader.dataset)

def reconstruct_and_save_batches(model, dataset, time_seq, res_dir, batch_size=200, device='cuda'):
    """
    Reconstruct dataset in batches, save temporary torch files, then return list of tmp filenames.
    """
    os.makedirs(res_dir, exist_ok=True)
    tmp_dir = os.path.join(res_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tmp_files = []
    model.eval()
    with torch.no_grad():
        idx = 0
        for batch in loader:
            batch = batch.to(torch.device('cpu'))
            inputs = batch[:, :, 1:].to(device).float()
            out, _, _ = model(inputs)
            fname = os.path.join(tmp_dir, f"reconstructed_part_{idx}.pt")
            torch.save(out.cpu(), fname)
            tmp_files.append(fname)
            idx += 1
            del out
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
    # Combine into list of numpy arrays trimmed by time_seq
    big_list = []
    start = 0
    for i, f in enumerate(tmp_files):
        part = torch.load(f)  # shape (B, T, F)
        B = part.size(0)
        t_chunk = time_seq[start:start+B]
        for j in range(B):
            tim = int(t_chunk[j])
            arr = part[j, :tim, :].cpu().numpy()
            big_list.append(arr)
        start += B
    # concatenate row-wise
    if len(big_list) > 0:
        big_array = np.concatenate(big_list, axis=0)
    else:
        big_array = np.empty((0, part.size(2)))
    # save combined numpy
    combined_file = os.path.join(res_dir, "reconstructed_combined.npy")
    np.save(combined_file, big_array)
    return tmp_files, combined_file