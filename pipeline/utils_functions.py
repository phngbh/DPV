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

# Mixed reconstruction loss: numeric columns -> MSE, discrete columns -> BCEWithLogits
def mixed_reconstruction_loss(output, target, mask, numeric_cols=None, discrete_cols=None, eps=1e-8):
    """
    output, target: (B, T, F)
    mask: (B, T, 1) with 1.0 for valid timesteps
    numeric_cols, discrete_cols: lists of indices in feature dim (0-based)
    returns: recon_loss scalar (torch)
    """
    device = output.device
    mask_f = mask  # (B, T, 1)
    total_loss = 0.0
    total_count = 0.0

    if numeric_cols:
        nc = torch.tensor(numeric_cols, dtype=torch.long, device=device)
        out_num = torch.index_select(output, 2, nc)    # (B,T,Num)
        tgt_num = torch.index_select(target, 2, nc)
        mse = (out_num - tgt_num) ** 2
        masked_mse = mse * mask_f
        num_count = mask_f.sum() * out_num.size(2) / mask_f.size(2) if mask_f.size(2)>0 else mask_f.sum()
        total_loss = total_loss + masked_mse.sum()
        total_count = total_count + masked_mse.numel() if masked_mse.numel()>0 else total_count

    if discrete_cols:
        dc = torch.tensor(discrete_cols, dtype=torch.long, device=device)
        out_dis = torch.index_select(output, 2, dc)   # logits
        tgt_dis = torch.index_select(target, 2, dc)
        bce = nn.BCEWithLogitsLoss(reduction='none')
        loss_dis = bce(out_dis, tgt_dis)
        masked_bce = loss_dis * mask_f
        total_loss = total_loss + masked_bce.sum()
        total_count = total_count + masked_bce.numel() if masked_bce.numel()>0 else total_count

    if total_count == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / (total_count + eps)

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

def train_epoch(model, loader, optimizer, numeric_cols, discrete_cols, global_time_seq, device, static_cols=None):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    for batch in loader:
        batch = batch.to(torch.device('cpu'))  # ensure tensor on CPU then move inputs to device
        inputs = batch[:, :, 1:].to(device).float()
        # build condition vector from static columns (take first timestep values)
        cond = None
        if static_cols is not None and len(static_cols) > 0:
            cond = inputs[:, 0, static_cols].to(device).float()
        mask = make_padding_mask(batch, global_time_seq, device=device)
        optimizer.zero_grad()
        output, mean, logvar = model(inputs, c=cond)
        recon = mixed_reconstruction_loss(output, inputs, mask, numeric_cols=numeric_cols, discrete_cols=discrete_cols)
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
def evaluate(model, loader, numeric_cols, discrete_cols, global_time_seq, device, static_cols=None):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    for batch in loader:
        batch = batch.to(torch.device('cpu'))
        inputs = batch[:, :, 1:].to(device).float()
        cond = None
        if static_cols is not None and len(static_cols) > 0:
            cond = inputs[:, 0, static_cols].to(device).float()
        mask = make_padding_mask(batch, global_time_seq, device=device)
        output, mean, logvar = model(inputs, c=cond)
        recon = mixed_reconstruction_loss(output, inputs, mask, numeric_cols=numeric_cols, discrete_cols=discrete_cols)
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / inputs.size(0)
        total_loss += (recon + kl).item() * inputs.size(0)
        total_recon += recon.item() * inputs.size(0)
        total_kl += kl.item() * inputs.size(0)
    return total_loss / len(loader.dataset), total_recon / len(loader.dataset), total_kl / len(loader.dataset)

def reconstruct_and_save_batches(
    model,
    dataset,
    time_seq,
    res_dir,
    batch_size=200,
    device='cuda',
    numeric_cols=None,
    scaler=None,
    discrete_cols=None,
    threshold=0.5,
    sample=False,
    rng_seed=42
):
    """
    Reconstruct dataset in batches, inverse-transform numeric columns using `scaler`,
    save temporary torch files, then return list of tmp filenames and combined numpy (unscaled).
    Params:
      numeric_cols: list of numeric column indices (0-based in model output)
      scaler: fitted sklearn StandardScaler (or object with inverse_transform)
      discrete_cols: list of column indices (0-based) within the model output to treat as binary.
      threshold / sample / rng_seed: for optional deterministic/stochastic binarization after calibration.
    Returns:
      tmp_files, combined_file (numpy .npy of concatenated reconstructed rows in original units)
    """
    os.makedirs(res_dir, exist_ok=True)
    tmp_dir = os.path.join(res_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tmp_files = []
    model.eval()
    device_t = torch.device(device) if isinstance(device, str) else device
    with torch.no_grad():
        idx = 0
        for batch in loader:
            batch = batch.to(torch.device('cpu'))
            inputs = batch[:, :, 1:].to(device_t).float()
            # build conditional vector if model expects it: we don't need it here for decode as model
            # expects c optionally; when reconstructing we will call model(inputs, c=...)
            # but inputs include static cols repeated so pass c below
            # derive cond if model has attribute cond_size > 0
            cond = None
            if hasattr(model, "cond_size") and model.cond_size > 0:
                # static columns are taken from first timestep of inputs
                # ensure static cols known outside; if not provided, assume first features are statics
                # the caller should set model.cond_size to match provided static_cols used during training
                cond = inputs[:, 0, :model.cond_size].to(device_t).float()
            out, _, _ = model(inputs, c=cond)
            fname = os.path.join(tmp_dir, f"reconstructed_part_{idx}.pt")
            torch.save(out.cpu(), fname)
            tmp_files.append(fname)
            idx += 1
            del out
            gc.collect()
            if device_t.type == 'cuda':
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
            arr = part[j, :tim, :].cpu().numpy()   # shape (tim, F), floats (scaled for numeric cols if scaler used)
            big_list.append(arr)
        start += B

    # concatenate row-wise
    if len(big_list) > 0:
        big_array = np.concatenate(big_list, axis=0)  # shape (total_tim, F)
    else:
        # if no data, create empty with zero feature dim fallback
        part = torch.zeros((0, 0))
        big_array = np.empty((0, part.size(1)))

    # inverse transform numeric columns if scaler provided
    if scaler is not None and numeric_cols is not None and len(big_array) > 0:
        numeric_cols = np.asarray(numeric_cols, dtype=int)
        # sklearn expects 2D array
        numeric_vals = big_array[:, numeric_cols]
        inv = scaler.inverse_transform(numeric_vals)
        big_array[:, numeric_cols] = inv

    # Note: calibration & binarization for discrete cols is handled externally (see caller),
    # but basic deterministic thresholding can be applied here if requested (not calibrated).
    if discrete_cols is not None and not sample:
        discrete_cols = np.asarray(discrete_cols, dtype=int)
        if len(big_array) > 0:
            big_array[:, discrete_cols] = (big_array[:, discrete_cols] >= float(threshold)).astype(big_array.dtype)

    combined_file = os.path.join(res_dir, "reconstructed_combined.npy")
    np.save(combined_file, big_array)
    return tmp_files, combined_file