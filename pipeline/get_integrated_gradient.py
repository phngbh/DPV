import os
import argparse
from datetime import datetime

import yaml
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoConfig
from models.prime_llm import PRIME_LLM
from scipy.stats import pearsonr

print('Start running script')
print('Starting time: ' + str(datetime.now().time()))

parser = argparse.ArgumentParser(description='Compute Integrated Gradients for PRIME_LLM')
parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
parser.add_argument('--steps', type=int, default=50, help='Number of IG steps (default: 50)')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for IG computation')
parser.add_argument('--use_abs_avg', action='store_true', help='Save average of absolute IGs (default: mean of signed IGs)')
parser.add_argument('--baseline_mode', type=str, choices=['zero', 'mean'], default=None, help='Baseline mode override (zero|mean)')
parser.add_argument('--target_index', type=int, default=None, help='Optional target index for multiclass outputs')
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

trained_model_path = cfg['trained_model']
data_path = cfg['data']
res_dir = cfg['res_dir']
suffix = cfg.get('suffix', '')
pretrained_model = cfg['pretrained_model']
best_params_pkl = cfg['best_params']
num_classes = cfg.get('num_classes', None)

os.makedirs(res_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# helper: load tensors saved as (X, time_seq, attention_mask, y) or similar
loaded = torch.load(data_path)
if isinstance(loaded, (list, tuple)):
    dataset = TensorDataset(*loaded)
elif isinstance(loaded, dict) and 'tensors' in loaded:
    dataset = TensorDataset(*loaded['tensors'])
else:
    raise RuntimeError("Expected data file to contain tensors compatible with TensorDataset (list/tuple or dict['tensors'])")

# get embedding dim from HF config
hf_config = AutoConfig.from_pretrained(pretrained_model, trust_remote_code=True)
embedding_dim = getattr(hf_config, "embedding_size", hf_config.hidden_size)

# build model
example_X = dataset.tensors[0]
input_dim = example_X.size(2)
with open(best_params_pkl, 'rb') as f:
    best_params = pickle.load(f)
print("Using best params:", best_params)

model = PRIME_LLM(
    input_dim=input_dim,
    hidden_dim=best_params['hidden_dim'],
    embedding_dim=embedding_dim,
    dropout=best_params.get('dropout', 0.0),
    pretrained_model=pretrained_model,
    use_lora=best_params.get('use_lora', True),
    lora_r=best_params.get('lora_r'),
    lora_alpha=best_params.get('lora_alpha'),
    lora_dropout=best_params.get('lora_dropout'),
    num_classes=num_classes
)
model.to(device)

# load weights robustly
state = torch.load(trained_model_path, map_location='cpu')
if isinstance(state, dict) and 'state_dict' in state:
    sd = state['state_dict']
else:
    sd = state
try:
    model.load_state_dict(sd)
except RuntimeError:
    model.load_state_dict(sd, strict=False)
model.eval()

# integrated gradients implementation (batch-wise)
def integrated_gradients_batch(model, inputs, baseline, attention_mask, steps=50, device='cpu', target_index=None):
    """
    inputs, baseline: tensors shape (B, T, F)
    attention_mask: tensor shape (B, T, ...) passed to model
    returns: integrated_grads tensor (B, T, F) on CPU
    """
    model.zero_grad()
    inputs = inputs.to(device).float()
    baseline = baseline.to(device).float()
    mask = attention_mask.to(device) if attention_mask is not None else None

    delta = inputs - baseline
    B = inputs.size(0)
    grads_accum = torch.zeros_like(inputs, device=device)

    for k in range(1, steps + 1):
        alpha = float(k) / steps
        scaled = (baseline + alpha * delta).requires_grad_(True)
        # forward (robust to signature)
        try:
            out = model(scaled, attention_mask=mask)
        except TypeError:
            out = model(scaled)
        # select scoring function
        if out.ndim == 2 and out.size(1) > 1:
            if target_index is None:
                scores = out.max(dim=1)[0]
            else:
                scores = out[:, target_index]
        else:
            scores = out.view(B, -1).sum(dim=1)
        grads = torch.autograd.grad(outputs=scores.sum(), inputs=scaled, create_graph=False, retain_graph=False, allow_unused=True)[0]
        if grads is None:
            grads = torch.zeros_like(scaled)
        grads_accum += grads
        # free
        del scaled, out, grads
    avg_grads = grads_accum / steps
    integrated = delta * avg_grads
    return integrated.detach().cpu()

# driver to compute IGs over dataset in batches
def compute_igs(model, dataloader, baseline_mode='zero', steps=50, device='cpu', target_index=None):
    all_igs = []
    for batch in dataloader:
        X = batch[0]
        attention_mask = None
        if len(batch) >= 3:
            if isinstance(batch[2], torch.Tensor) and batch[2].dim() == X.dim():
                attention_mask = batch[2]
        # baseline selection (override CLI -> args or cfg)
        bm = args.baseline_mode or baseline_mode
        if bm == 'zero' or bm is None:
            baseline = torch.zeros_like(X)
        elif bm == 'mean':
            baseline = torch.mean(X, dim=0, keepdim=True).expand_as(X)
        else:
            baseline = torch.zeros_like(X)
        ig_batch = integrated_gradients_batch(model, X, baseline, attention_mask, steps=steps, device=device, target_index=target_index)
        all_igs.append(ig_batch.numpy())
    all_igs = np.concatenate(all_igs, axis=0)
    if args.use_abs_avg:
        avg_ig = np.mean(np.abs(all_igs), axis=0)
    else:
        avg_ig = np.mean(all_igs, axis=0)
    return all_igs, avg_ig

# prepare subset (filter full-length if requested in config)
print('Load and filter data')
X_all, *rest = dataset.tensors
time_seq = None
attention_mask = None
y_all = None
if len(rest) == 3:
    time_seq, attention_mask, y_all = rest
elif len(rest) == 2:
    attention_mask, y_all = rest
elif len(rest) == 1:
    y_all = rest[0]

full_seq_len = cfg.get('filter_seq_len', None)
if full_seq_len is not None and time_seq is not None:
    idxs = (time_seq == full_seq_len).nonzero(as_tuple=True)[0]
else:
    idxs = torch.arange(X_all.size(0))

subset_n = int(min(len(idxs), cfg.get('max_samples', 1000)))
torch.manual_seed(cfg.get('seed', 93))
perm = torch.randperm(len(idxs))[:subset_n]
selected = idxs[perm]

selected_tensors = [t[selected] for t in dataset.tensors]
sub_dataset = TensorDataset(*selected_tensors)
dataloader = DataLoader(sub_dataset, batch_size=args.batch_size, shuffle=False)

print(f"Computing Integrated Gradients on {len(sub_dataset)} samples (steps={args.steps})")
all_igs, avg_ig = compute_igs(model, dataloader, baseline_mode=cfg.get('baseline_mode', 'zero'), steps=args.steps, device=device, target_index=args.target_index)

# save outputs
fname_all = os.path.join(res_dir, f"integrated_grads_all{('_'+suffix) if suffix else ''}.npy")
fname_avg = os.path.join(res_dir, f"integrated_grads_avg{('_'+suffix) if suffix else ''}.npy")
np.save(fname_all, all_igs)
np.save(fname_avg, avg_ig)

# also save indices and dataset tensors subset for reproducibility
torch.save(sub_dataset.tensors, os.path.join(res_dir, f"integrated_grads_dataset{('_'+suffix) if suffix else ''}.pth"))
with open(os.path.join(res_dir, f"integrated_grads_meta{('_'+suffix) if suffix else ''}.pkl"), 'wb') as fh:
    pickle.dump({'config': cfg, 'selected_idx': selected.cpu().numpy()}, fh)

print("Saved:", fname_all, fname_avg)
print('Finished running script')
print('Finishing time: ' + str(datetime.now().time()))