import argparse
import yaml
from datetime import datetime
import numpy as np
import os
import pickle
import gc
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW
from scipy.stats import pearsonr

from transformers import AutoConfig

from models.prime_llm import PRIME_LLM
from scripts.train_utils import train_model

print('Start running script')
print('Starting time: ' + str(datetime.now().time()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'The current device is {device}.')

# Parse arguments
parser = argparse.ArgumentParser(description='Arguments for running script')
parser.add_argument('--config', type=str, help='Path to the configuration file')
args = parser.parse_args()


# Load configuration
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# config options (CLI override for pretrained_model)
pretrained_model = args.pretrained_model or cfg['pretrained_model']
data_prefix = cfg['data_prefix']
target = cfg['target']
time_seq = cfg['time_seq']
subset_size = int(cfg['subset_size'])
train_size = int(cfg['train_size'])
val_size = int(cfg['val_size'])
res_dir = cfg['res_dir']
suffix = cfg['suffix']
tmp_dir = cfg.get('tmp_dir', './tmp/')
num_classes = cfg.get('num_classes', None)

use_lora = bool(cfg.get('use_lora', False))
n_resamples = int(cfg.get('n_resamples', 5))
epochs = int(cfg.get('epochs', 50))
patience = int(cfg.get('patience', 5))
best_params_path = cfg.get('best_params', None)
batch_size = int(cfg.get('batch_size', 16))

print(f"The LLM used is {pretrained_model}")

# Retrieve the hidden size from the model's configuration
hf_config = AutoConfig.from_pretrained(pretrained_model, trust_remote_code=True)
embedding_dim = getattr(hf_config, "embedding_size", hf_config.hidden_size)


# helper for loss selection
def get_criterion(num_classes):
    if num_classes is None:
        return MSELoss()
    elif num_classes == 2:
        return BCEWithLogitsLoss()
    else:
        return CrossEntropyLoss()

# load best params
if best_params_path is None:
    raise RuntimeError("No best parameters provided! Set best_params in the config or pass a path.")
with open(best_params_path, 'rb') as f:
    best_params = pickle.load(f)
print("Using provided best parameters for training:", best_params)

# cleaned model name for outputs
cleaned_pretrained_model = pretrained_model.split('/')[-1] if '/' in pretrained_model else pretrained_model

print('Start cross validation')
result_loss = []
result_corr = []
test_indices_list = []

# main resampling loop
for seed in range(5, n_resamples):
    print('Resample: ', seed)
    torch.manual_seed(seed)

    print("...Load data and set up input")
    data_0 = torch.load(os.path.join(data_prefix + '0.pth'))
    data_0 = data_0[:, 14:, :]

    y = torch.load(target)
    time_sequence = torch.load(time_seq) - 15

    print(f"...The remaining number of samples: {data_0.size(0)}. Shape of data is {data_0.shape}.")

    # build attention mask 
    max_seq = cfg.get('max_seq_len', data_0.shape[1])
    range_tensor = torch.arange(max_seq).unsqueeze(0).expand(len(time_sequence), max_seq)
    attention_mask_np = (range_tensor < time_sequence.unsqueeze(1)).int().numpy()
    attention_mask_np = np.fliplr(attention_mask_np)
    attention_mask = torch.from_numpy(attention_mask_np.copy())
    print("...The attention mask shape is ", attention_mask.shape)

    # subset sampling
    subset_indices = torch.randperm(data_0.size(0))[:subset_size]
    print(f'...Subset size is {subset_size}.')

    # deterministic split within subset (use different seeds if desired)
    rng = torch.Generator()
    rng.manual_seed(93)
    perm = subset_indices[torch.randperm(len(subset_indices), generator=rng)]
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    train_dataset = TensorDataset(data_0[train_idx], time_sequence[train_idx], attention_mask[train_idx], y[train_idx])
    val_dataset = TensorDataset(data_0[val_idx], time_sequence[val_idx], attention_mask[val_idx], y[val_idx])
    test_dataset = TensorDataset(data_0[test_idx], time_sequence[test_idx], attention_mask[test_idx], y[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_indices_list.append(test_idx.cpu().numpy())

    input_dim = data_0.size(2)
    print("...Initialize the model")

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
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        dropout=best_params['dropout'],
        pretrained_model=pretrained_model,
        num_classes=num_classes
    )
    model.to(device)

    criterion = get_criterion(num_classes)

    # parameter groups (same logic)
    param_groups = [
        {"params": model.embedding.parameters(), "lr": best_params['lr_embedding'], "weight_decay": best_params['weight_decay']},
        {"params": model.regressor.parameters(), "lr": best_params['lr_regressor'], "weight_decay": best_params['weight_decay']},
    ]
    if use_lora:
        adapter_params = [p for p in model.transformer.parameters() if p.requires_grad]
        param_groups.append({"params": adapter_params, "lr": best_params['lr_adapter'], "weight_decay": 0.0})
    else:
        param_groups.append({"params": model.transformer.parameters(), "lr": best_params['lr_adapter'], "weight_decay": best_params['weight_decay']})

    optimizer = AdamW(param_groups)

    print('...Train')
    model_trained, _ = train_model(
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

    # cleanup heavy objects we no longer need
    del data_0, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()

    # sliding-window evaluation (prediction points)
    result_loss_i = []
    result_corr_i = []
    for i in range(15):
        print(f"......Evaluate for prediction window {i}")
        if i == 0:
            test_loader_evaluate = test_loader
        else:
            data_i = torch.load(os.path.join(data_prefix + f"{i}.pth"))
            data_i = data_i[:, 14:, :]
            test_dataset_evaluate = TensorDataset(data_i[test_idx], time_sequence[test_idx], attention_mask[test_idx], y[test_idx])
            test_loader_evaluate = DataLoader(test_dataset_evaluate, batch_size=batch_size, shuffle=False)

        model_trained.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, seq_length, mask, targets in test_loader_evaluate:
                inputs = inputs.to(device).float()
                mask = mask.to(device).float()
                targets = targets.to(device).float()

                outputs = model_trained(inputs, attention_mask=mask)
                outputs = outputs.float().squeeze(-1)

                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())

                total_loss += criterion(outputs, targets).item()

        n_batches_eval = len(test_loader_evaluate) if len(test_loader_evaluate) > 0 else 1
        tloss = total_loss / n_batches_eval

        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        if torch.isnan(all_predictions).any() or torch.isinf(all_predictions).any():
            print("⚠️ NaN or Inf detected in predictions!")
            correlation = float('nan')
        else:
            correlation, _ = pearsonr(all_predictions.numpy(), all_targets.numpy())

        print(f"...Test Loss: {tloss}")
        print(f"...Test Pearson Correlation Coefficient: {correlation}")

        result_loss_i.append(tloss)
        result_corr_i.append(float(correlation))

        # free per-window data
        if i != 0:
            del data_i, test_dataset_evaluate, test_loader_evaluate
        gc.collect()
        torch.cuda.empty_cache()

    result_loss.append(result_loss_i)
    result_corr.append(result_corr_i)

    del model_trained, test_loader
    gc.collect()
    torch.cuda.empty_cache()

# save results
os.makedirs(res_dir, exist_ok=True)
np.save(os.path.join(res_dir, f"mse_{suffix}_{cleaned_pretrained_model}.npy"), np.array(result_loss, dtype=object))
np.save(os.path.join(res_dir, f"pearsonr_{suffix}_{cleaned_pretrained_model}.npy"), np.array(result_corr, dtype=object))
np.save(os.path.join(res_dir, f"test_indices_{suffix}_{cleaned_pretrained_model}.npy"), np.array(test_indices_list, dtype=object))

print("Finished.")