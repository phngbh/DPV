from typing import Callable, Tuple, Any, Dict
import copy
import pickle
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer

def train_model(
    model: torch.nn.Module,
    optimizer: Optimizer,
    criterion: Callable,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 50,
    patience: int = 5,
    use_amp: bool = True,
) -> Tuple[torch.nn.Module, float]:
    """
    Generic training loop with early stopping and optional AMP.
    Returns: (best_model (loaded weights), best_val_loss)
    """
    best_val_loss = float("inf")
    best_wts = copy.deepcopy(model.state_dict())
    no_improve = 0
    scaler = GradScaler() if use_amp and device.type == "cuda" else None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            # unpack; user may have different dataset tuple shapes
            inputs = batch[0].to(device).float()
            mask = None
            targets = None
            if len(batch) >= 3:
                # (inputs, seq_len, mask, targets) or (inputs, mask, targets)
                if batch[-1] is not None:
                    targets = batch[-1].to(device).float()
                if len(batch) >= 3:
                    mask = batch[2].to(device).float()
            # forward/backward with optional amp
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs, attention_mask=mask)
                    loss = criterion(outputs.squeeze(), targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs, attention_mask=mask)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_loader))

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device).float()
                mask = batch[2].to(device).float() if len(batch) >= 3 else None
                targets = batch[-1].to(device).float()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs, attention_mask=mask)
                        loss = criterion(outputs.squeeze(), targets)
                else:
                    outputs = model(inputs, attention_mask=mask)
                    loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / max(1, len(val_loader))

        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_wts)
    return model, best_val_loss

def objective_generic(
    trial,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    model_builder: Callable[[Any], Tuple[torch.nn.Module, list, Callable]],
    best_loss_path: str,
    best_model_path: str,
    epochs: int,
    patience: int,
) -> float:
    """
    Generic Optuna objective.
    model_builder(trial) -> (model, param_groups_for_optimizer, criterion)
    The caller chooses which trial.suggest_* to expose inside model_builder.
    """
    # load best overall (if exists)
    if os.path.exists(best_loss_path):
        with open(best_loss_path, "rb") as f:
            best_overall = pickle.load(f)
    else:
        best_overall = float("inf")

    model, param_groups, criterion = model_builder(trial)
    model.to(device)

    optimizer = torch.optim.AdamW(param_groups)

    model, best_val_loss = train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        patience=patience,
    )

    if best_val_loss < best_overall:
        with open(best_loss_path, "wb") as f:
            pickle.dump(best_val_loss, f)
        torch.save(model.state_dict(), best_model_path)

    return best_val_loss

class PaddedTSDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, indices=None):
        if indices is not None:
            X = X[indices]; y = y[indices]
        assert X.ndim == 3 and y.ndim == 1 and X.size(0) == y.size(0)
        self.X = X.float(); self.y = y.float()
    def __len__(self): return self.X.size(0)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_padding_mask(time_seq: torch.Tensor, max_len: int) -> torch.Tensor:
    if not torch.is_tensor(time_seq):
        time_seq = torch.as_tensor(time_seq, dtype=torch.long)
    time_seq = time_seq.long().clamp(min=0, max=max_len)
    range_tensor = torch.arange(max_len).unsqueeze(0).expand(time_seq.numel(), max_len)
    mask_np = (range_tensor < time_seq.unsqueeze(1)).int().numpy()
    mask_np = np.fliplr(mask_np)
    return torch.from_numpy(mask_np.copy()).long()

def train_one_epoch(model, loader, optim, criterion, device):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optim.zero_grad(set_to_none=True)
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X)
        loss = criterion(preds, y)
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)