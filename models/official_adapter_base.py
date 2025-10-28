# Common utilities and base adapter for official Autoformer/FEDformer wrappers
from types import SimpleNamespace
from typing import Optional, List, Tuple
import sys
import torch
import torch.nn as nn


@torch.no_grad()
def replace_left_padding(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    time_idx: Optional[int] = None,
    backfill_time_linearly: bool = True
) -> torch.Tensor:
    assert x.ndim == 3 and mask.ndim == 2, "x [B,L,D], mask [B,L]"
    B, L, D = x.shape
    device = x.device
    mask = mask.to(dtype=torch.long, device=device)

    L_valid = mask.sum(dim=1)
    first_idx = (L - L_valid).clamp(min=0, max=L)
    b_idx = torch.arange(B, device=device)
    has_valid = (L_valid > 0)

    first_rows = torch.zeros(B, D, device=device, dtype=x.dtype)
    if has_valid.any():
        first_rows[has_valid] = x[has_valid, first_idx[has_valid], :]

    out = x.clone()

    t = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    prefix = (t < first_idx.unsqueeze(1)) & has_valid.unsqueeze(1)
    if prefix.any():
        out[prefix] = first_rows.unsqueeze(1).expand(B, L, D)[prefix]

    if (time_idx is not None) and backfill_time_linearly:
        two_plus = (L_valid >= 2)
        if two_plus.any():
            fidx = first_idx[two_plus]
            b2 = b_idx[two_plus]
            t_first   = x[b2, fidx,     time_idx]
            t_second  = x[b2, (fidx+1).clamp(max=L-1), time_idx]
            dt_first  = t_second - t_first

            if prefix.any():
                idx_map = -torch.ones(B, dtype=torch.long, device=device)
                idx_map[b2] = torch.arange(b2.numel(), device=device)
                t_pos = t[prefix]
                fidx_rep = first_idx.unsqueeze(1).expand_as(t)[prefix]
                b_rep = b_idx.unsqueeze(1).expand_as(t)[prefix]
                sel = idx_map[b_rep]
                valid_sel = (sel >= 0)
                if valid_sel.any():
                    k_steps = (fidx_rep[valid_sel] - t_pos[valid_sel]).to(out.dtype)
                    t_first_sel = t_first[sel[valid_sel]]
                    dt_sel = dt_first[sel[valid_sel]]
                    t_pad = t_first_sel - k_steps * dt_sel
                    tmp = out[prefix]
                    row_idx = torch.arange(tmp.size(0), device=device)[valid_sel]
                    tmp[row_idx, time_idx] = t_pad
                    out[prefix] = tmp

    return out


class _BaseOfficialAdapter(nn.Module):
    """
    Tiny wrapper to mimic the official experiment harness but with our tensors.
    IMPORTANT: for official (Auto/FED) decoders, x_dec length should be label_len + pred_len.
    """
    def __init__(self, base_model: nn.Module, input_dim: int, seq_len: int, label_len: int, pred_len: int):
        super().__init__()
        self.base = base_model
        self.readout = nn.Linear(input_dim, 1)

        # keep lengths for proper decoder shapes
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.input_dim = input_dim

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        # x_enc: [B, L, D]
        B, L, D = x_enc.shape
        assert D == self.input_dim, "Input dim mismatch"

        # Build decoder inputs the way the official models expect:
        dec_len = self.label_len + self.pred_len
        x_dec = torch.zeros(B, dec_len, D, dtype=x_enc.dtype, device=x_enc.device)

        # time markers are unused in our use-case; keep shapes consistent
        x_mark_enc = torch.zeros(B, L, 4, dtype=torch.float32, device=x_enc.device)
        x_mark_dec = torch.zeros(B, dec_len, 4, dtype=torch.float32, device=x_enc.device)

        y_mv = self.base(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B, pred_len, D] for official repos
        # We only need the next-step prediction -> take the final step if pred_len>0, else first
        step = min(self.pred_len - 1, 0) if self.pred_len > 0 else 0
        return self.readout(y_mv[:, step, :]).squeeze(-1)


def _import_official(repo_path: str, candidates: List[Tuple[str, str]]):
    """
    Try multiple possible import locations for the official repo/class.
    If repo_path is provided, append to sys.path.
    """
    if repo_path:
        if repo_path not in sys.path:
            sys.path.append(repo_path)
    import_errors = []
    for module_path, class_name in candidates:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            return getattr(mod, class_name)
        except Exception as e:
            import_errors.append(f"{module_path}.{class_name}: {repr(e)}")
    raise ImportError("Could not import official Model. Tried:\n  - " + "\n  - ".join(import_errors))