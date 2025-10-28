# Autoformer adapter (uses common base/utils)
from types import SimpleNamespace
from typing import Optional
import torch

from .official_adapter_base import _BaseOfficialAdapter, _import_official


class AutoformerOfficialAdapter(_BaseOfficialAdapter):
    def __init__(
        self,
        repo_path: str,
        input_dim: int = 700,
        seq_len: int = 50,
        label_len: int = 1,
        pred_len: int = 1,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        embed: str = "fixed",
        freq: str = "h",
        factor: int = 1,
        moving_avg: int = 25,
    ):
        Model = _import_official(repo_path, [
            ("models.Autoformer", "Model"),
            ("Autoformer.models.Autoformer", "Model"),
            ("models.autoformer", "Model"),
        ])
        args = SimpleNamespace(
            enc_in=input_dim, dec_in=input_dim, c_out=input_dim,
            seq_len=seq_len, label_len=label_len, pred_len=pred_len,
            output_attention=False, factor=factor, d_model=d_model, n_heads=n_heads,
            e_layers=e_layers, d_layers=1, d_ff=d_ff, activation='gelu',
            features='M', embed=embed, freq=freq, dropout=dropout, num_workers=0,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            moving_avg=moving_avg,
        )
        base = Model(args)
        super().__init__(base, input_dim=input_dim, seq_len=seq_len, label_len=label_len, pred_len=pred_len)