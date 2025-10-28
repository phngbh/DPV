# FEDformer adapter (Wavelets mode) — uses common base/utils
from types import SimpleNamespace
from typing import Optional
import torch

from .official_adapter_base import _BaseOfficialAdapter, _import_official


class FEDformerOfficialAdapter(_BaseOfficialAdapter):
    """
    FEDformer in Wavelets mode: avoids FourierCorrelation (no FFT-bin index crashes).
    """
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
        modes: int = 16,
        mode_select: str = "low",
    ):
        Model = _import_official(repo_path, [
            ("models.FEDformer", "Model"),
            ("FEDformer.models.FEDformer", "Model"),
            ("models.fedformer", "Model"),
        ])
        args = SimpleNamespace(
            enc_in=input_dim, dec_in=input_dim, c_out=input_dim,
            seq_len=seq_len, label_len=label_len, pred_len=pred_len,
            d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_layers=1, d_ff=d_ff,
            activation='gelu', dropout=dropout, factor=factor, output_attention=False,
            features='M', embed=embed, freq=freq, num_workers=0,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            version='Wavelets',
            moving_avg=moving_avg,
            mode_select=mode_select,
            modes=modes,
            wavelet='db4',
            L=3,
            base='legendre',
            cross_activation='tanh',
            embed_type=0,
        )
        base = Model(args)
        super().__init__(base, input_dim=input_dim, seq_len=seq_len, label_len=label_len, pred_len=pred_len)