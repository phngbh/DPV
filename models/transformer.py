import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .embeddings import LinearEmbeddingLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Transformer(nn.Module):
    """
    Lightweight transformer encoder for time-series / tabular sequences.

    - Expects X: (batch, seq_len, input_dim)
    - timestamps: (batch, seq_len, 1)
    - attention_mask: (batch, seq_len) with 1 for valid tokens, 0 for padding (optional)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        output_dim: int = 1,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = LinearEmbeddingLayer(input_dim, hidden_dim, embedding_dim, dropout)
        self.timestamp_encoder = nn.Linear(1, embedding_dim)  # learnable timestamp encoding
        self.norm = nn.LayerNorm(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        if num_classes is None:
            self.regressor = nn.Linear(embedding_dim, output_dim)
            nn.init.kaiming_normal_(self.regressor.weight)
        else:
            self.classifier = nn.Linear(embedding_dim, num_classes)
            nn.init.kaiming_normal_(self.classifier.weight)

        self.to(device)

    def forward(self, X: torch.Tensor, timestamps: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Returns: (batch, output_dim) for regression or (batch, num_classes) for classification
        """
        # basic checks
        assert X.dim() == 3, "X must be (batch, seq_len, input_dim)"
        batch, seq_len, _ = X.shape

        # default attention mask -> all valid
        if attention_mask is None:
            attention_mask = torch.ones((batch, seq_len), dtype=X.dtype, device=X.device)

        # ensure mask types for downstream ops
        attn_float = attention_mask.to(dtype=X.dtype, device=X.device)  # used for masking embeddings
        padding_mask = ~(attention_mask.bool())  # True where padding (src_key_padding_mask expects True to mask)

        # Embedding and timestamp encoding
        X_embeds = self.embedding(X, attn_float)  # (batch, seq_len, embedding_dim)
        ts_enc = self.timestamp_encoder(timestamps)  # (batch, seq_len, embedding_dim)

        # combine, norm
        X_embeds = self.norm(F.relu(X_embeds + ts_enc))

        # transformer expects src_key_padding_mask with shape (batch, seq_len)
        transformer_output = self.transformer_encoder(X_embeds, src_key_padding_mask=padding_mask)

        # masked mean pooling
        attn_unsq = attn_float.unsqueeze(-1)  # (batch, seq_len, 1)
        masked_output = transformer_output * attn_unsq
        denom = attn_unsq.sum(dim=1, keepdim=True).clamp(min=1e-6)  # avoid div by zero
        pooled_output = masked_output.sum(dim=1) / denom  # (batch, embedding_dim)

        pooled_output = self.dropout(pooled_output)

        if self.num_classes is None:
            return self.regressor(pooled_output)
        else:
            return self.classifier(pooled_output)

