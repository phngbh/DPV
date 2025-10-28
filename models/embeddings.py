import torch.nn as nn
import torch.nn.functional as F

class LinearEmbeddingLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # weight init
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, X: torch.Tensor, attention_mask: torch.Tensor):
        # X: (batch, seq_len, input_dim)
        # attention_mask: (batch, seq_len) or broadcastable
        X = X * attention_mask.unsqueeze(-1)  # Apply mask
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = F.relu(self.fc2(X))
        X = self.dropout(X)
        return X