import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMForecaster(nn.Module):
    """
    LSTM-based forecaster for variable-length sequences.

    Args:
      input_dim: number of features per time step
      hidden_dim: hidden size of the LSTM
      dropout: dropout probability applied to final hidden state
      output_dim: regression output size (default 1)
      num_layers: number of LSTM layers
      num_classes: if provided, performs classification with this number of classes
      bidirectional: whether to use a bidirectional LSTM
    Forward:
      x: Tensor (batch, seq_len, input_dim)
      lengths: 1D tensor or list of sequence lengths (batch, )
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        output_dim: int = 1,
        num_layers: int = 2,
        num_classes: Optional[int] = None,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

        final_dim = hidden_dim * self.num_directions
        self.num_classes = num_classes
        if num_classes is None:
            self.regressor = nn.Linear(final_dim, output_dim)
            nn.init.kaiming_normal_(self.regressor.weight)
        else:
            self.classifier = nn.Linear(final_dim, num_classes)
            nn.init.kaiming_normal_(self.classifier.weight)

    def forward(self, x: torch.Tensor, lengths: Union[torch.Tensor, Sequence[int]]):
        # x: (batch, seq_len, input_dim)
        # lengths: (batch,)
        if isinstance(lengths, torch.Tensor):
            lengths_cpu = lengths.cpu()
        else:
            lengths_cpu = torch.tensor(lengths, dtype=torch.long)

        # pack padded sequence (enforce_sorted=False allows arbitrary order)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(packed_input)

        # ht: (num_layers * num_directions, batch, hidden_dim)
        if self.bidirectional:
            # last layer forward is ht[-2], last layer backward is ht[-1]
            last_hidden = torch.cat([ht[-2], ht[-1]], dim=1)  # (batch, hidden_dim * 2)
        else:
            last_hidden = ht[-1]  # (batch, hidden_dim)

        last_hidden = self.dropout(last_hidden)

        if self.num_classes is None:
            return self.regressor(last_hidden)
        else:
            return self.classifier(last_hidden)

