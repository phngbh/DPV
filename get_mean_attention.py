import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss
from sklearn.metrics import mean_squared_error as MSE
import gc
import numpy as np
import xgboost as xgb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attentions = torch.load("attentions_hba1c_18k_long.pth")

# Initialize a list to store the mean attention weights for each layer and head
mean_attentions = []

# Number of layers and heads in the model
num_layers = len(attentions[0])
num_heads = attentions[0][0].size(1)

# Loop through each layer and head
for layer in range(num_layers):
    layer_attentions = []
    for batch in attentions:
        layer_attentions.append(batch[layer])
    # Concatenate the attention tensors along the batch dimension
    layer_attentions = torch.cat(layer_attentions, dim=0)  # Shape: (total_samples, num_heads, seq_len, seq_len)
    mean_layer_attention = layer_attentions.mean(dim=0)  # Shape: (num_heads, seq_len, seq_len)
    mean_attentions.append(mean_layer_attention)
    
torch.save(mean_attentions,"mean_attentions_hba1c_18k_long.pth")