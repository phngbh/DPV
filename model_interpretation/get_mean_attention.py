import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import numpy as np
import argparse
from datetime import timedelta, datetime
import yaml

print('Start running script')
print('Starting time: ' + str(datetime.now().time()))

# Parse arguments
parser = argparse.ArgumentParser(description='Arguments for running script')
parser.add_argument('--config', type=str, help='Path to the configuration file')
args = parser.parse_args()

# Load configuration
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)['get_mean_attention']

# Set hyperparameters
attentions = config['attentions']
res_dir = config['res_dir']
suffix = config['suffix']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attentions = torch.load(attentions)

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
    
torch.save(mean_attentions,res_dir + "mean_attentions" + suffix + ".pth")