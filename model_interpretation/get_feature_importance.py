import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformers import AutoConfig
from models import TransformerWithLinearEmbedding
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
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
    config = yaml.safe_load(f)['get_feature_importance']

# Set hyperparameters
trained_model = config['trained_model']
data = config['data']
target = config['target']
time_seq = config['time_seq']
dropout = config['dropout']
hidden_dim = config['hidden_dim']
epochs = config['epochs']
patience = config['patience']
res_dir = config['res_dir']
suffix = config['suffix']
pretrained_model = config['pretrained_model']

# Retrieve the hidden size from the model's configuration
config = AutoConfig.from_pretrained(pretrained_model)
embedding_dim = config.hidden_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_feature_importance(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_gradients = []

    for data, seq_length, mask, targets in data_loader:
        data, mask, targets = data.to(device).float(), mask.to(device).float(), targets.to(device).float()
        
        # Enable gradients for input
        data.requires_grad = True

        # Forward pass
        outputs, _ = model(data, attention_mask=mask)
        outputs = outputs.squeeze(-1)  # Assuming a regression task with a single output

        # Compute gradients with respect to the input data
        gradients = torch.autograd.grad(outputs, data, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

        all_gradients.append(gradients.detach().cpu().numpy())

    # Concatenate gradients across all batches
    all_gradients = np.concatenate(all_gradients, axis=0)  # Shape: (total_samples, seq_len, num_features)
    
    # Compute mean absolute gradients as feature importance scores
    feature_importance = np.mean(np.abs(all_gradients), axis=0)  # Mean across all samples
    
    return feature_importance

# Load samples
X = torch.load(data)
y = torch.load(target)

# Parameters
input_dim = X.size(2)  # Number of features per time point (HbA1C: 760, LDL: 749)
hidden_dim = hidden_dim
embedding_dim = embedding_dim  # Reduced number of features to be compatible with BERT
output_dim = 1  # Output dimension for regression
global_max_length = 49  # Set this to the global maximum sequence length

# Instantiate the model
model = TransformerWithLinearEmbedding(input_dim = input_dim, embedding_dim = embedding_dim, output_dim=output_dim, dropout = dropout, pretrained_model=pretrained_model)

# Load the fine-tuned model
model.load_state_dict(torch.load(trained_model))
model.to(device)

## Make attention mask
time_sequence = torch.load(time_seq) - 1
# Create a range tensor that matches the sequence length dimension
range_tensor = torch.arange(49).unsqueeze(0).expand(len(time_sequence), 49)
# Create the mask by comparing the range tensor with lengths tensor
attention_mask_np = (range_tensor < time_sequence.unsqueeze(1)).int().numpy()
attention_mask_np = np.fliplr(attention_mask_np)
attention_mask = torch.from_numpy(attention_mask_np.copy())
torch.manual_seed(93)
test_indices = torch.randperm(X.size(0))[18000:]
test_dataset = TensorDataset(X[test_indices], time_sequence[test_indices], attention_mask[test_indices], y[test_indices])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Compute feature importance using gradients
feature_importance = compute_feature_importance(model, test_loader, device)

np.save(res_dir + "feature_importance" + suffix + ".npy", feature_importance)

print('Finished running script')
print('Finishing time: ' + str(datetime.now().time()))