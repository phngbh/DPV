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
    config = yaml.safe_load(f)['get_integrated_gradient']

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

print("Define gradient function")
def integrated_gradients(model, inputs, baseline, attention_mask, steps=50):
    # Scale inputs and compute gradients
    baseline = baseline.to(inputs.device).float()
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    grads = []

    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.to(inputs.device).float()
        scaled_input.requires_grad = True
        
        # Forward pass
        output, _ = model(scaled_input, attention_mask=attention_mask)
        output = output.squeeze(-1)  # Assuming a regression task with a single output
        
        # Zero gradients
        model.zero_grad()
        
        # Backward pass
        grad = torch.autograd.grad(outputs=output.sum(), inputs=scaled_input)[0]
        grads.append(grad.cpu().detach().numpy())
    
    grads = np.array(grads)  # Convert list of gradients to numpy array

    # Approximate the integral of gradients
    avg_grads = np.mean(grads[:-1] + grads[1:], axis=0) / 2.0
    integrated_grads = (inputs.cpu().detach().numpy() - baseline.cpu().detach().numpy()) * avg_grads

    return integrated_grads

def integrated_gradients_all_samples(model, data_loader, baseline, steps=50):
    model.eval()  # Set the model to evaluation mode
    all_integrated_grads = []
    
    for data, seq_length, mask, targets in data_loader:
        data, mask = data.to(device).float(), mask.to(device).float()
        
        # Compute integrated gradients for the current batch
        integrated_grads_batch = []
        for i in range(data.size(0)):  # Iterate over batch size
            input_sample = data[i:i+1]
            mask_sample = mask[i:i+1]
            baseline_sample = baseline[i:i+1]
            integrated_grads = integrated_gradients(model, input_sample, baseline_sample, attention_mask=mask_sample, steps=steps)
            integrated_grads_batch.append(integrated_grads)

        all_integrated_grads.append(np.concatenate(integrated_grads_batch, axis=0))

    # Concatenate integrated gradients across all batches
    all_integrated_grads = np.concatenate(all_integrated_grads, axis=0)  # Shape: (total_samples, seq_len, num_features)
    
    # Compute the average integrated gradients across all samples
    avg_integrated_grads = np.mean(np.abs(all_integrated_grads), axis=0)  # Shape: (seq_len, num_features)

    return all_integrated_grads, avg_integrated_grads

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

# Select relevant samples
time_sequence = torch.load(time_seq) - 1
indices_50 = np.where((time_sequence == 49))[0]
time_sequence = time_sequence[indices_50]
X = X[indices_50]
y = y[indices_50]

## Make attention mask
# Create a range tensor that matches the sequence length dimension
range_tensor = torch.arange(49).unsqueeze(0).expand(len(time_sequence), 49)
# Create the mask by comparing the range tensor with lengths tensor
attention_mask_np = (range_tensor < time_sequence.unsqueeze(1)).int().numpy()
attention_mask_np = np.fliplr(attention_mask_np)
attention_mask = torch.from_numpy(attention_mask_np.copy())

torch.manual_seed(93)
subset_indices = torch.randperm(len(X))[:500]
test_dataset = TensorDataset(X[subset_indices], time_sequence[subset_indices], attention_mask[subset_indices], y[subset_indices])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Compute gradient for a subset of data")
# Define a baseline (e.g., all-zero input)
baseline = torch.zeros_like(next(iter(test_loader))[0])

# Select a sample input from the test set
# data, seq_length, mask, targets = next(iter(test_loader))
# data, mask, targets = data.to(device).float(), mask.to(device).float(), targets.to(device).float()

# Compute integrated gradients
all_integrated_grads, avg_integrated_grads = integrated_gradients_all_samples(model, test_loader, baseline)

print("Save the gradients")
np.save(res_dir + "integrated_grads_avg" + suffix + ".npy", avg_integrated_grads)
np.save(res_dir + "integrated_grads_all" + suffix + ".npy", all_integrated_grads)
np.save(res_dir + "integrated_grads_y" + suffix + ".npy", y[subset_indices])

print('Finished running script')
print('Finishing time: ' + str(datetime.now().time()))