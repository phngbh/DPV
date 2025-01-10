import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig
from models import TransformerWithLinearEmbedding, TransformerWithLSTMEmbedding
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
    config = yaml.safe_load(f)['run_transformer']

# Set hyperparameters
data = config['data']
target = config['target']
time_seq = config['time_seq']
train_size = config['train_size']
val_size = config['val_size']
dropout = config['dropout']
hidden_dim = config['hidden_dim']
lstm_layers = config['lstm_layers']
lr = config['lr']
weight_decay = config['weight_decay']
epochs = config['epochs']
patience = config['patience']
res_dir = config['res_dir']
suffix = config['suffix']
pretrained_model = config['pretrained_model']

# Retrieve the hidden size from the model's configuration
config = AutoConfig.from_pretrained(pretrained_model)
embedding_dim = config.hidden_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print("Load data and set up input")
# Load data
X = torch.load(data)
y = torch.load(target)

## Make attention mask
time_sequence = torch.load(time_seq) - 1
# Create a range tensor that matches the sequence length dimension
range_tensor = torch.arange(49).unsqueeze(0).expand(len(time_sequence), 49)
# Create the mask by comparing the range tensor with lengths tensor
attention_mask_np = (range_tensor < time_sequence.unsqueeze(1)).int().numpy()
attention_mask_np = np.fliplr(attention_mask_np)
attention_mask = torch.from_numpy(attention_mask_np.copy())

# Set the seed for reproducibility and randomly choose 1000 indices without replacement
torch.manual_seed(93)
train_indices = torch.randperm(X.size(0))[:train_size]
torch.manual_seed(93)
val_indices = torch.randperm(X.size(0))[train_size:(train_size+val_size)]
torch.manual_seed(93)
test_indices = torch.randperm(X.size(0))[(train_size+val_size):]  # all indices except the training ones

# Create the dataset and data loader
train_dataset = TensorDataset(X[train_indices], time_sequence[train_indices], attention_mask[train_indices], y[train_indices])
val_dataset = TensorDataset(X[val_indices], time_sequence[val_indices], attention_mask[val_indices], y[val_indices])
test_dataset = TensorDataset(X[test_indices], time_sequence[test_indices], attention_mask[test_indices], y[test_indices])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Parameters for the model
input_dim = X.size(2)  # Number of features per time point (HbA1C: 760, LDL: 749)
hidden_dim = hidden_dim
lstm_layers = lstm_layers
embedding_dim = embedding_dim  # Reduced number of features to be compatible with BERT
output_dim = 1  # Output dimension for regression
global_max_length = 49  # Set this to the global maximum sequence length
learning_rate = lr
weight_decay = weight_decay
epochs = epochs
patience = patience

# Instantiate the model
model = TransformerWithLinearEmbedding(input_dim = input_dim, embedding_dim = embedding_dim, output_dim=output_dim, dropout = dropout, pretrained_model=pretrained_model)
#model = TransformerWithLinearEmbedding(input_dim = input_dim, hidden_dim = hidden_dim, lstm_layers = lstm_layers, embedding_dim = embedding_dim, output_dim=output_dim, global_max_length=global_max_length, pretrained_model=pretrained_model)
model.to(device)

print("Start training")
# Optimizer and loss function
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = MSELoss()

# Early stopping parameters
patience = patience
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

# Fine-tuning (simplified)
for epoch in range(epochs):  # Number of epochs
    
    if early_stop:
        print("Early stopping")
        break
        
    model.train()
    total_loss = 0
    for data, seq_length, mask, targets in train_loader:
        data, mask, targets = data.to(device).float(), mask.to(device).float(), targets.to(device).float()

        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions, _ = model(data, attention_mask = mask)
        predictions = predictions.squeeze(-1)
        
        # Loss calculation
        loss = criterion(predictions, targets)

        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()

        total_loss += loss.item()
    print(f'...Epoch {epoch + 1}: Loss = {total_loss / len(train_loader)}')
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, seq_length, mask, targets in val_loader:
            data, mask, targets = data.to(device).float(), mask.to(device).float(), targets.to(device).float()
            predictions, _ = model(data, attention_mask=mask)
            predictions = predictions.squeeze(-1)
            loss = criterion(predictions, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f'...Validation Loss: {val_loss}')

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            early_stop = True

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
            
# Function to compute correlation
def pearson_corrcoef(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x - mean_x
    ym = y - mean_y
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2)) * torch.sqrt(torch.sum(ym ** 2))
    r = r_num / r_den
    return r

print("Evaluate")
model.eval()
total_loss = 0
all_predictions = []
all_targets = []
all_attentions = []

with torch.no_grad():
    for inputs, seq_length, mask, targets in test_loader:
        inputs, targets = inputs.to(device).float(), targets.to(device).float()
        mask = mask.to(device).float()
        outputs, attentions = model(inputs, attention_mask = mask)
        outputs = outputs.squeeze(-1)

        # Collect predictions and targets for correlation calculation
        all_predictions.append(outputs)
        all_targets.append(targets)
        all_attentions.append(attentions)

        loss = criterion(outputs, targets)
        total_loss += loss.item()

print("Concatenate results")
# Concatenate all predictions and targets
all_predictions = torch.cat(all_predictions)
all_targets = torch.cat(all_targets)
#all_attentions = torch.cat(all_attentions)

# Calculate Pearson correlation coefficient
correlation = pearson_corrcoef(all_predictions, all_targets)

print(f"...Test Loss: {total_loss / len(test_loader)}")
print(f'...Test Pearson Correlation Coefficient: {correlation.item()}')

print("Save model and result")
torch.save(model.state_dict(), res_dir + '/transformer_' + suffix + '.pth')
torch.save(all_predictions, res_dir + "/predictions_transformer_" + suffix + ".pth")
torch.save(all_targets, res_dir + "/targets_transformer_" + suffix + ".pth")
torch.save(all_attentions, res_dir + "/attentions_" + suffix + ".pth")

print('Finished running script')
print('Finishing time: ' + str(datetime.now().time()))
