import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertConfig
from sklearn.metrics import mean_squared_error
import gc
import numpy as np
import argparse
from datetime import timedelta, datetime

print('Start running script')
print('Starting time: ' + str(datetime.now().time()))

# Parse arguments
parser = argparse.ArgumentParser(description='Arguments for running script')
parser.add_argument('--data', type=str, help='File directory to the processed input data')
parser.add_argument('--target', type=str, help='File directory to the target data')
parser.add_argument('--time_seq', type=str, help='File directory to the time sequence')
parser.add_argument('--train_size', type=int, help='Training size')
parser.add_argument('--val_size', type=int, help='Validation size')
parser.add_argument('--dropout', type=int, help='Fraction of parameters to drop out')
parser.add_argument('--hidden_dim', type=int, help='Hidden dimension of the LSTM embedding layer')
parser.add_argument('--lr', type=float, help='Learning rate during optimization')
parser.add_argument('--weight_decay', type=float, help='A regularization factor')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--patience', type=int, help='Number of epochs to wait before early stopping during optimization')
parser.add_argument('--res_dir', type=str, help='File directory to store the results')
parser.add_argument('--suffix', type=str, help='Name suffix of the results')

args = parser.parse_args()

# Set hyperparameters
data = args.data
target = args.target
time_seq = args.time_seq
train_size = args.train_size
val_size = args.val_size
dropout = args.dropout
hidden_dim = args.hidden_dim
lr = args.lr
weight_decay = args.weight_decay
epochs = args.epochs
patience = args.patience
res_dir = args.res_dir
suffix = args.suffix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

print("Define model")
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.regressor = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths):
        # Pack the sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(packed_input)
        
        # We use the hidden state from the last layer of the LSTM for prediction
        last_hidden_state = ht[-1]  # Take the last layer's hidden state
        output = self.regressor(self.dropout(last_hidden_state))
        return output

# Set the seed for reproducibility and randomly choose 1000 indices without replacement
torch.manual_seed(93)
train_indices = torch.randperm(X.size(0))[:train_size]
torch.manual_seed(93)
val_indices = torch.randperm(X.size(0))[train_size:(train_size+val_size)]
torch.manual_seed(93)
test_indices = torch.randperm(X.size(0))[(train_size+val_size):]  # all indices except the training ones

# Create the dataset and data loader
train_dataset = TensorDataset(X[train_indices], time_sequence[train_indices], y[train_indices])
val_dataset = TensorDataset(X[val_indices], time_sequence[val_indices], y[val_indices])
test_dataset = TensorDataset(X[test_indices], time_sequence[test_indices], y[test_indices])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model parameters
input_dim = X.size(2)  # As reduced or determined by your preprocessing (HbA1C: 766, LDL: 755)
hidden_dim = hidden_dim  # Size of LSTM hidden layers
output_dim = 1  # Predicting a single value

# Instantiate the model
model = LSTMRegressor(input_dim, hidden_dim, output_dim, dropout)
model.to(device)

# Loss function and optimizer
criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Early stopping parameters
patience = patience
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

# Training loop
for epoch in range(epochs):
    
    if early_stop:
        print("Early stopping")
        break
    
    model.train()
    total_loss = 0
    for batch in train_loader:
        data, seq_lengths, targets = batch
        data = data.to(device).float()
        targets = targets.to(device).float()

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data, seq_lengths)
        outputs = outputs.squeeze(-1)
        
        # Loss calculation and backpropagation
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}: Loss = {total_loss / len(train_loader)}')
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, seq_length, targets in val_loader:
            data, targets = data.to(device).float(), targets.to(device).float()
            predictions = model(data, seq_length)
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

# Evaluation (simplified)
model.eval()
total_loss = 0
all_predictions = []
all_targets = []
with torch.no_grad():
    for inputs, seq_lengths, targets in test_loader:
        inputs, targets = inputs.to(device).float(), targets.to(device).float()
        outputs = model(inputs, seq_lengths)
        outputs = outputs.squeeze(-1)

        # Collect predictions and targets for correlation calculation
        all_predictions.append(outputs)
        all_targets.append(targets)

        loss = criterion(outputs, targets)
        total_loss += loss.item()

# Concatenate all predictions and targets
all_predictions = torch.cat(all_predictions)
all_targets = torch.cat(all_targets)

# Calculate Pearson correlation coefficient
correlation = pearson_corrcoef(all_predictions, all_targets)

print(f"Test Loss: {total_loss / len(test_loader)}")
print(f'Test Pearson Correlation Coefficient: {correlation.item()}')

print("Save model and result")
torch.save(model, res_dir + '/lstm_' + suffix + '.pth')
torch.save(all_predictions, res_dir + "/predictions_lstm_" + suffix + ".pth")
torch.save(all_targets, res_dir + "/targets_lstm_" + suffix + ".pth")

print('Finished running script')
print('Finishing time: ' + str(datetime.now().time()))
