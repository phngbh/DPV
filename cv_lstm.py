import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from models import LSTMRegressor
from sklearn.metrics import mean_squared_error
import gc
import numpy as np
import argparse
import yaml

# Parse arguments
parser = argparse.ArgumentParser(description='Arguments for running script')
parser.add_argument('--config', type=str, help='Path to the configuration file')
args = parser.parse_args()

# Load configuration
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)['cv_lstm']

# Set hyperparameters
data = config['data']
target = config['target']
time_seq = config['time_seq']
train_size = config['train_size']
val_size = config['val_size']
dropout = config['dropout']
hidden_dim = config['hidden_dim']
lr = config['lr']
weight_decay = config['weight_decay']
epochs = config['epochs']
patience = config['patience']
res_dir = config['res_dir']
suffix = config['suffix']

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

train_size = list(map(int, train_size.split(',')))
def cross_validation(train_size = train_size, val_size = val_size):
    
    result_loss = []
    result_corr = []
    
    for i in range(len(train_size)):
        print('Train size: ',train_size[i])
        result_loss_i = []
        result_corr_i = []
        for seed in range(10):
            print('...Iteration: ',seed)
            # Set the seed for reproducibility and randomly choose 1000 indices without replacement
            torch.manual_seed(seed)
            train_indices = torch.randperm(X.size(0))[:train_size[i]]
            torch.manual_seed(seed)
            val_indices = torch.randperm(X.size(0))[train_size[i]:(train_size[i] + val_size)]
            torch.manual_seed(seed)
            test_indices = torch.randperm(X.size(0))[(train_size[i]+val_size):]  # all indices except the training ones

            # Create the dataset and data loader
            train_dataset = TensorDataset(X[train_indices], time_sequence[train_indices], y[train_indices])
            val_dataset = TensorDataset(X[val_indices], time_sequence[val_indices], y[val_indices])
            test_dataset = TensorDataset(X[test_indices], time_sequence[test_indices], y[test_indices])

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Parameters
            input_dim = X.size(2)  # Number of features per time point
            hidden_dim = hidden_dim  # Size of LSTM hidden layers
            output_dim = 1  # Predicting a single value

            # Instantiate the model
            model = LSTMRegressor(input_dim, hidden_dim, output_dim)
            model.to(device)

            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
                for data, seq_length, targets in train_loader:
                    data, targets = data.to(device).float(), targets.to(device).float()

                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    predictions = model(data, seq_length)
                    predictions = predictions.squeeze(-1)
                    
                    # Loss calculation
                    loss = criterion(predictions, targets)

                    # Backward pass
                    loss.backward()
                    
                    # Update parameters
                    optimizer.step()
                    
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
                print(f'......Validation Loss: {val_loss}')
                
                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), 'best_model' + suffix + '.pth')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        early_stop = True
            
            
            # Load the best model
            model.load_state_dict(torch.load('best_model' + suffix + '.pth'))

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
            
            with torch.no_grad():
                for inputs, seq_length, targets in test_loader:
                    inputs, targets = inputs.to(device).float(), targets.to(device).float()
                    outputs = model(inputs, seq_length)
                    outputs = outputs.squeeze(-1)

                    # Collect predictions and targets for correlation calculation
                    all_predictions.append(outputs)
                    all_targets.append(targets)

                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
            tloss = total_loss / len(test_loader)

            # Concatenate all predictions and targets
            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)

            # Calculate Pearson correlation coefficient
            correlation = pearson_corrcoef(all_predictions, all_targets)

            print(f"...Test Loss: {total_loss / len(test_loader)}")
            print(f'...Test Pearson Correlation Coefficient: {correlation.item()}')
            
            result_loss_i.append(tloss)
            result_corr_i.append(correlation)

        result_loss.append(result_loss_i)
        result_corr.append(result_corr_i)
    
    return([result_loss,result_corr])

result_lstm = cross_validation()
result_lstm = np.array(result_lstm, dtype = object)
np.save(res_dir + "/result_cv_lstm_" + suffix + ".npy", result_lstm)
np.savetxt(res_dir + "/result_cv_lstm_" + suffix + ".csv", result_lstm, delimiter=",")