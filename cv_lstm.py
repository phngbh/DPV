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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Load data and set up input")
# Load data
X = torch.load('/home/phong.nguyen/data_subset_ldl_long_full.pth')
y = torch.load('/home/phong.nguyen/target_subset_ldl_long_full.pth')

## Make attention mask
time_sequence = torch.load('/home/phong.nguyen/time_sequence_subset_ldl_long_full.pth') - 1
# Create a range tensor that matches the sequence length dimension
range_tensor = torch.arange(49).unsqueeze(0).expand(len(time_sequence), 49)
# Create the mask by comparing the range tensor with lengths tensor
attention_mask_np = (range_tensor < time_sequence.unsqueeze(1)).int().numpy()
attention_mask_np = np.fliplr(attention_mask_np)
attention_mask = torch.from_numpy(attention_mask_np.copy())

print("Define model")
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.regressor = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.7)
    
    def forward(self, x, lengths):
        # Pack the sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(packed_input)
        
        # We use the hidden state from the last layer of the LSTM for prediction
        last_hidden_state = ht[-1]  # Take the last layer's hidden state
        output = self.regressor(self.dropout(last_hidden_state))
        return output

def cross_validation(train_size = [3000,5000], val_size = 2000):
    
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
            input_dim = 755  # Number of features per time point
            hidden_dim = 200  # Size of LSTM hidden layers
            output_dim = 1  # Predicting a single value

            # Instantiate the model
            model = LSTMRegressor(input_dim, hidden_dim, output_dim)
            model.to(device)

            optimizer = Adam(model.parameters(), lr=2e-5, weight_decay=5e-3)
            criterion = MSELoss()
            
            # Early stopping parameters
            patience = 30
            best_val_loss = float('inf')
            epochs_no_improve = 0
            early_stop = False

            # Fine-tuning (simplified)
            for epoch in range(150):  # Number of epochs
                
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
                    torch.save(model.state_dict(), 'best_model_hba1c.pth')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        early_stop = True
            
            
            # Load the best model
            model.load_state_dict(torch.load('best_model_hba1c.pth'))

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
np.save("result_cv_lstm_hba1c_long.npy", result_lstm.cpu().numpy())
np.savetxt("result_cv_lstm_hba1c_long.csv", result_lstm.cpu().numpy(), delimiter=",")