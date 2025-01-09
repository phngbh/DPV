import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import argparse
import gc
import os
import yaml

import utils_functions as uf

# Parse arguments
parser = argparse.ArgumentParser(description='Arguments for LSTM-VAE')
parser.add_argument('--config', type=str, help='Path to the configuration file')
args = parser.parse_args()

# Load configuration
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)['LSTM_VAE']

# Set hyperparameters
data = config['data']
seq_len = config['seq_len']
time_seq = config['time_seq']
input_size = config['input_size']
hidden_size = config['hidden_size']
latent_size = config['latent_size']
num_layers = config['num_layers']
batch_size = config['batch_size']
num_epochs = config['epoch']
learning_rate = config['learning_rate']
data_type = config['data_type']
res_dir = config['res_dir']

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a custom dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, device):
        self.data = torch.tensor(data, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Define the Autoencoder model
class VAE(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, latent_size, num_layers, data_type):
        super(VAE, self).__init__()
        self.seq_len = seq_len
        self.data_type = data_type
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden2mean = nn.Linear(hidden_size, latent_size)
        self.hidden2logvar = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.LSTM(latent_size, hidden_size, num_layers, batch_first=True)
        self.hidden2input = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        _, state = self.encoder(x)
        #print(state[0].shape)
        #print(state[1].shape)
        #print(out.shape)
        #hidden = hidden.squeeze(0)  # Remove the num_layers dimension
        last_hidden = state[0][-1,:,:] # Only keep the last layer's hidden
        #print(last_hidden.shape)
        #out_new = out[:, -1, :]
        #print(out_new.shape)
        mean = self.hidden2mean(last_hidden)
        logvar = self.hidden2logvar(last_hidden)
        return mean, logvar, state

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + epsilon * std
        # print(z.shape)
        return z

    def decode(self, z, state):
        #z_hidden = self.latent2hidden(z)  # Map latent space back to hidden size
        z_hidden = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        # print(z_hidden.shape)
        # print(state[0].shape)
        # print(state[1].shape)
        output, _ = self.decoder(z_hidden, state)
        recons_output = self.hidden2input(output)
        if self.data_type == "discrete":
            recons_output = torch.sigmoid(recons_output)
            epsilon = 1e-8  # Small constant to avoid division by zero
            recons_output = torch.clamp(recons_output, epsilon, 1 - epsilon)  # Ensure probabilities are in (epsilon, 1-epsilon) range
        return recons_output

    def forward(self, x):
        mean, logvar, state = self.encode(x)
        z = self.reparameterize(mean, logvar)
        output = self.decode(z, state)
        return output, mean, logvar
    
# Load data
data = np.load(data, allow_pickle = True)
time_seq = np.load(time_seq, allow_pickle = True)

# Add an ID to the data
new_data = []
for i in range(len(data)):
    vec = np.full((data[i].shape[0], 1), i)
    tmp = np.concatenate((vec, data[i]), axis = 1) 
    new_data.append(tmp)
new_data = np.stack(new_data)

# Free up some memory
del data
gc.collect()

# Split the padded data into training and testing sets
train_data, test_data = train_test_split(new_data, test_size=0.2, random_state=42)

# Create training and testing datasets
train_dataset = TimeSeriesDataset(train_data, device)  
test_dataset = TimeSeriesDataset(test_data, device)  

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create VAE model
model = VAE(seq_len, input_size, hidden_size, latent_size, num_layers, data_type).to(device)

# Define loss function and optimizer
#criterion = nn.MSELoss() #nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # StepLR reduces the learning rate by a factor after a specified number of epochs
# scheduler = StepLR(optimizer, step_size=200, gamma=0.2)


# Training loop
for epoch in range(num_epochs):
    
    # if epoch == 201:  # Adjust the learning rate after a certain number of epochs
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = learning_rate/5  # Set a smaller learning rate
    # if epoch == 401:  # Adjust the learning rate after a certain number of epochs
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = learning_rate/25  # Set a smaller learning rate
    # if epoch == 1001:  # Adjust the learning rate after a certain number of epochs
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = learning_rate/50  # Set a smaller learning rate
    # if epoch == 5001:  # Adjust the learning rate after a certain number of epochs
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = learning_rate/100  # Set a smaller learning rate
    
    model.train()
    total_loss = 0
    total_recon_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Make padding masks 
        padding_masks = uf.make_padding_mask(batch, time_seq, device)
        
        # Forward
        new_batch = batch[:,:,1:]
        output, mean, logvar = model(new_batch)
        
        # Reconstruction loss
        if data_type == 'discrete':
            recon_loss = uf.custom_loss_discrete(output, new_batch, padding_masks)
        elif data_type == 'numeric':
            recon_loss = uf.custom_loss_numeric(output, new_batch, padding_masks)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # Total loss (combining reconstruction loss and KL divergence loss)
        loss = recon_loss + kl_loss
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
    # if total_loss/len(train_loader) < 100:  # Adjust the learning rate when loss is small
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = learning_rate/5  # Set a smaller learning 
    # if total_loss/len(train_loader) < 50:  # Adjust the learning rate when loss is small
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = learning_rate/10  # Set a smaller learning 

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Reconstruction loss: {total_recon_loss/len(train_loader):.4f}")
    
    # scheduler.step()  # Adjust the learning rate at the beginning of each epoch
    
# Evaluation loop
model.eval()
total_loss = 0.0

for batch in test_loader:
    
    # Make padding masks 
    padding_masks = uf.make_padding_mask(batch, time_seq, device)
        
    # Forward
    new_batch = batch[:,:,1:]
    output, mean, logvar = model(new_batch)

    # Reconstruction loss
    if data_type == 'discrete':
        recon_loss = uf.custom_loss_discrete(output, new_batch, padding_masks)
    elif data_type == 'numeric':
        recon_loss = uf.custom_loss_numeric(output, new_batch, padding_masks)

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    # Total loss (combining reconstruction loss and KL divergence loss)
    loss = recon_loss + kl_loss
    total_loss += loss.item()
    total_recon_loss += recon_loss.item()


# Compute the average loss for the test set
avg_loss = total_loss / len(test_loader)

# Print the average loss
print(f"Test Loss: {avg_loss:.4f}")

# Save the trained VAE model
torch.save(model.state_dict(), res_dir + 'lstm_vae_final_model_' + data_type + '.pt')

# Reconstruct test samples
sample_inputs = test_dataset[:,:,1:] 
with torch.no_grad():
    reconstructed_samples = model(sample_inputs)
torch.save(reconstructed_samples, res_dir + 'reconstructed_' + data_type + '_test.pt')
torch.save(test_dataset, res_dir + data_type + '_test.pt')

# Try to free up some memory
del reconstructed_samples
del test_dataset
del train_loader
del test_loader
gc.collect()

# # Reconstruct all samples
# sample_inputs = TimeSeriesDataset(data, device)[:len(data)] 

# # Try to free up some memory
# del data
# gc.collect()

# with torch.no_grad():
#     reconstructed_samples = model(sample_inputs)

# # Try to free up some memory
# del sample_inputs
# gc.collect()

# saved_checkpoint = '/home/phong.nguyen/lstm_vae_final_model_' + data_type + '.pt'

# # Load the saved model state_dict
# model.load_state_dict(torch.load(saved_checkpoint))

## Generate synthetic data
print("Generate synthetic data")
# Calculate the total number of samples
total_samples = len(data)

# Ensure the tmp directory exists within res_dir
tmp_dir = os.path.join(res_dir, 'tmp')
os.makedirs(tmp_dir, exist_ok=True)

# Initialize an empty list to store reconstructed samples
reconstructed_samples = []

# Split the data into smaller batches and generate synthetic data for each batch
for start_idx in range(0, total_samples, 200): # batch_size = 200
    end_idx = min(start_idx + 200, total_samples)
    
    # Get the subset of sample_inputs for the current batch
    batch_inputs = TimeSeriesDataset(data, device)[start_idx:end_idx]
    
    # Generate synthetic data for the current batch using the model
    with torch.no_grad():
        batch_reconstructed, _, _ = model(batch_inputs)
    
    # # Append the batch of reconstructed samples to the list
    # reconstructed_samples.append(batch_reconstructed)
    # Save part of resonstructed samples
    torch.save(batch_reconstructed, tmp_dir + 'reconstructed_' + data_type + str(start_idx) + '.pt')
    
    # Try to free up some memory
    del batch_inputs
    del batch_reconstructed
    gc.collect()
    
    torch.cuda.empty_cache()

# Initialize an empty list to store reconstructed samples
big_array_list = []

# Recover and combine the batches
for start_idx in range(0, total_samples, 200): # batch_size = 200
    
    print('File' + str(start_idx), end=" ")
    
    end_idx = min(start_idx + 200, total_samples)
    data = torch.load(tmp_dir + 'reconstructed_' + data_type + str(start_idx) + '.pt')
    t = time_seq[start_idx:end_idx]
    array_list = []
    for i in range(data.shape[0]):
        t_i = t[i]
        tensr = data[i,:t_i,:]
        arr = tensr.cpu().numpy()
        array_list.append(arr)
        
        # Try to free up some memory
        del tensr
        del arr
        gc.collect()
        torch.cuda.empty_cache()
    
    big_array_list = big_array_list + array_list
    # Try to free up some memory
    del data
    del array_list
    gc.collect()
    torch.cuda.empty_cache()
    
# Concatenate the arrays row-wise
big_array = np.concatenate(big_array_list, axis=0)

# Save reconstructed data
np.save(res_dir + 'reconstructed_' + data_type + '.npy', big_array)