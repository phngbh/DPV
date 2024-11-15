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

print("Define model")
# Embedding layers
class LinearEmbeddingLayer(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(LinearEmbeddingLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)  # Input layer to hidden layer
        self.fc2 = nn.Linear(50, embedding_dim) # Hidden layer to output layer
        self.dropout = nn.Dropout(0.7)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, X, attention_mask):
        X = X * attention_mask.unsqueeze(-1)  # Mask the input
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = self.fc2(X)
        
        # # Apply batch normalization along the feature dimension
        # X = X.permute(0, 2, 1)  # Permute to (batch_size, embedding_dim, seq_length)
        # X = self.bn(X)  # Apply BatchNorm1d
        # X = X.permute(0, 2, 1)  # Permute back to (batch_size, seq_length, embedding_dim)
        
        return X
        
class LSTMEmbeddingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers, embedding_dim, global_max_length):
        super(LSTMEmbeddingLayer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, embedding_dim)  # embedding_dim = LLM's hidden size
        self.global_max_length = global_max_length
        
    def forward(self, x, lengths):
        # x: (batch_size, seq_length, input_dim)
        # lengths: (batch_size)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)  # packed_output contains packed sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # (batch_size, seq_length, hidden_dim)

        # Manual padding to global_max_length
        batch_size = lstm_out.size(0)
        max_batch_seq_length = lstm_out.size(1)
        padded_lstm_out = torch.zeros(batch_size, self.global_max_length, lstm_out.size(2)).to(lstm_out.device)
        padded_lstm_out[:, -max_batch_seq_length:, :] = lstm_out
        
        # Transform LSTM output to the hidden size of BERT
        embeddings = self.linear(padded_lstm_out)  # embeddings: (batch_size, global_max_seq_length, embedding_dim)
        
        return embeddings

# Main functions
class BERTwithLinearEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim=1):
        super(BERTwithLinearEmbedding, self).__init__()
        self.embedding = LinearEmbeddingLayer(input_dim, embedding_dim)
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        self.dropout = nn.Dropout(0.7)
        # Assuming a regression task; change output features for classification
        self.regressor = nn.Linear(self.bert.config.hidden_size, output_dim)
        
    def forward(self, X, attention_mask=None): # X expected shape: [batch_size, seq_length, input_dim]
        # Apply the embedding layer
        X_embeds = self.embedding(X, attention_mask)  # [batch_size, seq_length, output_embedding_dim]

        # Process through BERT
        outputs = self.bert(inputs_embeds=X_embeds, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output # [batch_size, hidden_size]
        attentions = outputs.attentions  # Extract attention weights

        # Apply regression layer
        return self.regressor(self.dropout(pooled_output)), attentions

class BERTwithLSTMEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers, embedding_dim, global_max_length, output_dim=1):
        super(BERTwithLSTMEmbedding, self).__init__()
        self.embedding = LSTMEmbeddingLayer(input_dim, hidden_dim, lstm_layers, embedding_dim, global_max_length)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Assuming a regression task; change output features for classification
        self.regressor = nn.Linear(self.bert.config.hidden_size, output_dim)
        
    def forward(self, X, lengths, attention_mask=None): # X expected shape: [batch_size, seq_length, input_dim]
        # Apply the embedding layer
        X_embeds = self.embedding(X, lengths)  # [batch_size, seq_length, output_embedding_dim]

        # Process through BERT
        outputs = self.bert(inputs_embeds=X_embeds, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output # [batch_size, hidden_size]

        # Apply regression layer
        return self.regressor(pooled_output)

# Parameters
input_dim = 760  # Number of features per time point (HbA1C 760, LDL 749)
hidden_dim = 200
lstm_layers = 1
embedding_dim = 768  # Reduced number of features to be compatible with BERT
output_dim = 1  # Output dimension for regression
global_max_length = 49  # Set this to the global maximum sequence length

# Instantiate the model
model = BERTwithLinearEmbedding(input_dim = input_dim, embedding_dim = embedding_dim, output_dim=output_dim)

print("Load model and data")
# Load the fine-tuned model
model.load_state_dict(torch.load('/home/phong.nguyen/llm_hba1c_18k_long.pth'))
model.to(device)

# Load samples
X = torch.load('/home/phong.nguyen/data_subset_hba1c_long_full10.pth')
y = torch.load('/home/phong.nguyen/target_subset_hba1c_long_full10.pth')

# Select relevant samples
time_sequence = torch.load('/home/phong.nguyen/time_sequence_subset_hba1c_long_full10.pth') - 1
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
np.save("integrated_grads_avg_hba1c_long_full.npy", avg_integrated_grads)
np.save("integrated_grads_all_hba1c_long_full.npy", all_integrated_grads)
np.save("integrated_grads_y_hba1c_long_full.npy", y[subset_indices])

