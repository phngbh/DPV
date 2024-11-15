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
input_dim = 755  # Number of features per time point
hidden_dim = 200
lstm_layers = 1
embedding_dim = 768  # Reduced number of features to be compatible with BERT
output_dim = 1  # Output dimension for regression
global_max_length = 49  # Set this to the global maximum sequence length

# Instantiate the model
model = BERTwithLinearEmbedding(input_dim = input_dim, embedding_dim = embedding_dim, output_dim=output_dim)

# Load the fine-tuned model
model.load_state_dict(torch.load('/home/phong.nguyen/llm_ldl_18k_long.pth'))
model.to(device)

# Load samples
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
torch.manual_seed(93)
test_indices = torch.randperm(X.size(0))[18000:]
test_dataset = TensorDataset(X[test_indices], time_sequence[test_indices], attention_mask[test_indices], y[test_indices])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Compute feature importance using gradients
feature_importance = compute_feature_importance(model, test_loader, device)

np.save("feature_importance_ldl_long_full.npy", feature_importance)

