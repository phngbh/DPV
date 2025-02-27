import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from sklearn.metrics import mean_squared_error
import gc
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################
##################### Embedding layers ########################
###############################################################

class LinearEmbeddingLayer(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout):
        super(LinearEmbeddingLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)  # Input layer to hidden layer
        self.fc2 = nn.Linear(50, embedding_dim) # Hidden layer to output layer
        self.dropout = nn.Dropout(dropout)
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

##############################################################################
##################### Transformer with pretrained LLM ########################
##############################################################################

class TransformerWithLinearEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout, pretrained_model, output_dim=1, num_classes=None):
        super(TransformerWithLinearEmbedding, self).__init__()
        self.embedding = LinearEmbeddingLayer(input_dim, embedding_dim, dropout)
        self.transformer = AutoModel.from_pretrained(pretrained_model, output_attentions=True)
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        if num_classes is None:
            self.regressor = nn.Linear(self.transformer.config.hidden_size, output_dim)
        else:
            self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
    def forward(self, X, attention_mask=None): # X expected shape: [batch_size, seq_length, input_dim]
        # Apply the embedding layer
        X_embeds = self.embedding(X, attention_mask)  # [batch_size, seq_length, output_embedding_dim]

        # Process through LLM
        outputs = self.transformer(inputs_embeds=X_embeds, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output # [batch_size, hidden_size]
        attentions = outputs.attentions  # Extract attention weights

       # Apply regression or classification layer
        if self.num_classes is None:
            return self.regressor(self.dropout(pooled_output)), attentions
        else:
            return self.classifier(self.dropout(pooled_output)), attentions

class TransformerWithLSTMEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers, embedding_dim, global_max_length, pretrained_model, output_dim=1, num_classes=None):
        super(TransformerWithLSTMEmbedding, self).__init__()
        self.embedding = LSTMEmbeddingLayer(input_dim, hidden_dim, lstm_layers, embedding_dim, global_max_length)
        self.transformer = AutoModel.from_pretrained(pretrained_model, output_attentions=True)
        self.num_classes = num_classes
        if num_classes is None:
            self.regressor = nn.Linear(self.transformer.config.hidden_size, output_dim)
        else:
            self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
    def forward(self, X, lengths, attention_mask=None): # X expected shape: [batch_size, seq_length, input_dim]
        # Apply the embedding layer
        X_embeds = self.embedding(X, lengths)  # [batch_size, seq_length, output_embedding_dim]

        # Process through LLM
        outputs = self.transformer(inputs_embeds=X_embeds, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output # [batch_size, hidden_size]

       # Apply regression or classification layer
        if self.num_classes is None:
            return self.regressor(pooled_output)
        else:
            return self.classifier(pooled_output)
    

###############################################################
##################### LSTM ####################################
###############################################################

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, output_dim=1, num_layers=2):
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