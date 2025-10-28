import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, latent_size, num_layers, data_type="numeric"):
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
        last_hidden = state[0][-1,:,:]
        mean = self.hidden2mean(last_hidden)
        logvar = self.hidden2logvar(last_hidden)
        return mean, logvar, state

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode(self, z, state):
        z_hidden = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        output, _ = self.decoder(z_hidden, state)
        recons_output = self.hidden2input(output)
        if self.data_type == "discrete":
            recons_output = torch.sigmoid(recons_output).clamp(1e-8, 1 - 1e-8)
        return recons_output

    def forward(self, x):
        mean, logvar, state = self.encode(x)
        z = self.reparameterize(mean, logvar)
        output = self.decode(z, state)
        return output, mean, logvar