import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(
        self,
        seq_len,
        input_size,
        hidden_size,
        latent_size,
        num_layers,
        cond_size: int = 0,                # number of static conditioning features
    ):
        super(VAE, self).__init__()
        self.seq_len = seq_len
        self.cond_size = int(cond_size)

        # encoder / decoder sizes account for conditional inputs in linear heads
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden2mean = nn.Linear(hidden_size + self.cond_size, latent_size)
        self.hidden2logvar = nn.Linear(hidden_size + self.cond_size, latent_size)

        # decoder receives z (+ cond) as input features repeated in time
        # map z+cond -> decoder input dim (latent_size) before LSTM
        self.z_cond2dec = nn.Linear(latent_size + self.cond_size, latent_size)
        self.decoder = nn.LSTM(latent_size, hidden_size, num_layers, batch_first=True)
        self.hidden2input = nn.Linear(hidden_size, input_size)

    def encode(self, x, c=None):
        # x: (B, T, F_in)
        _, state = self.encoder(x)
        last_hidden = state[0][-1, :, :]  # (B, hidden_size)
        if c is not None:
            # c expected (B, cond_size)
            last_hidden = torch.cat([last_hidden, c], dim=1)
        mean = self.hidden2mean(last_hidden)
        logvar = self.hidden2logvar(last_hidden)
        return mean, logvar, state

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode(self, z, state, c=None):
        # z: (B, latent_size)
        if c is not None:
            z_in = torch.cat([z, c], dim=1)
        else:
            z_in = z
        # map to decoder latent dimension and repeat over time
        z_hidden = self.z_cond2dec(z_in).unsqueeze(1).repeat(1, self.seq_len, 1)
        output, _ = self.decoder(z_hidden, state)
        recons_output = self.hidden2input(output)
        return recons_output

    def forward(self, x, c=None):
        mean, logvar, state = self.encode(x, c=c)
        z = self.reparameterize(mean, logvar)
        output = self.decode(z, state, c=c)
        return output, mean, logvar