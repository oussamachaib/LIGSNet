import torch
from torch import nn

#%% Settings

#%% Convolutional autoencoder

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        torch.set_default_dtype(torch.float32)  # default dtype
        torch.set_float32_matmul_precision('high')  # precision

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=12, kernel_size=19, stride = 3),
            nn.LeakyReLU(),
            nn.Conv1d(12,12, 19, 2),
            nn.LeakyReLU(),
            nn.Conv1d(12, 24, 7, 2),
            nn.LeakyReLU(),
            nn.Conv1d(24, 24, 5, 2),
            nn.LeakyReLU(),
            nn.Conv1d(24, 48, 5, 2),
            nn.LeakyReLU(),
            nn.Conv1d(48, 48, 5, 2),
            nn.LeakyReLU(),
            nn.Conv1d(48, 48, 3, 2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1344, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
            )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(1, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1344),
            nn.Unflatten(1, (48, 28)),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(48, 48, 3, 2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(48, 48, 5, 2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(48, 24, 5, 2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(24, 24, 5, 2, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(24, 12, 7, 2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(12, 12, 19, 2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(12, 1, 19, 3, output_padding=2),
            )

    def forward(self,X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded

    def load_pretrained(selfs):
        return torch.load('checkpoints/ConvAutoencoder.pth')

#%% Loading the pretrained model

if __name__ == "__main__":
    network = ConvAutoencoder()
    model = network.load_pretrained()
