import torch
from torch import nn

#%% CNN

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.set_default_dtype(torch.float32)  # default dtype
        torch.set_float32_matmul_precision('high')  # precision

        # Backbone
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=12, kernel_size=19, stride=3),
            nn.LeakyReLU(),
            nn.Conv1d(12, 12, 19, 2),
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
            )
        # Dense layers
        self.fc = nn.Sequential(
            nn.Linear(1344,100),
            nn.LeakyReLU(),
            nn.Linear(100,100),
            nn.LeakyReLU(),
            nn.Linear(100,1),
            nn.Sigmoid(),
            )

    def forward(self,X):
        return self.fc(self.cnn(X))

    def load_pretrained(self):
        return torch.load('checkpoints/ConvClassifier.pth')

#%% Loading the pretrained model

if __name__ == "__main__":
    network = CNN()
    model = network.load_pretrained()