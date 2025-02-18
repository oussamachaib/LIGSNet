import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from loaders import load_preprocessed
import tqdm
from sklearn.model_selection import train_test_split
from models.ConvAutoencoder import ConvAutoencoder
import matplotlib.pyplot as plt

#%% Training autoencoder

class trainAE():
    def __init__(self,
                 lr = 1e-4,
                 batch_size = 667,
                 n_epochs = 200,
                 split_ratio = 0.8,
                 ):
        self.lr = lr # learning rate
        self.batch_size = batch_size # samples per batch
        self.n_epochs = n_epochs # number of training epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device
        self.criterion = nn.MSELoss() # loss function
        self.split_ratio = split_ratio # train/test split
        self.train_loss = [] # training set loss
        self.test_loss = [] # test/validation set loss

    def train(self, model):
        # Loading preprocessed data
        X, _ = load_preprocessed()

        # Splitting
        X_train, X_test = train_test_split(X, train_size = self.split_ratio)

        # Converting input to tensors
        X_train = torch.from_numpy(X_train.astype(np.float32)).unsqueeze(1)
        X_test = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(1)

        # Setting up optimizer and loader
        optimizer = torch.optim.Adam(lr=self.lr, params= model.parameters())  # Optimizer
        loader = DataLoader(X_train, batch_size = self.batch_size, shuffle = True)  # Loader

        # Moving model to device
        model.to(self.device)

        # Placeholders
        train_avgloss = []
        test_avgloss = []

        # Training model
        for epoch in tqdm.tqdm(range(self.n_epochs), desc = "Training"):
            # Placeholder lists for display
            batch_train_loss = []

            for batch_id, data in enumerate(loader):
                # Moving batch to GPU
                data = data.to(self.device)

                # Forward prop
                output = model.forward(data)

                # Computing loss
                training_loss = self.criterion(output, data)

                # Storing loss for display
                batch_train_loss.append(training_loss.item())

                # Back prop
                training_loss.backward()

                # Updating hyperparameters
                optimizer.step()

                # Resetting gradients in the optimizer for the next iteration
                optimizer.zero_grad()

            # Computing and saving average loss per epoch
            train_epoch_loss = np.mean(batch_train_loss)
            train_avgloss.append(train_epoch_loss)

            # Evaluating on test data set
            with torch.no_grad():
                test_loss = self.criterion(model.forward(X_test.to(self.device)), X_test)

            # Saving test loss per epoch
            test_avgloss.append(test_loss)

            # Tracking progress
            #tqdm.tqdm.write(f'(Train) MSE loss: {train_epoch_loss:.1} | (Test) MSE loss: {test_loss:.1}')

        # Saving train and test loss
        self.train_loss = train_avgloss
        self.test_loss = test_avgloss

    def loss_curves(self):
        # Some aesthetics
        plt.style.use('default')
        plt.rc('text', usetex=True)
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.train_loss, '-o', label=r'$\mathcal{L}_{\text{train}}$')
        ax.plot(self.test_loss, '-o', label=r'$\mathcal{L}_{\text{test}}$')
        ax.legend(fontsize=20, loc='upper right')
        ax.set_xlabel('Epoch', fontsize=20)
        ax.set_ylabel('Loss', fontsize=20)
        ax.tick_params(labelsize=20)
        fig.tight_layout()
        fig.show()

class trainCNN():
    def __init__(self):
        return 0


#%% Testing

if __name__ == "__main__":
    model = ConvAutoencoder()
    trainer = trainAE(lr = .1, batch_size=10)
    trainer.train(model)
    trainer.loss_curves()