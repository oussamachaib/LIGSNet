import numpy as np
import pickle

#%% Loading full raw LIGS data set

def load_raw():
    with open("data/raw/data.pkl", 'rb') as file:
        raw_data = pickle.load(file)
    return raw_data['X'], raw_data['y']

#%% Loading full preprocessed LIGS data set

def load_preprocessed():
    with open("data/preprocessed/data.pkl", 'rb') as file:
        preprocessed_data = pickle.load(file)
    return preprocessed_data['X'], preprocessed_data['y']

#%% Augmenting data with random translations

def augment(X, y, k = 1):
    n_random_samples = int(25000*k)
    padding = np.random.randint(1, 5999, size=(n_random_samples,))

    # Allocating memory
    X_augmented = np.zeros((n_random_samples, 6000), dtype= np.float32)
    y_augmented = np.zeros((n_random_samples,), dtype= np.float32)

    cnt = 0

    for i in range(n_random_samples):
        X_augmented[cnt, :] = np.hstack((X[i % 25000, -padding[i]:], X[i % 25000, 0:6000 - padding[i]]))
        y_augmented[cnt] = y[i % 25000]
        cnt += 1

    return X_augmented, y_augmented

