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

