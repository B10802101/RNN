import json
import numpy as np
from sklearn.model_selection import train_test_split

filepath = 'data.json'

def load_dataset(filepath):
    with open(filepath, 'w') as f:
        data = json.load(f)
    X = np.array(data['MFCCs'])
    Y = np.array(data['labels'])
    return X, Y
def get_data_splits(filepath, test_split, validation_split):
    X, Y = load_dataset(filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_split)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_split)
    #reshape