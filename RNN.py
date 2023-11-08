import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os

datapath = './data.json'
MFCC_feature = 13
epochs = 2
learningrate = 0.0001
batchsize = 1

def load_dataset(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    X = np.array(data['MFCCs'])
    Y = np.array(data['labels'])
    print('Data sets loaded!')
    return X, Y

def get_data_splits(filepath, test_split, validation_split):
    X, Y = load_dataset(filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_split)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_split)
    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test

def batch_the_data(batchsize, X_train, X_validation, X_test, Y_train, Y_validation, Y_test):
    X_train_batch = [X_train[i: i + batchsize] for i in range(0, len(X_train), batchsize)]
    X_validation_batch = [X_validation[i: i + batchsize] for i in range(0, len(X_validation), batchsize)]
    X_test_batch = [X_test[i: i + batchsize] for i in range(0, len(X_test), batchsize)]
    Y_train_batch = [Y_train[i: i + batchsize] for i in range(0, len(Y_train), batchsize)]
    Y_validation_batch = [Y_validation[i: i + batchsize] for i in range(0, len(Y_validation), batchsize)]
    Y_test_batch = [Y_test[i: i + batchsize] for i in range(0, len(Y_test), batchsize)]
    return X_train_batch, X_validation_batch, X_test_batch, Y_train_batch, Y_validation_batch, Y_test_batch

def RNN_model(inputsize, hiddensize, numlayers, batchsize):
    lstm = nn.LSTM(input_size=inputsize, hidden_size=hiddensize, num_layers=numlayers, batch_first=True)
    return lstm

def main():
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = get_data_splits(datapath, 0.1, 0.1)
    X_train_batch, X_validation_batch, X_test_batch, Y_train_batch, Y_validation_batch, Y_test_batch = batch_the_data(batchsize, X_train, X_validation, X_test, Y_train, Y_validation, Y_test)
    
    print(f'len(X_train_batch) {len(X_train_batch)}')
    print(f'len(X_validation_batch) {len(X_validation_batch)}')
    print(f'len(X_test_batch) {len(X_test_batch)}')
    print(f'len(Y_train_batch) {len(Y_train_batch)}')
    print(f'len(Y_validation_batch) {len(Y_validation_batch)}')
    print(f'len(Y_test_batch) {len(Y_test_batch)}')
    print(f'X_train_batch[0] shape {X_train_batch[0].shape}')
    print(f'X_test_batch[0] shape {X_test_batch[0].shape}')
    print(f'X_validation_batch[0] shape {X_validation_batch[0].shape}')
    print(f'Y_train_batch[0] shape {Y_train_batch[0].shape}')
    print(f'Y_test_batch[0] shape {Y_test_batch[0].shape}')
    print(f'Y_validation_batch[0] shape {Y_validation_batch[0].shape}')

    # To tensor
    X_train_batch = torch.Tensor(X_train_batch)
    X_validation_batch = torch.Tensor(X_validation_batch)
    X_test_batch = torch.Tensor(X_test_batch)
    Y_train_batch = torch.Tensor(Y_train_batch)
    Y_validation_batch = torch.Tensor(Y_validation_batch)
    Y_test_batch = torch.Tensor(Y_test_batch)

    # Set the model
    model = RNN_model(MFCC_feature, 64, 1, batchsize)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learningrate)
    
    # Start training
    for epoch in range(epochs):
        for X_train_subbatch, Y_train_subbatch in zip(X_train_batch, Y_train_batch):
            output, _ = model(X_train_subbatch)
            optimizer.zero_grad()
            print(f'X_train_subbatch = {X_train_subbatch.shape}')
            print(f'output shape = {len(output)}')
            print(f'Y_train_subbatch = {Y_train_subbatch}')
            loss = criterion(output, Y_train_subbatch)
            loss.backward()
            optimizer.step()

            print(f'{epoch+1}/{epochs}      loss: {loss.item():.4f}')
    print('====================Finish training=========================')

if __name__ == '__main__':
    main()
