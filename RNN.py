#Total 95400 data   Train 76000 Valid 9860 Test 9540
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from sklearn.model_selection import KFold

datapath = './data.npz'
MFCC_feature = 13
epochs = 200
learningrate = 0.0005
batchsize = 20
n_fold = 5
earlystop = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=MFCC_feature, hidden_size=64, num_layers=1, batch_first=True)
        self.linear = nn.Linear(64, 36)
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x[:, -1, :]                             # shape of x = batch size * 64
        x = nn.functional.leaky_relu(x)
        x = self.linear(x)
        return x

def load_dataset(filepath):
    data = np.load(filepath)
    X = data['MFCCs'].astype('float')
    Y = data['labels'].astype('float')
    print('Data sets loaded!')
    return X, Y

def get_data_splits(filepath, test_split):
    X, Y = load_dataset(filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_split)
    return X_train, X_test, Y_train, Y_test

def batch_the_data(batchsize, X_train, X_test, Y_train, Y_test):
    X_train_batch = np.array([X_train[i: i + batchsize] for i in range(0, len(X_train), batchsize)])
    X_test_batch = np.array([X_test[i: i + batchsize] for i in range(0, len(X_test), batchsize)])
    Y_train_batch = np.array([Y_train[i: i + batchsize] for i in range(0, len(Y_train), batchsize)])
    Y_test_batch = np.array([Y_test[i: i + batchsize] for i in range(0, len(Y_test), batchsize)])
    return X_train_batch, X_test_batch, Y_train_batch, Y_test_batch

def main():

    # Split the data
    X_train, X_test, Y_train, Y_test = get_data_splits(datapath, 0.1)
    X_train_batch, X_test_batch, Y_train_batch, Y_test_batch = batch_the_data(batchsize, X_train, X_test, Y_train, Y_test)


    # To tensor & GPU
    X_train_batch = torch.Tensor(X_train_batch)
    X_test_batch = torch.Tensor(X_test_batch[:len(X_test_batch)-1])
    Y_train_batch = torch.Tensor(Y_train_batch)
    Y_test_batch = torch.Tensor(Y_test_batch[:len(Y_test_batch)-1])

    # Set the model
    model = RNN_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learningrate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=4)

    # Start training
    training_loss, validation_loss, training_accuracys, validation_accuracys = [], [], [], []
    best_accuracy = 0
    for fold, (train_index, val_index)in enumerate(kf.split(X_train_batch)):
        X_train_batch_fold = X_train_batch[train_index].to(device)
        Y_train_batch_fold = Y_train_batch[train_index].to(device)
        X_validation_batch_fold = X_train_batch[val_index].to(device)
        Y_validation_batch_fold = Y_train_batch[val_index].to(device)
        for epoch in range(epochs):

            # Training
            start_time = time.time()
            iter = 0
            correct_recognized = 0
            total_train = 0
            correct_recognized_val = 0
            total_validation =0
            earlystop_patience = 0
            for X_train_subbatch, Y_train_subbatch in zip(X_train_batch_fold, Y_train_batch_fold):
                X_train_subbatch.to(device)
                Y_train_subbatch.to(device)
                output = model(X_train_subbatch)
                optimizer.zero_grad()
                loss = criterion(output, Y_train_subbatch.long())                 #output 100*36, target 100*36
                loss.backward()
                optimizer.step()

                _, train_predicts = torch.max(output, 1)

                for i in range(0, batchsize):
                    if train_predicts[i] == Y_train_subbatch[i]:
                        correct_recognized += 1
                total_train += Y_train_subbatch.size(0)
                training_accuracy = correct_recognized / total_train
                iter +=1
                if (iter + 1) % (1000) == 0:
                    print(f'            {fold + 1} fold    {epoch + 1}/{epochs} epoch    {iter + 1}/{len(Y_train_batch_fold)}   loss: {loss.item():.4f}')
            training_loss.append(loss.item())
            training_accuracys.append(training_accuracy)

            # Validation
            model.eval()
            with torch.no_grad():
                for X_validation_subbatch, Y_validation_subbatch in zip(X_validation_batch_fold, Y_validation_batch_fold):
                    X_validation_subbatch.to(device)
                    Y_validation_subbatch.to(device)
                    output = model(X_validation_subbatch)
                    _, valid_predict = torch.max(output, 1)
                    loss = criterion(output, Y_validation_subbatch.long())
                    for i in range(0, batchsize):
                        if valid_predict[i] == Y_validation_subbatch[i]:
                            correct_recognized_val += 1
                    total_validation += Y_validation_subbatch.size(0)
                    validation_accuracy = correct_recognized_val / total_validation
                if validation_accuracy > best_accuracy:
                    best_accuracy = correct_recognized_val / total_validation
                    print('Saving better model parameters...')
                    torch.save(model.state_dict(), 'best_model.pth')
                else:
                    earlystop_patience += 1
                    if earlystop_patience == earlystop:
                        print('/////////No improve after 20 epochs, Skip to next fold/////////')
                        break
            validation_loss.append(loss.item())
            validation_accuracys.append(validation_accuracy)
            print(f'            Validation loss: {loss}')
            print(f'            Training accuracy {(training_accuracy) * 100}%    Validation accuracy: {(validation_accuracy) * 100}%')
            print(f'            Training time of {epoch + 1} epoch is {time.time() - start_time}')
            print('//////////////////////////////////////////////////////////')
            #Update learning rate
            scheduler.step()
    print('====================Finish training=========================')

    #Show the Training history
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 8))
    ax1.plot(range(len(training_loss)), training_loss, 'b', label = 'training loss')
    ax1.plot(range(len(validation_loss)), validation_loss, 'g', label = 'validation loss')
    ax1.set_xlabel('Number of steps')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(range(len(validation_accuracys)), validation_accuracys, 'g', label = 'validation accuracy')
    ax2.plot(range(len(training_accuracys)), training_accuracys, 'b', label = 'training accuracy')
    ax2.set_xlabel('Number of steps')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.show()

if __name__ == '__main__':
    main()

# output shape = (10, 36)