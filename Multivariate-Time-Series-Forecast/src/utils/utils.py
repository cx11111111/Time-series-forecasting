from globals import *
from .visualisation import *

import torch
import torch.optim as optim

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from datetime import datetime


def evaluate(net, val_set, history):
    ''' Evaluates the performance of the
        RNN on the test set.

    Arguments:
        net (nn.Module): RNN net
        val_set (dict): validation input and target output
        history (dict): dict used for loss log
    Returns:
        val_loss (float): loss on the validation set
    '''
    net.eval()

    val_predict = net(val_set['X'])
    val_loss = loss_func(val_predict, val_set['Y'])
    history['val_loss'].append(val_loss.item())

    return val_loss.item()


def train(net, train_loader, optimizer, history):
    ''' Evaluates the performance of the
        RNN on the test set.

    Arguments:
        net (nn.Module): RNN net
        train_loader (DataLoader): train input and 
                                   target output
        optimizer: optimizer object (Adam)
        history (dict): dict used for loss log
    Returns:
        test_loss (float): loss on the test set
    '''
    net.train()

    total_num = 0
    train_loss = 0
    for input, target in train_loader:
        optimizer.zero_grad()
        loss = loss_func(net(input), target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(target)
        total_num += len(target)

    history['train_loss'].append(train_loss / total_num)

    return train_loss / total_num


def train_loop(net, epochs, lr, wd, train_loader, val_set,i, debug=True):
    ''' Performs the training of the RNN using Adam optimizer.
        Train and evaluation losses are being logged.

    Arguments:
        net (nn.Module): RNN to be trained
        epochs (int): number of epochs we wish to train
        lr (float): max learning rate for Adam optimizer
        wd (float): L2 regularization weight decay
        train_loader (DataLoader): train input and target output
        val_set (dict): validation input and target output
        debug (bool): Should we display train progress?
    '''
    history = dict()
    history['train_loss'] = list()
    history['val_loss'] = list()

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epochs):
        train_loss = train(net, train_loader, optimizer, history)

        with torch.no_grad():
            val_loss = evaluate(net, val_set, history)

        if debug and (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.8f}",
                  f" |  Validation Loss: {val_loss:.8f}")

    if debug:
        show_loss(history,i)


def train_val_test_split(subsequences):
    ''' Splits the loaded subsequences into train, validation and test set w.r.t. split ratio.s
    Arguments:
        subsequences (numpy.ndarray): input-output pairs
    Returns:
        train_loader (DataLoader): train set inputs and target outputs
        val_set (dict): validation set inputs and target outputs
        test_set (dict): test set inputs and target outputs
    '''
    # Get the length of the train set
    TRAIN_SPLIT = int(config.train_ratio*len(subsequences))
    VAL_SPLIT = TRAIN_SPLIT + int(config.val_ratio*len(subsequences))

    # Extract train set
    trainX = torch.Tensor(subsequences[:TRAIN_SPLIT, :-1]).to(device)
    trainY = torch.Tensor(subsequences[:TRAIN_SPLIT, -1]).to(device)
    #trainY=trainY.reshape(-1,1)
    # Extract validation set
    valX = torch.Tensor(subsequences[TRAIN_SPLIT:VAL_SPLIT, :-1]).to(device)
    valY = torch.Tensor(subsequences[TRAIN_SPLIT:VAL_SPLIT, -1]).to(device)
    #valY=valY.reshape(-1,1)
    # Extract test set
    testX = torch.Tensor(subsequences[VAL_SPLIT:, :-1]).to(device)
    testY = torch.Tensor(subsequences[VAL_SPLIT:, -1]).to(device)
    #testY=testY.reshape(-1,1)

    # Adapt train/val/test inputs to (batch, sequence len, input_dim) shape
    COL_NUM = int(trainX.shape[-1]/config.lag)
    trainX = trainX.view(-1, config.lag, COL_NUM)
    valX = valX.view(-1, config.lag, COL_NUM)
    testX = testX.view(-1, config.lag, COL_NUM)

    # Create train set loader
    train_set = torch.utils.data.TensorDataset(trainX, trainY)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.bs)

    # Create test set
    val_set = dict()
    val_set['X'] = torch.Tensor(valX).to(device)
    val_set['Y'] = torch.Tensor(valY).to(device)

    # Create test set
    test_set = dict()
    test_set['X'] = torch.Tensor(testX).to(device)
    test_set['Y'] = torch.Tensor(testY).to(device)

    return train_loader, val_set, test_set


def extract_subsequences(sequence, cols, lag):
    ''' Extracts subsequences in order to create
        input - output pairs for the train / test data.
        We extract input subsequences of lag length
        ex: sequence[i:i + lab] and set it's target
        output as the next value in the sequence.
        Time-lagged values are taken for each input feature.

    Arguments:
        sequence(numpy.ndarray): entire dataset sequence
        cols (list): column(feature) names of the original dataset
        lag (int): number of previous values we use as input
    Returns:
        subseqs (numpy.ndarray): list of extracted input - output pairs
    '''
    # Recreate the original dataset
    dataset = pd.DataFrame(sequence)
    dataset.columns = cols

    lagged_features = list()
    for i in range(lag, 0, -1):
        # Get the time lagged features: Features(t-i)
        lagged_features.append(dataset.shift(i))
    #print("1:{}".format(lagged_features))

    # Remember the original sequence (which we are trying to predict)
    lagged_features.append(dataset['Patv'])
    #print("2:{}".format(lagged_features))

    # Create new dataframe which constis of lagged features and target sequence
    subseqs = pd.concat(lagged_features, axis=1)
    #print("3:{}".format(subseqs))

    # When shifting rows, missing values get NA assigned so we remove rows which contain NA values
    subseqs.dropna(inplace=True)

    #print(subseqs)

    return subseqs.values


def load_dataset(dataset_path, i,show_data=True):
    ''' Loads the dataset stored as the csv file.
        Displays the loaded data if desired.

    Arguments:
        dataset_path(string): path to the dataset file
        show_data(bool): should we show loaded data?
    Returns:
        data (dict): loaded & scaled dataset, with some
                     additionally extracted information
        scaler (MinMaxScaler): normalizes dataset values
    '''
    # Column which contains the values which we are trying
    # to predict
    TARGET_COL = 'Patv'

    # Load the dataset as DataFrame
    dataset = pd.read_csv(dataset_path)
    dataset=dataset[dataset[TARGET_COL]!=0]
    dataset = dataset[dataset[TARGET_COL] != -0.3]
    dataset.dropna(inplace=True)
    groups = dataset.groupby('TurbID')
    dataframe = []
    for name, group in groups:
        group = group.reset_index(drop=True)
        dataframe.append(group)
    dataset=dataframe[i]
    # 'No' is not a necessary feature
    dataset.drop(columns=['TurbID'], inplace=True)
    dataset.drop(columns=['Day'], inplace=True)
    dataset.drop(columns=['Tmstamp'], inplace=True)
    corr_matrix = dataset.corr()
    print(corr_matrix)

    # Extract column names
    cols = list(dataset.columns)
    # Extract target values (used for plotting later)
    target = dataset[TARGET_COL].values
    if show_data:
        display_dataset(target,i)

    # We normalize the dataset values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(dataset)
    
    data = dict()
    data['original'] = dataset
    data['scaled'] = scaled
    data['column_names'] = cols
    data['target'] = target
    #print(data)
    return data, scaler
