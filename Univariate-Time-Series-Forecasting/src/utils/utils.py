from globals import *
from .visualisation import *

import torch
import torch.optim as optim

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def evaluate(net, valid_set, history):
    ''' 评估RNN在验证集上的性能
    参数:
        net (nn.Module): RNN net
        test_set (dict): test input and target output
        history (dict): dict used for loss log
    返回:
        test_loss (float): loss on the test set
    '''
    net.eval()

    val_predict = net(valid_set['X'])
    val_loss = loss_func(val_predict, valid_set['Y'])
    history['val_loss'].append(val_loss.item())

    return val_loss.item()


def train(net, train_loader, optimizer, history):
    ''' 评估RNN在训练集上的性能
    参数:
        net (nn.Module): RNN net
        train_loader (DataLoader): train input and target output
        optimizer: optimizer object (Adam)
        history (dict): dict used for loss log
    返回:
        train_loss (float): loss on the train_loader
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


def train_loop(net, epochs, lr, wd, train_loader, valid_set, i,debug=True):
    ''' 使用Adam优化器执行RNN的训练.
        记录训练和评估损失.
    参数:
        net (nn.Module): RNN to be trained
        epochs (int): number of epochs we wish to train
        lr (float): max learning rate for Adam optimizer
        wd (float): L2 regularization weight decay
        train_loader (DataLoader): train input and target output
        test_set (dict): test input and target output
        debug (bool): Should we display train progress?
    '''
    history = dict()
    history['train_loss'] = list()
    history['val_loss'] = list()

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epochs):
        train_loss = train(net, train_loader, optimizer, history)

        with torch.no_grad():
            val_loss = evaluate(net, valid_set, history)

        if debug and (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.8f}",
                  f" |  Val Loss: {val_loss:.8f}")

    if debug:
        show_loss(history,i)


def train_test_split(subsequences):
    ''' 将数据集划分为训练集和测试集.
    参数:
        subsequences (): input-output pairs
    返回:
        train_loader (DataLoader): train set inputs and target outputs
        test_set (dict): test set inputs and target outputs
    '''
    #训练集的长度
    TRAIN_SIZE = int(config.split_ratio * len(subsequences))
    valid_size=int(config.split_ratio1*len(subsequences))
    train_seqs = subsequences[:TRAIN_SIZE]
    valid_seqs=subsequences[TRAIN_SIZE:valid_size]
    test_seqs = subsequences[valid_size:]

    # Divide inputs and target outputs
    trainX, trainY = [torch.Tensor(list(x)).to(device)
                      for x in zip(*train_seqs)]
    validX,validY=[torch.Tensor(list(x)).to(device)
                   for x in zip(*valid_seqs)]
    testX, testY = [torch.Tensor(list(x)).to(device)
                    for x in zip(*test_seqs)]

    train_set = torch.utils.data.TensorDataset(trainX, trainY)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.bs)

    valid_set = dict()
    valid_set['X'] = torch.Tensor(validX).to(device)
    valid_set['Y'] = torch.Tensor(validY).to(device)

    test_set = dict()
    test_set['X'] = torch.Tensor(testX).to(device)
    test_set['Y'] = torch.Tensor(testY).to(device)
    print(valid_set)

    return train_loader, valid_set, test_set


def extract_subsequences(sequence, lag):
    ''' 划分输入数据和目标值
    参数:
        sequence(numpy.ndarray): 整个数据集
        lag(int): number of previous values we use as input
        pred(int):the length we use as the forecast
    返回:
        subseqs(list): 输入输出对列表
    '''
    subseqs=list()
    for i in range(len(sequence) - lag - 1):
        input = sequence[i:i + lag]
        output = sequence[i + lag]

        subseqs.append((input, output))

    return subseqs


def load_dataset(dataset_path,i, show_data=True):
    ''' 加载数据集.
    参数:
        dataset_path(string): path to the dataset file
        show_data(bool): should we show loaded data?
    返回:
        dataset (numpy.ndarray): loaded dataset
        scaler (MinMaxScaler): normalizes dataset values
    '''
    # Load the dataset as DataFrame
    dataset = pd.read_csv(dataset_path)
    #print(dataset)
    columns=["TurbID","Patv"]
    dataset=dataset[columns]
    print(dataset)
    target_col='Patv'
    dataset = dataset[dataset[target_col] != 0]
    dataset = dataset[dataset[target_col] != -0.3]
    # 处理异常值
    dataset = dataset.dropna()
    groups=dataset.groupby('TurbID')
    dataframe=[]
    for name,group in groups:
        group=group.reset_index(drop=True)
        dataframe.append(group)
    #print(dataframe)

    #xlabels = dataset.iloc[:, 2].values
    dataset = dataframe[i].iloc[:, 1:].values

    if show_data:
        display_dataset(dataset,i)

    # 归一化
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    return dataset, scaler
