import math
import os.path

from globals import *

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


x_ticks = list()
tick_positions = list()


def show_evaluation(net,test_set,scaler,i,test_MSE,test_RMSE,test_MAE,test_R2,debug=True):
    ''' 评估 RNN 在测试集上的性能，并显示预测值和目标值.
    参数:
        net (nn.Module): RNN to evaluate
        dataset (numpy.ndarray): dataset
        scaler (MinMaxScaler): 反归一化
        debug (bool): should we calculate/display eval.MSE/MAE
    '''
    #预测测试数据集
    net.eval()
    test_predict = net(test_set['X'],test_set['Y'])

    # 对实际值和预测值反归一化
    test_predict = scaler.inverse_transform(test_predict.cpu().data.numpy())
    test_set['Y'] = scaler.inverse_transform(test_set['Y'].cpu().squeeze(-1).data.numpy().reshape(-1,1))

    print(test_predict)
    print(test_set['Y'])

    tensor1=np.array(test_set['Y'])
    tensor2=np.array(test_predict)
    column_data1=tensor1.flatten()
    column_data2 = tensor2.flatten()
    data=np.column_stack((column_data1,column_data2))
    df = pd.DataFrame(data, columns=['real','predict'])
    df.to_csv('C:/Users/22279/Desktop/transformerdata.csv',index=False)

    # 绘制原始序列与预测序列
    plt.figure(figsize=(20,10))
    plt.plot(test_set['Y'],label='real')
    plt.plot(test_predict,label='predict')
    plt.ylabel("Patv")
    plt.title('Forecast and Real')
    plt.legend()
    save_dir_path='C:/Users/22279/Desktop/imagesgru/predict/'
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    plt.savefig('C:/Users/22279/Desktop/imagesgru/predict/Turb{}predict.png'.format(i + 1))
    plt.clf()
    #plt.show()

    if debug:
        #计算测试集的MSE、MAE
        test_mse = mean_squared_error(test_predict, test_set['Y'])
        test_rmse = math.sqrt(test_mse)
        test_mae = mean_absolute_error(test_predict, test_set['Y'])
        test_r2 = r2_score(test_predict, test_set['Y'])

        print(f"Test MSE: {test_mse:.4f} | Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f} | Test R2: {test_r2:.4f}")

        test_MSE.append(test_mse)
        test_RMSE.append(test_rmse)
        test_MAE.append(test_mae)
        test_R2.append(test_r2)
    return test_MSE,test_RMSE,test_MAE,test_R2



def show_loss(history,i):
    ''' Display train and evaluation loss

    Arguments:
        history(dict): Contains train and test loss logs
    '''
    plt.figure(figsize=(20, 10))
    plt.plot(history['train_loss'], label='Train loss')
    plt.plot(history['val_loss'], label='Val loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    save_dir_path = 'C:/Users/22279/Desktop/imagesgru/loss/'
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    plt.savefig('C:/Users/22279/Desktop/imagesgru/loss/Turb{}loss.png'.format(i + 1))
    plt.clf()
    #plt.show()


def display_dataset(dataset,i):
    ''' Displays the loaded data

    Arguments:
        dataset(numpy.ndarray): loaded data
        xlabels(numpy.ndarray): strings representing according dates
    '''
    #global x_ticks
    #global tick_positions
    # 我们无法在 x 轴上显示数据集中的每个日期，因为我们无法清楚地看到任何标签。所以我们提取每个第 n 个标签/刻度
    #segment = int(len(dataset) / 6)

    #for i, date in enumerate(xlabels):
        #if i > 0 and (i + 1) % segment == 0:
            #x_ticks.append(date)
            #tick_positions.append(i)
        #elif i == 0:
            #x_ticks.append(date)
            #tick_positions.append(i)

    # Display loaded data
    plt.figure(figsize=(20, 10))
    plt.plot(dataset)
    plt.title(f'Turb[{i+1}]Data')
    plt.ylabel("Patv")
    #plt.xticks(tick_positions, x_ticks, size='small')
    plt.legend()
    #plt.show()
    save_dir_path = 'C:/Users/22279/Desktop/imagesgru/all data/'
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    plt.savefig('C:/Users/22279/Desktop/imagesgru/all data/Turb{}all.png'.format(i+1))
    plt.clf()


    #时间序列分解
    decomp=seasonal_decompose(dataset,model='additive',period=1008)
    trend=decomp.trend
    seasonal=decomp.seasonal
    residual=decomp.resid

    #绘制分解图
    plt.figure(figsize=(20, 10))
    plt.subplot(411)
    plt.plot(dataset,label='Original')
    plt.legend()

    plt.subplot(412)
    plt.plot(trend,label='Trend')
    plt.legend()

    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend()

    plt.subplot(414)
    plt.plot(residual,label='Residuals')
    plt.legend()
    save_dir_path = 'C:/Users/22279/Desktop/imagesgru/decompose/'
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    plt.savefig('C:/Users/22279/Desktop/imagesgru/decompose/Turb{}decompose.png'.format(i + 1))
    plt.clf()
    #plt.show()
