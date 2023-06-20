import math
import os.path
from globals import *

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def show_evaluation(net, test_set,subsequences, scaler,i,test_MSE,test_RMSE,test_MAE,test_R2,debug=True):
    ''' Evaluates performance of the RNN on the entire
        dataset, and shows the prediction as well as
        target values.

    Arguments:
        net (nn.Module): RNN to evaluate
        dataset (numpy.ndarray): dataset
        target (numpy.ndarray): target values for prediction,
                                original (unscaled)
        scaler (MinMaxScaler): used for denormalization
        debug (bool): should we calculate/display eval.
                      MSE/MAE
    '''
    net.eval()
    TRAIN_SPLIT=int(config.train_ratio*len(subsequences))
    VAL_SPLIT=TRAIN_SPLIT+int(config.val_ratio*len(subsequences))
    # Prediction on the entire dataset

    prediction = net(test_set['X']).unsqueeze(-1).cpu().data.numpy()
    print(prediction.shape)
    scaling_temp=np.concatenate([subsequences[VAL_SPLIT:,-10:-1],prediction],axis=1)
    prediction = scaler.inverse_transform(scaling_temp)[:,-1]


    scaling_temp1 = np.concatenate([subsequences[VAL_SPLIT:, -10:-1],test_set['Y'].cpu().numpy().reshape(-1,1)], axis=1)
    test_real = scaler.inverse_transform(scaling_temp1)[:,-1]

    # Plotting the original sequence vs. predicted
    plt.figure(figsize=(20, 10))
    plt.plot(test_real,label='real')
    plt.plot(prediction,label='predict')
    plt.title('Multivariate Time-Series Forecast')
    plt.ylabel("Patv")
    plt.legend()
    save_dir_path = 'C:/Users/cx/Desktop/images1gru/predict/'
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    plt.savefig('C:/Users/cx/Desktop/images1gru/predict/Turb{}predict.png'.format(i + 1))
    plt.clf()
    #plt.show()

    if debug:
        # Calculating test MSE & MAE
        test_mse = mean_squared_error(prediction, test_real)
        test_rmse=math.sqrt(test_mse)
        test_mae = mean_absolute_error(prediction, test_real)
        test_r2=r2_score(prediction,test_real)

        print(f"Test MSE: {test_mse:.4f} | Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f} | Test R2: {test_r2:.4f}")
        test_MSE.append(test_mse)
        test_RMSE.append(test_rmse)
        test_MAE.append(test_mae)
        test_R2.append(test_r2)
    return test_MSE, test_RMSE, test_MAE, test_R2



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
    plt.ylabel('MSELoss')
    plt.legend()
    save_dir_path = 'C:/Users/cx/Desktop/images1gru/loss/'
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    plt.savefig('C:/Users/cx/Desktop/images1gru/loss/Turb{}loss.png'.format(i + 1))
    plt.clf()
    #plt.show()


def display_dataset(dataset,i):
    ''' Displays the loaded data

    Arguments:
        dataset(numpy.ndarray): loaded data
        xlabels(numpy.ndarray): strings representing
                                 according dates
    '''
    #global x_ticks
    #global tick_positions

    # Remove information about hours (only for plotting purposes)
    #xlabels = [x[:10] for x in xlabels]
    # We can't show every date in the dataset
    # on the x axis because we couldn't see
    # any label clearly. So we extract every
    # n-th label/tick
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
    #plt.xlabel('Time')
    plt.ylabel("Patv")
    #plt.xticks(tick_positions, x_ticks, size='small')
    plt.legend()
    save_dir_path = 'C:/Users/cx/Desktop/images1gru/all data/'
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    plt.savefig('C:/Users/cx/Desktop/images1gru/all data/Turb{}all.png'.format(i + 1))
    plt.clf()
    #plt.show()
