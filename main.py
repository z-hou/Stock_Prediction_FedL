import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from model import LSTM
import time
import torch
import torch.nn as nn
import math
from sklearn.metrics import mean_squared_error


stock_data = pd.read_excel('./yahoo_data.xlsx')


'''
##Visiualize close price
plt.figure(figsize=(15, 9))
plt.plot(stock_data[['Close*']])
plt.xticks(range(0, stock_data.shape[0], 100), stock_data['Date'].loc[::20], rotation=45)
#plt.xticks(stock_data["Close*"], stock_data['Date'], rotation=45)
plt.title("****** Stock Price", fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price (USD)', fontsize=18)
plt.savefig('StockPrice.jpg')
plt.show()
'''
def data_preprocess(data_path, seq_length):
    stock_data = pd.read_excel(data_path)
    price = stock_data[['Adj Close**']]
    print(price.info())

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_raw = scaler.fit_transform(price["Adj Close**"].values.reshape(-1, 1))

    #data_raw = price.to_numpy()
    data_raw = np.flip(data_raw)
    print("Check data_raw: ", data_raw)
    dataset = []
    
    for index in range(len(data_raw) - seq_length):
        dataset.append(data_raw[index: index + seq_length])
	
    dataset = np.array(dataset)
    #dataset = np.expand_dims(dataset, axis=-1)
    print("Check dataset: ", type(dataset), "Shape: ", dataset.shape)  # (232, 20, 1)
    # 按照8:2进行训练集、测试集划分
    test_set_size = int(np.round(0.2 * dataset.shape[0]))
    train_set_size = dataset.shape[0] - (test_set_size)

    x_train = dataset[:train_set_size, :-1, :]
    y_train = dataset[:train_set_size, -1, :]

    x_test = dataset[train_set_size:, :-1]
    y_test = dataset[train_set_size:, -1, :]

    print("x_train's shape: ", x_train.shape)
    print("y_train's shape: ", y_train.shape)
    print("x_test's shape: ", x_test.shape)
    print("y_test's shape: ", y_test.shape)

    return [x_train, y_train, x_test, y_test, scaler]


def build_model(input_dim, hidden_dim, num_layers, output_dim):
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    return model


def train(model, num_epochs, x_train, y_train_lstm, criterion, optimiser):
    loss_list=[]
    for t in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", t, "MSE: ", loss.item())
        loss_list.append(loss.item())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()  
    return y_train_pred, loss_list


def evaluation(model, x_test, test_label, scaler):
    model = model.eval()
    y_test_pred = model(x_test)
    y_test_pred = y_test_pred.detach().numpy()
    #print("check y_test_pred: ", y_test_pred)
    print("predcition is: ", y_test_pred.shape, type(y_test_pred), type(test_label), test_label.shape)
    testScore = math.sqrt(mean_squared_error(test_label[:,0], y_test_pred[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    plt.figure()
    plt.plot(y_test_pred, color='b', label='predict_price')
    plt.plot(test_label, color='g', label='groundTruch_price')
    plt.xlabel('date')
    plt.ylabel('price')
    plt.legend()
    plt.show()

    return y_test_pred

    
if __name__ == '__main__':
    x_train, y_train, x_test, y_test, scaler = data_preprocess('./yahoo_data.xlsx', seq_length=20)

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    # 真实的数据标签
    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
    y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

    input_dim = 1
    # 隐藏层特征的维度
    hidden_dim = 32
    # 循环的layers
    num_layers = 2
    # 预测后一天的收盘价
    output_dim = 1
    num_epochs = 500

    model = build_model(input_dim, hidden_dim, num_layers, output_dim)
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    #hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []

    y_train_pred, loss_list = train(model, num_epochs, x_train, y_train_lstm, criterion, optimiser)
    plt.figure()
    plt.plot(loss_list, color='b', label='loss curve')
    plt.savefig('loss_curve.png')


    training_time = time.time() - start_time
    print("Training time: {}".format(training_time)) 

    predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    print(predict)  # 预测值
    original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))
    print(original)  # 真实值

    evaluation(model, x_test, y_test, scaler)



