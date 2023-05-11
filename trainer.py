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

    train_data = dataset[:train_set_size, :-1, :]
    train_label = dataset[:train_set_size, -1, :]

    test_data = dataset[train_set_size:, :-1]
    test_label = dataset[train_set_size:, -1, :]

    print("x_train's shape: ", train_data.shape)
    print("y_train's shape: ", train_label.shape)
    print("x_test's shape: ", test_data.shape)
    print("y_test's shape: ", test_label.shape)

    return train_data, train_label, test_data, test_label, scaler


def build_model(input_dim, hidden_dim, num_layers, output_dim):
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    return model


def train(model, num_epochs, x_train, y_train_lstm, criterion, optimiser):
    loss_list=[]
    for i in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", i, "MSE: ", loss.item())
        loss_list.append(loss.item())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    torch.save(model, "price_predictor_epo{}.pth".format(num_epochs))
    return y_train_pred, loss_list


def evaluation(model, x_test, test_label, scaler):
    model = model.eval()
    #torch.onnx.export(model, x_test, "model.onnx", verbose=True)
    test_pred = model(x_test)
    test_pred = test_pred.detach().numpy()
    #print("check y_test_pred: ", y_test_pred)
    print("predcition is: ", test_pred.shape, type(test_pred), type(test_label), test_label.shape)
    rme_value = math.sqrt(mean_squared_error(test_label[:,0], test_pred[:,0]))
    print('RME value: %.2f RMSE' % (rme_value))

    #plt.figure()
    #plt.plot(test_pred, color='b', label='predict_price')
    #plt.plot(test_label, color='g', label='groundTruch_price')
    #plt.xlabel('date')
    #plt.ylabel('price')
    #plt.legend()
    #plt.show()

    return test_pred

    
if __name__ == '__main__':
    train_data, train_label, test_data, test_label, scaler = data_preprocess('./yahoo_data.xlsx', seq_length=20)

    train_data = torch.from_numpy(train_data).type(torch.Tensor)
    test_data = torch.from_numpy(test_data).type(torch.Tensor)
    # 真实的数据标签
    train_label_lstm = torch.from_numpy(train_label).type(torch.Tensor)
    test_label_lstm = torch.from_numpy(test_label).type(torch.Tensor)
    train_label_gru = torch.from_numpy(train_label).type(torch.Tensor)
    test_label_gru = torch.from_numpy(test_label).type(torch.Tensor)

    input_dim = 1
    # 隐藏层特征的维度
    hidden_dim = 32
    # 循环的layers
    num_layers = 2
    # 预测后一天的收盘价
    output_dim = 1
    num_epochs = 100

    model = build_model(input_dim, hidden_dim, num_layers, output_dim)
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    #hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []

    y_train_pred, loss_list = train(model, num_epochs, train_data, train_label_lstm, criterion, optimiser)
    plt.figure()
    plt.plot(loss_list, color='b', label='loss curve')
    plt.savefig('loss_curve.png')


    training_time = time.time() - start_time
    print("Training time: {}".format(training_time)) 

    #predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    #print(predict)  # 预测值
    #original = pd.DataFrame(scaler.inverse_transform(train_label_lstm.detach().numpy()))
    #print(original)  # 真实值
    
    test_pred = evaluation(model, test_data, test_label, scaler)
    test_predict = pd.DataFrame(scaler.inverse_transform(test_pred))
    test_label = pd.DataFrame(scaler.inverse_transform(test_label))
    #print(test_predict)
    #print(test_label)
    plt.figure()
    plt.plot(test_predict, color='b', label='predict_price')
    plt.plot(test_label, color='g', label='groundTruch_price')
    plt.xlabel('date')
    plt.ylabel('price')
    plt.legend()
    plt.show()


