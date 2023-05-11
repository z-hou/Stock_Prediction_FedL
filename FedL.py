from trainer import data_preprocess, build_model

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
import numpy as np
from model import LSTM
import copy


stock_data = pd.read_excel('./yahoo_data.xlsx')

class EdgeSystem(object):
    def __init__(self, model, train_data, train_label, eval_dataset, client_number, commu_round):
        self.global_model = model
        self.eval_dataset = eval_dataset
        self.local_models = []
        self.client_number = client_number
        self.commu_round = commu_round
        self.train_data = train_data
        self.train_label = train_label
        for i in range(client_number):
            temp = copy.deepcopy(model)
            self.local_models.append(temp)

    def workflow(self):
        for i in range(self.commu_round):
            self.send_weights()
            self.clients_work()
            self.aggregation()

    def clients_work(self):
        for i in range(self.client_number):
            self.local_models[i] = self.train(self.local_models[i], self.train_data[i], self.train_label[i], 20)


    def train(self, model, train_data, train_label, num_epochs):
        criterion = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for i in range(num_epochs):
            train_prediction = model(train_data)
            loss = criterion(train_prediction, train_label)
            print("local model {}".format(j), " | Epoch ", i, " | MSE: ", loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        return model

    def send_weights(self):
        params = {}
        with torch.no_grad():
            for k, v in self.global_model.named_parameters():
                params[k] = copy.deepcopy(v)
        for j in range(self.client_number):
            with torch.no_grad():
                for k, v in self.local_models[j].named_parameters():
                     v.copy_(params[k])



    def aggregation(self):
        #s = 0
        #for j in index:
        #    # normal
        #    s += self.local_models[j].len
        params = {}
        with torch.no_grad():
            for k, v in self.local_models[0].named_parameters():
                params[k] = copy.deepcopy(v)
                params[k].zero_()
        for j in range(self.client_number):
            with torch.no_grad():
                for k, v in self.local_models[j].named_parameters():
                    #params[k] += v * (self.local_models[j].len / s)
                    params[k] += v * 0.2
        with torch.no_grad():
            for k, v in self.global_model.named_parameters():
                v.copy_(params[k])



'''
train(model, num_epochs, x_train, y_train_lstm, criterion, optimiser):
    loss_list=[]
    for i in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", i, "MSE: ", loss.item*())
        loss_list.append(loss.item())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    torch.save(model, "price_predictor_epo{}.pth".format(num_epochs))
    return y_train_pred, loss_list
'''

def Niid_data(client_number, train_data, train_label):
    sample_numbers = train_label.shape[0]
    print("Check sample_numbers: ", sample_numbers)

    Partitioned_data = np.split(train_data, client_number, axis=0)
    Partitioned_label = np.split(train_label, client_number, axis=0)
    
    #for item in all_data:
    #    print(item.shape)
    return Partitioned_data, Partitioned_label


if __name__ == '__main__':
    train_data, train_label, test_data, test_label, scaler = data_preprocess('./yahoo_data.xlsx', seq_length=20)
    Partitioned_data, Partitioned_label = Niid_data(5, train_data, train_label)

    input_dim = 1
    # 隐藏层特征的维度
    hidden_dim = 32
    # 循环的layers
    num_layers = 2
    # 预测后一天的收盘价
    output_dim = 1

    model = build_model(input_dim, hidden_dim, num_layers, output_dim)
    EDsys = EdgeSystem(model, Partitioned_data, Partitioned_label, test_data, client_number=5, commu_round=25)
    EDsys.workflow()


    #print(model.len)
    #for k,v in model.named_parameters():
    #    print(k)