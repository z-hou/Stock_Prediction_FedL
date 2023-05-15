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
samples_data = [60, 203, 246, 214, 267]

def train(model, model_index, train_data, train_label, num_epochs):
        criterion = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        
        train_data = torch.from_numpy(train_data).type(torch.Tensor)
        train_label = torch.from_numpy(train_label).type(torch.Tensor)
        #print("train data: ", train_data, train_data.shape)
        for i in range(num_epochs):
            train_prediction = model(train_data)
            loss = criterion(train_prediction, train_label)
            #print("local model {}".format(model_index), " | Epoch ", i, " | Train MSE: ", loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        return model

class EdgeSystem(object):
    def __init__(self, model, train_data, train_label, eval_dataset, eval_datalabel, client_number, commu_round):
        self.global_model = model
        self.eval_dataset = eval_dataset
        self.eval_datalabel = eval_datalabel
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
            #self.evaluation()
        self.evaluation()

    def clients_work(self):
        for i in range(self.client_number):
            print("train model {}".format(i+1))
            self.local_models[i] = train(self.local_models[i], i, self.train_data[i], self.train_label[i], 20)


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
                    params[k] += v * samples_data[j]/990
        with torch.no_grad():
            for k, v in self.global_model.named_parameters():
                v.copy_(params[k])

    def evaluation(self):
        model = self.global_model.eval()
        test_data = torch.from_numpy(self.eval_dataset).type(torch.Tensor)
        #torch.onnx.export(model, x_test, "model.onnx", verbose=True)
        test_pred = model(test_data)
        test_pred = test_pred.detach().numpy()
        #print("check y_test_pred: ", y_test_pred)
        print("Global_model predcition is: ", test_pred.shape, type(test_pred), type(self.eval_datalabel), test_label.shape)
        rme_value = math.sqrt(mean_squared_error(self.eval_datalabel[:,0], test_pred[:,0]))
        print('RME value: %.2f RMSE' % (rme_value))

        plt.figure()
        plt.plot(test_pred, color='b', label='predict_price')
        plt.plot(test_label, color='g', label='groundTruch_price')
        plt.xlabel('date')
        plt.ylabel('price')
        plt.legend()
        plt.savefig('./global.png')
        plt.close()

        for i in range(self.client_number):
            model_ = self.local_models[i].eval()
            test_pred_local = model(test_data)
            test_pred_local = test_pred_local.detach().numpy()
            rme_value_local= math.sqrt(mean_squared_error(self.eval_datalabel[:,0], test_pred_local[:,0]))
            print('RME value for client {}: {} RMSE'.format(i, rme_value_local))
            plt.figure()
            plt.plot(test_pred, color='b', label='predict_price')
            plt.plot(test_label, color='g', label='groundTruch_price')
            plt.xlabel('date')
            plt.ylabel('price')
            plt.legend()
            plt.savefig('./client_{}.png'.format(i))
            plt.close()


        




def divide_data(client_number, train_data, train_label, NIID=False):

    if NIID == False:
        sample_numbers = train_label.shape[0]
        print("Check sample_numbers: ", sample_numbers)

        Partitioned_data = np.split(train_data, client_number, axis=0)
        Partitioned_label = np.split(train_label, client_number, axis=0)
    
        for item in Partitioned_data:
            print(item.shape)
        return Partitioned_data, Partitioned_label

    else:
        print("Check train_data: ", train_data.shape)
        print("Check train_label: ", train_label.shape)
        #print("train label: ", train_label)
        train_label_new = np.sort(train_label, axis=0) ##按列排序
        print("Check train_label_new: ", train_label_new.shape)
        sort_indx = np.argsort(train_label, axis=0)
        #print("index after argsort: ", sort_indx)
        train_data_new_list = []
        for ind in sort_indx:
            train_data_new_list.append(train_data[ind])
        train_data_new = np.squeeze(np.array(train_data_new_list), axis=1)
        print("Check train_data_new: ", train_data_new.shape)
        #for item in np.around(train_label, decimals=4):
        #    print(item)
        ##add categories
        categ = []
        for item in train_label:
            if item >= -10.0 and item < -0.8:
                categ.append(0)
            if item >= -0.8 and item < -0.6:
                categ.append(1)
            if item >= -0.6 and item < -0.4:
                categ.append(2)
            if item >= -0.4 and item < -0.2:
                categ.append(3)
            if item >= -0.2 and item < 0:
                categ.append(4)
            if item >= 0 and item < 0.2:
                categ.append(5)
            if item >= 0.2 and item < 0.4:
                categ.append(6)
            if item >= 0.4 and item < 0.6:
                categ.append(7)
            if item >= 0.6 and item < 0.8:
                categ.append(8)
            if item >= 0.8 and item <= 10.0:
                categ.append(9)
            #if item <= -1 or item > 1.0:
            #    print("out of range: ", item, item.dtype)
        print("0: ", categ.count(0))
        print("1: ", categ.count(1))
        print("2: ", categ.count(2))
        print("3: ", categ.count(3))
        print("4: ", categ.count(4))
        print("5: ", categ.count(5))
        print("6: ", categ.count(6))
        print("7: ", categ.count(7))
        print("8: ", categ.count(8))
        print("9: ", categ.count(9))
        print("categ len: ", len(categ))
        print(categ)
        slice_point1 = categ.count(0)+categ.count(1)+categ.count(2)
        slice_point2 = categ.count(0)+categ.count(1)+categ.count(2)+categ.count(3)
        slice_point3 = categ.count(0)+categ.count(1)+categ.count(2)+categ.count(3)+categ.count(4)
        slice_point4 = categ.count(0)+categ.count(1)+categ.count(2)+categ.count(3)+categ.count(4)+categ.count(5)+categ.count(6)+categ.count(7)
    
        print(slice_point1," ", slice_point2," ", slice_point3," ", slice_point4)
        train_data_new = np.split(train_data_new, [slice_point1, slice_point2, slice_point3, slice_point4], axis=0)
        train_label_new = np.split(train_label_new, [slice_point1, slice_point2, slice_point3, slice_point4], axis=0)
        
        for item in train_data_new:
            print(item.shape)
        for ite in train_label_new:
            print(ite.shape)

        return train_data_new, train_label_new



if __name__ == '__main__':
    train_data, train_label, test_data, test_label, scaler = data_preprocess('./yahoo_data.xlsx', seq_length=20)
    Partitioned_data, Partitioned_label = divide_data(5, train_data, train_label, NIID=True)


    
    input_dim = 1
    # 
    hidden_dim = 32
    # LSTM layers
    num_layers = 2
    # prediction
    output_dim = 1

    model = build_model(input_dim, hidden_dim, num_layers, output_dim)
    EDsys = EdgeSystem(model, Partitioned_data, Partitioned_label, test_data, test_label, client_number=5, commu_round=25)
    EDsys.workflow()
    

    #print(model.len)
    #for k,v in model.named_parameters():
    #    print(k)