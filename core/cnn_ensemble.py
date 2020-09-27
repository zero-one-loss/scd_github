import sys
sys.path.append('..')
import pickle
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split
from skorch.helper import predefined_split
from skorch.dataset import Dataset
# from core.cifar10_net import CNN, LeNet, SimpleNet
import numpy as np

class CNN(object):
    def __init__(self, round=1, SimpleNet=None):
        self.round = round
        self.scd = {}
        for i in range(self.round):
            self.scd[i] = NeuralNetClassifier(
                SimpleNet[i] if type(SimpleNet) is list else SimpleNet,
                classes=2,
                max_epochs=100,
                lr=0.001,
                criterion=torch.nn.CrossEntropyLoss,
                # Shuffle training data on each epoch
                iterator_train__shuffle=True,
                # train_split=0.1,
                batch_size=64,
                optimizer=torch.optim.Adam,
                device='cuda',
                verbose=1,
                warm_start=False,
            )

    def fit(self, data, label):
        for i in range(self.round):
            print('round %d: ' % i)

            self.scd[i].fit(data, label)
            # self.scd[i].device = 'cpu'
            # self.scd[i].module_.cpu()


    def predict(self, data, best_index=None, all=False):
        if best_index is not None:
            yp = self.scd[best_index].predict(data)
            return yp
        else:
            yp = np.zeros((data.shape[0], self.round))
            for i in range(self.round):
                yp[:, i] = self.scd[i].predict(data)
            if all:
                return yp
            yp = yp.mean(axis=1).round()

            return yp
        

class CNNVote(object):
    def __init__(self, path, group):
        self.path = [path + '_%d.pkl' % i for i in range(group)]
        self.group = group
        
    def predict(self, data):
        yp = []
        for i in range(self.group):
            with open(self.path[i], 'rb') as f:
                scd = pickle.load(f)
                yp.append(scd.predict(data, all=True))
                del scd
        yp = np.concatenate(yp, axis=1)
        yp = yp.mean(axis=1).round()

        return yp