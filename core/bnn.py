import tensorflow as tf
import larq as lq
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle








class BNN(object):
    def __init__(self, path):
        if isinstance(path, list):
            self.models = [tf.keras.models.load_model(i) for i in path]
            self.best_model = self.models[0]
        elif isinstance(path, str):
            self.best_model = tf.keras.models.load_model(path)
            self.models = [tf.keras.models.load_model(path)]


    def predict(self, data, best_index=None):
        if best_index is not None:
            yp = self.models[best_index].predict(data).argmax(axis=1)
            return yp
        else:
            yp = np.zeros((data.shape[0], len(self.models)))
            for i in range(len(self.models)):
                yp[:, i] = self.models[i].predict(data).argmax(axis=1)
            return yp.mean(axis=1).round()

    def predict_proba(self, data):
        yp = np.zeros((data.shape[0], 2))
        for i in range(len(self.models)):
            yp += self.models[i].predict(data)
        return yp / len(self.models)