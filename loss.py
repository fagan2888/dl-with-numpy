# Loss Functions

import numpy as np

class SquareLoss(Loss):
    def __init__(self):
        pass
    def loss(self, y_true, y_pred):
        return 0.5 * np.power((y_true - y_pred), 2)
    def gradient(self, y_true, y_pred):
        return -(y_true - y_pred)

class MeanSquareError():
    def __init__(self):
         pass
    def loss(self,y_true,y_pred):
        return np.mean(np.power((y_true - y_pred), 2))
    def gradient(self, y_true, y_pred):
        return -(np.mean((2 * (y_true - y_pred))))

class BinaryCrossEntropyLoss():
    def __init__(self):
        pass
    def loss(self, y_true, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    def acc(self, y_true, y_pred):
        return accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    def gradient(self, y_true, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)

class CategoricalCrossEntropy():
    def __init__(self):
        pass
    def loss(self,y_true,y_pred):
        # Avoid divison by zero
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.sum(y_true * np.log(y_pred+1e-9)) / y_pred.shape[0]
    def gradient(self,y_true,y_pred):
        # TODO
        NotImplementedError()

class MeanAbsoluteError():
    def __init__(self):
        pass
    def loss(self,y_true,y_pred):
        return np.mean(np.abs(y_pred - y_true))
    def gradient():
        # TODO
        NotImplementedError()
