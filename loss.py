# Loss Functions

from activation_functions import *

class SquareLoss(Loss):
    def __init__(self): pass
    def loss(self, y_true, y_pred):
        return 0.5 * np.power((y_true - y_pred), 2)
    def gradient(self, y_true, y_pred):
        return -(y_true - y_pred)

class MeanSquareLoss():
    def __init__(self):
         pass
    def loss(self,y_true,y_pred):
        return np.power((y_true - y_pred), 2) / len(y_true)
    def gradient(self, y_true, y_pred):
        return -((2 * (y_true - y_pred)) / len(y_true))

class CrossEntropyLoss():
    def __init__(self): pass
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
