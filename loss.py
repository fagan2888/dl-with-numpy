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
