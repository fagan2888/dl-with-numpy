# Activation Functions
import numpy as np

class Identity():
    def __call__(self,x):
        return x
    def gradient(self,x):
        return 1

class Sigmoid():
    def __call__(self,x):
        return 1 / (1 + np.exp(-x))
    def gradient(self,x):
        return self.__call__(x) * (1 - self.__call__(x))

class TanH():
    def __call__(self,x):
        return (np.exp(2x) - 1) * (np.exp(2x) + 1)
    def gradient(self,x):
        return 1 - np.pow(self.__call__(x),2)

class ReLu():
    def __call__(self,x):
        return np.where(x>=0,x,0)
    def gradient(self,x):
        return np.where(x>=0,1,0)

class LeakyReLu():
    def __call__(self,x):
        return np.where(x>=0,x,0.01 * x)
    def gradient(self,x):
        return np.where(x>=0,1,0.01)

class ELU():
    def __init__(self,alpha = 0.01):
        self.alpha = alpha
    def __call__(self,x):
        return np.where(x>=0,x,self.aplha * (np.exp(x) - 1))
    def gradient(self,x):
        return np.where(x>=0,1,self.__call__(x) + self.alpha)

class SELU():
    def __init__(self):
        self.alpha = 1.67326
        self.scale = 1.0507

    def __call__(self, x):
        return self.scale * np.where(x >= 0.0, x, self.alpha*(np.exp(x)-1))

    def gradient(self, x):
        return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))
