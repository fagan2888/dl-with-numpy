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
