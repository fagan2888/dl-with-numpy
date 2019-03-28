# Activation Functions
import numpy as np

class identity():
    def __call__(self,x):
        return x
    def gradient(self,x):
        return 1

class sigmoid():
    def __call__(self,x):
        return 1 / (1 + np.exp(-x))
    def gradient(self,x):
        return self.__call__(x) * (1 - self.__call__(x))
