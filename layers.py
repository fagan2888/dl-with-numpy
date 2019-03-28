# Layers

from activation_functions import *
from loss import *

import numpy as np
import math

class Layer(object):
    def set_input_shape(self,shape):
        self.input_shape = shape
    def layer_name(self):
        return self.__class__.__name__
    
