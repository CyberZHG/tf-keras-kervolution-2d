from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K


__all__ = ['LinearKernel']


class LinearKernel(Layer):

    def __init__(self, **kwargs):
        super(LinearKernel, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return K.dot(inputs[0], inputs[1])
