from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K


__all__ = ['LinearKernel', 'L1Kernel', 'L2Kernel']


class LinearKernel(Layer):

    def __init__(self, **kwargs):
        super(LinearKernel, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return K.dot(inputs[0], inputs[1])


class L1Kernel(Layer):

    def __init__(self, **kwargs):
        super(L1Kernel, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x, kernel = K.expand_dims(inputs[0], axis=-1), inputs[1]
        return K.mean(K.abs(x - kernel), axis=-2)


class L2Kernel(Layer):

    def __init__(self, **kwargs):
        super(L2Kernel, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x, kernel = K.expand_dims(inputs[0], axis=-1), inputs[1]
        return K.mean(K.square(x - kernel), axis=-2)
