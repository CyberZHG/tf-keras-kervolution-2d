import os
import tempfile
from unittest import TestCase
import numpy as np
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.models import Sequential, load_model
from tf_keras_kervolution_2d import KernelConv2D, GaussianKernel


class TestGaussianKernel(TestCase):

    def test_fit(self):
        x = np.random.standard_normal((1024, 3, 5, 5))
        y = (x[:, 1, 2, 3] > 0).astype('int32')

        model = Sequential()
        model.add(Dense(
            input_shape=(3, 5, 5),
            units=4,
            activation='tanh',
        ))
        model.add(KernelConv2D(
            filters=4,
            kernel_size=3,
            kernel_function=GaussianKernel(gamma=0.5),
        ))
        model.add(Flatten())
        model.add(Dense(units=2, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary()

        model.fit(x, y, epochs=30)

        model_path = os.path.join(tempfile.gettempdir(), 'test_knn_%f.h5' % np.random.random())
        model.save(model_path)
        model = load_model(model_path, custom_objects={
            'KernelConv2D': KernelConv2D,
            'GaussianKernel': GaussianKernel,
        })

        predicted = model.predict(x).argmax(axis=-1)
        self.assertLess(np.sum(np.abs(y - predicted)), 300)
