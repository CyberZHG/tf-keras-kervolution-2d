import os
import tempfile
from unittest import TestCase
import numpy as np
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
from tensorflow.python.keras.models import Sequential, load_model
from tf_keras_kervolution_2d import KernelConv2D, LinearKernel


class TestLinearKernel(TestCase):

    def test_same(self):
        kernel = np.random.standard_normal((3, 3, 5, 11))
        x = np.random.standard_normal((2, 7, 13, 5))

        model = Sequential()
        model.add(KernelConv2D(
            input_shape=(7, 13, 5),
            filters=11,
            kernel_size=3,
            kernel_function=LinearKernel(),
            padding='same',
            use_bias=False,
            weights=[kernel],
        ))
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        model_path = os.path.join(tempfile.gettempdir(), 'test_knn_%f.h5' % np.random.random())
        model.save(model_path)
        model = load_model(model_path, custom_objects={
            'KernelConv2D': KernelConv2D,
            'LinearKernel': LinearKernel,
        })

        kernel_output = model.predict(x)

        model = Sequential()
        model.add(Conv2D(
            input_shape=(7, 13, 5),
            filters=11,
            kernel_size=3,
            padding='same',
            use_bias=False,
            weights=[kernel],
        ))
        model.compile(optimizer='adam', loss='mse')
        normal_output = model.predict(x)

        self.assertTrue(np.allclose(normal_output, kernel_output))

    def test_fit(self):
        x = np.random.standard_normal((1024, 3, 5, 5))
        y = (x[:, 1, 2, 3] > 0).astype('int32')

        model = Sequential()
        model.add(KernelConv2D(
            input_shape=(3, 5, 5),
            filters=4,
            kernel_size=3,
            kernel_function=LinearKernel(),
            data_format='channels_first'
        ))
        model.add(KernelConv2D(
            filters=3,
            kernel_size=3,
            kernel_function=LinearKernel(),
            padding='same',
            data_format='channels_last'
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
            'LinearKernel': LinearKernel,
        })

        predicted = model.predict(x).argmax(axis=-1)
        self.assertLess(np.sum(np.abs(y - predicted)), 200)

    def test_invalid_channel_dim(self):
        with self.assertRaises(ValueError):
            model = Sequential()
            model.add(KernelConv2D(
                input_shape=(None, 5, 5),
                filters=4,
                kernel_size=3,
                kernel_function=LinearKernel(),
                data_format='channels_first'
            ))

    def test_output_shape(self):
        conv = KernelConv2D(
            filters=11,
            kernel_size=3,
            kernel_function=LinearKernel(),
            padding='same',
        )
        self.assertEqual((None, 7, 13, 11), conv.compute_output_shape((None, 7, 13, 5)))

        conv = KernelConv2D(
            filters=11,
            kernel_size=3,
            kernel_function=LinearKernel(),
            padding='valid',
            data_format='channels_first'
        )
        self.assertEqual((None, 11, 11, 3), conv.compute_output_shape((None, 7, 13, 5)))
