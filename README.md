# Tf-Keras Kervolution 2D

[![Travis](https://travis-ci.org/CyberZHG/tf-keras-kervolution-2d.svg)](https://travis-ci.org/CyberZHG/tf-keras-kervolution-2d)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/tf-keras-kervolution-2d/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/tf-keras-kervolution-2d)
[![996.ICU](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://996.icu) 

Unofficial implementation of [Kervolutional Neural Networks](https://arxiv.org/pdf/1904.03955.pdf).

## Install

```bash
python setup.py install
```

## Usage

### Basic

```python
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten, Dense
from tf_keras_kervolution_2d import KernelConv2D, PolynomialKernel


model = Sequential()
model.add(KernelConv2D(
    input_shape=(3, 5, 5),
    filters=4,
    kernel_size=3,
    kernel_function=PolynomialKernel(p=3, trainable_c=True),
))
model.add(Flatten())
model.add(Dense(units=2, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()
```

### Kernels

```python
from tf_keras_kervolution_2d import LinearKernel      # Equivalent to normal convolution
from tf_keras_kervolution_2d import L1Kernel          # Manhattan distance
from tf_keras_kervolution_2d import L2Kernel          # Euclidean distance
from tf_keras_kervolution_2d import PolynomialKernel  # Polynomial
from tf_keras_kervolution_2d import GaussianKernel    # Gaussin / RBF
```
