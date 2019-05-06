#!/usr/bin/env bash
pycodestyle --max-line-length=120 tf_keras_kervolution_2d tests && \
    nosetests --nocapture --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --cover-package=tf_keras_kervolution_2d tests
