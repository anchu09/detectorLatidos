#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:27:37 2022

@author: dani
"""

import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
# from keras.integration_test.models.translation import TransformerDecoder
from scipy.signal import find_peaks
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import random
from tensorflow import keras
from keras.utils import Sequence
import tensorflow as tf
from datetime import datetime
import shutil
import keras_nlp
from keras_nlp.layers import SinePositionEncoding, TransformerDecoder,TransformerEncoder

from keras.layers import Input, Conv1D, MaxPooling1D, LayerNormalization, Flatten, Dense, Dropout, MultiHeadAttention, \
    GlobalMaxPooling1D, Reshape, UpSampling1D, Conv1DTranspose

#para un latido normal en vez de darme un pico me da tres
def transformer1(train_data_generator, test_data_generator, epochs):

    # Transformer
    input_shape = (5000, 2)
    # Capas convolucionales de extracción de características
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu")(inputs)
    x = Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = Conv1D(512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = Conv1D(512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    # Capa del transformer

    sine_encoder = SinePositionEncoding(max_wavelength=5000)
    sine_encoding = sine_encoder(x)
    encoded=x+sine_encoding
    encoder_output=TransformerEncoder(num_heads=2, intermediate_dim=8, dropout=0.3)(encoded)

    #classifier layers
    # classif=UpSampling1D(8)(decoder_output)
    classif = Conv1DTranspose(256, kernel_size=3, strides=8, padding="same", activation="relu")(encoder_output)
    classif = Dense(128, activation="relu")(classif)
    classif = Dropout(0.3)(classif)
    classif = Dense(6, activation="softmax")(classif)
    model = tf.keras.Model(inputs=inputs, outputs=classif)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.fit_generator(train_data_generator, epochs=epochs)
    test_predictions = model.predict(test_data_generator)

    return test_predictions

#va bien para losnonrmales
input_shape = (5000, 2)
# Capas convolucionales de extracción de características
inputs = Input(shape=input_shape)
x = Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu")(inputs)
x = Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
# x = Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
# x = Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = Conv1D(512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = Conv1D(512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
# x = Conv1D(512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = MaxPooling1D(pool_size=2)(x)
# Capa del transformer

sine_encoder = SinePositionEncoding(max_wavelength=x.shape[1])
sine_encoding = sine_encoder(x)
encoded=x+sine_encoding
encoder_output=TransformerEncoder(num_heads=2, intermediate_dim=4, dropout=0.3)(encoded)

classif = Conv1DTranspose(256, kernel_size=8, strides=2, padding="same", activation="relu")(encoder_output)
classif = Conv1DTranspose(256, kernel_size=8, strides=2, padding="same", activation="relu")(classif)
classif = Conv1DTranspose(256, kernel_size=8, strides=2, padding="same", activation="relu")(classif)
# classif = Dense(256, activation="relu")(classif)
# classif = Dropout(0.3)(classif)
classif = Dense(128, activation="relu")(classif)
classif = Dropout(0.3)(classif)
classif = Dense(6, activation="softmax")(classif)
model = tf.keras.Model(inputs=inputs, outputs=classif)

model.compile(optimizer="adam", loss="categorical_crossentropy")

# model.fit_generator(train_data_generator, epochs=20)
#overfitear con todo eltest
model.fit_generator(test_data_generator, epochs=20)

test_predictions = model.predict(test_data_generator)

# # # aquí mooving average NO AYUDA ASI QUE DE MOMENTO NO LA USO
# window=15
# test_predictions = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window), mode='same') / window, axis=0, arr=test_predictions)
#