#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:01:42 2023

@author: dani
"""
from keras import regularizers
from keras.layers import Input, Conv1D, MaxPooling1D, LayerNormalization, Dense, Dropout, Conv1DTranspose, \
    BatchNormalization
import tensorflow as tf

def model_build_func(input_shape, kernel_conv=7, strides_conv=1, dilation_rate=2):
    inputs = Input(shape=input_shape, name='input_lay')
    x = Conv1D(64, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate, padding="same",
               activation="elu", kernel_regularizer=regularizers.l2(0.0001))(inputs)
    x = LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate, padding="same",
               activation="elu", kernel_regularizer=regularizers.l2(0.0001))(x)
    x = LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate, padding="same",
               activation="elu", kernel_regularizer=regularizers.l2(0.0001))(x)
    x = LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate, padding="same",
               activation="elu", kernel_regularizer=regularizers.l2(0.0001))(x)
    x = LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate, padding="same",
               activation="elu", kernel_regularizer=regularizers.l2(0.0001))(x)
    x = LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate, padding="same",
               activation="elu", kernel_regularizer=regularizers.l2(0.0001))(x)

    x = LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate, padding="same",
               activation="elu", kernel_regularizer=regularizers.l2(0.0001))(x)
    x = LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate, padding="same",
               activation="elu", kernel_regularizer=regularizers.l2(0.0001))(x)
    x = LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(512, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate, padding="same",
               activation="elu", kernel_regularizer=regularizers.l2(0.0001))(x)
    x = LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(512, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate, padding="same",
               activation="elu", kernel_regularizer=regularizers.l2(0.0001))(x)
    x = LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(512, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate, padding="same",
               activation="elu", kernel_regularizer=regularizers.l2(0.0001))(x)
    x = LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(512, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate, padding="same",
               activation="elu", kernel_regularizer=regularizers.l2(0.0001), name='last_conv')(x)
    x = LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=2)(x)

    classif = Conv1DTranspose(256, kernel_size=8, strides=2, padding="same", activation="elu",
                              kernel_regularizer=regularizers.l2(0.0001))(x)
    classif = Conv1DTranspose(256, kernel_size=8, strides=2, padding="same", activation="elu",
                              kernel_regularizer=regularizers.l2(0.0001))(classif)
    classif = Conv1DTranspose(256, kernel_size=8, strides=2, padding="same", activation="elu",
                              kernel_regularizer=regularizers.l2(0.0001))(classif)

    classif = Dense(256, activation="elu")(classif)
    classif = LayerNormalization(-2)(classif)
    classif = BatchNormalization()(classif)
    classif = Dropout(0.3)(classif)

    classif = Dense(128, activation="elu")(classif)
    classif = LayerNormalization(-2)(classif)

    classif = BatchNormalization()(classif)
    classif = Dropout(0.3)(classif)
    classif = Dense(6, activation="softmax")(classif)
    model = tf.keras.Model(inputs=inputs, outputs=classif)
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    return model
