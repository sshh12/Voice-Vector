"""
Voice Verification Models

Many snippets borrowed from:
https://github.com/akshaysharma096/Siamese-Networks/
"""
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers import Input, Lambda, Dense, MaxPooling2D, Conv2D, Flatten
import keras.backend as K
import numpy as np


def _init_weights(shape, name=None):
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def _init_bias(shape, name=None):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def create_feature_model(input_shape):

    model = Sequential()
    # TODO optimize for mfcc data / "images"
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
                   kernel_initializer=_init_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu',
                     kernel_initializer=_init_weights,
                     bias_initializer=_init_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=_init_weights,
                     bias_initializer=_init_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=_init_weights,
                     bias_initializer=_init_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=_init_weights, bias_initializer=_init_weights))

    return model


def make_siamese(input_shape, feat_model, compile=True):

    a_input = Input(input_shape)
    b_input = Input(input_shape)

    a_encoded = feat_model(a_input)
    b_encoded = feat_model(b_input)

    L1Layer = Lambda(lambda feats: K.abs(feats[0] - feats[1]))
    dist_tensor = L1Layer([a_encoded, b_encoded])

    pred = Dense(1, activation='sigmoid', bias_initializer=_init_bias)(dist_tensor)

    siamese_model = Model(inputs=[a_input, b_input], outputs=pred)

    if compile:
        optimizer = Adam(lr=0.00006)
        siamese_model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return siamese_model


def make_vox_model():
    shape = (300, 400, 1)
    feat_model = create_feature_model(shape)
    model = make_siamese(shape, feat_model)
    return model
