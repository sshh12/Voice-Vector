"""
Voice Verification Models

Many snippets borrowed from:
https://github.com/akshaysharma096/Siamese-Networks/
"""
from keras.utils.generic_utils import get_custom_objects
from keras.models import Sequential, Model, load_model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers import Input, Lambda, Dense, MaxPooling2D, Conv2D, Flatten
import keras.backend as K
import numpy as np
import click


INPUT_SHAPE = (400, 252, 1)


def _init_weights(shape, name=None):
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def _init_bias(shape, name=None):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


get_custom_objects().update({
    '_init_weights': lambda: _init_weights,
    '_init_bias': lambda: _init_bias
})


def create_feature_model(input_shape):

    model = Sequential()
    model.add(Conv2D(64, (11, 11), activation='relu', input_shape=input_shape,
                   kernel_initializer=_init_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu',
                     kernel_initializer=_init_weights,
                     bias_initializer=_init_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer=_init_weights,
                     bias_initializer=_init_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer=_init_weights,
                     bias_initializer=_init_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=_init_weights, bias_initializer=_init_weights))

    return model


def make_siamese(input_shape, feat_model, compile_model=True):

    a_input = Input(input_shape)
    b_input = Input(input_shape)

    a_encoded = feat_model(a_input)
    b_encoded = feat_model(b_input)

    L1Layer = Lambda(lambda feats: K.abs(feats[0] - feats[1]))
    dist_tensor = L1Layer([a_encoded, b_encoded])

    pred = Dense(1, activation='sigmoid', bias_initializer=_init_bias)(dist_tensor)

    siamese_model = Model(inputs=[a_input, b_input], outputs=pred)

    if compile_model:
        optimizer = Adam(lr=0.0001)
        siamese_model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return siamese_model


def make_vox_model():
    feat_model = create_feature_model(INPUT_SHAPE)
    model = make_siamese(INPUT_SHAPE, feat_model)
    return model


@click.command()
@click.option('--model_path',
              required=True,
              help='Keras model (from training).',
              type=click.Path())
@click.option('--save_model',
              default='vvmodel.h5',
              help='Where to save new model.',
              type=click.Path())
def siamese_to_feature_model(model_path, save_model):

    siamese_model = load_model(model_path)

    input_layer = siamese_model.layers[2].get_input_at(0)
    output_layer = siamese_model.layers[2].get_output_at(0)

    feat_model = Model(inputs=input_layer, outputs=output_layer)
    feat_model.save(save_model)


if __name__ == '__main__':
    siamese_to_feature_model()
