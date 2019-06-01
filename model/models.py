"""
Voice Verification Models

Many snippets borrowed from:
https://github.com/akshaysharma096/Siamese-Networks/
"""
from keras.models import Sequential, Model, load_model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers import Input, Lambda, Dense, MaxPooling2D, Conv2D, Flatten
import keras.backend as K
import numpy as np
import click


INPUT_SHAPE = (252, 400, 1)


def _add_conv_block(model, filters, size, **kwargs):
    model.add(Conv2D(filters, (size, size),
                     activation='relu',
                     kernel_regularizer=l2(2e-4),
                     **kwargs))
    model.add(MaxPooling2D())


def create_feature_model(input_shape):

    model = Sequential()
    _add_conv_block(model, 64, 11, input_shape=input_shape)
    _add_conv_block(model, 128, 7)
    _add_conv_block(model, 256, 5)
    _add_conv_block(model, 256, 3)
    model.add(Flatten())
    model.add(Dense(1024, activation='sigmoid'))

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
        siamese_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

    return siamese_model


def get_siamese_model():
    feat_model = create_feature_model(INPUT_SHAPE)
    model = make_siamese(INPUT_SHAPE, feat_model)
    return model


@click.command()
@click.option('--model_path',
              required=True,
              help='Keras model (from training).',
              type=click.Path())
@click.option('--save_model',
              default='vv_model.h5',
              help='Where to save new model.',
              type=click.Path())
def siamese_to_feature_model(model_path, save_model):
    """Convert the siamese architecture to a feature detector"""
    siamese_model = load_model(model_path)

    input_layer = siamese_model.layers[2].get_input_at(0)
    output_layer = siamese_model.layers[2].get_output_at(0)

    feat_model = Model(inputs=input_layer, outputs=output_layer)
    feat_model.save(save_model)


if __name__ == '__main__':
    siamese_to_feature_model()
