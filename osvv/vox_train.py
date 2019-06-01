"""
Training

Generator adapted from:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.utils import Sequence
from keras.models import load_model
from collections import defaultdict
import numpy as np
import functools
import random
import click
import glob
import os

from models import get_siamese_model


INPUT_SHAPE = (252, 400, 1)
SPEC_MEAN = 0.305
SPEC_STD = 0.177
FEATURE_CACHE_SIZE = 256


class VoxCelebDataGenerator(Sequence):

    def __init__(self, dataset_path, vox_ids, vox_paths, batch_size=32):
        self.path = dataset_path
        self.vox_ids = vox_ids
        self.vox_paths = vox_paths
        self.batch_size = batch_size
        self.batches_per_epoch = max(len(vox_ids) // batch_size, 1)
        self.on_epoch_end()

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, index):
        XA, XB, y = self._bake_batch()
        return [XA, XB], y

    def on_epoch_end(self):
        pass

    def _load_random_feat(self, vox_id):
        feats_path = random.choice(self.vox_paths[vox_id])
        feats = _load_features(feats_path)
        if len(feats) == 0:
            return self._load_random_feat(vox_id)
        else:
            feat = random.choice(feats)
            return feat.reshape(*INPUT_SHAPE)

    def _bake_batch(self):

        XA = np.zeros((self.batch_size, *INPUT_SHAPE))
        XB = np.zeros((self.batch_size, *INPUT_SHAPE))

        # Binary array
        y = np.random.randint(0, 2, (self.batch_size,))

        ids = np.random.choice(self.vox_ids, size=(self.batch_size,), replace=False)

        for i in range(self.batch_size):

            a_id = ids[i]
            a_feat = self._load_random_feat(a_id)

            if y[i] == 1:
                b_id = a_id
            else:
                b_id = _pick_random_other(a_id, self.vox_ids)

            b_feat = self._load_random_feat(b_id)

            XA[i] = a_feat
            XB[i] = b_feat

        return XA, XB, y


@functools.lru_cache(FEATURE_CACHE_SIZE)
def _load_features(feats_fn):
    return (np.load(feats_fn) - SPEC_MEAN) / SPEC_STD


def _pick_random_other(value, all_values):
    num_vals = len(all_values)
    other_idx = all_values.index(value) + np.random.randint(1, num_vals)
    return all_values[other_idx % num_vals]


def _get_train_dev_ids(dataset_path, shuffle=True, val_split=0.8):

    vox_data = defaultdict(list)
    for fn in glob.iglob(os.path.join(dataset_path, '_data', '*.npy')):
        vox_id = fn.split('.')[-2]
        vox_data[vox_id].append(fn)
    vox_ids = list(vox_data)

    if shuffle:
        random.shuffle(vox_ids)

    split_idx = int(len(vox_ids) * val_split)
    return vox_ids[:split_idx], vox_ids[split_idx:], vox_data


def _get_model(weights_folder):

    old_weights = {int(fn.split('-')[1]): fn for fn in glob.glob(os.path.join(weights_folder, '*.h5'))}
    if len(old_weights) != 0:
        latest_fn = old_weights[max(old_weights)]
        if 'y' in input('Use prev weights from {}? (y/n): '.format(latest_fn)).lower():
            return load_model(latest_fn)

    return get_siamese_model()


@click.command()
@click.option('--dataset_path',
              default=os.environ.get('VOXCELEB1_PATH', '.'),
              help='Path to vox dataset.',
              type=click.Path())
@click.option('--epochs',
              default=20,
              help='Epochs.',
              type=int)
@click.option('--batch_size',
              default=32,
              help='Batch size.',
              type=int)
@click.option('--weights_folder',
              default='weights',
              help='Path to save weights.',
              type=click.Path())
def train(dataset_path, epochs=20, batch_size=32, weights_folder='weights'):

    print('Finding Ids...', end='')
    train_ids, dev_ids, vox_paths = _get_train_dev_ids(dataset_path)
    print('found {}+{}'.format(len(train_ids), len(dev_ids)))

    train_gen = VoxCelebDataGenerator(dataset_path, train_ids, vox_paths, batch_size)
    val_gen = VoxCelebDataGenerator(dataset_path, dev_ids, vox_paths, batch_size)

    model = _get_model(weights_folder)

    save_checkpoint = ModelCheckpoint(os.path.join(weights_folder, 'siamese-{epoch:02d}-{val_loss:.3f}.h5'),
                                      monitor='val_loss', save_best_only=True)
    lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    tf_board = TensorBoard(log_dir='./logs')

    model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        epochs=epochs,
                        workers=4,
                        callbacks=[save_checkpoint, lr_plateau, tf_board])

if __name__ == '__main__':
    train()
