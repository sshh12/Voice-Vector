"""
Training

Generator adapted from:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
import numpy as np
import functools
import pickle
import random
import click
import glob
import os

from vox_features import FEATS_FN
from models import make_vox_model


class VoxCelebDataGenerator(Sequence):

    def __init__(self, dataset_path, vox_ids, batch_size=16, batches_per_epoch=32):
        self.path = dataset_path
        self.vox_ids = vox_ids
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.on_epoch_end()

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, index):
        XA, XB, y = self._bake_batch()
        return [XA, XB], y

    def on_epoch_end(self):
        pass

    def _load_random_feat(self, vox_id):
        feats_path = os.path.join(self.path, vox_id, FEATS_FN)
        feats = _load_features(feats_path)
        return random.choice(feats).reshape(300, 400, 1)

    def _bake_batch(self):

        XA = np.zeros((self.batch_size, 300, 400, 1))
        XB = np.zeros((self.batch_size, 300, 400, 1))
        y = np.zeros((self.batch_size,), dtype=int)

        y[self.batch_size//2:] = 1

        ids = np.random.choice(self.vox_ids, size=(self.batch_size,), replace=False)

        for i in range(self.batch_size):

            a_id = ids[i]
            a_feat = self._load_random_feat(a_id)

            if i >= self.batch_size // 2:
                b_id = a_id
            else:
                b_id = _pick_random_other(a_id, self.vox_ids)

            b_feat = self._load_random_feat(b_id)

            XA[i] = a_feat
            XB[i] = b_feat

        return XA, XB, y


@functools.lru_cache(128)
def _load_features(feats_fn):
    with open(feats_fn, 'rb') as feats_file:
        return pickle.load(feats_file)


def _pick_random_other(value, all_values):
    num_vals = len(all_values)
    other_idx = all_values.index(value) + np.random.randint(1, num_vals)
    return all_values[other_idx % num_vals]

def _get_train_dev_ids(dataset_path, shuffle=True, val_split=0.8):

    vox_ids = []
    for fn in glob.glob(os.path.join(dataset_path, '*', FEATS_FN)):
        fn_parts = fn.split('\\')
        vox_ids.append(fn_parts[-2])

    if shuffle:
        random.shuffle(vox_ids)

    split_idx = int(len(vox_ids) * val_split)
    return vox_ids[split_idx:], vox_ids[:split_idx]


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
              default=16,
              help='Batch size.',
              type=int)
@click.option('--weights_path',
              default='weights',
              help='Path to save weights.',
              type=click.Path())
def train(dataset_path, epochs=20, batch_size=16, weights_path='weights'):

    print('Finding Ids...', end='')
    dev_ids, train_ids = _get_train_dev_ids(dataset_path)
    print('found {}+{}'.format(len(train_ids), len(dev_ids)))

    train_gen = VoxCelebDataGenerator(dataset_path, train_ids, batch_size)
    val_gen = VoxCelebDataGenerator(dataset_path, dev_ids, batch_size)

    model = make_vox_model()

    save_checkpoint = ModelCheckpoint(os.path.join(weights_path, 'siamese-{epoch:02d}-{val_loss:.3f}.h5'),
                                      monitor='val_loss', save_best_only=True)

    model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        epochs=epochs,
                        callbacks=[save_checkpoint])

if __name__ == '__main__':
    train()
