"""
Training

Generator adapted from:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from collections import defaultdict
import numpy as np
import functools
import pickle
import random
import click
import glob
import os

from models import make_vox_model


INPUT_SHAPE = (400, 252, 1)


class VoxCelebDataGenerator(Sequence):

    def __init__(self, dataset_path, vox_ids, vox_paths, batch_size=16):
        self.path = dataset_path
        self.vox_ids = vox_ids
        self.vox_paths = vox_paths
        self.batch_size = batch_size
        self.batches_per_epoch = max(len(vox_ids) * 2 // batch_size, 1)
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
        y = np.zeros((self.batch_size,), dtype=int)

        y[self.batch_size//2:] = 1

        ids = np.random.choice(self.vox_ids, size=(self.batch_size,), replace=True)

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


@functools.lru_cache(256)
def _load_features(feats_fn):
    with open(feats_fn, 'rb') as feats_file:
        return pickle.load(feats_file)


def _pick_random_other(value, all_values):
    num_vals = len(all_values)
    other_idx = all_values.index(value) + np.random.randint(1, num_vals)
    return all_values[other_idx % num_vals]

def _get_train_dev_ids(dataset_path, shuffle=True, val_split=0.8):

    vox_data = defaultdict(list)
    for fn in glob.glob(os.path.join(dataset_path, '*', '*', '*.pkl')):
        # VoxCeleb\v1\txt\<vox_id>\<vid_id>\<vid_id>.<vox_id>.pkl
        vox_id = fn.split('.')[-2]
        vox_data[vox_id].append(fn)
    vox_ids = list(vox_data)

    if shuffle:
        random.shuffle(vox_ids)

    split_idx = int(len(vox_ids) * val_split)
    return vox_ids[:split_idx], vox_ids[split_idx:], vox_data


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

    model = make_vox_model()

    save_checkpoint = ModelCheckpoint(os.path.join(weights_folder, 'siamese-{epoch:02d}-{val_loss:.3f}.h5'),
                                      monitor='val_loss', save_best_only=True)

    model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        epochs=epochs,
                        callbacks=[save_checkpoint])

if __name__ == '__main__':
    train()
