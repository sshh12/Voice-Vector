"""
Compute audio features from wav files.

set VOXCELEB1_PATH=/path/to/dataset
"""
from multiprocessing import Pool, Lock
import numpy as np
import random
import pickle
import click
import glob
import tqdm
import os

from utils import melspec, read_wav, read_vox_txt


FEATS_FN = 'data.feats.pkl'
WRITE_LOCK = Lock()


def _compute(wav_fn, save=True, window_size=300):

    vid_dir = os.path.dirname(wav_fn)
    audio_data = read_wav(wav_fn)
    spec_data = melspec(audio_data)

    feats = []

    for txt_fn in glob.iglob(os.path.join(vid_dir, '*.txt')):
        
        props, frames = read_vox_txt(txt_fn)
        start_frame, end_frame = frames[0], frames[-1]
        
        # TODO find where this `5` comes from
        # sorta computed from ~ len(spec_data) / (vid_seconds * fps)
        start_frame *= 5
        end_frame *= 5
        
        # add every other window of data
        for i in range(start_frame, end_frame - window_size * 2, window_size * 2):
            feats.append(spec_data[i:i+window_size])

    if save:
        save_fn = os.path.join(vid_dir, os.pardir, FEATS_FN)
        with WRITE_LOCK:
            if os.path.exists(save_fn):
                with open(save_fn, 'rb') as save_data:
                    feats.extend(pickle.load(save_data))
            with open(save_fn, 'wb') as save_data:
                pickle.dump(feats, save_data)
                
    return feats


def _trim_features(feat_fn, max_feats):

    with open(feat_fn, 'rb') as feat_data:
        feats = pickle.load(feat_data)

    random.shuffle(feats)
    feats = feats[:max_feats]

    with open(feat_fn, 'wb') as feat_data:
        pickle.dump(feats, feat_data)


@click.command()
@click.option('--dataset_path',
              default=os.environ.get('VOXCELEB1_PATH', '.'),
              help='Path to vox dataset.',
              type=click.Path())
@click.option('--processes',
              default=1,
              help='Number of processes to use (0 for max)',
              type=int)
@click.option('--max_feats',
              default=20,
              help='Limit number of features stored per id (0 for no limit)',
              type=int)
@click.option('--max_wavs',
              default=5000,
              help='Limit number of wave files that will have features computed',
              type=int)
def compute_features(dataset_path, processes=1, max_feats=20, max_wavs=5000):
    """Convert wav to audio features"""
    if processes <= 0:
        processes = None

    wav_files = glob.glob(os.path.join(dataset_path, '*', '*', '*.wav'))[:max_wavs]

    bar = tqdm.tqdm(total=len(wav_files), desc='Audio Files', unit='wav')
    with Pool(processes=processes) as pool:
        for i in pool.imap(_compute, wav_files):
            bar.update()

    # reduce feat file size by randomly removing features
    if max_feats > 0:

        feat_files = glob.glob(os.path.join(dataset_path, '*', FEATS_FN))

        bar = tqdm.tqdm(total=len(feat_files), desc='Feature Files', unit='feats.pkl')
        with Pool(processes=processes) as pool:
            for i in pool.starmap(_trim_features, zip(feat_files, [max_feats] * len(feat_files))):
                bar.update()

if __name__ == '__main__':
    compute_features()
