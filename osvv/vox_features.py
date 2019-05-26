"""
Compute audio features from wav files.

set VOXCELEB1_PATH=/path/to/dataset
"""
from multiprocessing import Pool
import numpy as np
import random
import pickle
import click
import glob
import tqdm
import os

from utils import melspec, read_wav, read_vox_txt


FPS = 25                                         # VoxCelebv1 dataset FPS
RATE = 16000                                     # Assume 16k sample rate
SECS_PER_EXAMPLE = 2                             # Training example size in seconds
FRAMES_PER_EXAMPLE = SECS_PER_EXAMPLE * FPS      # Number of video frames per example
SPEC_PER_SEC = 126                               # Number of spectrogram vectors per second
FRAME_TO_SPEC = (1 / FPS) * RATE / SPEC_PER_SEC  # Convert frame number to spectrogram index


def _compute(wav_fn, save=True, window_size=300):

    vid_dir = os.path.dirname(wav_fn)
    vox_id, vid_id = vid_dir.split(os.sep)[-2:]
    audio_data = read_wav(wav_fn)
    spec_data = melspec(audio_data)

    feats = []

    for txt_fn in glob.iglob(os.path.join(vid_dir, '*.txt')):

        props, frames = read_vox_txt(txt_fn)
        start_frame, end_frame = frames[0], frames[-1]

        for i in range(start_frame, end_frame, FRAMES_PER_EXAMPLE):
            start_spec = int(i * FRAME_TO_SPEC)
            end_spec = start_spec + SPEC_PER_SEC * SECS_PER_EXAMPLE
            spec_segment = spec_data[:, start_spec:end_spec]
            if spec_segment.shape != (400, SPEC_PER_SEC * SECS_PER_EXAMPLE):
                break
            feats.append(spec_segment)

    if save and len(feats) > 0:
        save_fn = os.path.join(vid_dir, '{}.{}.pkl'.format(vid_id, vox_id))
        if os.path.exists(save_fn):
            print(save_fn, 'already exists...replacing it.')
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


if __name__ == '__main__':
    compute_features()
