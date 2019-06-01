"""
Utilities
"""
from librosa.feature import melspectrogram
from librosa.core import power_to_db
import scipy.io.wavfile as wav
import numpy as np


def read_vox_txt(data_fn):
    """Read VoxCelebv1 data format"""
    with open(data_fn, 'r') as data_file:
        txt = data_file.read().split('\n')

    props = {}
    frames = []

    for i in range(5):
        key, val = txt[i].split(':')
        props[key.strip()] = val.strip()

    for i in range(7, len(txt) - 1):
        frame = txt[i].split('\t')[0]
        frames.append(int(frame))

    return props, frames


def read_wav(fn, mono=True):
    """Read audio data from wav file"""
    sr, sig = wav.read(fn)

    # Enforce sample rate
    assert sr == 16000

    if mono and len(sig.shape) == 2:
        sig = sig[:, 0]
    return sig.astype(np.float32)


def melspec(data, sr=16000):
    """Compute melspectrogram from raw audio signal"""
    raw_mel = melspectrogram(y=data, sr=sr,
                             n_mels=416, fmax=4000,
                             hop_length=128)
    feat = power_to_db(raw_mel, ref=np.max)
    feat = 1 - (feat / -80.0)
    feat = feat[:-16, :]

    return feat
