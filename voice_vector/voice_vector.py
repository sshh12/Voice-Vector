"""
A class to generate voice embeddings.
"""
from pkg_resources import resource_filename
from keras.models import load_model
import numpy as np

import librosa
do_melspec = librosa.feature.melspectrogram
pwr_to_db = librosa.core.power_to_db


VV_MODEL_FN = resource_filename(__name__, 'data/vv_best.h5')
SHAPE = (251, 400, 1)
RATE = 16000
WINDOW_SIZE = 2
SPEC_SIZE = SHAPE[0]

# Precomputed stats for normalization
SPEC_MEAN = 0.301
SPEC_STD = 0.181


class VoiceEmbeddings:

    def __init__(self, model_fn=None, rate=16000):
        """
        Create a voice vector generator.
        """
        # For now only this is supported.
        assert rate == RATE
        self.rate = rate

        if model_fn is None:
            model_fn = VV_MODEL_FN

        self.model = load_model(model_fn)

    def _compute_spectrogram(self, frame):
        melspec = do_melspec(y=frame.astype(np.float32), sr=self.rate, n_mels=416, fmax=4000, hop_length=128)
        norm_melspec = pwr_to_db(melspec, ref=np.max)
        return np.transpose((1 - (norm_melspec / -80.0))[:-16, :])

    def get_mean_vec(self, large_frame):
        """
        Compute the mean voice vector for audio frame.

        `large_frame` is a numpy array of >=2 seconds of single channel
        audio data at a 16k sample rate. See `demo.py` for a basic example.
        """
        if len(large_frame) < RATE * WINDOW_SIZE:
            return None

        large_spec = self._compute_spectrogram(large_frame)
        size = len(large_spec)
        specs = []

        for k in range(0, size, SPEC_SIZE // 2):

            if k + SPEC_SIZE > size:
                k = size - SPEC_SIZE

            spec_part = large_spec[k:k+SPEC_SIZE]
            spec_part = spec_part.reshape(*SHAPE)

            specs.append(spec_part)

        specs = (np.array(specs) - SPEC_MEAN) / SPEC_STD

        preds = self.model.predict(specs)

        return np.mean(preds, axis=0)

    def get_vecs(self, frames):
        """
        Convert 2 second frames of voice data to voice vectors.

        `frames` must be an array of numpy arrays containing 2 seconds of single channel
        audio data at a 16k sample rate.
        """
        specs = []

        for frame in frames:
            assert len(frame) == RATE * WINDOW_SIZE
            spectrogram = self._compute_spectrogram(frame).reshape(*SHAPE)
            specs.append((spectrogram - SPEC_MEAN) / SPEC_STD)

        preds = self.model.predict(np.array(specs))

        return preds

    def get_vec(self, frame):
        """Like get_vecs(...) for for a single frame."""
        assert len(frame) == RATE * WINDOW_SIZE
        return self.get_vecs([frame])[0]
