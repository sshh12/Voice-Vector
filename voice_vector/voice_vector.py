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
SHAPE = (252, 400, 1)

# Precomputed stats for normalization
SPEC_MEAN = 0.305
SPEC_STD = 0.177


class VoiceEmbeddings:

    def __init__(self, model_fn=None, rate=16000):
        """
        Create a voice vector generator.
        """
        # For now only this is supported.
        assert rate == 16000
        self.rate = rate

        if model_fn is None:
            model_fn = VV_MODEL_FN

        self.model = load_model(model_fn)

    def _patch_shape(self, spec):
        """Fix for bug where shape is one off"""
        if len(spec) == SHAPE[0] - 1:
            fixed_spec = np.append(spec, [spec[-1, :]], axis=0)
        else:
            fixed_spec = spec
        return fixed_spec.reshape(*SHAPE)

    def get_vecs(self, frames):
        """
        Convert 2 second frames of voice data to voice vectors.

        `frames` must be an array of numpy arrays containing 2 seconds of single channel
        audio data at a 16k sample rate. See `demo.py` for a basic example.
        """
        specs = []

        for frame in frames:
            melspec = do_melspec(y=frame.astype(np.float32), sr=self.rate, n_mels=416, fmax=4000, hop_length=128)
            norm_melspec = pwr_to_db(melspec, ref=np.max)
            spectrogram = np.transpose((1 - (norm_melspec / -80.0))[:-16, :])
            spectrogram = self._patch_shape(spectrogram)
            specs.append((spectrogram - SPEC_MEAN) / SPEC_STD)

        preds = self.model.predict(np.array(specs))

        return preds

    def get_vec(self, frame):
        """Like get_vecs(...) for for a single frame."""
        return self.get_vecs([frame])[0]
