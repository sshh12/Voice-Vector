from keras.models import load_model
import matplotlib.pyplot as plt
from scipy.io import wavfile
from osvv import models, utils
from sklearn.manifold import TSNE
import numpy as np
import conv_vad
from mpldatacursor import datacursor
import csv
import os


COMMONVOICE_PATH = os.environ.get('COMMONVOICE_PATH', 'commonvoice')

model = load_model('vv_model.h5')

audio_files = []
meta_data = []
colors = []
labels = []

with open(os.path.join(COMMONVOICE_PATH, 'dev.tsv'), 'r', encoding='utf-8') as tsv_file:
    reader = csv.reader(tsv_file, delimiter='\t')
    for row in reader:
        path = row[1]
        age, gender, accent = row[5:8]
        if '' in [path, age, gender, accent] or path == 'path':
            continue
        audio_files.append(os.path.join(COMMONVOICE_PATH, 'clips', path + '.wav'))
        meta_data.append((age, gender, accent))

voice_vectors = []

print(len(audio_files))
print(len(meta_data))

for i, audio_fn in enumerate(audio_files):

    audio_data = utils.read_wav(audio_fn)
    spec_data = np.transpose(utils.melspec(audio_data))
    spec_data = spec_data[100:-100,:]

    specs = []
    for k in range(0, len(spec_data) - 252, 120):
        data = spec_data[k:k+252,:].reshape(252, 400, 1)
        specs.append(data)
    specs = np.array(specs)

    if 0 in specs.shape:
        continue

    out = model.predict(specs)

    print(specs.shape, '->', out.shape)

    voice_vectors.append(np.median(out, axis=0))
    labels.append(audio_fn)

    if meta_data[i][1] == 'female':
        colors.append('r')
    else:
        colors.append('b')

voice_vectors = np.array(voice_vectors)

print(voice_vectors.shape)

vv_trans = TSNE(n_components=2, n_iter=10000).fit_transform(voice_vectors)

def stuff(**args):
    idx = args['ind'][0]
    os.system('start ' + labels[idx])
    return str(labels[idx])

plt.scatter(vv_trans[:, 0], vv_trans[:, 1], c=colors, marker='.')

datacursor(formatter=stuff)
plt.show()
