from mpldatacursor import datacursor # pip install mpldatacursor
from sklearn.manifold import TSNE    # pip install scikit-learn
import matplotlib.pyplot as plt      # pip install matplotlib
import scipy.io.wavfile as wav       # pip install scipy
from scipy.io import wavfile
from tqdm import tqdm                # pip install tqdm
import numpy as np                   # pip install numpy
import voice_vector
import csv
import os


# Path to the Common Voice dataset
COMMONVOICE_PATH = os.environ.get('COMMONVOICE_PATH', 'commonvoice')
 # A sample rate of 16k is required
RATE = 16000
# The current model uses 2 second frames of audio
WINDOW_SIZE = 2

# Create a embeddings generator and load the keras model
voice_embs = voice_vector.VoiceEmbeddings()

def common_voice_data():

    audio_files, meta_data = [], []

    with open(os.path.join(COMMONVOICE_PATH, 'dev.tsv'), 'r', encoding='utf-8') as tsv_file:

        reader = csv.reader(tsv_file, delimiter='\t')

        for row in reader:

            path = row[1]
            age, gender, accent = row[5:8]
            if '' in [path, age, gender, accent] or path == 'path':
                continue

            # all the audio files were converted to a .wav
            # with a sample rate of 16k and 1 channel
            audio_files.append(os.path.join(COMMONVOICE_PATH, 'clips', path + '.wav'))
            meta_data.append((age, gender, accent))

    return audio_files, meta_data

print('Finding Common Voice files...', end='')
audio_files, meta_data = common_voice_data()
print('found {}'.format(len(audio_files)))

embs = []
idxs = []
labels = []

for i, audio_fn in tqdm(enumerate(audio_files), total=len(audio_files)):

    # Read audio data as numpy array
    audio_data = wav.read(audio_fn)[1].astype(np.uint16)

    # Trim since speakers normally have a buffer before/after
    audio_data = audio_data[1000:-1000]

    speaker_emb = voice_embs.get_median_vec(audio_data)

    if speaker_emb is None:
        continue

    # Use the median embedding to represent the speaker
    embs.append(speaker_emb)
    labels.append(audio_fn)
    idxs.append(i)

# Use TSNE to visualize embeddings in 2D
embs_2d = TSNE(n_components=2, n_iter=10000).fit_transform(np.array(embs))

# A func to play audio files from the visualization
def on_click(**args):
    idx = args['ind'][0]
    os.system('start ' + labels[idx])
    return str(labels[idx])

data_by_gender = {'female': [], 'male': [], 'other': []}
for i in range(len(embs_2d)):
    idx = idxs[i]
    data = data_by_gender[meta_data[idx][1]]
    data.append(embs_2d[i])

max_displayed = min(len(data_by_gender['female']), len(data_by_gender['male']))
for gender in data_by_gender:
    data = np.array(data_by_gender[gender])[:max_displayed]
    plt.scatter(data[:, 0], data[:, 1], marker='.', label=gender)

datacursor(formatter=on_click)
plt.title('Voice Embeddings by Gender')
plt.legend()
plt.show()
