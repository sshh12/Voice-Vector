# Voice Vector

> A one-shot siamese approach to generating voice embeddings.

## Usage

##### Install

`pip install https://github.com/sshh12/Voice-Vector/releases/download/v0.1.0/voice-vector-0.1.0.tar.gz`

##### API

```python
import voice_vector

voice_embs = voice_vector.VoiceEmbeddings()

# Frames are numpy arrays of 16k single channel audio data.
mean_emb = voice_embs.get_mean_vec(large_frame)
embs     = voice_embs.get_vecs(frames)
emb      = voice_embs.get_vec(frame)
```

##### Demo

See [demo.py](https://github.com/sshh12/Voice-Vector/blob/master/demo.py) for the complete script. Uses the [Common Voice Dataset](https://voice.mozilla.org/en).

```python
...
import voice_vector

 # A sample rate of 16k is required
RATE = 16000
# The current model uses 2 second frames of audio
WINDOW_SIZE = 2

audio_files, meta_data = common_voice_data()

voice_embs = voice_vector.VoiceEmbeddings()

embs = []
colors = []
labels = []

for i, audio_fn in enumerate(audio_files):

    meta = meta_data[i]

    # Read audio data as numpy array
    audio_data = wav.read(audio_fn)[1].astype(np.uint16)

    # Trim since speakers normally have a buffer before/after
    audio_data = audio_data[1000:-1000]

    speaker_embs = []

    for k in range(0, len(audio_data), RATE):

        # Generate embeddings for subsections of audio file
        audio_part = audio_data[k:k+RATE*WINDOW_SIZE]
        if len(audio_part) != RATE * WINDOW_SIZE:
            break

        # Generate an embedding for this 2s of audio data
        speaker_embs.append(voice_embs.get_vec(audio_part))

    if len(speaker_embs) == 0:
        continue

    # Use the mean embedding to represent the speaker
    embs.append(np.mean(speaker_embs, axis=0))
    labels.append(audio_fn)
    colors.append('r' if meta[1] == 'female' else 'b')

# Use TSNE to visualize embeddings in 2D
embs_2d = TSNE(n_components=2, n_iter=10000).fit_transform(np.array(embs))

plt.scatter(embs_2d[:, 0], embs_2d[:, 1], c=colors, marker='.')
plt.show()
```

![TSNE](https://user-images.githubusercontent.com/6625384/58922114-2fe79780-86ff-11e9-9b45-33ee1c7a342d.png)

## DIY

##### Training with VoxCeleb1

1. Download [VoxCelebv1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)
2. Set `VOXCELEB1_PATH` to where its saved (`ls` in this path should produce a list of ids)
3. `pip install -r model/train-requirements.txt`
4. Download audio files and convert them to wavs `python model\vox_audio_dl.py`
5. Compute and save spectrograms `python model\vox_features.py --processes 4`
6. Train `python model\vox_train.py --epochs 1000 --batch_size 32 --run_name run1`
7. Convert the training model into an embeddings model `python model\models.py --model_path weights\siamese-xx-xxxx.h5 --save_model my_vv_model.h5`

## Related

#### References
* [Siamese Neural Networks for One-Shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
* [github.com/akshaysharma096/Siamese-Networks/](https://github.com/akshaysharma096/Siamese-Networks/)
* [stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
* [VoxCeleb: a large-scale speaker identification dataset](https://arxiv.org/abs/1706.08612v1)

#### Code
* [andabi/voice-vector](https://github.com/andabi/voice-vector)
* [prajual/Master-Voice_Prints](https://github.com/prajual/Master-Voice_Prints)
