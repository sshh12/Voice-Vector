# Voice Vector

> TODO

## Usage

#### Install

`TODO`

#### Script

`TODO`

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
