# One Shot Voice Verification

### Training
1. Download (VoxCelebv1)[http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html]
2. Download YouTube videos (as wav) `python osvv/vox_audio_dl.py`
3. Download compute features `python osvv/vox_features.py`
4. Train `python osvv/vox_train.py --epoch xxx`
5. Convert model to a feature predictor `python osvv/models.py --model_path weights/siamese-xxx.h5`

### References
* [Siamese Neural Networks for One-Shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
* [github.com/akshaysharma096/Siamese-Networks/](https://github.com/akshaysharma096/Siamese-Networks/)
* [stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
