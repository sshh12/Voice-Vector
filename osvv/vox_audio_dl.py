"""
Download VoxCeleb1 youtube videos as wav files.

set VOXCELEB1_PATH=/path/to/dataset
"""
import youtube_dl
import shutil
import click
import tqdm
import glob
import os


YDL_OPTS = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192'
    }],
    'postprocessor_args': [
        '-ar', None
    ],
    'prefer_ffmpeg': True,
    'keepvideo': True,
    'quiet': True
}


def _delete_file(fn):
    if os.path.exists(fn):
        os.remove(fn)


def _download(vid_id, output_fn, sr):
    opts = dict(YDL_OPTS)
    opts['outtmpl'] = output_fn + '.%(ext)s'
    opts['postprocessor_args'][1] = str(sr)
    with youtube_dl.YoutubeDL(opts) as ydl:
        info_dict = ydl.extract_info('http://www.youtube.com/watch?v={}'.format(vid_id), download=True)
        _delete_file(output_fn + '.webm')
        _delete_file(output_fn + '.m4a')
        return info_dict


@click.command()
@click.option('--dataset_path',
              default=os.environ.get('VOXCELEB1_PATH', '.'),
              help='Path to vox dataset.',
              type=click.Path())
@click.option('--replace',
              default=False,
              help='Replace already downloaded wavs.',
              type=bool)
@click.option('--remove_bad',
              default=True,
              help='Delete folders of bad video links.',
              type=bool)
@click.option('--sr',
              default=16000,
              help='Wave sample rate.',
              type=int)
def download_videos(dataset_path, replace=False, remove_bad=True, sr=16000):
    """Download videos in VoxCelebv1"""
    yt_folders = glob.glob(os.path.join(dataset_path, '*', '*'))

    for fn in tqdm.tqdm(yt_folders):

        fn_parts = fn.split('\\')
        vox_id = fn_parts[-2]
        vid_id = fn_parts[-1]

        new_file_path = os.path.join(dataset_path, vox_id, vid_id, vid_id)

        # already exists
        if not replace and os.path.exists(new_file_path + '.wav'):
            continue

        try:
            _download(vid_id, new_file_path, sr)
        except Exception as err:
            # delete folders of vids that connot be downloaded
            # b/c of region lock, copyright, etc
            if remove_bad:
                shutil.rmtree(os.path.join(dataset_path, vox_id, vid_id))


if __name__ == '__main__':
    download_videos()
