import numpy as np
import librosa
from scipy.io import wavfile
import soundfile as sf

def read_fromBaseline(wav_path, length=100000, random_start=False):
    try:
        x, sr = librosa.load(wav_path, sr=None)
        assert sr == 16000
        if length > len(x):
            x = np.concatenate([x] * int(np.ceil(length/len(x))))
        feature = x[:length]
        return feature / np.max(np.abs(feature))
    except Exception as e:
        print("Error with getting feature from %s: %s" % (wav_path, str(e)))
        return None


def read_scipy(wav_path, length=100000):
    try:
        sr, x = wavfile.read(wav_path)
        assert sr == 16000
        if length > len(x):
            x = np.concatenate([x] * int(np.ceil(length/len(x))))
        feature = x[:length]
        return feature / np.max(np.abs(feature))
    except Exception as e:
        print("Error with getting feature from %s: %s" % (wav_path, str(e)))
        return None


def read_test(wav_path):
    try:
        sr, x = wavfile.read(wav_path)
        assert sr == 16000
        return x / np.max(np.abs(x))
    except Exception as e:
        print("Error with getting feature from %s: %s" % (wav_path, str(e)))
        return None


def universe_reader(file_path, length=None):
    if '.flac' in file_path:
        try:
            x, sr = sf.read(file_path)
            assert sr == 16000
            if length is not None and length > len(x):
                x = np.concatenate([x] * int(np.ceil(length/len(x))))
            elif length is None:
                length = len(x)
            feature = x[:length]
            return feature / np.max(np.abs(feature))
        except Exception as e:
            print("Error with getting feature from %s: %s" % (file_path, str(e)))
            return None
    elif ".wav" in file_path:
        try:
            sr, x = wavfile.read(file_path)
            assert sr == 16000
            if length is not None and length > len(x):
                x = np.concatenate([x] * int(np.ceil(length/len(x))))
            elif length is None:
                length = len(x)
            feature = x[:length]
            return feature / np.max(np.abs(feature))
        except Exception as e:
            print("Error with getting feature from %s: %s" % (file_path, str(e)))
            return None
    else:
        raise NotImplementedError("Only wav and flac formats supported, file_path=" + file_path)