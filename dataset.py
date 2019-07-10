from torch.utils.data import Dataset
import glob
import os
import random
from librosa.feature import mfcc
import pdb

class VoiceAntiSpoofDataset(Dataset):
    def __init__(self, dataset_dir, mode, reading_fn, train_val_index=None,
                 seed=1, transform=[], add_TData=None):
        """

        :param dataset_dir: path to training data
        :param mode: one of ('train', val')
        :param reading_fn: function to read .waf files, returns np.arrays
        :param train_val_index: dict, TODO after
        """
        assert mode in set(['val', 'train', 'all']), "mode must" \
                                                     " be one of (train, val, all)"

        self.transform = transform
        self.reading_fn = reading_fn
        if train_val_index is None:
            wav_paths = sorted(glob.glob(os.path.join(dataset_dir, '**/*.wav'), recursive=True))
            random.shuffle(wav_paths)
            train_paths = wav_paths[:40000]
            val_paths = wav_paths[40000:]

            train_human = sorted(filter(lambda path: "human" in path, train_paths))
            train_spoof = sorted(filter(lambda path: "spoof" in path, train_paths))

            val_human = sorted(filter(lambda path: "human" in path, val_paths))
            val_spoof = sorted(filter(lambda path: "spoof" in path, val_paths))
            if mode == 'train':
                self.labels = [0] * len(train_human) + [1] * len(train_spoof)
                self.data = train_human + train_spoof

                weight_h = len(self.data) / len(train_human)
                weight_s = len(self.data) / len(train_spoof)
                self.weights = [weight_h] * len(train_human) + [weight_s] * len(train_spoof)
                print("len train", len(self.data))
            elif mode == 'val':
                self.labels = [0] * len(val_human) + [1] * len(val_spoof)
                self.data = val_human + val_spoof
                print("len val", len(val_human), len(val_spoof))
                self.weights = None
            else:
                human = sorted(filter(lambda path: "human" in path, wav_paths))
                print(len(human), "len human")
                spoof = sorted(filter(lambda path: "spoof" in path, wav_paths))
                print(len(spoof), "len spoof")
                self.labels = ([0] * len(human) + [1] * len(spoof))
                self.data = human + spoof
                weight_h = len(self.data) / len(human)
                weight_s = len(self.data) / len(spoof)
                self.weights = [weight_h] * len(human) + [weight_s] * len(spoof)
            if add_TData is not None:
                wav_paths = sorted(glob.glob(os.path.join(dataset_dir, '**/*.wav'), recursive=True))
                human = sorted(filter(lambda path: "human" in path, wav_paths))
                print(len(human), "len additional human")
                spoof = sorted(filter(lambda path: "spoof" in path, wav_paths))
                print(len(spoof), "len spoof")
                self.data += human + spoof
                self.labels += ([0] * len(human) + [1] * len(spoof))
                self.weights += [self.weights[0]] * len(human) + [self.weights[-1]] * len(spoof)


        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        data = self.reading_fn(self.data[idx])
        reverse = data[::-1]
        mfcc_data = mfcc(data, sr=16000, n_mfcc=128)
        mfcc_data_reverse = mfcc(reverse, sr=16000, n_mfcc=128)
        for t in self.transform:
            data = t(data)
            mfcc_data = t(mfcc_data)
            reverse = t(reverse)
            mfcc_data_reverse = t(mfcc_data_reverse)
        label = self.labels[idx]
        return {'data': data, 'mfcc': mfcc_data, 'label': label,
                'reverse': reverse, "mfcc_reverse": mfcc_data_reverse}

    def __len__(self):
        return len(self.data)
