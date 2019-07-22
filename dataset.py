from torch.utils.data import Dataset
import glob
import os
import random
from librosa.feature import mfcc
import numpy as np
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
                wav_paths = sorted(glob.glob(os.path.join(add_TData, '**/*.wav'), recursive=True))
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
        mfcc_data = mfcc(data, sr=16000, n_mfcc=45)
        for t in self.transform:
            data = t(data)
            mfcc_data = t(mfcc_data)

        label = self.labels[idx]
        return {'data': data, 'mfcc': mfcc_data, 'label': label}

    def __len__(self):
        return len(self.data)


class MixDataset(Dataset):
    def __init__(self, paths, reading_fn, mfcc_function, transform=[]):
        """
        :param paths: list of absolute path to files

        :param reading_fn:  function to read .waf files, returns np.arrays
        :param mfcc_function: function to construct mfcc
        :param transform:  list of transforms to apply to output data
        """
        super(MixDataset, self).__init__()
        ishuman = [1 if '/human/' in fp else 0 for fp in paths]
        isspoof = [1 if '/spoof/' in fp else 0 for fp in paths]
        assert all(np.logical_xor(ishuman, isspoof)), " Wrong paths given"
        self.data = paths
        self.labels = isspoof
        self.reading_fn = reading_fn
        self.transform = transform
        self.mfcc_func = mfcc_function
        weights_h = len(self.data) / (len(self.data) - sum(isspoof))
        weights_s = len(self.data) / sum(isspoof)
        print('#class weights:', (weights_h, weights_s))
        self.weights = [weights_s if l == 1 else weights_h for l in self.labels]

    def __getitem__(self, idx):
        print("in item")
        data = self.reading_fn(self.data[idx])
        assert False, print(type(data))
        print(type(data))
        mfcc_data = self.mfcc_func(data)
        for t in self.transform:
            data = t(data)
            mfcc_data = t(mfcc_data)

        label = self.labels[idx]
        return {'data': data, 'mfcc': mfcc_data, 'label': label}

    def __len__(self):
        return len(self.data)