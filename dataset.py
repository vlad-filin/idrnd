from torch.utils.data import Dataset
import glob
import os
import random


class VoiceAntiSpoofDataset(Dataset):
    def __init__(self, dataset_dir, mode, reading_fn, train_val_index=None,
                 seed=1, transform=None):
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
            random.seed(seed)
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
            elif mode == 'val':
                self.labels = [0] * len(val_human) + [1] * len(val_spoof)
                self.data = val_human + val_spoof
            else:
                self.labels = ([0] * len(train_human) + [1] * len(train_spoof) +
                               [0] * len(val_human) + [1] * len(val_spoof))
                self.data = (train_human + train_spoof +
                             val_human + val_spoof)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        data = self.reading_fn(self.data[idx])
        for t in self.transform:
            data = t(data)
        label = self.labels[idx]
        return {'data': data, 'label': label}

    def __len__(self):
        return len(self.data)
