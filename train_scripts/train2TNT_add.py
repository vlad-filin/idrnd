import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.insert(0, "../")
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mobilenetV3 import mobilenetv3

from models import TwoBranchTNT
from dataset import VoiceAntiSpoofDataset
from reading_utils import read_fromBaseline, read_scipy
from Metrics import compute_err
import DftSpectrogram_pytorch

import glob
from kekas import Keker, DataOwner

from kekas.metrics import accuracy
from utils import step_fn, ScoreCallback, exp_decay
import torch.autograd as autograd
from librosa.feature import melspectrogram

dft_conf0 = {"length": 512,
            "shift": 256,
            "nfft": 512,
            "mode": 'log',
            "normalize_feature": True}
dft_conf1 = {"length": 512,
            "shift": 256,
            "nfft": 512,
            "mode": 'log',
            "normalize_feature": False,
            "trainable":True}
dft_pytorchNT = DftSpectrogram_pytorch.DftSpectrogram(**dft_conf0)
dft_pytorchT = DftSpectrogram_pytorch.DftSpectrogram(**dft_conf1)

NetMfcc = mobilenetv3(in_channels=1, get_weights=False)
NetDft = mobilenetv3(in_channels=2, get_weights=False)
model = TwoBranchTNT(NetMfcc, NetDft, extractorNT=dft_pytorchNT, extractorT=dft_pytorchT, num_features=2560)


mfcc_function = lambda data: melspectrogram(data, n_mels=100)
dataset_dir = '../../Training_Data/'
add_data_dir = '../../validationASV/'
print("Num samples:", len(glob.glob(os.path.join(dataset_dir, '**/*.wav'), recursive=True)))
dataset = VoiceAntiSpoofDataset(dataset_dir, 'all', read_scipy,
                               transform=[lambda x: x[None, ...].astype(np.float32)],
                                add_TData=add_data_dir,
                                mfcc_function=mfcc_function)


"""
dataset_val = VoiceAntiSpoofDataset(dataset_dir, 'val', read_scipy,
                                 transform=[lambda x: x[None, ...].astype(np.float32)])
"""
dataset_val_dir = '../../ASV2017DEV/'
dataset_val = VoiceAntiSpoofDataset(dataset_val_dir, 'all', read_scipy,
                                   transform=[lambda x: x[None, ...].astype(np.float32)],
                                    mfcc_function=mfcc_function)


batch_size = 256
num_workers = 16
"""
dataset.data = dataset.data[0:24] + dataset.data[-24:]
dataset.labels = dataset.labels[0:24]  + dataset.labels[-24:]
dataset_val.data = dataset_val.data[0:24] + dataset_val.data[-24:]
dataset_val.labels = dataset_val.labels[0:24]  + dataset_val.labels[-24:]
dataset.weights = dataset.weights[0:24] + dataset.weights[-24:]

sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.weights, len(dataset.weights))
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
val_dl = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
len(dataset), len(dataset_val), len(np.unique(dataset.data))
"""
dataowner = DataOwner(dataloader, val_dl, None)
criterion = nn.CrossEntropyLoss(weight=None)

folder = '2TNeNormNTNormMFCC100'
keker = Keker(model=model,
              dataowner=dataowner,
              criterion=criterion,
              step_fn=step_fn,                    # previosly defined step function
              target_key="label",                 # remember, we defined it in the reader_fn for DataKek?
              opt=torch.optim.Adam,               # optimizer class. if note specifiyng,
                                                  # an SGD is using by default
              opt_params={"weight_decay": 1e-3},
              callbacks=[ScoreCallback('preds', 'label', compute_err,
                                       'checkpoints/' + folder,
                                       logdir='tensorboard/' + folder)],
                 metrics={"acc": accuracy})
with autograd.detect_anomaly():
    keker.kek(lr=1e-3,
              epochs=50,
              sched=torch.optim.lr_scheduler.MultiStepLR,       # pytorch lr scheduler class
              sched_params={"milestones": [15, 25, 35, 45], "gamma": 0.5},
             cp_saver_params={
                  "savedir": "checkpoints/" + folder,
             "metric":"acc",
             "mode":'max'},
              logdir="tensorboard/" + folder)


torch.save(model.state_dict(), "checkpoints/" + folder + "/final.pt")