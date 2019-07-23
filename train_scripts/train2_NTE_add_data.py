import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.insert(0, "../")
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from densenet import densenet169

from models import TwoBranchModelNTE
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


dft_pytorchNT = DftSpectrogram_pytorch.DftSpectrogram(**dft_conf0)

DN2_Dft = densenet169()
DN2_Dft.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
DN2_MFCC = densenet169()
DN2_MFCC.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
model = TwoBranchModelNTE(DN2_MFCC, DN2_Dft, dft_pytorchNT, num_features=3328).to('cuda')



#mfcc_function = lambda data: mfcc(data, sr=16000, n_mfcc=45)
mel_function = lambda data: melspectrogram(data, sr=16000, n_mels=128)
dataset_dir = '../../Training_Data/'
add_data_dir = '../../validationASV/'
print("Num samples:", len(glob.glob(os.path.join(dataset_dir, '**/*.wav'), recursive=True)))
dataset = VoiceAntiSpoofDataset(dataset_dir, 'all', read_scipy,
                               transform=[lambda x: x[None, ...].astype(np.float32)],
                                add_TData=add_data_dir,
                                mfcc_function=mel_function)


"""
dataset_val = VoiceAntiSpoofDataset(dataset_dir, 'val', read_scipy,
                                 transform=[lambda x: x[None, ...].astype(np.float32)])
"""
dataset_val_dir = '../../ASV2017DEV/'
dataset_val = VoiceAntiSpoofDataset(dataset_val_dir, 'all', read_scipy,
                                   transform=[lambda x: x[None, ...].astype(np.float32)])

sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.weights, len(dataset.weights))
batch_size = 192
num_workers = 16

#dataset.data = dataset.data[0:24] + dataset.data[-24:]
#dataset.labels = dataset.labels[0:24]  + dataset.labels[-24:]
#dataset_val.data = dataset_val.data[0:24] + dataset_val.data[-24:]
#dataset_val.labels = dataset_val.labels[0:24]  + dataset_val.labels[-24:]
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
val_dl = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
len(dataset), len(dataset_val), len(np.unique(dataset.data))


dataowner = DataOwner(dataloader, val_dl, None)
criterion = nn.CrossEntropyLoss(weight=None)


keker = Keker(model=model,
              dataowner=dataowner,
              criterion=criterion,
              step_fn=step_fn,                    # previosly defined step function
              target_key="label",                 # remember, we defined it in the reader_fn for DataKek?
              opt=torch.optim.Adam,               # optimizer class. if note specifiyng,
                                                  # an SGD is using by default
              opt_params={"weight_decay": 1e-3},
              callbacks=[ScoreCallback('preds', 'label', compute_err,
                                       'checkpoints/2branch_mel128_addData_DN',
                                       logdir='tensorboard/2branch_mel128_addData_DN')],
                 metrics={"acc": accuracy})
with autograd.detect_anomaly():
    keker.kek(lr=1e-3,
              epochs=50,
              sched=torch.optim.lr_scheduler.MultiStepLR,       # pytorch lr scheduler class
              sched_params={"milestones": [15, 25, 35, 45], "gamma": 0.5},
             cp_saver_params={
                  "savedir": "checkpoints/2branch_mel128_addData_DN",
             "metric":"acc",
             "mode":'max'},
              logdir="tensorboard/2branch_mel128_addData_DN")


torch.save(model.state_dict(), "checkpoints/2branch_mel128_addData_DN/final.pt")