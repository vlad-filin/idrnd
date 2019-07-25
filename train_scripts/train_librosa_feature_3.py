import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.insert(0, "../")
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from MobileNetV2_torchvision import MobileNetV2

from models import LibrosaFeaturesModel
from dataset import VoiceAntiSpoofDataset
from reading_utils import read_fromBaseline, read_scipy
from Metrics import compute_err
import DftSpectrogram_pytorch

import glob
from kekas import Keker, DataOwner

from kekas.metrics import accuracy
from utils import step_fn, ScoreCallback, exp_decay
import torch.autograd as autograd
from librosa.feature import mfcc




MN2_MEL = MobileNetV2()
MN2_MEL.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
model = LibrosaFeaturesModel(MN2_MEL)
mfcc_function = lambda data: mfcc(data, sr=16000, n_mfcc=128)
dataset_dir = '../../Training_Data/'
print("Num samples:", len(glob.glob(os.path.join(dataset_dir, '**/*.wav'), recursive=True)))
dataset = VoiceAntiSpoofDataset(dataset_dir, 'all', read_scipy,
                               transform=[lambda x: x[None, ...].astype(np.float32)],
                                mfcc_function=mfcc_function)
"""
dataset_val = VoiceAntiSpoofDataset(dataset_dir, 'val', read_scipy,
                                 transform=[lambda x: x[None, ...].astype(np.float32)])
"""
dataset_val_dir = '../../validationASV/'
dataset_val = VoiceAntiSpoofDataset(dataset_val_dir, 'all', read_scipy,
                                   transform=[lambda x: x[None, ...].astype(np.float32)],
                                    mfcc_function=mfcc_function)

sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.weights, len(dataset.weights))
batch_size = 196
num_workers = 16

#dataset.data = dataset.data[0:24] + dataset.data[-24:]
#dataset.labels = dataset.labels[0:24]  + dataset.labels[-24:]
#dataset_val.data = dataset_val.data[0:24] + dataset_val.data[-24:]
#dataset_val.labels = dataset_val.labels[0:24]  + dataset_val.labels[-24:]
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
val_dl = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
print(len(dataset), len(dataset_val), len(np.unique(dataset.data)))



dataowner = DataOwner(dataloader, val_dl, None)
criterion = nn.CrossEntropyLoss(weight=None)

folder = 'MFCC128_MN2'
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