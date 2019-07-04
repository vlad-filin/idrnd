import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.insert(0, "../")
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from MobileNetV2_torchvision import MobileNetV2

from models import TorchVisionNet_with_2exctractor
import DftSpectrogram_pytorch
from dataset import VoiceAntiSpoofDataset
from utils import read_fromBaseline, read_scipy
from Metrics import compute_err

import glob
from kekas import Keker, DataOwner

from kekas.metrics import accuracy
from utils import step_fn, ScoreCallback, exp_decay

dft_conf0 = {"length": 512,
            "shift": 256,
            "nfft": 512,
            "mode": 'log',
            "normalize_feature": False}
dft_conf1 = {"length": 512,
            "shift": 256,
            "nfft": 512,
            "mode": 'log',
            "normalize_feature": False,
            "trainable":True}


dft_pytorchNT = DftSpectrogram_pytorch.DftSpectrogram(**dft_conf0)
dft_pytorchT = DftSpectrogram_pytorch.DftSpectrogram(**dft_conf1)
MN2 = MobileNetV2()
MN2.classifier[1] = nn.Linear(1280, 2, bias=False)
MN2.features[0][0] = nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
model = TorchVisionNet_with_2exctractor(MN2, dft_pytorchNT, dft_pytorchT).to('cuda')


dataset_dir = '../../Training_Data/'
print("Num samples:", len(glob.glob(os.path.join(dataset_dir, '**/*.wav'), recursive=True)))
dataset = VoiceAntiSpoofDataset(dataset_dir, 'all', read_scipy,
                                transform=[lambda x: x[None, ...].astype(np.float32)])
dataset_val_dir = '../../validationASV/'
dataset_val = VoiceAntiSpoofDataset(dataset_val_dir, 'all', read_scipy,
                                   transform=[lambda x: x[None, ...].astype(np.float32)])
batch_size = 64
num_workers = 16


dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_dl = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
len(dataset), len(dataset_val), len(np.unique(dataset.data))


dataowner = DataOwner(dataloader, val_dl, None)
weights = torch.tensor([5, 1.25])
weights = weights / weights.sum()
weights = weights.to('cuda')
criterion = nn.CrossEntropyLoss(weight=weights)


keker = Keker(model=model,
              dataowner=dataowner,
              criterion=criterion,
              step_fn=step_fn,                    # previosly defined step function
              target_key="label",                 # remember, we defined it in the reader_fn for DataKek?
              opt=torch.optim.Adam,               # optimizer class. if note specifiyng,
                                                  # an SGD is using by default
              opt_params={"weight_decay": 1e-3},
              callbacks=[ScoreCallback('preds', 'label', compute_err,
                                       'checkpoints_3rdBL', logdir='tensorboard/tensorboard4BL')],
                 metrics={"acc": accuracy})

keker.kek(lr=1e-3,
          epochs=50,
          sched=torch.optim.lr_scheduler.MultiStepLR,       # pytorch lr scheduler class
          sched_params={"milestones": [15, 25, 40], "gamma": 0.5},
         cp_saver_params={
              "savedir": "./checkpoints_4BL",
         "metric":"acc",
         "mode":'max'},
          logdir="tensorboard/tensorboard4BL")