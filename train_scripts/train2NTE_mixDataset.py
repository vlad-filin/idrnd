import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.insert(0, "../")
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from MobileNetV2_torchvision import MobileNetV2

from models import TwoBranchModelNTE
from dataset import MixDataset, VoiceAntiSpoofDataset
from reading_utils import universe_reader, read_scipy
from Metrics import compute_err
import DftSpectrogram_pytorch
from librosa.feature import mfcc
import glob
from kekas import Keker, DataOwner

from kekas.metrics import accuracy
from utils import step_fn, ScoreCallback, exp_decay, jointer, roc_auc
import torch.autograd as autograd


dft_conf0 = {"length": 512,
            "shift": 256,
            "nfft": 512,
            "mode": 'log',
            "normalize_feature": True}

dft_pytorchNT = DftSpectrogram_pytorch.DftSpectrogram(**dft_conf0)

MN2_Dft = MobileNetV2()
MN2_Dft.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
MN2_MFCC = MobileNetV2()
MN2_MFCC.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
model = TwoBranchModelNTE(MN2_MFCC, MN2_Dft, dft_pytorchNT, num_features=2560).to('cuda')

train_files = [("../../ASV2019_human/", "../train_asv.txt"),
               ("../../Training_Data", "../train_idrnd.txt"),
               ("../../Training_Data", "../val_idrnd.txt")]
dataset_val_dir = '../../validationASV/'
val_dataset = VoiceAntiSpoofDataset(dataset_val_dir, 'all', read_scipy,
                                   transform=[lambda x: x[None, ...].astype(np.float32)])
train_data = []
for tpl in train_files:
    train_data += jointer(*tpl)

mfcc_function = lambda x: mfcc(x, sr=16000, n_mfcc=25)
reading_fn = lambda x: universe_reader(x, length=100000)

train_dataset = MixDataset(train_data, reading_fn, mfcc_function,
                           transform=[lambda x: x[None, ...].astype(np.float32)])

#train_dataset.data = train_dataset.data[0:25] + train_dataset.data[-24:]
#train_dataset.labels = train_dataset.labels[0:25]  + train_dataset.labels[-24:]
#val_dataset.data = val_dataset.data[0:24] + val_dataset.data[-24:]
#val_dataset.labels = val_dataset.labels[0:24]  + val_dataset.labels[-24:]
#train_dataset.weights = train_dataset.weights[0:25] + train_dataset.weights[-24:]

sampler = torch.utils.data.sampler.WeightedRandomSampler(train_dataset.weights, len(train_dataset.weights))
batch_size = 32
num_workers = 8

print("###", len(train_dataset), len(val_dataset), len(np.unique(train_dataset.data)))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

dataowner = DataOwner(train_dataloader, val_dataloader, None)
criterion = nn.CrossEntropyLoss()

keker = Keker(model=model,
              dataowner=dataowner,
              criterion=criterion,
              step_fn=step_fn,                    # previosly defined step function
              target_key="label",                 # remember, we defined it in the reader_fn for DataKek?
              opt=torch.optim.Adam,               # optimizer class. if note specifiyng,
                                                  # an SGD is using by default
              opt_params={"weight_decay": 1e-3},
              callbacks=[ScoreCallback('preds', 'label', compute_err,
                                       'checkpoints/2branchNTE_Mix', logdir='tensorboard/2branchNTE_Mix')],
              metrics={"acc": accuracy})
keker.kek(lr=1e-3,
          epochs=50,
          sched=torch.optim.lr_scheduler.MultiStepLR,  # pytorch lr scheduler class
          sched_params={"milestones": [15, 25, 35, 45], "gamma": 0.5},
          cp_saver_params={
              "savedir": "checkpoints/2branchNTE_Mix",
              "metric": "acc",
              "mode": 'max'},
          logdir="tensorboard/2branchNTE_Mix")
torch.save(model.state_dict(), "checkpoints/2branchNTE_Mix/final.pt")
