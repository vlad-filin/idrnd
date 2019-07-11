import os
import pandas as pd
import torch
import tqdm
from reading_utils import read_test
from models import TwoBranchModelNTE
import DftSpectrogram_pytorch
from MobileNetV2_torchvision import  MobileNetV2
import torch.nn as nn
from librosa.feature import mfcc

print("Done!")

dataset_dir = "."
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


model.load_state_dict((torch.load("24.7368.18.pt")))
model.eval()
print("model Done!")
eval_protocol_path = "protocol_test.txt"
eval_protocol = pd.read_csv(eval_protocol_path, sep=" ", header=None)
eval_protocol.columns = ['path', 'key']
eval_protocol['score'] = 0.0

print(eval_protocol.shape)
print(eval_protocol.sample(5).head())

for protocol_id, protocol_row in tqdm.tqdm(list(eval_protocol.iterrows())):
    feature = read_test(os.path.join(dataset_dir, protocol_row['path']))
    mfcc_feature = mfcc(feature, sr=16000, n_mfcc=128)
    with torch.no_grad():
        feature = torch.tensor(feature, dtype=torch.float32).to('cuda')[None, None,...]
        mfcc_feature = torch.tensor(mfcc_feature, dtype=torch.float32).to('cuda')[None, None,...]
        score = model.softmax(model(feature, mfcc_feature)).cpu()
    eval_protocol.at[protocol_id, 'score'] = score[0][0]
eval_protocol[['path', 'score']].to_csv('answers.csv', index=None)
print(eval_protocol.sample(5).head())