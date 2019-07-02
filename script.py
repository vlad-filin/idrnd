import os
import pandas as pd
import torch
import tqdm
from utils import read_scipy
from models import TorchVisionNet_with_exctractor
import DftSpectrogram_pytorch
from MobileNetV2_torchvision import  MobileNetV2
import torch.nn as nn
from librosa.feature import mfcc

print("Done!")

dataset_dir = "."
dft_conf = {"length": 512,
            "shift": 256,
            "nfft": 512,
            "mode": 'log',
            "normalize_feature": True}


dft_pytorch = DftSpectrogram_pytorch.DftSpectrogram(**dft_conf)
MN2 = MobileNetV2()
MN2.classifier[1] = nn.Linear(1280, 2, bias=False)
MN2.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
model = TorchVisionNet_with_exctractor(MN2, dft_pytorch).to('cuda')


model.load_state_dict((torch.load("weights.pt")))
model.eval()
print("model Done!")
eval_protocol_path = "protocol_test.txt"
eval_protocol = pd.read_csv(eval_protocol_path, sep=" ", header=None)
eval_protocol.columns = ['path', 'key']
eval_protocol['score'] = 0.0

print(eval_protocol.shape)
print(eval_protocol.sample(5).head())

for protocol_id, protocol_row in tqdm.tqdm(list(eval_protocol.iterrows())):
    feature = read_scipy(os.path.join(dataset_dir, protocol_row['path']))
    with torch.no_grad():
        feature = torch.tensor(feature, dtype=torch.float32).to('cuda')
        score = model.softmax(model(feature)).cpu()
    eval_protocol.at[protocol_id, 'score'] = score[0][0]
eval_protocol[['path', 'score']].to_csv('answers.csv', index=None)
print(eval_protocol.sample(5).head())