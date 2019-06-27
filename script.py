import os
import pandas as pd
import torch
import tqdm
from utils import read_scipy
from models import ModelWrapper_TorchVisionNet
from MobileNetV2_torchvision import  MobileNetV2
import torch.nn as nn
from librosa.feature import mfcc

print("Done!")
dataset_dir = "."
MN2 = MobileNetV2()
MN2.classifier[1] = nn.Linear(1280, 2, bias=False)
model = ModelWrapper_TorchVisionNet(MN2).to('cuda')
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
    feature = mfcc(feature)[None, None,...]
    feature = torch.tensor(feature, dtype=torch.float32).to('cuda')
    score = model.softmax(model(feature)).cpu()
    eval_protocol.at[protocol_id, 'score'] = score[0][0]
eval_protocol[['path', 'score']].to_csv('answers.csv', index=None)
print(eval_protocol.sample(5).head())