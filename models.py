import torch.nn as nn


class ModelWrapper_TorchVisionNet(nn.Module):
    def __init__(self, Net):
        super(ModelWrapper_TorchVisionNet, self).__init__()

        self.conv1x1 = nn.Conv2d(1, 3, kernel_size=1)
        self.Net = Net
        self.softmax = nn.Softmax(dim=0)
    def forward(self, x):

        x = self.conv1x1(x)
        x = self.Net(x)
        #x = self.softmax(x)
        return x

