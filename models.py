import torch.nn as nn
import torch

class ModelWrapper_TorchVisionNet(nn.Module):
    def __init__(self, Net):
        super(ModelWrapper_TorchVisionNet, self).__init__()

        self.conv1x1 = nn.Conv2d(1, 3, kernel_size=1)
        self.Net = Net
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):

        x = self.conv1x1(x)
        x = self.Net(x)
        #x = self.softmax(x)
        return x
class TorchVisionNet_with_exctractor(nn.Module):
    def __init__(self, Net, extractor):
        super(TorchVisionNet_with_exctractor, self).__init__()

        self.Net = Net
        self.extractor = extractor
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        if self.extractor.trainable is False:
            with torch.no_grad():
                x = self.extractor(x)
        else:
            x = self.extractor(x)
        x = self.Net(x)
        #x = self.softmax(x)
        return x


class TorchVisionNet_with_2exctractor(nn.Module):

    def __init__(self, Net, extractorNT, extractorT):
        super(TorchVisionNet_with_2exctractor, self).__init__()

        self.Net = Net
        self.extractorNT = extractorNT
        self.extractorT = extractorT

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x0 = self.extractorNT(x)
        x1 = self.extractorT(x)
        feature = torch.cat((x0, x1), dim=1)
        x = self.Net(feature)
        # x = self.softmax(x)
        return x


class TwoBranchModel(nn.Module):

    def __init__(self, NetMfcc, NetDft, extractorNT,
                 extractorT, num_features=1024):
        super(TwoBranchModel, self).__init__()

        self.NetMfcc = NetMfcc
        self.NetDft = NetDft
        self.extractorNT = extractorNT
        self.extractorT = extractorT
        self.linear = nn.Linear(num_features, 2, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mfcc):

        x0 = self.extractorNT(x)
        x1 = self.extractorT(x)
        feature = torch.cat((x0, x1), dim=1)
        v0 = self.NetDft(feature)
        v1 = self.NetMfcc(mfcc)
        v =  torch.cat((v0, v1), dim=1)
        res = self.linear(v)
        # x = self.softmax(x)
        return res


class TwoBranchModelNTE(nn.Module):

    def __init__(self, NetMfcc, NetDft, extractorNT, num_features=1024):
        super(TwoBranchModelNTE, self).__init__()

        self.NetMfcc = NetMfcc
        self.NetDft = NetDft
        self.extractorNT = extractorNT
        self.linear = nn.Linear(num_features, 2, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mfcc):

        x0 = self.extractorNT(x)
        v0 = self.NetDft(x0)
        v1 = self.NetMfcc(mfcc)
        v =  torch.cat((v0, v1), dim=1)
        res = self.linear(v)
        # x = self.softmax(x)
        return res