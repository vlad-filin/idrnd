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
        with torch.no_grad():
            x0 = self.extractorNT(x)
        x1 = self.extractorT(x)
        feature = torch.stack((x0, x1), dim=1)
        x = self.Net(feature)
        # x = self.softmax(x)
        return x
