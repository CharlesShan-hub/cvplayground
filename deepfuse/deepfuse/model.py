import torch.nn as nn
import torch

def load_model(opts):
    model = DeepFuse(device=opts.device)
    model.load_state_dict(torch.load(opts.pre_trained, map_location=opts.device)['model'])
    return model

"""
    This script defines the DeepFuse model and related module

    Author: SunnerLi
"""
# -------------------------------------------------------------------------------------------------------
#   Define layers
# -------------------------------------------------------------------------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 16, kernel_size = 5, last = nn.ReLU):
        super().__init__()
        if kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = padding),
            nn.BatchNorm2d(out_channels),
            last()
        )

    def forward(self, x):
        out = self.main(x)
        return out

class FusionLayer(nn.Module):
    def forward(self, x, y):
        return x + y

# -------------------------------------------------------------------------------------------------------
#   Define model
# -------------------------------------------------------------------------------------------------------
class DeepFuse(nn.Module):
    def __init__(self, device = torch.device('cpu')):
        super().__init__()
        self.layer1 = ConvLayer(1, 16, 5, last = nn.LeakyReLU) # type: ignore
        self.layer2 = ConvLayer(16, 32, 7)
        self.layer3 = FusionLayer()
        self.layer4 = ConvLayer(32, 32, 7, last = nn.LeakyReLU) # type: ignore
        self.layer5 = ConvLayer(32, 16, 5, last = nn.LeakyReLU) # type: ignore
        self.layer6 = ConvLayer(16, 1, 5, last = nn.Tanh) # type: ignore
        self.device = device
        self.to(self.device)

    def forward(self,y_1,y_2):
        # c11 = self.layer1(y_1[:, 0:1])
        # c12 = self.layer1(y_2[:, 0:1])
        c11 = self.layer1(y_1)
        c12 = self.layer1(y_2)
        c21 = self.layer2(c11)
        c22 = self.layer2(c12)
        f_m = self.layer3(c21, c22)
        c3  = self.layer4(f_m)
        c4  = self.layer5(c3)
        c5  = self.layer6(c4)
        return c5