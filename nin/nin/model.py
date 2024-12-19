import torch
import torch.nn as nn
import torch.nn.functional as F

class NiN(nn.Module):
    def __init__(self, num_classes=10):
        super(NiN, self).__init__()
        self.num_classes = num_classes

        self.net = nn.Sequential(
            self._mlp_block(96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            self._mlp_block(256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            self._mlp_block(384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            self._mlp_block(num_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def _mlp_block(self, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size, strides, padding),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    # 实例化NiN网络
    nininet = NiN(num_classes=10)  # 假设我们只需要10个输出类别
    print(nininet)
