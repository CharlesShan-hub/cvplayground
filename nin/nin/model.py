import torch
import torch.nn as nn
import torch.nn.functional as F

class NiN(nn.Module):
    def __init__(self, num_classes=10):
        super(NiN, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.mlp1 = self._mlp_block(96, 96, 1)
        self.mlp2 = self._mlp_block(256, 256, 1)
        self.mlp3 = self._mlp_block(384, 384, 1)
        self.mlp4 = self._mlp_block(384, 384, 1)
        self.mlp5 = self._mlp_block(256, 256, 1)

    def _mlp_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.lrn1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.mlp1(x)

        x = F.relu(self.conv2(x))
        x = self.lrn2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.mlp2(x)

        x = F.relu(self.conv3(x))
        x = self.mlp3(x)

        x = F.relu(self.conv4(x))
        x = self.mlp4(x)

        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.mlp5(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc8(x)
        return x

# 实例化NiN网络
nininet = NiN(num_classes=10)  # 假设我们只需要10个输出类别
print(nininet)
