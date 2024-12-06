import torch.nn as nn

class VGGBase(nn.Module):
    def __init__(self, cfg, num_classes=1000, init_weights=True):
        super(VGGBase, self).__init__()
        self.features = self.make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-2], 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG11(VGGBase):
    def __init__(self, num_classes=1000, init_weights=True):
        super(VGG11, self).__init__(cfgs['VGG11'], num_classes, init_weights)

class VGG13(VGGBase):
    def __init__(self, num_classes=1000, init_weights=True):
        super(VGG16, self).__init__(cfgs['VGG13'], num_classes, init_weights)

class VGG16(VGGBase):
    def __init__(self, num_classes=1000, init_weights=True):
        super(VGG11, self).__init__(cfgs['VGG16'], num_classes, init_weights)

class VGG19(VGGBase):
    def __init__(self, num_classes=1000, init_weights=True):
        super(VGG16, self).__init__(cfgs['VGG19'], num_classes, init_weights)

# # 实例化VGG11和VGG16网络
# vgg11 = VGG11(num_classes=10)  # 假设我们只需要10个输出类别
# vgg16 = VGG16(num_classes=10)  # 假设我们只需要10个输出类别

# print(vgg11)
# print(vgg16)
