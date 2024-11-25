import torch.nn as nn
from sklearn import svm
import torch


class AlexNet(nn.Module):

    def __init__(self, num_classes, classify=True, save_feature=False, init_weight=None, device='cpu'):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.classify = classify
        self.save_feature = save_feature
        self.device = device

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
     
        self.drop8 = nn.Dropout()
        self.fn8 = nn.Linear(256 * 6 * 6, 4096)
        self.active8 = nn.ReLU(inplace=True)
        
        self.drop9 = nn.Dropout()
        self.fn9 = nn.Linear(4096, 4096)
        self.active9 = nn.ReLU(inplace=True)
        
        self.fn10 = nn.Linear(4096, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.drop8(x)
        x = self.fn8(x)
        x = self.active8(x)

        x = self.drop9(x)
        x = self.fn9(x)
        
        feature = self.active9(x)
        final = self.fn10(feature)

        if self.classify:
            return final
        if self.save_feature:
            return feature
    
    def init_weights(self,pre_trained_url,pre_train_save_path=""):
        from urllib.parse import urlparse

        def is_url(path):
            parsed_url = urlparse(path)
            return bool(parsed_url.scheme)

        if is_url(pre_trained_url):
            from torch.hub import load_state_dict_from_url
            state_dict=load_state_dict_from_url(
                url=pre_trained_url, 
                model_dir=pre_train_save_path,
                progress=True
            )
            current_state = self.state_dict()
            keys = list(state_dict.keys())
            for key in keys:
                if key.startswith('features'):
                    current_state[key] = state_dict[key]
            current_state['fn8.weight'] = state_dict['classifier.1.weight']
            current_state['fn8.bias'] = state_dict['classifier.1.bias']
            current_state['fn9.weight'] = state_dict['classifier.4.weight']
            current_state['fn9.bias'] = state_dict['classifier.4.bias']
        else:
            state_dict = torch.load(pre_trained_url, map_location=self.device)['model_state_dict']            
            keys = list(state_dict.keys())
            for key in keys:
                if not key.startswith('fn10'):
                    self.state_dict()[key] = state_dict[key]


class SVM:
    def __init__(self):
        pass

    def train(self, features, labels):
        clf = svm.LinearSVC()
        clf.fit(features, labels)
        return clf


class RegNet(nn.Module):

    def __init__(self):
        super(RegNet, self).__init__()

        layers = []
        fc1 = nn.Linear(4096, 4096)
        fc1.weight.data.normal_(0.0, 0.01)
        layers.append(fc1)
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Tanh())
        fc2 = nn.Linear(4096, 5)# 1(is object or not) + 4(X,Y,H,W)
        fc2.weight.data.normal_(0.0, 0.01)
        layers.append(fc2)
        layers.append(nn.Tanh())
        
        self.logits = nn.Sequential(*layers)

    def forward(self, x):
        return self.logits(x)

  
        
    
