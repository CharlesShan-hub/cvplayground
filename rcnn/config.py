from torch.cuda import is_available
from clib.utils import Options


class TrainAldexNetOptions(Options):
    def __init__(self):
        super().__init__("RCNN - AlexNet")
        self.update(
            {
                # Utils
                "comment": "",
                "device": "cuda" if is_available() else "cpu",
                "model_base_path": "path/to/folder/to/save/ckpt",
                "dataset_path": "path/to/dataset",

                # Model Option
                "pre_trained": True,
                "pre_trained_path": "path/to/save/pre_trained.pth",
                "pre_trained_url":"https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
                "num_classes": 17,
                "image_size": 224,
                
                # Train Config
                "seed": 42,
                "batch_size": 64,
                "epochs": 2,
                "optimizer": 'SGD',
                "lr": 0.01,
                "lr_scheduler": 'ReduceLROnPlateau',
                "max_epoch": {
                    1: 0,   # unlimit max epoch (need ReduceLROnPlateau and limit reduce)
                    2: 100,
                }[2],
                "max_reduce": {
                    1: 0,   # unlimit max reduce (need limit reduce)
                    2: 4,
                }[2], # when ReduceLROnPlateau, max reduce number
                "factor": 0.1,  # when ConstantLR | ReduceLROnPlateau, lr_new  = lr_old * factor
                "train_mode": {
                    1:"Holdout", 
                    2:"K-fold"
                }[1],
                "val": 0.2,  # when Holdout, precent to validate (not for train!)
            }
        )


class TrainSVMOptions(Options):
    def __init__(self):
        super().__init__("RCNN - SVM")
        self.update(
            {
                # Utils
                "comment": "",
                "device": "cuda" if is_available() else "cpu",
                "model_base_path": "path/to/folder/to/save/ckpt",
                "dataset_path": "path/to/dataset",
            }
        )


class TrainBoxOptions(Options):
    def __init__(self):
        super().__init__("RCNN - Box")
        self.update(
            {
                # Utils
                "comment": "",
                "device": "cuda" if is_available() else "cpu",
                "model_base_path": "path/to/folder/to/save/ckpt",
                "dataset_path": "path/to/dataset",

                "lr_scheduler": 'ReduceLROnPlateau',
            }
        )


class TestAldexNetOptions(Options):
    def __init__(self):
        super().__init__('RCNN - AlexNet')
        self.update(
            {
                # Utils
                'comment': '',
                'device': 'cuda' if is_available() else 'cpu',
                'model_path': 'path/to/model.pth',
                'dataset_path': 'path/to/dataset',
                
                # Model Option
                'num_classes': 17,

                # Test Config
                'batch_size': 8,
                
            }
        )