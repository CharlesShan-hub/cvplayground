import os
import torchvision.transforms as T


DATA_PATH = 'data'
CLASSES_PATH = os.path.join(DATA_PATH, 'classes.json')

BATCH_SIZE = 64
EPOCHS = 135
WARMUP_EPOCHS = 0
LEARNING_RATE = 1E-4

EPSILON = 1E-6

from torch.cuda import is_available
from cslib.utils import Options

class TrainOptions(Options):
    def __init__(self):
        super().__init__("YOLOv1")
        self.update(
            {
                # Utils
                "comment": "",
                "device": "cuda" if is_available() else "cpu",
                "model_base_path": "path/to/folder/to/save/ckpt",
                "dataset_path": "path/to/dataset",

                # Model Option
                "image_size": 448,
                "num_classes": 10,
                "S": 7, # Divide each image into a SxS grid
                "B": 2, # Number of bounding boxes to predict
                "C": 20, # Number of classes in the dataset

                # Train Config
                "seed": 42,
                "batch_size": 64,
                "optimizer":{
                    1: 'ASGD', 
                    2: 'Adadelta', 
                    3: 'Adagrad', 
                    4: 'Adam', 
                    5: 'AdamW', 
                    6: 'Adamax', 
                    7: 'LBFGS', 
                    8: 'NAdam', 
                    9: 'Optimizer', 
                    10: 'RAdam', 
                    11: 'RMSprop', 
                    12: 'Rprop', 
                    13: 'SGD', 
                    14: 'SparseAdam'
                }[13],
                "lr": 0.01,
                "lr_scheduler": {
                    1: 'LambdaLR', 
                    2: 'MultiplicativeLR', 
                    3: 'StepLR', 
                    4: 'MultiStepLR', 
                    5: 'ConstantLR', 
                    6: 'LinearLR',
                    7: 'ExponentialLR', 
                    8: 'SequentialLR', 
                    9: 'CosineAnnealingLR', 
                    10: 'ChainedScheduler', 
                    11: 'ReduceLROnPlateau',# only support
                    12: 'CyclicLR', 
                    13: 'CosineAnnealingWarmRestarts', 
                    14: 'OneCycleLR', 
                    15: 'PolynomialLR', 
                    16: 'LRScheduler',
                }[11],
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


class TestOptions(Options):
    def __init__(self):
        super().__init__("YOLOv1")
        self.update(
            {
                # Utils
                "comment": "",
                "device": "cuda" if is_available() else "cpu",
                "model_path": "path/to/model.ckpt",
                "dataset_path": "path/to/dataset",

                # Model Option
                "image_size": 448,
                "num_classes": 10,
                "S": 7, # Divide each image into a SxS grid
                "B": 2, # Number of bounding boxes to predict
                "C": 20, # Number of classes in the dataset

                # Test Config
                "batch_size": 32,
            }
        )


class DisplayOptions(Options):
    def __init__(self):
        super().__init__("YOLOv1")
        self.update(
            {
                # Utils
                "device": "cuda" if is_available() else "cpu",
                "dataset_path": "path/to/dataset",

                # Model Option
                "normalize": False,
                "augment": False,
                "image_size": (448,448),
                "S": 7, # Divide each image into a SxS grid
                "B": 2, # Number of bounding boxes to predict
                "C": 20, # Number of classes in the dataset
            }
        )