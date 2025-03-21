from torch.cuda import is_available
from cslib.utils import Options

class TestOptions(Options):
    def __init__(self):
        super().__init__("GoogLeNet")
        self.update(
            {
                # Utils
                "comment": "",
                "device": "cuda" if is_available() else "cpu",
                "model_path": "path/to/model/folder",
                "dataset_path": "path/to/dataset",

                # Model Option
                "model_name":{
                    0:"vgg11",
                    1:"vgg11_bn",
                    2:"vgg13",
                    3:"vgg13_bn",
                    4:"vgg16",
                    5:"vgg16_bn",
                    6:"vgg19",
                    7:"vgg19_bn"
                }[0],

                # Test Config
                "batch_size": 32,
            }
        )
