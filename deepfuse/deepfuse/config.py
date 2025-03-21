from torch.cuda import is_available
from cslib.utils import Options  

class TestOptions(Options):
    def __init__(self):
        super().__init__('DeepFuse')
        self.update({
            'pre_trained': 'model.pth',
            'H': 400,
            'W': 600,
            'resize': False,
            'device': 'cuda' if is_available() else 'cpu'
        })