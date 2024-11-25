from clib.algorithms import selective_search

from torchvision.transforms import Compose, Resize, ToTensor

def transform(image_size):
    return Compose([Resize((image_size, image_size)), ToTensor()])