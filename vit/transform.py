from torchvision.transforms import Compose, Resize, ToTensor

transform = Compose([ToTensor(), Resize((224, 224), antialias=True)]) # type: ignore