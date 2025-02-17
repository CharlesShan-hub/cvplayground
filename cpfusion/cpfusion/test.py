import click
from clib.utils import glance
from clib.dataset.fusion import TNO, LLVIP
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from utils import *
from model import *
from inference import fusion

@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--layer", type=int, default=4)
def main(**kwargs):
    dataset = LLVIP(root=kwargs['dataset_path'], transform=Compose([ToTensor()]), download=True, train=False)
    # dataset = TNO(root=kwargs['dataset_path'], transform=Compose([ToTensor()]), download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for ir,vis in dataloader:
        fused = fusion(ir,vis,kwargs['layer'])
        glance([ir,vis,fused])

if __name__ == '__main__':
    main()
