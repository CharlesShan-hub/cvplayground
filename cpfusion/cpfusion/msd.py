'''
Transform to pyramid by adaptive Gaussian kernel
'''

import click
from clib.utils import glance
from clib.dataset.fusion import LLVIP, TNO
from clib.algorithms.msd import Laplacian
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
def main(**kwargs):
    dataset = LLVIP(root=kwargs['dataset_path'], transform=Compose([ToTensor()]), download=True, train=False)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    for ir,vis in dataloader:
        pyr = Laplacian(image = ir)
        layers = pyr.pyramid
        recon = pyr.recon
        glance([ir,recon])
        glance(layers)
        break


if __name__ == '__main__':
    main()