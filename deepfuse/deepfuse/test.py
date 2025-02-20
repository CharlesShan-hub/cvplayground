import click
from pathlib import Path
from clib.dataset.fusion import LLVIP
from config import TestOptions
from inference import inference
from model import load_model
from utils import *

@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--pre_trained", type=click.Path(exists=True), required=True)
@click.option("--save_path", type=click.Path(exists=True), required=True)
def main(**kwargs):
    opts = TestOptions().parse(kwargs)
    dataset = LLVIP(root=opts.dataset_path, transform=None, download=True, train=False)
    model = load_model(opts)
    for i in range(len(dataset)):
        if i%50 != 0:
            continue
        ir,vis = dataset[i]
        fused = Path(kwargs['save_path']) / Path(ir).name
        ir_img = to_tensor(path_to_gray(ir)).unsqueeze(0)
        vis_img = to_tensor(path_to_rgb(vis)).unsqueeze(0)
        fused_img = inference(model, ir_img, vis_img, opts)
        # glance([ir_img, rgb_to_gray(vis_img), fused_img],clip=True)
        save_array_to_img(fused_img, filename=fused)

if __name__ == '__main__':
    main()