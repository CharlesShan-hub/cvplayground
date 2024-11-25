import click
import torch
from torch.utils.data import DataLoader
from config import TestAldexNetOptions
from dataset import Flowers2
from transform import transform
from model import AlexNet
from clib.inference import BaseInferencer
import numpy as np
from pathlib import Path
from tqdm import tqdm

class FeatureSaver(BaseInferencer):
    def __init__(self, opts):
        super().__init__(opts)

        self.model = AlexNet(
            num_classes=opts.num_classes,
            classify=False,
            save_feature=True
        ).to(opts.device)
        self.load_checkpoint()

        self.criterion = torch.nn.CrossEntropyLoss()

        self.transform = transform(opts.image_size)

        dataset = Flowers2(
            root=opts.dataset_path,
            image_size=opts.image_size,
            transform=self.transform
        )

        self.loader = DataLoader(
            dataset=dataset,
            batch_size=opts.batch_size,
            shuffle=False
        )

    def test(self):
        assert self.model is not None
        assert self.loader is not None
        self.model.eval()
        all_outputs = []
        all_labels = []
        all_keys = []
        pbar = tqdm(self.loader, total=len(self.loader))
        with torch.no_grad():
            for images, labels, key , _  in pbar:
                images = images.to(self.opts.device)
                labels = labels.to(self.opts.device)
                outputs = self.model(images)
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_keys.append(key.cpu().numpy())

            all_outputs = np.vstack(all_outputs)
            all_labels = np.concatenate(all_labels)
            all_keys = np.concatenate(all_keys)

            np.save(Path(self.opts.dataset_path) / 'flowers-17' /'feature_for_svm.npy', all_outputs)
            np.save(Path(self.opts.dataset_path) / 'flowers-17' /'label_for_svm.npy', all_labels)
            np.save(Path(self.opts.dataset_path) / 'flowers-17' /'key_for_svm.npy', all_keys)

@click.command()
@click.option("--model_path", type=click.Path(exists=True), required=True)
@click.option("--comment", type=str, default="", show_default=False)
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--num_classes", type=int, default=10, show_default=True)
@click.option("--image_size", type=int, default=224, show_default=True)
@click.option("--batch_size", type=int, default=8, show_default=True, required=False)
@click.option("--seed", type=int, default=42, show_default=True, required=False)
def test(**kwargs):
    opts = TestAldexNetOptions().parse(kwargs,present=True)
    tester = FeatureSaver(opts)
    tester.test()

if __name__ == "__main__":
    test()
