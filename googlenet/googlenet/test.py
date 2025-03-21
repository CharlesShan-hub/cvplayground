import click
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import TestOptions
from dataset import ImageNetVal
from transform import transform
from model import googlenet, GoogLeNet_Weights
from cslib.inference import BaseInferencer
from cslib.utils import glance


class GoogLeNetTester(BaseInferencer):
    def __init__(self, opts):
        super().__init__(opts)

        self.model = googlenet(weights=GoogLeNet_Weights,progress=True, model_path=opts.model_path).to(opts.device)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.transform = transform

        dataset = ImageNetVal(
            root=opts.dataset_path, transform=self.transform, download=True
        )

        self.test_loader = DataLoader(dataset=dataset, batch_size=opts.batch_size, shuffle=False)
    
    def test(self):
        assert self.model is not None
        assert self.test_loader is not None
        self.model.eval()
        pbar = tqdm(self.test_loader, total=len(self.test_loader))
        correct = total = 0
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.opts.device)
                labels = labels.to(self.opts.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix(acc=(correct / total))

            print(
                f"Accuracy of the model on the {total} test images: {100 * correct / total:.2f}%"
            )

@click.command()
@click.option("--model_path", type=click.Path(exists=True), required=True)
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--batch_size", type=int, default=8, show_default=True, required=False)
@click.option("--comment", type=str, default="", show_default=False)
def test(**kwargs):
    opts = TestOptions().parse(kwargs,present=True)
    tester = GoogLeNetTester(opts)
    tester.test()


if __name__ == "__main__":
    test()
