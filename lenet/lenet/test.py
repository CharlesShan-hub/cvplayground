import click
import torch
from torch.utils.data import DataLoader
from config import TestOptions
from dataset import MNIST
from transform import transform
from model import LeNet
from clib.inference import BaseInferencer
from clib.utils import glance


class LeNetTester(BaseInferencer):
    def __init__(self, opts):
        super().__init__(opts)

        self.model = LeNet(
            num_classes=opts.num_classes,
            use_max_pool=opts.use_max_pool,
            use_relu=opts.use_relu
        ).to(opts.device)
        self.load_checkpoint()

        self.criterion = torch.nn.CrossEntropyLoss()

        self.transform = transform(opts.image_size)

        dataset = MNIST(
            root=opts.dataset_path, train=False, transform=self.transform, download=True
        )

        self.test_loader = DataLoader(dataset=dataset, batch_size=opts.batch_size, shuffle=False)
    
    def test(self):
        assert self.model is not None
        assert self.test_loader is not None
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.opts.device)
                labels = labels.to(self.opts.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(
                f"Accuracy of the model on the {total} test images: {100 * correct / total:.2f}%"
            )

@click.command()
@click.option("--model_path", type=click.Path(exists=True), required=True)
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--image_size", type=int, default=28, show_default=True)
@click.option("--batch_size", type=int, default=8, show_default=True, required=False)
@click.option("--num_classes", type=int, default=10, show_default=True)
@click.option("--use_relu", type=bool, default=False, show_default=True)
@click.option("--use_max_pool", type=bool, default=False, show_default=True)
@click.option("--comment", type=str, default="", show_default=False)
def test(**kwargs):
    opts = TestOptions().parse(kwargs,present=True)
    tester = LeNetTester(opts)
    tester.test()


if __name__ == "__main__":
    test()
