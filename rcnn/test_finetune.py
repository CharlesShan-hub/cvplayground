import click
import torch
from torch.utils.data import DataLoader, random_split
from config import TestAldexNetOptions
from dataset import Flowers2
from transform import transform
from model import AlexNet
from clib.inference import BaseInferencer
# from clib.utils import glance

class FinetuneTester(BaseInferencer):
    def __init__(self, opts):
        super().__init__(opts)

        self.model = AlexNet(
            num_classes=opts.num_classes,
            classify=True,
            save_feature=False
        ).to(opts.device)
        self.load_checkpoint()

        self.criterion = torch.nn.CrossEntropyLoss()

        self.transform = transform(opts.image_size)

        dataset = Flowers2(
            root=opts.dataset_path,
            image_size=opts.image_size,
            transform=self.transform
        )

        val_size = int(opts.val_size * len(dataset))
        test_size = int(opts.test_size * len(dataset))
        train_size = len(dataset) - val_size - test_size
        _, _, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(opts.seed),
        )

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=opts.batch_size,
            shuffle=False
        )

    def test(self):
        assert self.model is not None
        assert self.test_loader is not None
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels, _ , _  in self.test_loader:
                images = images.to(self.opts.device)
                labels = labels.to(self.opts.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # glance(images,title=labels)
                # print('---------------------------------')
                # print(labels)
                # print(predicted)

            print(
                f"Accuracy of the model on the {total} test images: {100 * correct / total:.2f}%"
            )

@click.command()
@click.option("--model_path", type=click.Path(exists=True), required=True)
@click.option("--comment", type=str, default="", show_default=False)
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--num_classes", type=int, default=10, show_default=True)
@click.option("--image_size", type=int, default=224, show_default=True)
@click.option("--batch_size", type=int, default=8, show_default=True, required=False)
@click.option("--val_size", type=float, default=0.2, show_default=True, required=False)
@click.option("--test_size", type=float, default=0.2, show_default=True, required=False)
@click.option("--seed", type=int, default=42, show_default=True, required=False)
def test(**kwargs):
    opts = TestAldexNetOptions().parse(kwargs,present=True)
    tester = FinetuneTester(opts)
    tester.test()

if __name__ == "__main__":
    test()
