import click
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import TestOptions
from dataset import ImageNetVal
from transform import transform
import model as models
from cslib.inference import BaseInferencer
from cslib.utils import glance


class VggTester(BaseInferencer):
    def __init__(self, opts):
        super().__init__(opts)

        if opts.model_name == "vgg11":
            self.model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1,progress=True, model_path=opts.model_path)
        elif opts.model_name == "vgg11_bn":
            self.model = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1,progress=True, model_path=opts.model_path)
        elif opts.model_name == "vgg13":
            self.model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1,progress=True, model_path=opts.model_path)
        elif opts.model_name == "vgg13_bn":
            self.model = models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1,progress=True, model_path=opts.model_path)
        elif opts.model_name == "vgg16":
            self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1,progress=True, model_path=opts.model_path)
        elif opts.model_name == "vgg16_bn":
            self.model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1,progress=True, model_path=opts.model_path)
        elif opts.model_name == "vgg19":
            self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1,progress=True, model_path=opts.model_path)
        elif opts.model_name == "vgg19_bn":
            self.model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1,progress=True, model_path=opts.model_path)
        else:
            raise ValueError(f"Invalid model name: {opts.model_name}")
        self.model = self.model.to(opts.device)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.transform = transform

        dataset = ImageNetVal(
            root=opts.dataset_path, transform=self.transform, download=True
        )

        self.test_loader = DataLoader(dataset=dataset, batch_size=opts.batch_size, shuffle=False)
    
    def test(self) -> None:
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
@click.option("--model_name", type=str, default="", show_default=False)
@click.option("--batch_size", type=int, default=8, show_default=True, required=False)
@click.option("--comment", type=str, default="", show_default=False)
def test(**kwargs):
    opts = TestOptions().parse(kwargs,present=True)
    tester = VggTester(opts)
    tester.test()


if __name__ == "__main__":
    test()
