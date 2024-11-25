import click
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transform import transform
from model import LeNet
from config import TrainOptions
from clib.train import BaseTrainer


class LeNetTrainer(BaseTrainer):
    def __init__(self, opts):
        super().__init__(opts)

        self.model = LeNet(
            num_classes=opts.num_classes,
            use_max_pool=opts.use_max_pool,
            use_relu=opts.use_relu,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.opts.optimizer = "SGD"
        self.optimizer = optim.SGD(
            params=self.model.parameters(), 
            lr=opts.lr
        )

        self.opts.lr_scheduler = "ReduceLROnPlateau"
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, 
            mode='max', 
            factor=opts.factor, 
            patience=2
        )

        self.transform = transform

        train_dataset = datasets.MNIST(
            root=opts.dataset_path, train=True, download=True, transform=self.transform
        )
        test_dataset = datasets.MNIST(
            root=opts.dataset_path, train=False, download=True, transform=self.transform
        )
        val_size = int(opts.val * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(opts.seed),
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opts.batch_size,
            shuffle=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=opts.batch_size,
            shuffle=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=opts.batch_size,
            shuffle=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )

    def holdout_train(self, epoch):
        assert self.optimizer is not None
        assert self.model is not None
        assert self.criterion is not None
        assert self.train_loader is not None
        pbar = tqdm(self.train_loader, total=len(self.train_loader))
        running_loss = torch.tensor(0.0).to(self.opts.device)
        batch_index = 0
        correct = total = 0
        self.model.train()
        for images, labels in pbar:
            images = images.to(self.opts.device)
            labels = labels.to(self.opts.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            batch_index += 1
            running_loss += loss
            pbar.set_description(
                f"Epoch [{epoch}/{self.opts.max_epoch if self.opts.max_epoch != 0 else '∞'}]"
            )
            pbar.set_postfix(loss=(running_loss.item() / batch_index))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_accuracy = correct / total
        self.loss = loss

        self.writer.add_scalar(
            tag = "lr", 
            scalar_value = self.get_lr(), 
            global_step = epoch
        )
        self.writer.add_scalar(
            tag = "Loss/train", 
            scalar_value = train_loss, 
            global_step = epoch
        )
        self.writer.add_scalar(
            tag = "Accuracy/train", 
            scalar_value = train_accuracy, 
            global_step = epoch
        )

        print(f"Epoch [{epoch}/{self.opts.max_epoch if self.opts.max_epoch != 0 else '∞'}]", \
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        
        return train_loss

    def holdout_validate(self,epoch):
        assert self.model is not None
        assert self.criterion is not None
        assert self.val_loader is not None
        assert self.scheduler is not None
        running_loss = torch.tensor(0.0).to(self.opts.device)
        correct = total = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.opts.device)
                labels = labels.to(self.opts.device)
                outputs = self.model(images)
                running_loss += self.criterion(outputs, labels)
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / total
        val_accuracy = float(correct / total)
        assert isinstance(self.scheduler, ReduceLROnPlateau)
        self.scheduler.step(metrics=val_accuracy)
    
        self.writer.add_scalar(
            tag="Loss/val", 
            scalar_value=val_loss, 
            global_step=epoch
        )
        self.writer.add_scalar(
            tag = "Accuracy/val", 
            scalar_value = val_accuracy, 
            global_step = epoch
        )

        print(f"Epoch [{epoch}/{self.opts.max_epoch if self.opts.max_epoch != 0 else '∞'}]", \
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        return val_loss
    
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
@click.option("--comment", type=str, default="", show_default=False)
@click.option("--model_base_path", type=click.Path(exists=True), required=True)
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--num_classes", type=int, default=10, show_default=True)
@click.option("--use_relu", type=bool, default=False, show_default=True)
@click.option("--use_max_pool", type=bool, default=False, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True, required=False)
@click.option("--batch_size", type=int, default=8, show_default=True, required=False)
@click.option("--lr", type=float, default=0.03, show_default=True, required=False)
@click.option("--max_epoch", type=int, default=100, show_default=True, required=False)
@click.option("--max_reduce", type=int, default=6, show_default=True, required=False)
@click.option("--factor", type=float, default=0.1, show_default=True, required=False)
@click.option("--train_mode", type=str, default="Holdout", show_default=False)
@click.option("--val", type=float, default=0.2, show_default=True, required=False)
def train(**kwargs):
    opts = TrainOptions().parse(kwargs)
    trainer = LeNetTrainer(opts)
    trainer.train()


if __name__ == "__main__":
    train()
