import click
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch import nn, optim
from torch.autograd import Variable

from config import TrainBoxOptions
from model import RegNet
from dataset import Flowers2_Box
from clib.train import BaseTrainer

class Regloss(nn.Module):
    def __init__(self):
        super(Regloss, self).__init__()
    
    def forward(self, y_true, y_pred):
        no_object_loss = torch.pow((1 - y_true[:, 0]) * y_pred[:, 0],2).mean()
        object_loss = torch.pow((y_true[:, 0]) * (y_pred[:, 0] - 1),2).mean()

        reg_loss = (y_true[:, 0] * (torch.pow(y_true[:, 1:5] - y_pred[:, 1:5],2).sum(1))).mean()    
        
        loss = no_object_loss + object_loss + reg_loss
        return loss
    
class BoxTrainer(BaseTrainer):
    def __init__(self, opts):
        super().__init__(opts)

        self.model = RegNet().to(opts.device)

        self.criterion = Regloss().to(opts.device)

        self.opts.optimizer = "SGD"
        self.optimizer = optim.SGD(
            params=self.model.parameters(), 
            lr=opts.lr
        )

        self.opts.lr_scheduler = "ReduceLROnPlateau"
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, 
            mode='min', 
            factor=opts.factor, 
            patience=2
        )

        dataset = Flowers2_Box(
            root=opts.dataset_path,
            image_size=opts.image_size
        )

        val_size = int(opts.val_size * len(dataset))
        test_size = int(opts.test_size * len(dataset))
        train_size = len(dataset) - val_size - test_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
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
        total = 0
        self.model.train()
        for features, _, _ , rects  in pbar:
            features = features.to(self.opts.device)
            rects = Variable(rects).to(self.opts.device)
            outputs = self.model(features)
            loss = self.criterion(rects, outputs)
            total += rects.size(0)
            loss.backward()
            self.optimizer.step()

            # print(outputs)
            # print(rects)
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.norm()
            #         print(f'Parameter: {name}, Gradient Norm: {grad_norm}')
            #         if grad_norm > 1e6:  # 这里的阈值是示例，实际情况可能需要调整
            #             print(f'Gradient explosion detected for {name}')

            self.optimizer.zero_grad()
            batch_index += 1
            running_loss += loss
            pbar.set_description(
                f"Epoch [{epoch}/{self.opts.max_epoch if self.opts.max_epoch != 0 else '∞'}]"
            )
            pbar.set_postfix(loss=(running_loss.item() / batch_index))
            total += rects.size(0)


        train_loss = running_loss / total
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

        print(f"Epoch [{epoch}/{self.opts.max_epoch if self.opts.max_epoch != 0 else '∞'}]", \
              f"Train Loss: {train_loss:.4f}")
        
        return train_loss

    def holdout_validate(self,epoch):
        assert self.model is not None
        assert self.criterion is not None
        assert self.val_loader is not None
        assert self.scheduler is not None
        running_loss = torch.tensor(0.0).to(self.opts.device)
        total = 0
        self.model.eval()
        for features, _, _ , rects  in self.val_loader:
            features = features.to(self.opts.device)
            rects = Variable(rects).to(self.opts.device)
            outputs = self.model(features)
            running_loss += self.criterion(rects, outputs)
            # print("------------")
            # print(outputs)
            # print(rects)
            total += rects.size(0)

        val_loss = running_loss / total
        assert isinstance(self.scheduler, ReduceLROnPlateau)
        self.scheduler.step(metrics=val_loss)
    
        self.writer.add_scalar(
            tag="Loss/val", 
            scalar_value=val_loss, 
            global_step=epoch
        )

        print(f"Epoch [{epoch}/{self.opts.max_epoch if self.opts.max_epoch != 0 else '∞'}]", \
              f"Val Loss: {val_loss:.4f}")

        return val_loss
    
    def test(self):
        assert self.model is not None
        assert self.test_loader is not None
        assert self.criterion is not None
        self.model.eval()
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for features, _, _ , rects  in self.test_loader:
                features = features.to(self.opts.device)
                rects = rects.to(self.opts.device)
                outputs = self.model(features)
                running_loss += self.criterion(rects, outputs)
                total += rects.size(0)
                running_loss/=total
            print(
                f"Loss of the model on the {total} test images: {running_loss:.4f}"
            )

@click.command()
@click.option("--comment", type=str, default="", show_default=False)
@click.option("--model_base_path", type=click.Path(exists=True), required=True)
@click.option("--dataset_path", type=click.Path(exists=True), required=True)

@click.option("--num_classes", type=int, default=3, show_default=True)
@click.option("--image_size", type=int, default=224, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True, required=False)
@click.option("--batch_size", type=int, default=8, show_default=True, required=False)
@click.option("--lr", type=float, default=0.03, show_default=True, required=False)
@click.option("--max_epoch", type=int, default=100, show_default=True, required=False)
@click.option("--max_reduce", type=int, default=6, show_default=True, required=False)
@click.option("--factor", type=float, default=0.1, show_default=True, required=False)
@click.option("--cooldown", type=float, default=5, show_default=True, required=False)
@click.option("--train_mode", type=str, default="Holdout", show_default=False)
@click.option("--val_size", type=float, default=0.2, show_default=True, required=False)
@click.option("--test_size", type=float, default=0.2, show_default=True, required=False)
def train(**kwargs):
    opts = TrainBoxOptions().parse(kwargs)
    trainer = BoxTrainer(opts)
    trainer.train()

if __name__ == "__main__":
    train()
