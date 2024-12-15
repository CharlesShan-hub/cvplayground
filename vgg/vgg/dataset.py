from clib.dataset import TinyImageNet
from clib.utils import glance
from torch.utils.data import DataLoader
import click

__all__ = ['TinyImageNet']

@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
def main(dataset_path):
    train_dataset = TinyImageNet(
        root = dataset_path,
        split = "train",
        download = True,
    )
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class names (first 10): {train_dataset.classes[:10]}")
    
    for i, (image, label) in enumerate(train_dataset):
        print(f"Sample {i}: Label {label}, Image shape {image.size}")
        glance(image)
        if i == 4:  # 只遍历前5个样本
            break
    
    val_dataset = TinyImageNet(
        root = dataset_path,
        split = "val",
        download = True,
    )
    print(f"Number of classes: {len(val_dataset.classes)}")
    print(f"Class names (first 10): {val_dataset.classes[:10]}")
    
    for i, (image, label) in enumerate(val_dataset):
        print(f"Sample {i}: Label {label}, Image shape {image.size}")
        if i == 4:  # 只遍历前5个样本
            break
    return
    train_loader = DataLoader(train_dataset)
    for item in train_loader:
        breakpoint()
    # glance(train_loader[0][0])


if __name__ == "__main__":
    main()