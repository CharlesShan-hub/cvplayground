from clib.dataset import ImageNetVal
from clib.utils import glance
import click

__all__ = ['ImageNetVal']

@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
def main(dataset_path):
    # ImageNet 的验证集
    val_dataset = ImageNetVal(
        root = dataset_path,
        download = True,
    )

    print(f"Number of classes: {len(val_dataset.classes)}")
    print(f"Class names (first 10): {val_dataset.classes[:10]}")
    
    for i, (image, label) in enumerate(val_dataset):
        print(f"Sample {i}: Label {label}, Image shape {image.size}")
        glance(image)
        # if i == 4:  # 只遍历前5个样本
        break


if __name__ == "__main__":
    main()