import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets.voc import VOCDetection
import click
from config import DisplayOptions
from utils import plot_boxes,get_bounding_boxes,scale_bbox_coord

class YoloPascalVocDataset(VOCDetection):
    def __init__(self, set_type, normalize, augment, DATA_PATH, IMAGE_SIZE, S, B, C):
        assert set_type in {'train', 'test'}
        super(YoloPascalVocDataset, self).__init__(
            root=DATA_PATH, 
            year='2007',
            image_set=('train' if set_type == 'train' else 'val'),
            download=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Resize(IMAGE_SIZE)
            ])
        )

        self.normalize = normalize
        self.augment = augment
        self.IMAGE_SIZE = IMAGE_SIZE
        self.S = S
        self.B = B
        self.C = C
        self.grid_size_x = IMAGE_SIZE[0] / self.S
        self.grid_size_y = IMAGE_SIZE[1] / self.S
        self.classes = {
            "car": 0,
            "person": 1,
            "horse": 2,
            "bicycle": 3,
            "aeroplane": 4,
            "train": 5,
            "diningtable": 6,
            "dog": 7,
            "chair": 8,
            "cat": 9,
            "bird": 10,
            "boat": 11,
            "pottedplant": 12,
            "tvmonitor": 13,
            "sofa": 14,
            "motorbike": 15,
            "bottle": 16,
            "bus": 17,
            "sheep": 18,
            "cow": 19
        }

    def __getitem__(self, i):
        data, label = super(YoloPascalVocDataset, self).__getitem__(i)
        original_data = data.clone()  # 保持原始数据不变

        # 数据增强和归一化
        if self.augment:
            # Apply augmentation transformations
            x_shift = int((0.2 * torch.rand(1) - 0.1) * self.IMAGE_SIZE[0])
            y_shift = int((0.2 * torch.rand(1) - 0.1) * self.IMAGE_SIZE[1])
            scale = 1 + 0.2 * torch.rand(1)

            data = TF.affine(data, angle=0.0, scale=scale.item(), translate=(x_shift, y_shift), shear=0.0)
            data = TF.adjust_hue(data, 0.2 * torch.rand(1) - 0.1)
            data = TF.adjust_saturation(data, 0.2 * torch.rand(1) + 0.9)

        if self.normalize:
            data = TF.normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Resize image to desired size
        # data = TF.resize(data, self.IMAGE_SIZE)

        grid_size_x = data.size(dim=2) / self.S  # Images in PyTorch have size (channels, height, width)
        grid_size_y = data.size(dim=1) / self.S
        # Process bounding boxes into the SxSx(5*B+C) ground truth tensor
        boxes = {}
        class_names = {}                    # Track what class each grid cell has been assigned to
        depth = 5 * self.B + self.C     # 5 numbers per bbox, then one-hot encoding of label
        ground_truth = torch.zeros((self.S, self.S, depth))
        for j, bbox_pair in enumerate(get_bounding_boxes(label,self.IMAGE_SIZE)):
            name, coords = bbox_pair
            assert name in self.classes, f"Unrecognized class '{name}'"
            class_index = self.classes[name]
            x_min, x_max, y_min, y_max = coords

            # Augment labels
            if self.augment:
                half_width = self.IMAGE_SIZE[0] / 2
                half_height = self.IMAGE_SIZE[1] / 2
                x_min = scale_bbox_coord(x_min, half_width, scale) + x_shift
                x_max = scale_bbox_coord(x_max, half_width, scale) + x_shift
                y_min = scale_bbox_coord(y_min, half_height, scale) + y_shift
                y_max = scale_bbox_coord(y_max, half_height, scale) + y_shift

            # Calculate the position of center of bounding box
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            col = int(mid_x // grid_size_x)
            row = int(mid_y // grid_size_y)

            if 0 <= col < self.S and 0 <= row < self.S:
                cell = (row, col)
                if cell not in class_names or name == class_names[cell]:
                    # Insert class one-hot encoding into ground truth
                    one_hot = torch.zeros(self.C)
                    one_hot[class_index] = 1.0
                    ground_truth[row, col, :self.C] = one_hot
                    class_names[cell] = name

                    # Insert bounding box into ground truth tensor
                    bbox_index = boxes.get(cell, 0)
                    if bbox_index < self.B:
                        bbox_truth = (
                            (mid_x - col * grid_size_x) / self.IMAGE_SIZE[0],     # X coord relative to grid square
                            (mid_y - row * grid_size_y) / self.IMAGE_SIZE[1],     # Y coord relative to grid square
                            (x_max - x_min) / self.IMAGE_SIZE[0],                 # Width
                            (y_max - y_min) / self.IMAGE_SIZE[1],                 # Height
                            1.0                                                     # Confidence
                        )

                        # Fill all bbox slots with current bbox (starting from current bbox slot, avoid overriding prev)
                        # This prevents having "dead" boxes (zeros) at the end, which messes up IOU loss calculations
                        bbox_start = 5 * bbox_index + self.C
                        ground_truth[row, col, bbox_start:] = torch.tensor(bbox_truth).repeat(self.B - bbox_index)
                        boxes[cell] = bbox_index + 1


        return data, ground_truth, original_data

@click.command()
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--normalize", type=bool, default=False, show_default=True)
@click.option("--augment", type=bool, default=False, show_default=True)
@click.option("--width", type=int, default=448, show_default=True)
@click.option("--height", type=int, default=448, show_default=True)
@click.option("--S", type=int, default=7, show_default=True)
@click.option("--B", type=int, default=2, show_default=True)
@click.option("--C", type=int, default=20, show_default=True)
def test(**kwargs):
    kwargs['image_size'] = (kwargs['width'],kwargs['height'])
    del kwargs['width']
    del kwargs['height']
    opts = DisplayOptions().parse(kwargs,present=True)
    # Display data
    train_set = YoloPascalVocDataset('train', opts.normalize, opts.augment,\
        opts.dataset_path, opts.image_size, opts.S, opts.B, opts.C)

    negative_labels = 0
    smallest = 0
    largest = 0
    for data, label, datao in train_set:
        negative_labels += torch.sum(label < 0).item()
        smallest = min(smallest, torch.min(data).item())
        largest = max(largest, torch.max(data).item())
        plot_boxes(data, label, train_set.classes, max_overlap=float('inf'))
        plot_boxes(datao, label, train_set.classes, max_overlap=float('inf'))
        break
    print('num_negatives', negative_labels)
    print('dist', smallest, largest)

if __name__ == "__main__":
    test()
