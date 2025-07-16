import torch
from torchvision.datasets.voc import VOCDetection
import click
from transform import transform
from config import DisplayOptions
from utils import plot_boxes
from cslib.utils import to_numpy
import numpy as np

class YoloPascalVocDataset(VOCDetection):
    def __init__(self, set_type, normalize, augment, data_path, width, height, S, B, C):
        assert set_type in {'train', 'val'}
        super(YoloPascalVocDataset, self).__init__(data_path, year='2007',image_set=set_type,download=True)

        self.classes = [
            "car", "person", "horse", "bicycle", "aeroplane",
            "train", "diningtable", "dog", "chair", "cat",
            "bird", "boat", "pottedplant", "tvmonitor", "sofa", 
            "motorbike", "bottle", "bus", "sheep", "cow"
        ]
        self.transform = transform(augment,normalize,width,height)
        self.width = width
        self.height = height
        self.S = S
        self.B = B
        self.C = C

    def __getitem__(self, i):
        data, label = super(YoloPascalVocDataset, self).__getitem__(i)
        transformed = self.transform(
            image=to_numpy(data).astype(np.float32),
            bboxes=self.get_bounding_boxes(label),
            category_ids=self.get_category_ids(label)
        )
        image = transformed['image']
        bboxes = transformed['bboxes']
        category_ids = transformed['category_ids']

        grid_size_x = image.size(dim=2) / self.S  # Images in PyTorch have size (channels, height, width)
        grid_size_y = image.size(dim=1) / self.S

        # Process bounding boxes into the SxSx(5*B+C) ground truth tensor
        boxes = {}
        class_names = {}                    # Track what class each grid cell has been assigned to
        depth = 5 * self.B + self.C     # 5 numbers per bbox, then one-hot encoding of label
        ground_truth = torch.zeros((self.S, self.S, depth))
        
        for class_index, coords in zip(category_ids,bboxes):
            name = self.classes[int(class_index)]

            # Calculate the position of center of bounding box
            x_min, y_min, x_max, y_max = coords
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            col = int(mid_x // grid_size_x)
            row = int(mid_y // grid_size_y)
            assert(0 <= col < self.S and 0 <= row < self.S)
            cell = (row, col)

            if cell in class_names and name != class_names[cell]:
                continue
            # Insert class one-hot encoding into ground truth
            one_hot = torch.zeros(self.C)
            one_hot[class_index] = 1.0
            ground_truth[row, col, :self.C] = one_hot
            class_names[cell] = name

            # Insert bounding box into ground truth tensor
            bbox_index = boxes.get(cell, 0)
            if bbox_index >= self.B:
                continue
            bbox_truth = (
                (mid_x - col * grid_size_x) / self.width,     # X coord relative to grid square
                (mid_y - row * grid_size_y) / self.height,     # Y coord relative to grid square
                (x_max - x_min) / self.width,                 # Width
                (y_max - y_min) / self.height,                 # Height
                1.0                                                     # Confidence
            )

            # Fill all bbox slots with current bbox (starting from current bbox slot, avoid overriding prev)
            # This prevents having "dead" boxes (zeros) at the end, which messes up IOU loss calculations
            bbox_start = 5 * bbox_index + self.C
            ground_truth[row, col, bbox_start:] = torch.tensor(bbox_truth).repeat(self.B - bbox_index)
            boxes[cell] = bbox_index + 1

        return image, ground_truth
    
    def get_bounding_boxes(self, label):
        boxes = []
        for obj in label['annotation']['object']:
            box = obj['bndbox']
            coords = (
                int(box['xmin']),
                int(box['ymin']),
                int(box['xmax']),
                int(box['ymax'])
            )
            boxes.append(coords)
        
        return boxes
    
    def get_lables(self, label):
        return [i['name'] for i in label['annotation']['object']]
    
    def get_category_ids(self,label):
        return [self.classes.index(i) for i in self.get_lables(label)]

    def get_dimensions(self, label):
        size = label['annotation']['size']
        return int(size['width']), int(size['height'])

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
    opts = DisplayOptions().parse(kwargs,present=True)
    # Display data
    train_set = YoloPascalVocDataset('train', opts.normalize, opts.augment,\
        opts.dataset_path, opts.width, opts.height, opts.S, opts.B, opts.C)

    negative_labels = 0
    smallest = 0
    largest = 0
    for data, label in train_set:
        negative_labels += torch.sum(label < 0).item()
        smallest = min(smallest, torch.min(data).item())
        largest = max(largest, torch.max(data).item())
        plot_boxes(data, label, train_set.classes, opts.S, opts.C, (opts.width,opts.height),max_overlap=float('inf'))
        break
    print('num_negatives', negative_labels)
    print('dist', smallest, largest)

if __name__ == "__main__":
    test()
