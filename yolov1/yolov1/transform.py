import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import normalize as torch_normalize
import random
import numpy as np

def transform(augment: bool, normalize: bool, width:int, height:int):
    trans_list = [A.Resize(width=width, height=height)]
    if augment:
        trans_list.extend([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=1),
            A.HueSaturationValue( # only support np.float32! not np.float64!
                hue_shift_limit=0.2 * random.random() - 0.1,  # Randomize hue shift
                sat_shift_limit=0.2 * random.random() + 0.9,  # Randomize saturation shift
                val_shift_limit=0,  # Optionally add value shift if needed
                p=1  # Apply the transformation with probability 1
            )
        ])
    trans_list.append(ToTensorV2())
    if normalize:
        trans_list.append(A.Lambda(image=lambda img, **kwargs: torch_normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )))
    return A.Compose(
        trans_list,
        bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.5, label_fields=['category_ids'])
    )
