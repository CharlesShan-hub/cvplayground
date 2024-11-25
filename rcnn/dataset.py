from clib.dataset import Flowers17
from clib.algorithms.selective_search import selective_search
from clib.utils import path_to_rgb, to_image
from utils import crop_and_filter_regions

import torch
from typing import Callable, Optional
import numpy as np
from random import random
from tqdm import tqdm

class Flowers2(Flowers17):
    def __init__(self,
        root: str, 
        image_size: int,
        threshold: float = 0.5,
        sigma: float = 0.9,
        scale: int = 500,
        min_size: int = 10,
        reset_patch: bool = False, 
        transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(root,transform=transform,download=download)

        self.names = {
            0: 'Background',  
            1: 'Tulip',       # 561
            2: 'Pansy',       # 1281
        }
        self._set_box_dict()
        self.image_size = image_size
        self.threshold = threshold
        self.sigma = sigma
        self.scale = scale
        self.min_size = min_size
        self._patch_folder = self._base_folder / 'patches'
        self.transform = transform
        self.images = []
        self.labels = []
        self.image_names = []
        self.rects = []
        
        if (self._patch_folder.exists()==False) or (reset_patch):
            self.save_to_numpy() 
        self.load_from_npy()
        

    def save_to_numpy(self):
        self._patch_folder.mkdir()

        pbar = tqdm(self.d.items(), total=len(self.d.items()))
        for (key, box) in pbar:
            img = path_to_rgb(self._images_folder / f"image_{key:04d}.jpg")
            _, regions = selective_search(img, scale=self.scale, sigma=self.sigma, min_size=self.min_size)
            images,labels,rects = crop_and_filter_regions(img, key, regions, box, self.image_size, self.threshold)
            np.save(self._patch_folder / f'image_{key}.npy', images)
            np.save(self._patch_folder / f'label_{key}.npy', labels)
            np.save(self._patch_folder / f'rects_{key}.npy', rects)
    
    def load_from_npy(self):
        for key in self.d:
            i = np.load(self._patch_folder / f'image_{key}.npy')
            l = np.load(self._patch_folder / f'label_{key}.npy')
            r = np.load(self._patch_folder / f'rects_{key}.npy')
            for n in range(i.shape[0]):
                if l[n] == 0:
                    if random() < 0.95:
                        continue
                self.images.extend([i[n,:,:,:], i[n,::-1,:,:], i[n,:,::-1,:]])
                self.labels.extend([l[n],l[n],l[n]])  
                self.image_names.extend([key,key,key])  
                self.rects.extend([r[n,:],r[n,:],r[n,:]])
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image, label = to_image(self.images[idx]), self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label, self.image_names[idx], self.rects[idx]
    
    def _set_box_dict(self):
        self.d = {
            561:[90,126,350,434],
            562:[160,15,340,415],
            563:[42,25,408,405],
            564:[66,48,384,318],
            565:[17,208,363,354],
            566:[42,20,398,310],
            567:[40,60,410,290],
            568:[40,60,360,380],
            573:[140,80,360,360],
            574:[50,80,360,360],
            575:[140,80,400,350],
            576:[140,80,400,350],
            577:[100,200,400,200],
            578:[200,100,380,160],
            579:[20,180,520,180],
            580:[20,10,420,450],
            581:[60,100,400,300],
            582:[152,35,398,435],
            583:[40,45,380,315],
            584:[40,45,410,395],
            608:[200,40,200,250],
            610:[180,105,300,300],
            613:[180,105,150,150],
            617:[70,120,210,280],
            627:[90,80,230,150],
            630:[270,140,165,108],
            633:[140,200,120,120],
            634:[220,180,220,150],
            636:[220,88,200,232],
            637:[290,42,110,158],
            1281:[90,66,330,374],
            1282:[90,35,330,425],
            1283:[90,45,316,435],
            1284:[26,16,408,574],
            1285:[50,55,375,420],
            1286:[40,63,347,487],
            1287:[60,80,340,360],
            1288:[34,27,403,535],
            1289:[40,60,360,380],
            1290:[36,5,399,525],
            1291:[40,105,298,371],
            1292:[40,60,362,460],
            1293:[40,20,394,512],
            1294:[84,30,329,470],
            1295:[93,92,330,420],
            1296:[64,250,379,466],
            1297:[106,40,277,392],
            1298:[72,60,348,540],
            1305:[50,17,350,543],
            1306:[103,100,308,379],
            1307:[27,28,403,572],
            1314:[30,14,420,516],
            1315:[38,80,332,426],
            1316:[72,32,354,568],
            1317:[86,70,264,421],
            1318:[95,110,285,430],
            1319:[111,50,273,405],
            1323:[16,122,465,502],
            1324:[40,60,380,440],
            1325:[114,37,242,356],
        }


class Flowers2_Box(Flowers2):
    def __init__(self,
        root: str,
        image_size: int,
        threshold: float = 0.5,
        transform: Optional[Callable] = None,
        ):
        # Has already load from numpy
        super().__init__(root,image_size,threshold,transform=transform)
        # Replace images to features
        self.features = np.load(self._base_folder / f'feature_for_svm.npy')
        
    def __getitem__(self, idx: int):
        label = self.labels[idx]
        rects = torch.tensor(self.rects[idx]) / self.image_size
        rects[0] *= self.image_size
        return self.features[idx,:], label, self.image_names[idx], rects
    
    