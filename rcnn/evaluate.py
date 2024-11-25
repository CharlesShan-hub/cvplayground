from clib.utils import glance, path_to_rgb, to_tensor
from clib.algorithms.selective_search import selective_search
from model import AlexNet, RegNet
from utils import crop_and_filter_regions,show_rect

import numpy as np
import torch
import joblib
device = 'cpu'

INPUT_GT = [90,66,330,374]
INPUT_KEY = 1281
INPUT_PATH = f"/Users/kimshan/resources/DataSets/torchvision/flowers-17/jpg/image_{INPUT_KEY}.jpg"
FINETUNE_PATH = r"/Users/kimshan/resources/DataSets/Model/RCNN/Flowers17/AlexNet_Finetune/checkpoints/51.pt"
SVM1_PATH = r"/Users/kimshan/resources/DataSets/Model/RCNN/Flowers17/SVM/checkpoints/1_svm.pkl"
SVM2_PATH = r"/Users/kimshan/resources/DataSets/Model/RCNN/Flowers17/SVM/checkpoints/2_svm.pkl"
REG_PATH = r"/Users/kimshan/resources/DataSets/Model/RCNN/Flowers17/REG/checkpoints/21.pt"

def test():
    # Load Finetune Model
    finetune_model = AlexNet(
            num_classes=3,
            classify=False,
            save_feature=True
        )
    params = torch.load(
        FINETUNE_PATH, 
        map_location=device,
        weights_only=True
    )
    finetune_model.load_state_dict(params['model_state_dict'])

    # Load SVM Model
    svm1_model = joblib.load(SVM1_PATH)
    svm2_model = joblib.load(SVM2_PATH)

    # Load Reg Model
    reg_model = RegNet()
    params = torch.load(
        REG_PATH, 
        map_location=device,
        weights_only=True
    )
    reg_model.load_state_dict(params['model_state_dict'])


    # Get Input picture
    image = path_to_rgb(INPUT_PATH)
    # glance(image)

    # Selected Search
    _, regions = selective_search(image, scale=500, sigma=0.9, min_size=10)
    images,labels,rects = crop_and_filter_regions(image, INPUT_KEY, regions, INPUT_GT, 224, 0.5)
    # glance(images,shape=(2,29),title=labels)

    # Get Features
    images_tensor = torch.stack([to_tensor(i) for i in images])
    features = finetune_model(images_tensor)

    # 
    results = []
    results_old = []
    results_label = []
    flower = {1:'pancy', 2:'Tulip'}

    for i in range(len(images)):
        for svm in [svm1_model,svm2_model]:
            feature = features[i,:].data.cpu().numpy()
            pred = svm.predict([feature.tolist()])
            if pred[0] == 0:
                continue
            if rects[i][0] == 0:
                continue
            box = reg_model(features[i,:]).data.cpu().numpy()
            # if box[0]<0.3:
            if box[0]<0.8:
                continue
            (_, bx, by, bw, bh) = box # Eg. box [0.9368, 0.1490, 0.1342, 0.9999, 1.0000]
            (_, px, py, pw, ph) = rects[i] # Eg. rects[i] [1, 60, 15, 403, 368]
            new_x = px + pw*bx
            new_y = py + ph*by
            new_w = bw*pw
            new_h = bh*ph
            results.append([new_x, new_y , new_w, new_h])
            results_label.append(pred[0])
            # print(box)
            # print(rects[i])
            # print([new_x+new_w/2, new_y+new_h/2, new_w, new_h])
            # show_rect(image, [rects[i][1:]], flower[pred[0]])

    average_center_x, average_center_y, average_w,average_h = 0, 0, 0, 0
    #use average values to represent the final result
    for vert in results:
        average_center_x += vert[0]
        average_center_y += vert[1]
        average_w += vert[2]
        average_h += vert[3]
    average_center_x = average_center_x / len(results)
    average_center_y = average_center_y / len(results)
    average_w = average_w / len(results)
    average_h = average_h / len(results)
    average_result = [[average_center_x, average_center_y, average_w, average_h]]
    result_label = max(results_label, key=results_label.count)
    print(result_label,average_result)
    # show_rect(image, results, ' ')
    show_rect(image, average_result, flower[result_label])
    # show_rect(image, [INPUT_GT], flower[result_label])

if __name__ == "__main__":
    test()