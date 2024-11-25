from skimage.transform import resize
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_rect(img, regions, message):
    '''
    :param img_path: input image
    :param regions: rects
    :param message: annotation
    :return: 
    '''
    for x, y, w, h in regions:
        x, y, w, h =int(x),int(y),int(w),int(h)
        rect = cv2.rectangle(
            img,(x, y), (x+w, y+h), (0,255,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, message, (x+20, y+40),font, 1,(255,0,0),2)
    plt.imshow(img)
    plt.show()


def crop_and_filter_regions(image, key, regions, box, image_size, iou_threshold):
    """ Rect Selection

    Args:
        image (np.ndarray): Origin Image
        key (int): image ID, to calculate image catgray
        regions (list): Regions before Selection
        box (tuple): Ground Truth of box in image
        image_size (int): image size

    """
    images = []
    labels = []
    rects = []
    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding small regions
        if r['size'] < 220:
            continue
        if (r['rect'][2] * r['rect'][3]) < 500:
            continue
        # resize to 227 * 227 for input
        proposal_img, proposal_vertice = clip_pic(image, r['rect'])
        # Delete Empty array
        if len(proposal_img) == 0:
            continue
        # Ignore things contain 0 or not C contiguous array
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        # Check if any 0-dimension exist
        [a, b, c] = proposal_img.shape
        if a == 0 or b == 0 or c == 0:
            continue
        resized_proposal_img = resize(
            proposal_img, 
            (image_size, image_size), 
            mode='reflect', 
            anti_aliasing=True
        )
        candidates.add(r['rect'])
        img_float = np.asarray(resized_proposal_img, dtype="float32")
        images.append(img_float)
        # IOU
        iou_val = IOU(box, proposal_vertice)
        # labels, let 0 represent default class, which is background
        index = 1 if key > 1280 else 2 # 1 and 2 is two kind of flowers
        if iou_val < iou_threshold:
            rects.append([0]+list(r['rect']))
            labels.append(0)
        else:
            rects.append([1]+list(r['rect']))
            labels.append(index)
    
    return images, labels, rects

def clip_pic(img, rect):
    [x, y, w, h] = rect
    x_1 = x + w
    y_1 = y + h
    # return img[x:x_1, y:y_1, :], [x, y, x_1, y_1, w, h]   
    return img[y:y_1, x:x_1, :], [x, y, x_1, y_1, w, h]

def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return if_intersect
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter

def IOU(ver1, vertice2):
    # vertices in four points
    vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False