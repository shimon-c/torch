import numpy as np
import cv2
def read_color_img(imgname, to_rgb=True):
    img = cv2.imread(imgname, cv2.IMREAD_COLOR)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def convert_to_ternsor(img):
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    return img