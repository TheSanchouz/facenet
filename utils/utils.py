import cv2
import numpy as np


def pad_img(img, pad_value, target_dims):
    h, w, _ = img.shape
    pads = []
    pads.append(int(np.floor((target_dims[0] - h) / 2.0)))
    pads.append(int(np.floor((target_dims[1] - w) / 2.0)))
    pads.append(int(target_dims[0] - h - pads[0]))
    pads.append(int(target_dims[1] - w - pads[1]))
    padded_img = cv2.copyMakeBorder(
        img, pads[0], pads[2], pads[1], pads[3], cv2.BORDER_CONSTANT, value=pad_value,
    )
    return padded_img, pads


def scale_img(img, target_dims):
    height, width, _ = img.shape
    if target_dims[0] / target_dims[1] < height / width:
        scale = target_dims[0] / height
    else:
        scale = target_dims[1] / width

    scaled_img = cv2.resize(
        img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
    )
    return scaled_img, scale
