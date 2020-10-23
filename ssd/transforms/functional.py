import cv2

import torch
import numpy as np


MEANS = (104, 117, 123)


def caffe_preprocessing(image, size, mean=MEANS):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= np.array(mean)
    x = x.astype(np.float32)
    return x


def np_to_tensor(image):
    return torch.from_numpy(image).permute(2, 0, 1)
