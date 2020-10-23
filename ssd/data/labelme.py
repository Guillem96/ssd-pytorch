import os
import json
from pathlib import Path

import torch
import torch.utils.data as data

import cv2
import numpy as np


class LabelmeAnnotationTransform(object):

    def __init__(self, class2idx):
        self.class2idx = class2idx

    def __call__(self, target, width, height):
        res = []
        for shape in target['shapes']:
            pts = sum(shape['points'], [])
            pts = [pts[0] / width, pts[1] / height,
                   pts[2] / width, pts[3] / height]
            bndbox = pts + [self.class2idx[shape['label']]]
            res.append(bndbox)

        return res 


class LabelmeDataset(data.Dataset):

    def __init__(self, 
                 root,
                 classes,
                 transform=None, 
                 target_transform=None,
                 dataset_name='Labelme'):

        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        self.classes = classes
        self.class_2_idx = {c: i for i, c in enumerate(self.classes)}

        self.annots = list(self.root.glob('*.json'))
        self.images = [Path(str(o).replace('.json', '.jpg')) 
                       for o in self.annots]

        if self.target_transform is None:
            self.target_transform = LabelmeAnnotationTransform(self.class_2_idx)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.annots)

    def pull_item(self, index):
        annot_path = self.annots[index]
        im_path = self.images[index]

        target = json.load(annot_path.open())
        img = cv2.imread(str(im_path))
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        im_path = self.images[index]
        return cv2.imread(str(im_path))

    def pull_anno(self, index):
        annot_path = self.annots[index]
        im_name = annot_path.stem + '.jpg'
        target = json.load(annot_path.open())
        gt = self.target_transform(anno, 1, 1)
        return im_name, gt

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
