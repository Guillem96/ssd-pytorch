import numpy as np
import torch.utils.data


def convert_to_coco_api(ds):
    try:
        from pycocotools.coco import COCO
        from pycocotools import mask as coco_mask
    except ImportError:
        print('Install pycocotools following the instructions defined at:'
                ' https://github.com/cocodataset/cocoapi/tree/master/PythonAPI')
        sys.exit(1)

    coco_ds = COCO()

    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for image_id in range(len(ds)):
        img, targets, h, w = ds.pull_item(image_id)
        bboxes, labels, areas = _parse_targets((h, w), targets)

        img_dict = dict(id=image_id, height=h, width=w)
        dataset['images'].append(img_dict)

        for i in range(len(bboxes)):
            ann = {
                'image_id': image_id,
                'bbox': bboxes[i],
                'category_id': labels[i],
                'area': areas[i],
                'iscrowd': 0,
                'id': ann_id
            }

            categories.add(labels[i])
            dataset['annotations'].append(ann)
            ann_id += 1

    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset

    return convert_to_coco_api(dataset)


def _parse_targets(im_shape, targets):
    targets = np.array(targets)

    scale = np.array([im_shape[1], im_shape[0]] * 2).reshape(-1, 4)

    boxes = targets[..., :4]
    boxes = boxes * scale
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    labels = targets[..., -1].astype('int32')

    areas = boxes[..., 2] * boxes[..., 3]

    return boxes.tolist(), labels.tolist(), areas.tolist()
