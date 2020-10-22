# SSD: Single Shot MultiBox Object Detector, in PyTorch

Modern PyTorch SSD implementation.

This implementation is completely based from the one made by [amdegroot](https://github.com/amdegroot/ssd.pytorch), therefore all kudos for him.


## Training SSD

1. Navigate to `models` directory and download all the pretrained models running:

```bash
$ download.sh
```

2. This will download two models, for now, we'll only pay attention to the `vgg16_reducedfc.pth`. This file contains the pretrained weights of the SSD backbone (VGG)

3. Run the training script with your preferred parameters and dataset (By now, this repository supports COCO, VOC or labelme formats).

```$bash
$ python -m ssd train --dataset <labelme|VOC|COCO> \
    --dataset-root path-to-my-data \
    --config <yaml-file-with-prior-boxes-config> \
    --basenet models/vgg16_reducedfc.pth \ # Downloaded in the previous step
    --epochs 8 \
    --batch-size 32 \
    --save-dir data-to-store-checkpoints
```

### Configuration files

To train a model, you need to provide a configuration file to define the prior boxes
configuration and some metadata regarding to the dataset. For some famous datasets
there are already pre-defined configurations in `configs` dir.

In case you are training with a custom dataset, copy and paste a pre-defined configuration
file and modify according your preferences.

A configuration file looks as follows:

```yaml
config:
  name: <dataset-name>
  num-classes: <number of classes>
  image-size: 300
  prior-boxes: # Prior boxes configuration (do not touch it if you don't know what you are doing 😀)
    feature-maps: [38, 19, 10, 5, 3, 1]
    min-sizes: [30, 60, 111, 162, 213, 264]
    max-sizes: [60, 111, 162, 213, 264, 315]
    aspect-ratios: [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    steps: [8, 16, 32, 64, 100, 300]
    clip: true
```
