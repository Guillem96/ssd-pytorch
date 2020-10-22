# SSD: Single Shot MultiBox Object Detector, in PyTorch

Modern PyTorch SSD implementation.

This implementation is completely based from the one made by [amdegroot](https://github.com/amdegroot/ssd.pytorch), therefore all kudos for him.


## Training SSD
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              
https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth


```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python -m ssd train <options>
```