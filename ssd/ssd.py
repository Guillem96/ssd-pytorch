import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as zoo

from .nn import L2Norm
from .boxes import Detect, PriorBox


_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}

_extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}

_mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


class SSD300(nn.Module):

    def __init__(self, cfg):
        super(SSD300, self).__init__()
        self.cfg = cfg
        self.size = 300
        self.num_classes = cfg['num-classes']
        
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()

        # SSD network
        base = _vgg(_base[str(self.size)], 3)
        extras = _add_extras(_extras[str(self.size)], 1024)
        loc_head, conf_head = _multibox(base, 
                                        extras, 
                                        _mbox[str(self.size)], 
                                        self.num_classes)

        self.vgg = nn.ModuleList(base)

        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(loc_head)
        self.conf = nn.ModuleList(conf_head)

        self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)

        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def _post_process_inference(self, detections):
        # detections: [BATCH, CLASSES, PRIORS, COORDS + SCORE]
        # scores: [BATCH, PRIORS, CLASSES]
        scores = detections[..., 0].permute(0, 2, 1)
        
        # boxes: [BATCH, PRIORS, CLASSES, COORDS]
        boxes = detections[..., 1:].permute(0, 2, 1, 3)

        # scores: [BATCH, PRIORS]
        # classes: [BATCH, PRIORS]
        scores, classes = scores.max(-1)
        
        if torch.jit.is_tracing():
            # TODO: Build an indexer
            boxes = torch.stack([b[torch.arange(len(b)), l] for b, l in zip(boxes, classes)])
            return boxes, classes, scores

        return [dict(scores=s, boxes=b[torch.arange(len(s)), l], labels=l) 
                for s, b, l in zip(scores, boxes, classes)]

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch, 3, 300, 300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        device = x.device
        sources = []

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        loc = [l(x).permute(0, 2, 3, 1).contiguous() 
               for x, l in zip(sources, self.loc)]
        conf = [c(x).permute(0, 2, 3, 1).contiguous() 
                for x, c in zip(sources, self.conf)]

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        if not self.training:
            conf = F.softmax(conf, -1)
            output = self.detect(loc, conf, self.priors.float().to(device))
            output = self._post_process_inference(output)
        else:
            output = loc, conf, self.priors.to(device)

        return output


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def _vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    layers += [pool5, conv6, nn.ReLU(inplace=True), 
               conv7, nn.ReLU(inplace=True)]
    return layers


def _add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def _multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]

    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]

    return loc_layers, conf_layers
