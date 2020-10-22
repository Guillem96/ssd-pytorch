import click
import cv2

import torch

import ssd
import ssd.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@click.command()
@click.argument('im_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-c', '--checkpoint', required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--config',
              required=True, type=click.Path(exists=True, dir_okay=False))
def inference(im_path, checkpoint, config):

    cfg = yaml.safe_load(open(config))
    idx_to_class = ssd.data.VOC_CLASSES

    im = cv2.imread(im_path)
    im_in = T.get_transforms(cfg['image-size'], inference=True)(im)
    im_in = im_in.unsqueeze(0).to(device)

    model = ssd.ssd(cfg, cfg['image-size'], cfg['num-classes'])
    model.eval()

    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    with torch.no_grad():
        detections = model(im_in)[0]

    scores = detections[..., 0].t()
    boxes = detections[..., 1:].permute(1, 0, 2)

    scale = torch.as_tensor([im.shape[1], im.shape[0]] * 2, device=device)
    scale.unsqueeze_(0)

    scores, classes = scores.max(-1)
    true_mask = scores > .5
    scores = scores[true_mask]
    
    classes = classes[true_mask]
    names = [idx_to_class[i - 1] for i in classes.cpu().tolist()]
    boxes = boxes[true_mask, classes] * scale
    boxes = boxes.int().cpu().numpy().tolist()

    im = ssd.viz.draw_boxes(im, boxes, names)
    cv2.imshow('prediction', im)
    cv2.waitKey(0)
