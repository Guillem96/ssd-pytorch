import yaml

import click
import cv2

import torch

import ssd
import ssd.transforms as T


@click.command()
@click.argument('im_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-c', '--checkpoint', required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--traced/--no-traced', default=False)
@click.option('--config', default=None, type=click.Path(dir_okay=False))
@click.option('-o', '--output',
              default=None, type=click.Path(dir_okay=False))
def inference(im_path, checkpoint, config, traced, output):

    if not traced:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        cfg = yaml.safe_load(open(config))['config']
        idx_to_class = cfg['classes']

        model = ssd.SSD300(cfg)
        model.eval()

        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
    else:
        device = torch.device('cpu')
        files = {'classes': ''}
        model = torch.jit.load(checkpoint, 
                               map_location=device, 
                               _extra_files=files)
        model.to(device)
        model.eval()
        idx_to_class = files['classes'].decode().split(',')

    im = cv2.imread(im_path)
    im_in = T.get_transforms(300, inference=True)(im)
    im_in = im_in.unsqueeze(0).to(device)

    with torch.no_grad():
        detections = model(im_in)

    scale = torch.as_tensor([im.shape[1], im.shape[0]] * 2)
    scale.unsqueeze_(0)

    if not traced:
        detections = detections[0]
        true_mask = detections['scores'] > .5
        scores = detections['scores'][true_mask].cpu().tolist()
        boxes = (detections['boxes'][true_mask].cpu() * scale).int().tolist()
        labels = detections['labels'][true_mask].cpu().tolist()
    else:
        boxes, labels, scores = detections
        true_mask = scores[0] > .5
        boxes = (boxes[0][true_mask] * scale).cpu().int().tolist()
        labels = labels[0][true_mask].cpu().tolist()

    names = [idx_to_class[i - 1] for i in labels]

    im = ssd.viz.draw_boxes(im, boxes, names)
    if output is not None:
        cv2.imwrite(output, im)

    cv2.imshow('prediction', im)
    cv2.waitKey(0)
