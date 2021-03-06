import yaml
import click

import cv2
import torch

import ssd


@click.command()
@click.option('--image', default=None)
@click.option('--config', required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--checkpoint', required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--output', required=True,
              type=click.Path(dir_okay=False))
def trace_ssd(image, config, checkpoint, output):
    device = torch.device('cpu')

    cfg = yaml.safe_load(open(config))['config']
    checkpoint = torch.load(checkpoint, map_location=device)

    model = ssd.SSD300(cfg)
    model.eval()
    model.load_state_dict(checkpoint)
    model.to(device)

    if image is None:
        example = torch.randn(1, 3, cfg['image-size'], cfg['image-size'])
    else:
        im = cv2.imread(image)
        example = ssd.transforms.get_transforms(cfg['image-size'], 
                                                inference=True)(im)
        example.unsqueeze_(0)

    traced_ssd = torch.jit.script(model)
    files = {'classes': ','.join(cfg['classes'])}
    traced_ssd.save(output, files)

    # Sanitize check
    files['classes'] = ''
    loaded_model = torch.jit.load(output,
                                  map_location=device, 
                                  _extra_files=files)
    loaded_classes = files['classes'].decode().split(',')
    if image is not None:
        with torch.no_grad():
            boxes, labels, scores = loaded_model(example)
        scale = torch.as_tensor([im.shape[1], im.shape[0]] * 2, device=device)
        scale.unsqueeze_(0)

        true_mask = scores[0] > .5
        scores = scores[0][true_mask].cpu().tolist()
        boxes = (boxes[0][true_mask].cpu() * scale).int().tolist()
        labels = labels[0][true_mask].cpu().tolist()
        names = [loaded_classes[i - 1] for i in labels]

        im = ssd.viz.draw_boxes(im, boxes, names)
        cv2.imshow('traced', im)
        cv2.waitKey(0)


if __name__ == "__main__":
    trace_ssd()