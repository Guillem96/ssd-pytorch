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

    model = ssd.ssd(cfg, cfg['image-size'], cfg['num-classes'])
    model.eval()
    model.load_state_dict(checkpoint)

    if image is None:
        example = torch.randn(1, 3, cfg['image-size'], cfg['image-size'])
    else:
        im = cv2.imread(image)
        example = ssd.transforms.get_transforms(cfg['image-size'], 
                                                inference=True)(im)
        example.unsqueeze_(0)

    traced_ssd = torch.jit.trace(model, example)
    files = torch.jit.DEFAULT_EXTRA_FILES_MAP
    files['config'] = open(config).read()
    traced_ssd.save(output, files)

    # Sanitize check
    files['config'] = ''
    loaded_model = torch.jit.load(output, 
                                  map_location=device, 
                                  _extra_files=files)
    loaded_cfg = yaml.safe_load(files['config'])['config']
    if image is not None:
        with torch.no_grad():
            detections = loaded_model(example)[0]

        scores = detections[..., 0].t()
        boxes = detections[..., 1:].permute(1, 0, 2)

        scale = torch.as_tensor([im.shape[1], im.shape[0]] * 2, device=device)
        scale.unsqueeze_(0)

        scores, classes = scores.max(-1)
        true_mask = scores > .5
        scores = scores[true_mask]
        
        classes = classes[true_mask]
        names = [loaded_cfg['classes'][i - 1] for i in classes.cpu().tolist()]
        boxes = boxes[true_mask, classes] * scale
        boxes = boxes.int().cpu().numpy().tolist()

        im = ssd.viz.draw_boxes(im, boxes, names)
        cv2.imshow('traced', im)
        cv2.waitKey(0)

if __name__ == "__main__":
    trace_ssd()