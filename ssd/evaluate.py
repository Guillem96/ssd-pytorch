import click
import yaml

import torch

import ssd
import ssd.transforms as T

from ssd.coco.coco_utils import get_coco_api_from_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@click.command()
@click.option('--dataset', required=True,
              type=click.Choice(['labelme', 'VOC', 'COCO']))
@click.option('--dataset-root', required=True,
              type=click.Path(exists=True, file_okay=False))
@click.option('--config', required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--checkpoint', required=True,
              type=click.Path(exists=True, dir_okay=False))
def evaluate(dataset, dataset_root, config, checkpoint):

    cfg = yaml.safe_load(open(config))['config']
    transform = T.get_transforms(cfg['image-size'], training=False)

    if dataset == 'COCO':
        dataset = ssd.data.COCODetection(root=dataset_root,
                                         classes=cfg['classes'],
                                         transform=transform)
    elif dataset == 'VOC':
        dataset = ssd.data.VOCDetection(root=dataset_root,
                                        classes=cfg['classes'],
                                        transform=transform)
    else:
        dataset = ssd.data.LabelmeDataset(root=dataset_root,
                                          classes=cfg['classes'],
                                          transform=transform)

    print('Generating COCO dataset...',  end=' ')
    coco_dataset = get_coco_api_from_dataset(dataset)
    print(coco_dataset.dataset['images'])
    print('done')

    model = ssd.ssd(cfg, cfg['image-size'], cfg['num-classes'])
    model.eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)

    ssd.engine.evaluate(model, dataset, coco_dataset, device)


if __name__ == "__main__":
    evaluate()