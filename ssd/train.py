import click
import yaml
from pathlib import Path

import torch

import ssd
import ssd.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@click.command()
@click.option('-d', '--dataset',
              required=True, type=click.Choice(['VOC', 'COCO', 'labelme']),
              help='VOC, COCO or labelme')
@click.option('--dataset-root', required=True, 
              type=click.Path(exists=True, file_okay=False),
              help='Dataset root directory path')
@click.option('--config',
              required=True, type=click.Path(exists=True, dir_okay=False),
              help='Configuration regarding dataset')
@click.option('--basenet', 
              default='vgg16_reducedfc.pth', 
              type=click.Path(exists=True, dir_okay=False),
              help='Pretrained base model')
@click.option('--checkpoint', default=None, type=click.Path(dir_okay=False),
              help='Checkpoint state_dict file to resume training from')
@click.option('--save-dir', required=True,
              type=click.Path(file_okay=False),
              help='Directory for saving checkpoint models')
@click.option('--epochs', default=8, type=int,
              help='Times to iterate over the whole dataset')
@click.option('--batch-size', default=16, type=int,
              help='Batch size for training')
@click.option('--num_workers', default=4, type=int,
              help='Number of workers used in dataloading')
@click.option('--lr', '--learning-rate', default=1e-3, type=float,
              help='initial learning rate')
@click.option('--momentum', default=0.9, type=float,
              help='Momentum value for optim')
@click.option('--wd', default=5e-4, type=float,
              help='Weight decay for SGD')
@click.option('--gamma', default=0.1, type=float,
              help='Gamma update for SGD')
@click.option('--step-size', default=4, type=int,
              help='Decrease the learning rate by gamma after n epochs steps')

def train(dataset, dataset_root, config,
          basenet, checkpoint, save_dir,
          epochs, batch_size, num_workers,
          lr, momentum, wd, gamma, step_size):

    dataset_root = Path(dataset_root)
    basenet = Path(basenet)
    checkpoint = Path(checkpoint) if checkpoint is not None else None
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    cfg = yaml.safe_load(open(config))
    transform = T.get_transforms(cfg['image-size'], training=True)

    if dataset == 'COCO':
        dataset = ssd.data.COCODetection(root=dataset_root,
                                         transform=transform)
    elif dataset == 'VOC':
        dataset = ssd.data.VOCDetection(root=dataset_root,
                                        transform=transform)
    else:
        dataset = ssd.data.LabelmeDataset(root=dataset_root,
                                          transform=transform)

    model = ssd.ssd(cfg, cfg['image-size'], cfg['num-classes'])
    if checkpoint is not None:
        print(f'Resuming training, loading {str(checkpoint)}...')
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint)
    else:
        vgg_weights = torch.load(basenet)
        print('Loading base network...')
        model.vgg.load_state_dict(vgg_weights)

    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=lr, 
                                momentum=momentum,
                                weight_decay=wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=step_size,
                                                   gamma=gamma)
    criterion = ssd.nn.MultiBoxLoss(0.5, 3)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True, 
        collate_fn=ssd.data.detection_collate,
        pin_memory=True)

    steps_per_epoch = len(dataset) // batch_size
    for epoch in range(1, epochs + 1):
        metrics = {}
        for i, (images, targets) in enumerate(data_loader):
            images = images.to(device)
            targets = [o.to(device) for o in targets]

            out = model(images)

            loss_l, loss_c = criterion(out, targets)
            metrics['loc_loss'] = metrics.get('loc_loss', 0.) + loss_l.item()
            metrics['conf_loss'] = metrics.get('conf_loss', 0.) + loss_c.item()

            loss = loss_l + loss_c
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                means = {k: v / i for k, v in metrics.items()}
                logs = ' '.join(f'{k}: {v:.4f}' for k, v in means.items())
                print(f'Epoch [{epoch} {i}/{steps_per_epoch}] {logs}')

        lr_scheduler.step()
        model_f = save_dir / f'{dataset.name}_{epoch}.pt'
        torch.save(model.state_dict(), model_f)


if __name__ == '__main__':
    train()
