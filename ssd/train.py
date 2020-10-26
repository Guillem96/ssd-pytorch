import click
import yaml
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

import ssd
import ssd.transforms as T
import ssd.transforms.functional as F


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
              default='models/vgg16_reducedfc.pth', 
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
@click.option('--num-workers', default=4, type=int,
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
@click.option('--logdir', default=None, 
              type=click.Path(file_okay=False),
              help='Dataset root directory path')
def train(dataset, dataset_root, config,
          basenet, checkpoint, save_dir,
          epochs, batch_size, num_workers,
          lr, momentum, wd, gamma, step_size,
          logdir):
    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    dataset_root = Path(dataset_root)
    basenet = Path(basenet)
    logdir = (Path(logdir) / now) if logdir is not None else None
    checkpoint = Path(checkpoint) if checkpoint is not None else None

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    cfg = yaml.safe_load(open(config))['config']
    transform = T.get_transforms(cfg['image-size'], training=True)

    if dataset == 'COCO':
        viz_dataset = ssd.data.COCODetection(root=dataset_root,
                                             classes=cfg['classes'])
        dataset = ssd.data.COCODetection(root=dataset_root,
                                         classes=cfg['classes'],
                                         transform=transform)
    elif dataset == 'VOC':
        viz_dataset = ssd.data.VOCDetection(root=dataset_root,
                                            classes=cfg['classes'])

        dataset = ssd.data.VOCDetection(root=dataset_root,
                                        classes=cfg['classes'],
                                        transform=transform)
    else:
        viz_dataset = ssd.data.LabelmeDataset(root=dataset_root,
                                              classes=cfg['classes'])
        dataset = ssd.data.LabelmeDataset(root=dataset_root,
                                          classes=cfg['classes'],
                                          transform=transform)

    if logdir is not None:
        tb_writter = SummaryWriter(str(logdir), flush_secs=10)
    else:
        tb_writter = None

    model = ssd.SSD300(cfg)
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
        ssd.engine.train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion_fn=criterion,
            data_loader=data_loader,
            epoch=epoch,
            device=device,
            tb_writer=tb_writter)

        _log_predictions(model, viz_dataset, epoch, tb_writter)
        lr_scheduler.step()
        model_f = save_dir / f'{dataset.name}_{epoch}.pt'
        torch.save(model.state_dict(), model_f)


@torch.no_grad()
def _log_predictions(model, dataset, epoch, tb_writer=None):
    if tb_writer is None: 
        return

    import matplotlib.pyplot as plt

    model.eval()

    tfm = T.get_transforms(300, inference=True)
    rand_idx = torch.randint(size=(4,), high=len(dataset)).tolist()
    images = [dataset.pull_item(i) for i in rand_idx]
    images_in = [tfm(F.tensor_to_np(o)).unsqueeze(0).to(device)
                 for o, *_ in images]
    predictions = [model(im)[0] for im in images_in]

    plt.figure(figsize=(20, 20))

    for i, (im, p) in enumerate(zip(images, predictions), start=1):
        im, _, h, w = im
        scale = torch.as_tensor([w, h, w, h]).unsqueeze(0)

        im = F.tensor_to_np(im)

        true_mask = p['scores'] > 0.8
        boxes = p['boxes'][true_mask].cpu() * scale

        plt.subplot(2, 2, i)
        viz_im = ssd.viz.draw_boxes(im.copy(), boxes.int().tolist())
        plt.imshow(viz_im[...,::-1])
        plt.axis('off')

    tb_writer.add_figure('Prediction Epoch ' + str(epoch), plt.gcf())


if __name__ == '__main__':
    train()
