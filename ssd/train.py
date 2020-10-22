import argparse
from pathlib import Path

import torch

import ssd
import ssd.transforms as T

# TODO: SSDAugmentation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')ç
parser.add_argument('--epochs', default=8, type=int,
                    help='Times to iterate over the whole dataset')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')

def train(args):
    if args.dataset == 'COCO':
        cfg = ssd.data.config.coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=T.SSDAugmentation(cfg['min_dim']))
    elif args.dataset == 'VOC':
        cfg = ssd.data.config.voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim']))
    else:
        pass # TODO: labelme

    model = ssd.ssd('train', cfg, cfg['min_dim'], cfg['num_classes'])
    model.to(device)
    model.train()

    if args.resume:
        print(f'Resuming training, loading {args.resume}...')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet, 
                                 map_location=device)
        print('Loading base network...')
        model.vgg.load_state_dict(vgg_weights)

    if not args.resume:
        print('Initializing weights...')
        model.extras.apply(weights_init)
        model.loc.apply(weights_init)
        model.conf.apply(weights_init)

    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr, 
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    criterion = ssd.nn.MultiBoxLoss(0.5, 3)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True, 
        collate_fn=ssd.data.detection_collate,
        pin_memory=True)

    metrics = {}
    for epoch in range(1, args.epochs + 1):
        for i, (images, targets) in enumerate(data_loader):
            images = images.to(device)
            targets = targets.to(device)

            out = model(images)

            loss_l, loss_c = criterion(out, targets)
            metrics['loc_loss'] = metrics['loc_loss'] + loss_l.item()
            metrics['conf_loss'] = metrics['conf_loss'] + loss_c.item()

            loss = loss_l + loss_c
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                means = {k: v / i for k, v in metrics.items()}
                logs = ' '.join(f'{k}: {v:.4f}' for k, v in means.items())
                print(f'Epoch [{epoch}] {logs}')

        torch.save(model.state_dict(), 
                   str(Path(args.save_folder, (args.dataset + '.pth'))))

def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    Path(args.save_folder).mkdir(exist_ok=True, parents=True)
    train(parser.parse_args())
