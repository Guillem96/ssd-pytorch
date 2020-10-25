import math
import time

import torch
import numpy as np


def train_one_epoch(model, 
                    optimizer, 
                    criterion_fn,
                    data_loader, 
                    epoch, 
                    device,
                    print_freq=10,
                    tb_writer=None):
    model.train()

    metrics = dict()

    for i, (images, targets) in enumerate(data_loader):
        x = images.to(device)
        y = [t.to(device) for t in targets]

        out = model(x)
        loss_l, loss_c = criterion_fn(out, y)
        loss = loss_l + loss_c
        metrics['loc_loss'] = metrics.get('loc_loss', 0.) + loss_l.item()
        metrics['conf_loss'] = metrics.get('conf_loss', 0.) + loss_c.item()
        metrics['loss'] = metrics.get('loss', 0.) + loss.item()

        # Summary for tensorboard
        if tb_writer is not None:
            summary = {
                'regression_loss': loss_l.item(),
                'classification_loss': loss_c.item(),
                'loss': loss.item()
            }
            summary['learning_rate'] = optimizer.param_groups[0]["lr"]
            tb_writer.add_scalars(
                'train', summary, global_step=len(data_loader) * epoch + i)

        if not math.isfinite(loss.item()):
            print("Loss is NaN, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()

        if (i + 1) % print_freq == 0:
            means_dict = {k: v / i for k, v in metrics.items()}
            lr = optimizer.param_groups[0]["lr"]

            print(f'Epoch[{epoch}] [{i}/{len(data_loader)}]', end=' ')
            print(' '.join([f'{k}: {v:.4f}' for k, v in means_dict.items()]), 
                  end=' ')
            print(f' lr: {lr}')


@torch.no_grad()
def evaluate(model, dataset, coco, device):

    from ssd.coco.coco_eval import CocoEvaluator

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)

    model.eval()

    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    running_time = 0.
    running_eval_time = 0.

    for image_id in range(len(dataset)):
        image, targets = dataset[image_id]
        image = image.unsqueeze(0).to(device)
        coco_im = coco.dataset['images'][image_id]
        # targets = _parse_targets(targets, coco_im)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)
        detections = _parse_detections(outputs, coco_im, device)
        model_time = time.time() - model_time
        running_time += model_time

        evaluator_time = time.time()

        coco_evaluator.update({image_id: detections})
        evaluator_time = time.time() - evaluator_time
        running_eval_time += evaluator_time

    mean_time = running_time / len(dataset)
    mean_eval_time = running_eval_time / len(dataset)

    print(f'Validation: Mean Inference time: {mean_time:.4f} '
          f'Mean Evaluation time : {mean_eval_time:.4f}')

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    return coco_evaluator


def _parse_detections(detections, coco_im, device):
    detections = detections[0]
    scores = detections[..., 0].t()
    boxes = detections[..., 1:].permute(1, 0, 2)

    scale = torch.as_tensor([coco_im['width'], 
                             coco_im['height']] * 2, 
                             device=device)
    scale.unsqueeze_(0)

    scores, classes = scores.max(-1)
    boxes = boxes[torch.arange(len(scores)), classes] * scale

    boxes = boxes.int().cpu()
    classes = classes.cpu() - 1
    scores = scores.cpu()

    return dict(boxes=boxes, labels=classes, scores=scores)
