"""Pre-train encoder and classifier for source dataset."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
from PIL import Image
import time
import os.path as osp
from torch.autograd import Variable
from torchvision import transforms as T

from reid.utils.serialization import save_checkpoint
from reid.utils.meters import AverageMeter
from reid.evaluation_metrics.classification import *
from reid.loss import CapsuleLoss

def train(args, model, train_loader, start_epoch):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    base_param_ids = set(map(id, model.module.base.parameters()))
    new_params = [p for p in model.parameters() if
                  id(p) not in base_param_ids]

    param_groups = [
        {'params': model.module.base.parameters(), 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}]


    optimizer = optim.Adam(param_groups, lr=args.lr)

    # Criterion
    criterion = CapsuleLoss()
    criterion2 = nn.CrossEntropyLoss().cuda()
    criterion3 = nn.CrossEntropyLoss().cuda()

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr * (0.1 ** (epoch // args.step_size))

        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    ####################
    # 2. train network #
    ####################

    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)

        print_freq = 1
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions_id = AverageMeter()
        precisions_id2 = AverageMeter()
        precisions_id3 = AverageMeter()

        end = time.time()

        for i, inputs in enumerate(train_loader):
            data_time.update(time.time() - end)

            imgs, pids, imgname = inputs

            inputs = Variable(imgs.cuda())
            labels = torch.eye(args.true_class).index_select(dim=0, index=pids)
            labels = Variable(labels.cuda())
            targets = Variable(pids.cuda())
            results, y, y2 = model(inputs)

            loss1 = criterion(imgs, labels, results)
            loss2 = criterion2(y, targets)
            loss3 = criterion3(y2, targets)

            prec, = accuracy_capsule(results.data, targets.data, args.true_class)
            prec = prec[0]

            prec2, = accuracy(y.data, targets.data)
            prec2 = prec2[0]

            prec3, = accuracy(y2.data, targets.data)
            prec3 = prec3[0]

            loss = loss1 + 0.5*loss2 + 0.5*loss3

            # update the re-id model
            losses.update(loss.data.item(), targets.size(0))

            precisions_id.update(prec, targets.size(0))
            precisions_id2.update(prec2, targets.size(0))
            precisions_id3.update(prec3, targets.size(0))

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec_capslue {:.2%} ({:.2%})\t'
                      'Prec_ID2 {:.2%} ({:.2%})\t'
                      'Prec_ID3 {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(train_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions_id.val, precisions_id.avg,
                              precisions_id2.val, precisions_id2.avg,
                              precisions_id3.val, precisions_id3.avg))

        # save model
        if (epoch+1) % 5 == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
            }, fpath=osp.join(args.logs_dir, 'checkpoint'+str(epoch+1)+'.pth.tar'))

        print('\n * Finished epoch {:3d} \n'.format(epoch))

    return model



