
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os.path as osp


def init_model(net, restore):
    # restore model weights
    if restore is not None and os.path.exists(restore):
        if osp.isfile(restore):
            checkpoint = torch.load(restore)
            print("=> Loaded checkpoint '{}'".format(restore))
        else:
            raise ValueError("=> No checkpoint found at '{}'".format(restore))

        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {} "
              .format(start_epoch))

        # net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))
    else:
        net.restored = False

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    # net = nn.DataParallel(net).cuda()
    return net