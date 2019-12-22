
from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import sys

from torch.backends import cudnn

from reid.evaluators_1stream import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.core.train_1stream_capsule import *
from reid.models.dense1stream_capsule import DenseNet
from reid.data_loader_1stream import *


def main(args):
    cudnn.benchmark = True
    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    train_loader = get_loader(args.train_path, args.height, args.width, relabel=True,
                                   batch_size=args.batch_size, mode='train', num_workers=args.workers, name_pattern = args.name_pattern)

    gallery_loader = get_loader(args.gallery_path, args.height, args.width, relabel=False,
                                   batch_size=args.batch_size, mode='test', num_workers=args.workers, name_pattern = args.name_pattern)

    query_loader = get_loader(args.query_path, args.height, args.width, relabel=False,
                                   batch_size=args.batch_size, mode='test', num_workers=args.workers, name_pattern = args.name_pattern)

    # Create model
    model = DenseNet(num_feature=args.num_feature, num_classes=args.true_class, num_iteration = args.num_iteration)

    # Load from checkpoint
    start_epoch = args.start_epoch
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    model = nn.DataParallel(model).cuda()

    # Evaluator
    if args.evaluate:
        evaluator = Evaluator(model)
        print("Test:")
        evaluator.evaluate(query_loader, gallery_loader, query_loader.dataset.ret, gallery_loader.dataset.ret, args.output_feature)
        return

    # Start training
    model= train(args, model, train_loader, start_epoch)
    save_checkpoint({'state_dict': model.module.state_dict()}, fpath=osp.join(args.logs_dir, 'model.pth.tar'))

    evaluator = Evaluator(model)
    print("Test:")
    evaluator.evaluate(query_loader, gallery_loader, query_loader.dataset.ret, gallery_loader.dataset.ret, args.output_feature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReIDCaps")
    # data
    parser.add_argument('--train_path', type=str, default='/home/yan/datasets/celeb/train', help="train image with soft mask")
    parser.add_argument('--gallery_path', type=str, default='/home/yan/datasets/celeb/gallery', help="gallery image with soft mask")
    parser.add_argument('--query_path', type=str, default='/home/yan/datasets/celeb/query', help="query image with soft mask")

    parser.add_argument('--name_pattern', type=str, default='celeb', help="celeb or market")

    parser.add_argument('-b', '--batch-size', type=int, default=20)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=224, help="input height, default: 224")
    parser.add_argument('--width', type=int, default=224, help="input width, default: 224")

    # model
    parser.add_argument('--num_feature', type=int, default=1024)
    parser.add_argument('--num_iteration', type=int, default=4)

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--step_size', type=int, default=40)
    parser.add_argument('--print-freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=5, help="model save frequence")
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'log_celeb_all'))
    parser.add_argument('--output_feature', type=str, default='pool5')
    # ground-truth class number
    parser.add_argument('--true_class', type=int, default=632)

    main(parser.parse_args())
