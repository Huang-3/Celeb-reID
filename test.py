
from __future__ import print_function, absolute_import
import argparse

from torch.backends import cudnn
from reid.utils.serialization import load_checkpoint
from reid.core.train_1stream_capsule import *
from reid.models.dense1stream_capsule import DenseNet
from reid.data_loader_1stream import *
from reid.evaluator import *
from pathlib import Path
from sklearn.externals import joblib


def main(args):
    cudnn.benchmark = True
    extractor = Evaluator()

    gallery_loader = get_loader(args.gallery_path + args.part_gallery[0], args.height, args.width, relabel=False,
                                batch_size=args.batch_size, mode='test', num_workers=args.workers,
                                name_pattern=args.name_pattern)

    query_loader = get_loader(args.query_path + args.part_query[0], args.height, args.width, relabel=False,
                              batch_size=args.batch_size, mode='test', num_workers=args.workers,
                              name_pattern=args.name_pattern)

    distmat_all = torch.zeros(len(args.part_train), query_loader.dataset.num_data, gallery_loader.dataset.num_data)

    for i in range(len(args.part_train)):

        gallery_loader = get_loader(args.gallery_path + args.part_gallery[i], args.height, args.width, relabel=False,
                                       batch_size=args.batch_size, mode='test', num_workers=args.workers, name_pattern = args.name_pattern)

        query_loader = get_loader(args.query_path + args.part_query[i], args.height, args.width, relabel=False,
                                       batch_size=args.batch_size, mode='test', num_workers=args.workers, name_pattern = args.name_pattern)

        my_distmat = Path(args.part_train[i] + '/distmat.pkl')

        if my_distmat.is_file():
            distmat_all[i, :, :] = joblib.load(my_distmat)
            print('open distmat' + args.part_train[i])
        else:
            my_query = Path(args.part_train[i] + '/query_features.pkl')
            my_gallery = Path(args.part_train[i] + '/gallery_features.pkl')

            if my_query.is_file() and my_gallery.is_file():
                query_features = joblib.load(my_query)
                print('open query feature')
                gallery_features = joblib.load(my_gallery)
                print('open gallery feature')
            else:
                # Create model
                model = DenseNet(num_feature=args.num_feature, num_classes=args.true_class, num_iteration = args.num_iteration)

                if args.resume:
                    checkpoint = load_checkpoint(args.part_train[i] + args.resume)
                    model.load_state_dict(checkpoint['state_dict'])

                model = nn.DataParallel(model).cuda()

                # Evaluator
                if args.evaluate:
                    query_features, gallery_features = extractor.extract(query_loader, gallery_loader, model, args.output_feature)

                    joblib.dump(query_features, args.part_train[i] + '/query_features.pkl')
                    joblib.dump(gallery_features, args.part_train[i] + '/gallery_features.pkl')


            print('calculate distance for ' + args.part_train[i])
            distmat = extractor.distance(query_features, gallery_features, query_loader.dataset.ret, gallery_loader.dataset.ret, rerank=False)
            distmat_all[i,:,:] = distmat
            joblib.dump(distmat, args.part_train[i] + '/distmat.pkl')

    print('evaluate')
    distanceMat = distmat_all[0,:,:] + 0.5*(distmat_all[1,:,:] + distmat_all[2,:,:]) + distmat_all[3,:,:] + 0.5*distmat_all[4,:,:] + distmat_all[5,:,:]
    extractor.evaluate(distanceMat, query = query_loader.dataset.ret, gallery=gallery_loader.dataset.ret)
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReIDCaps")

    parser.add_argument('--part_train', type=str,
                        default=['log_celeb_11', 'log_celeb_12', 'log_celeb_13', 'log_celeb_21',
                                 'log_celeb_22', 'log_celeb_all'], nargs='+', help="name of train")

    parser.add_argument('--part_query', type=str,
                        default=['query_1_1', 'query_1_2', 'query_1_3', 'query_2_1',
                                 'query_2_2', 'query'], nargs='+', help="name of query")

    parser.add_argument('--part_gallery', type=str,
                        default=['gallery_1_1', 'gallery_1_2', 'gallery_1_3', 'gallery_2_1',
                                 'gallery_2_2', 'gallery'], nargs='+', help="name of gallery")

    parser.add_argument('--train_path', type=str, default='/home/yan/datasets/celeb/', help="train image with soft mask")
    parser.add_argument('--gallery_path', type=str, default='/home/yan/datasets/celeb/', help="gallery image with soft mask")
    parser.add_argument('--query_path', type=str, default='/home/yan/datasets/celeb/', help="query image with soft mask")

    parser.add_argument('--name_pattern', type=str, default='celeb', help="celeb or market")

    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=224, help="input height, default: 256")
    parser.add_argument('--width', type=int, default=224, help="input width, default: 128")

    # model
    parser.add_argument('--num_feature', type=int, default=1024)
    parser.add_argument('--num_iteration', type=int, default=4)

    # training configs
    parser.add_argument('--resume', type=str, default='/model.pth.tar', metavar='PATH')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--evaluate', action='store_true', default=True, help="evaluation only")

    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--output_feature', type=str, default='pool5')

    # ground-truth class number
    parser.add_argument('--true_class', type=int, default=632)

    main(parser.parse_args())
