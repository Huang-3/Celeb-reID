from torch.utils import data
# from reid.utils.data import transforms as T
from PIL import Image
import os.path as osp
from glob import glob
import re
from torchvision import transforms as T


class ReidDataset(data.Dataset):
    def __init__(self, img_path, transform, relabel, name_pattern):
        self.img_path = img_path
        self.transform = transform
        self.fnames = []
        self.pids = []
        self.ret = []
        self.preprocess(relabel = relabel, name_pattern = name_pattern)
        self.num_data = int(len(self.fnames))

    def preprocess(self, relabel, name_pattern):
        if name_pattern=='celeb':
            pattern = re.compile(r'([-\d]+)_(\d)')
        else:
            pattern = re.compile(r'([-\d]+)_c(\d)')
        fpaths = sorted(glob(osp.join(self.img_path, '*.jpg')))
        i = 0
        all_pids = {}
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored

            self.fnames.append(fpath)

            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            cam -= 1
            pid = all_pids[pid]
            self.pids.append(pid)
            i = i+1
            self.ret.append((fname, pid, cam))

    def __getitem__(self, index):
        img = Image.open(self.fnames[index])
        pids = self.pids[index]
        names = osp.basename(self.fnames[index])

        return self.transform(img), pids, names

    def __len__(self):
        return self.num_data


def get_loader(img_path, height, width, batch_size=16, relabel=True, mode='train', num_workers=4, name_pattern = ''):
    """Build and return a data loader."""

    if mode == 'train':

        transform = T.Compose([
            T.Resize(size=(height, width), interpolation=3),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = T.Compose([
            T.Resize(size=(height, width), interpolation=3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    dataset = ReidDataset(img_path, transform, relabel, name_pattern)

    data_loader = data.DataLoader(dataset=dataset,
                             batch_size=batch_size, num_workers=num_workers,
                             shuffle=(mode == 'train'), pin_memory=True, drop_last=(mode == 'train'))
    return data_loader