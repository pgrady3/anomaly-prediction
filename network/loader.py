from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data import DataLoader
import pickle
import time
from tqdm import tqdm
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


class ForecastingDataset(Dataset):
    def __init__(self, data_pkl, train):
        self.train = train
        self.dataset = data_pkl

        split_idx = int(0.8 * len(self.dataset))
        if train:
            self.dataset = self.dataset[:split_idx]     # 80 percent
        else:
            self.dataset = self.dataset[split_idx:]     # 20 percent

        self.transforms = self.get_transforms()
        print('Dataset loaded {} samples, train={}'.format(len(self.dataset), train))

    def get_transforms(self):
        if self.train:
            t = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            t = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        return t

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        out = dict()
        out['FDE'] = sample['FDE']
        out['class'] = self.fde_to_class(sample['FDE'])

        tformed_images = [self.transforms(im) for im in sample['images']]
        out['images'] = torch.stack(tformed_images)     # Stack tensors to make new (6,3,244,244) array

        return out

    def fde_to_class(self, fde):
        # Convert FDE in meters to class. Super hacky
        cutoffs = [1, 2, 4, 7, 10, 14, 20, 30, 40, 200]
        for i, c in enumerate(cutoffs):
            if fde < c:
                return i


if __name__ == '__main__':
    # dataset = ForecastingDataset('forecast/dataset_big.pkl', train=True)
    dataset = ForecastingDataset('forecast/dataset_small.pkl', train=True)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=6)

    start_time = time.time()
    print('start', len(dataloader))

    all_fde = []
    for sample in tqdm(dataloader):
        all_fde.extend(sample['class'].tolist())

    # all_fde = np.array(all_fde)
    # splits = np.split(all_fde, 10)
    # for s in splits:
    #     print(s.max())

    # n, bins, patches = plt.hist(all_fde, 50)
    # plt.xlabel('FDE')
    # plt.show()

    print('Epoch dataload time: ', time.time() - start_time)
