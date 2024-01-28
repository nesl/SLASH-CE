import torch
from torch.utils.data import Dataset
import numpy as np

class IMU_CE_Pattern(Dataset):

    def __init__(self, dataset, examples, num_i, flat_for_pc):
        self.data = list()
        self.dataset = dataset
        self.num_i = num_i
        self.flat_for_pc = flat_for_pc

        with open(examples) as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))

    
    def __getitem__(self, index):
        if self.num_i == 2:
            i1, i2, l = self.data[index]
            l = ':- not event(i1, i2, {}).'.format(l)
            if self.flat_for_pc:
                return {'i1': self.dataset[i1][0].flatten(), 'i2': self.dataset[i2][0].flatten()}, l
            else:
                return {'i1': self.dataset[i1][0], 'i2': self.dataset[i2][0]}, l

        elif self.num_i == 3:
            i1, i2, i3, l = self.data[index]
            l = ':- not event(i1, i2, i3, {}).'.format(l)
            if self.flat_for_pc:
                return {'i1': self.dataset[i1][0].flatten(), 'i2': self.dataset[i2][0].flatten(), 'i3': self.dataset[i3][0].flatten()}, l
            else:
                return {'i1': self.dataset[i1][0], 'i2': self.dataset[i2][0], 'i3': self.dataset[i3][0]}, l

        elif self.num_i == 4:
            i1, i2, i3, i4, l = self.data[index]
            l = ':- not event(i1, i2, i3, i4, {}).'.format(l)
            if self.flat_for_pc:
                return {'i1': self.dataset[i1][0].flatten(), 'i2': self.dataset[i2][0].flatten(), 'i3': self.dataset[i3][0].flatten(), 'i4': self.dataset[i4][0].flatten()}, l
            else:
                return {'i1': self.dataset[i1][0], 'i2': self.dataset[i2][0], 'i3': self.dataset[i3][0], 'i4': self.dataset[i4][0]}, l
    
        elif self.num_i == 6:
            i1, i2, i3, i4, i5,i6, l = self.data[index]
            l = ':- not event(i1, i2, i3, i4, i5, i6, {}).'.format(l)
            if self.flat_for_pc:
                return {'i1': self.dataset[i1][0].flatten(), 'i2': self.dataset[i2][0].flatten(), 'i3': self.dataset[i3][0].flatten(), 'i4': self.dataset[i4][0].flatten(), 'i5': self.dataset[i5][0].flatten(), 'i6': self.dataset[i6][0].flatten()}, l
            else:
                return {'i1': self.dataset[i1][0], 'i2': self.dataset[i2][0], 'i3': self.dataset[i3][0], 'i4': self.dataset[i4][0], 'i5': self.dataset[i5][0], 'i6': self.dataset[i6][0]}, l
    

    def __len__(self):
        return len(self.data)


class IMUDataset(Dataset):
    def __init__(self, data, labels, transforms=[]):
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imu = self.data[idx]
        label = self.labels[idx]
        return self.preprocess(imu.astype(np.float32)), label.astype(np.int64)
    
    def preprocess(self, data):
    # normalize sound 
    # factor = 32768.0 # TODO: change this to np.max(), by changeing self.sounds and self.labels into tensors (instead of list of tensors)
    # sound = normalize(factor)(sound)
        if self.transforms != []:
            data = self.transforms(data).squeeze(dim=0)
        return data