import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
from CLOCs_code.data_preprocess import _read_info


class KITTIDataset(Dataset):
    def __init__(self, root_path, info_path, used_classes):
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        self._root_path = root_path
        self._kitti_infos = infos
        self._used_classes = used_classes
        print("number of infos:", len(self._kitti_infos))

    def __len__(self):
        return len(self._kitti_infos)
        #return 150

    @property
    def kitti_infos(self):
        return self._kitti_infos

    def __getitem__(self, index):
        return _read_info(self._kitti_infos[index], self._used_classes)
