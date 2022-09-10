import pandas as pd
from torch.utils.data import Dataset
import torch
from skimage import io
import numpy as np
import os
import sys
sys.path.insert(0,'AI4AI4Climate')
sys.path.insert(0,'AI4AI4Climate/com')
from com.common import *

class FrameLandmarksDataset():

    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.

        """

        t = COLOR.Green + 80*'=' + COLOR.END
        print(t)
        """
        root_dir is the path to dataset
        csv_file annotation file name
        output is the output dir for renamed images
        data_source is the name of the dataset (same as folder name)
        """
        check_if_dir_existed(root_dir)
        self.path2dataset = root_dir
        self.path2csv = os.path.dirname(root_dir)
        check_if_file_existed(os.path.join(self.path2csv, csv_file))
        self.landmarks_frame = pd.read_csv(os.path.join(self.path2csv, csv_file))
        self.transform = transform

    def __len__(self):
        """
        len(dataset) returns the size of the dataset.
        """
        return len(self.landmarks_frame)


    def __getitem__(self, idx):
        """
        Support the indexing such that dataset[i] can be used to get i^th sample.
        """
        mobile = "mobile"

        if torch.is_tensor(idx):
            idx = idx.tolist()
            print("idx----------", idx)

        #------------------------------------#
        id = int(self.landmarks_frame.iloc[idx,0])
        t = int(self.landmarks_frame.iloc[idx,1])
        lat = float(self.landmarks_frame.iloc[idx,2])
        lon = float(self.landmarks_frame.iloc[idx,3])
        hW = int(self.landmarks_frame.iloc[idx,4])


        fname = f'{id:04n}_{mobile}_{hW}_{t}_{lat}_{lon}.png'
        print(new_fname)
        check_if_file_existed(os.path.join(self.path2dataset,fname))
        image = io.open(os.path.join(self.path2dataset,fname))
        landmarks = np.array([id, nW, t, lat, lon])

        landmarks = landmarks.reshape(-1, 5)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
