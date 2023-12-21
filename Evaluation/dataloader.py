from torch.utils.data import Dataset
from torchvision import transforms
from os.path import join
from PIL import Image
import pandas as pd
import numpy as np
import torch


class TestDataset(Dataset):
    def __init__(self, img_dir, data_loc, clive= False):
        """
        Args:
            img_dir (string): Directory with all the images.
            data_loc (string): Location of the csv file with image names and corresponding MOS.
            clive (bool): If True, specific images of CLIVE database are resized to 500x500.
        """
        
        self.img_dir = img_dir
        self.clive = clive
        self.tensorizer  = transforms.ToTensor()

        self.data = pd.read_csv(data_loc)
        self.data = self.data.astype({'im_loc': str, 'mos': np.float32})

        if self.clive:
            self.resizer = transforms.Resize((500,500))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im_loc = self.data.iloc[idx]['im_loc']
        x = Image.open(join(self.img_dir, im_loc))
        x = self.tensorizer(x)
        
        if self.clive:
            if im_loc == '1024.JPG' or im_loc == '1113.JPG':
                x = self.resizer(x)

        if x.shape[0] <3:
            x = torch.cat([x]*3, dim=0)

        return x, self.data.iloc[idx]['mos'], im_loc
