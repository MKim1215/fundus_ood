import os
import pandas as pd
from torchvision.io import read_image
# import torch
from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
from PIL import Image

class Ood_Dataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file, usecols=['filename', 'in', 'out'])
        self.img_labels = pd.read_csv(annotations_file, usecols=['filename', 'label'])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img_path = self.img_labels['filename'].iloc[idx]
        # image = read_image(img_path)
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels['label'].iloc[idx]
        # label = self.img_labels[self.labels].iloc[idx].to_numpy()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class ID_Dataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        # self.class_list = ["Hemorrhage", "HardExudate", "CWP", "Drusen", "VascularAbnormality", "Membrane",
        #                     "ChroioretinalAtrophy", "MyelinatedNerveFiber", "RNFLDefect", "GlaucomatousDiscChange",
        #                     "NonGlaucomatousDiscChange", "MacularHole"]
        # self.labels = ['in', 'out']
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img_path = self.img_labels['filename'].iloc[idx]
        # image = read_image(img_path)
        image = Image.open(img_path).convert('RGB')
        # label = self.img_labels.iloc[idx, 1]
        # label = self.img_labels['label'].iloc[idx].to_numpy()
        label = self.img_labels['label'].iloc[idx]
        # label = self.img_labels[self.labels].iloc[idx].to_numpy()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

# class total_daset(Dataset):
#     def __init__(self, annotations_file, transform=None, target_transform=None):
#         self.df_ind = pd.read_csv(annotations_file, usecols=['filename'])
#         self.df_out1= pd.read_csv(annotations_file, usecols=['filename'])
#         self.df_out2 = pd.read_csv(annotations_file, usecols=['filename'])
#         self.df_out3 = pd.read_csv(annotations_file, usecols=['filename'])
#         self.df_out_total = pd.concat()
        

#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         img_path = self.img_labels['filename'][idx]
#         df_out_total[idx % ]
#         # image = read_image(img_path)
#         image = Image.open(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return ind_img, out_img

class TotalDataset(Dataset):
    def __init__(self, in_file, out_file, transform=None, target_transform=None):
        self.df_in = pd.read_csv(in_file, usecols=['filename'])
        self.df_out = self.concat_out(out_file)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df_in) + len(self.df_out)

    def __getitem__(self, idx):
        in_path = self.df_in['filename'][idx]
        out_path = self.df_out['filename'][idx]
        
        in_img = Image.open(in_path)
        in_label = 1
        out_img = Image.open(out_path)
        out_label = 0

        if self.transform:
            in_img = self.transform(in_img)
            out_img = self.transform(out_img)

        if self.target_transform:
            in_label = self.target_transform(in_label)
            out_label = self.target_transform(out_label)

        in_sample = [in_img, in_label]
        out_sample = [out_img, out_label]

        return in_sample, out_sample

    def concat_out(self, out_file):
        out_total = []
        for filename in out_file:
            df = pd.read_csv(filename, usecols=['filename'])
            out_total.append(df)

        total_df = pd.concat(out_total, ignore_index=True)
        return total_df