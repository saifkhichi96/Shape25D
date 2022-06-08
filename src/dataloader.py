import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from utils.io import ls


class TLessDataset(Dataset):
    """Dataset class for loading the texture-less data from memory."""

    def __init__(self, path, transform=None):
        """
        Args:
            path (string): Path to the dataset.
        """
        print("Loading texture-less dataset from:", path)
        self.transform = transform
        self.images = []
        self.labels = []

        objects = sorted(os.listdir(path))
        for o in objects:
            obj_dir = os.path.join(path, o)
            if os.path.isdir(obj_dir):
                sequences = os.listdir(obj_dir)
                for s in sequences:
                    seq_dir = os.path.join(obj_dir, s)
                    if os.path.isdir(seq_dir):
                        self.images += [os.path.join(seq_dir, p) for p in ls(seq_dir, '.png')]
                        self.labels += [os.path.join(seq_dir, p) for p in ls(seq_dir, '.npz')]

    def __len__(self):
        """Return the size of dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get the item at index idx."""

        # Get the data and label
        data = cv2.imread(self.images[idx])
        dmap = np.load(self.labels[idx])['dmap'].astype(np.float32)
        nmap = np.load(self.labels[idx])['nmap'].astype(np.float32)
        mask = dmap >= 1

        # Apply transformation if any
        if self.transform:
            data = self.transform(data)

        # Return the data and label
        return data, (dmap, nmap, mask)


class TransparentDataset(Dataset):
    """Dataset class for loading the transparent data from memory."""

    def __init__(self, path, single_object=False, envs=None, seqs=None, transform=None):
        """
        Args:
            path (string): Path to the dataset.
            single_object (bool): If True, only load data from one object. If False, 'path' is the path to the
                                  directory containing all objects. Default: False.
            envs (list): List of world environments to load. If None, load all environments. Default: None.
            seqs (list): List of sequences to load. If None, load all sequences. Default: None.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        print("Loading transparent dataset from:", path)
        self.transform = transform
        self.images = []
        self.labels = []

        if single_object:
            objects = [path]
        else:
            objects = [os.path.join(path, o) for o in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, o))]

        for o in objects:
            sequences = [os.path.join(o, s) for s in sorted(os.listdir(o)) if
                         os.path.isdir(os.path.join(o, s)) and (seqs is None or s in seqs)]
            for s in sequences:
                images_path = os.path.join(s, 'images')
                images = ls(images_path, '.png')

                if envs is not None:
                    # Keep only the specified environments
                    images = [i for i in images if i.split('/')[-1].split('_')[-1].split('.')[0] in envs]

                labels_path = s
                labels = []
                for y in ls(labels_path, '.npz'):
                    labels += [y] * (len(envs) if envs is not None else 5)

                if len(images) == len(labels):
                    self.images += [os.path.join(images_path, x) for x in images]
                    self.labels += [os.path.join(labels_path, y) for y in labels]

    def __len__(self):
        """Return the size of dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get the item at index idx."""

        # Get the data and label
        data = cv2.imread(self.images[idx])
        dmap = np.load(self.labels[idx])['dmap'].astype(np.float32)
        nmap = np.load(self.labels[idx])['nmap'].astype(np.float32)
        mask = dmap >= 1

        # Apply transformation if any
        if self.transform:
            data = self.transform(data)

        # Return the data and label
        return data, (dmap, nmap, mask)
