# coding: utf-8
"""Visualize samples from dataset."""

import argparse
import time

import cv2
import numpy as np
from torch.utils.data import DataLoader

from dataloader import TLessDataset, TransparentDataset


def read_dataset(path, b, dataset='notex'):
    dataset = TLessDataset(path) if dataset == 'notex' else TransparentDataset(path)
    print(len(dataset), "samples in dataset.")

    return DataLoader(dataset, batch_size=b, shuffle=True)


def create_view(frame):
    """Show current frame of the RGB-D dataset as images.

    :param frame:
    :return:
    """
    image, depth, norms, mask = frame

    # convert normals from [-1, 1] to [0, 255]
    norms = ((norms + 1) / 2) * 255

    # apply a colormap on grayscale depth map, makes easier to see depth changes
    depth = cv2.applyColorMap((depth * 255.0).astype(np.uint8), cv2.COLORMAP_JET)

    masked_image = image.copy()

    bg_color = 128  # gray window background
    masked_image[mask, :] = bg_color
    depth[mask, :] = bg_color
    norms[mask, :] = bg_color

    dst = np.hstack((image, masked_image, depth, norms))
    return dst.astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', '-d', type=str, default='../out')
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--dataset', '-ds', type=str, default='notex', choices=['notex', 'trans'])
    return parser.parse_args()


def main(args):
    data = read_dataset(args.dataset_dir, args.batch_size, args.dataset)
    for it in data:
        image, label = it
        dmap, nmap, mask = label

        rows = []
        for i in range(image.shape[0]):
            im = image[i].numpy()
            dm = dmap[i].numpy()
            nm = nmap[i].numpy()
            ma = mask[i].numpy()

            rows.append(create_view(frame=(im, dm, nm, ma)))

        cv2.imshow('Dataset', np.vstack(rows))
        if cv2.waitKey(delay=1) == ord('q'):
            raise KeyboardInterrupt

        time.sleep(3)


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print("Visualization interrupted.")
