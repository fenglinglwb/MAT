import cv2
import os
import sys
import numpy as np
import math
import glob
import pyspng
import PIL.Image

import torch
import lpips


def read_image(image_path):
    with open(image_path, 'rb') as f:
        if pyspng is not None and image_path.endswith('.png'):
            image = pyspng.load(f.read())
        else:
            image = np.array(PIL.Image.open(f))
    if image.ndim == 2:
        image = image[:, :, np.newaxis] # HW => HWC
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    image = image.transpose(2, 0, 1) # HWC => CHW
    image = torch.from_numpy(image).float().unsqueeze(0)
    image = image / 127.5 - 1

    return image


def calculate_metrics(folder1, folder2):
    l1 = sorted(glob.glob(folder1 + '/*.png') + glob.glob(folder1 + '/*.jpg'))
    l2 = sorted(glob.glob(folder2 + '/*.png') + glob.glob(folder2 + '/*.jpg'))
    assert(len(l1) == len(l2))
    print('length:', len(l1))

    # l1 = l1[:3]; l2 = l2[:3];

    device = torch.device('cuda:0')
    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()
    # loss_fn = lpips.LPIPS(net='vgg').to(device)

    lpips_l = []
    with torch.no_grad():
        for i, (fpath1, fpath2) in enumerate(zip(l1, l2)):
            print(i)
            _, name1 = os.path.split(fpath1)
            _, name2 = os.path.split(fpath2)
            name1 = name1.split('.')[0]
            name2 = name2.split('.')[0]
            assert name1 == name2, 'Illegal mapping: %s, %s' % (name1, name2)

            img1 = read_image(fpath1).to(device)
            img2 = read_image(fpath2).to(device)
            assert img1.shape == img2.shape, 'Illegal shape'
            lpips_l.append(loss_fn(img1, img2).mean().cpu().numpy())

    res = sum(lpips_l) / len(lpips_l)

    return res


if __name__ == '__main__':
    folder1 = 'path to the inpainted result'
    folder2 = 'path to the gt'

    res = calculate_metrics(folder1, folder2)
    print('lpips: %.4f' % res)
    with open('lpips.txt', 'w') as f:
        f.write('lpips: %.4f' % res)
