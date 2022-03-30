import cv2
import os
import sys
import numpy as np
import math
import glob
import pyspng
import PIL.Image


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def calculate_l1(img1, img2):
    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0
    l1 = np.mean(np.abs(img1 - img2))

    return l1


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
    # image = image.transpose(2, 0, 1) # HWC => CHW

    return image


def calculate_metrics(folder1, folder2):
    l1 = sorted(glob.glob(folder1 + '/*.png') + glob.glob(folder1 + '/*.jpg'))
    l2 = sorted(glob.glob(folder2 + '/*.png') + glob.glob(folder2 + '/*.jpg'))
    assert(len(l1) == len(l2))
    print('length:', len(l1))

    # l1 = l1[:3]; l2 = l2[:3];

    psnr_l, ssim_l, dl1_l = [], [], []
    for i, (fpath1, fpath2) in enumerate(zip(l1, l2)):
        print(i)
        _, name1 = os.path.split(fpath1)
        _, name2 = os.path.split(fpath2)
        name1 = name1.split('.')[0]
        name2 = name2.split('.')[0]
        assert name1 == name2, 'Illegal mapping: %s, %s' % (name1, name2)

        img1 = read_image(fpath1).astype(np.float64)
        img2 = read_image(fpath2).astype(np.float64)
        assert img1.shape == img2.shape, 'Illegal shape'
        psnr_l.append(calculate_psnr(img1, img2))
        ssim_l.append(calculate_ssim(img1, img2))
        dl1_l.append(calculate_l1(img1, img2))

    psnr = sum(psnr_l) / len(psnr_l)
    ssim = sum(ssim_l) / len(ssim_l)
    dl1 = sum(dl1_l) / len(dl1_l)

    return psnr, ssim, dl1


if __name__ == '__main__':
    folder1 = 'path to the inpainted result'
    folder2 = 'path to the gt'

    psnr, ssim, dl1 = calculate_metrics(folder1, folder2)
    print('psnr: %.4f, ssim: %.4f, l1: %.4f' % (psnr, ssim, dl1))
    with open('psnr_ssim_l1.txt', 'w') as f:
        f.write('psnr: %.4f, ssim: %.4f, l1: %.4f' % (psnr, ssim, dl1))

