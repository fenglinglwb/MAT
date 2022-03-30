import numpy as np
import scipy.linalg
from . import metric_utils
import math
import cv2


def compute_psnr(opts, max_real):
    # stats: numpy, [N, 3]
    stats = metric_utils.compute_image_stats_for_generator(opts=opts, capture_all=True, max_items=max_real).get_all()

    if opts.rank != 0:
        return float('nan'), float('nan'), float('nan')

    print('Number of samples: %d' % stats.shape[0])
    avg_psnr = stats[:, 0].sum() / stats.shape[0]
    avg_ssim = stats[:, 1].sum() / stats.shape[0]
    avg_l1 = stats[:, 2].sum() / stats.shape[0]
    return avg_psnr, avg_ssim, avg_l1