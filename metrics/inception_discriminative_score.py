
import numpy as np
import scipy.linalg
from . import metric_utils
import sklearn.svm

#----------------------------------------------------------------------------

def compute_ids(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    real_activations = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_all=True, max_items=max_real).get_all()

    fake_activations = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen).get_all()

    if opts.rank != 0:
        return float('nan')

    svm = sklearn.svm.LinearSVC(dual=False)
    svm_inputs = np.concatenate([real_activations, fake_activations])
    svm_targets = np.array([1] * real_activations.shape[0] + [0] * fake_activations.shape[0])
    print('Fitting ...')
    svm.fit(svm_inputs, svm_targets)
    u_ids = 1 - svm.score(svm_inputs, svm_targets)
    real_outputs = svm.decision_function(real_activations)
    fake_outputs = svm.decision_function(fake_activations)
    p_ids = np.mean(fake_outputs > real_outputs)

    return float(u_ids), float(p_ids)

#----------------------------------------------------------------------------
