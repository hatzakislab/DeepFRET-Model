import os
import random
import sys

import numpy as np
import itertools
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def min_none(ls):
    """Returns minimum value of list, and None if all elements are None"""
    v = [x for x in ls if x is not None]
    return np.min(v) if v else None


def random_seed_mp(verbose=False):
    """Initializes a pseudo-random seed for multiprocessing use"""
    seed_val = int.from_bytes(os.urandom(4), byteorder="little")
    np.random.seed(seed_val)
    if verbose:
        print("Random seed value: {}".format(seed_val))


def sample_max_normalize_3d(X):
    """
    Sample-wise max-value normalization of 3D array (tensor). This is not
    feature-wise normalization, to keep the ratios between features intact!
    """
    if len(X.shape) == 2:
        X = X[np.newaxis, :, :]
    assert len(X.shape) == 3
    arr_max = np.max(X, axis=(1, 2), keepdims=True)
    return np.squeeze(X / arr_max)


def count_adjacent_values(arr):
    """
    Returns start index and length of segments of equal values.

    Example for plotting several axvspans:
    --------------------------------------
    adjs, lns = lib.count_adjacent_true(score)
    t = np.arange(1, len(score) + 1)

    for ax in axes:
        for starts, ln in zip(adjs, lns):
            alpha = (1 - np.mean(score[starts:starts + ln])) * 0.15
            ax.axvspan(
            xmin = t[starts], xmax = t[starts] + (ln - 1),
            alpha = alpha, color = "red", zorder = -1)
    """
    arr = arr.ravel()

    n = 0
    same = [(g, len(list(l))) for g, l in itertools.groupby(arr)]
    starts = []
    lengths = []
    for v, l in same:
        _len = len(arr[n : n + l])
        _idx = n
        n += l
        lengths.append(_len)
        starts.append(_idx)
    return starts, lengths


def load_npz_data(path, set_names=("X", "y"), top_percentage=100):
    """Loads .npy formatted simulated data"""
    sets = [
        np.load(os.path.join(str(path), file + ".npz"))["arr_0"]
        for file in set_names
    ]
    fraction = int(1 / (top_percentage * 0.01))
    X, y = [s[0 : (s.shape[0] // fraction), :, :] for s in (sets)]
    return X, y


def swap_integers(arr, x, y):
    """
    Swaps two integers in array inplace,
    by tuple swaping (e.g. swap 1 and 5).
    """
    a, b = (arr == x), (arr == y)
    arr[a], arr[b] = y, x
    return arr


def generate_name(length=10, module=None):
    """
    Generates a random ID for a module
    """
    if module is None:
        module = sys.modules[__name__]
    while True:
        name = "id{:0{length}d}".format(
            random.randint(0, 10 ** length - 1), length=length
        )
        if not hasattr(module, name):
            return name


def global_function(func):
    """
    Decorate a local function to make it global, thus enabling
    multiprocessing pickling of it
    """
    module = sys.modules[func.__module__]
    func.__name__ = func.__qualname__ = generate_name(module=module)
    setattr(module, func.__name__, func)
    return func
