import os.path
from warnings import warn

import numpy as np
import scipy.signal
import sklearn.model_selection
import sklearn.utils
from send2trash import send2trash
from tensorflow import keras


def smoothe_one_hot_labels(y, amount):
    """
    Smoothes labels towards 0.5 for all classes.
    """
    return y * (1 - amount) + 0.5 * amount


def labels_to_binary(y, one_hot, to_ones):
    """Converts group labels to binary labels, given desired targets"""
    if one_hot:
        y = y.argmax(axis=2)
    y[~np.isin(y, to_ones)] = -1
    y[y != -1] = 1
    y[y != 1] = 0
    return y


def preprocess_2d_timeseries_seq2seq(
    X, y, n_timesteps,
):
    """Preprocess X and y for seq2seq learning"""
    X = X.reshape(-1, n_timesteps, X.shape[1])
    y = y.reshape(-1, n_timesteps, 1)
    return X, y


def class_to_one_hot(*y, num_classes):
    """Turns classes [1,2,3...] into one-hot encodings"""
    y_cat = [keras.utils.to_categorical(yi, num_classes) for yi in list(y)]
    if len(y_cat) == 1:
        y_cat = np.squeeze(y_cat)
    return y_cat


def gpu_model_to_cpu(trained_gpu_model, untrained_cpu_model, outdir, modelname):
    """
    Loads a keras GPU model and saves it as a CPU-compatible model.
    The models must be exactly alike.
    """
    weights = os.path.join(str(outdir), "weights_temp.h5")
    trained_gpu_model.save_weights(weights)
    untrained_cpu_model.load_weights(weights)
    keras.models.save_model(
        untrained_cpu_model, os.path.join(str(outdir), modelname)
    )
    try:
        send2trash(weights)
    except OSError:
        warn(
            "Didn't trash file (probably because of Google Drive)",
            RuntimeWarning,
        )


def balance_classes(X, y, frame=0, exclude_label_from_limiting=0, shuffle=True):
    """
    Parameters
    ----------
    y:
        Tensor with labels [0, 1, 2, ...]
    exclude_label_from_limiting:
        Label(s) to exclude as limiting factors
    shuffle:
        Whether to reshuffle classes

    Returns
    -------
    Balanced classes
    """
    assert len(X) == len(y)

    prebalance = np.bincount(y[:, frame, 0].astype(int))
    scores = np.zeros(len(prebalance))
    if exclude_label_from_limiting is not None:
        prebalance = np.delete(prebalance, exclude_label_from_limiting)
    limiting = np.min(prebalance)

    balanced_X = []
    balanced_y = []
    for n in range(len(X)):
        yi = y[n, :, :]
        li = np.int(yi[frame])
        if scores[li] < limiting:
            xi = X[n, :, :]
            balanced_X.append(xi)
            balanced_y.append(yi)
            scores[li] += 1
    balanced_X, balanced_y = [np.array(arr) for arr in (balanced_X, balanced_y)]
    if shuffle:
        sklearn.utils.shuffle(balanced_X, balanced_y)
    return balanced_X, balanced_y


def generate_callbacks(
    outdir, patience, name, monitor="val_loss", mode="min", verbose=1
):
    """Generate callbacks for model"""
    checkpoint_params = dict(
        save_best_only=True, monitor=monitor, mode=mode, verbose=False
    )

    log = keras.callbacks.CSVLogger(
        filename=os.path.join(str(outdir), name + "_training.log"), append=False
    )
    early_stopping = keras.callbacks.EarlyStopping(
        patience=patience, monitor=monitor, verbose=verbose, mode=mode
    )

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(str(outdir), name + "_best_model.h5"), **checkpoint_params
    )
    weight_checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(str(outdir), name + "_best_model_weights.h5"),
        save_weights_only=True,
        **checkpoint_params
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=2,
        mode="auto",
        min_delta=0.0001,
        cooldown=1,
        min_lr=0,
    )
    return [
        log,
        early_stopping,
        model_checkpoint,
        weight_checkpoint,
        reduce_lr,
    ]


def seq_probabilities(yi, target_values, bleach_skip_threshold=0.5, skip_column=0):
    """
    Calculates class-wise probabilities over the entire trace for a one-hot encoded
    sequence prediction. Skips values where the first value is above threshold (bleaching)
    """
    assert len(yi.shape) == 2

    p = yi[yi[:, skip_column] < bleach_skip_threshold]  # Discard rows above threshold
    if len(p) > 0:
        p = p.sum(axis=0) / len(p)  # Sum rows for each class
        p = p / p.sum()  # Normalize probabilities to 1
        # p[skip_column] = 0
    else:
        p = np.zeros(yi.shape[1])
    confidence = p[target_values].sum()
    return p, confidence


def find_bleach(p_bleach, threshold=0.5, window=7):
    """
    Finds bleaching given a list of frame-wise probabilities.
    The majority of datapoints in a given window must be above the threshold
    """
    is_bleached = scipy.signal.medfilt(p_bleach > threshold, window)
    bleach_frame = np.argmax(is_bleached)
    if bleach_frame == 0:
        bleach_frame = None
    return bleach_frame