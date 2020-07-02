import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import tensorflow as tf
import argparse

running_on_google_colab = "google.colab" in sys.modules
if running_on_google_colab:
    # Must come before custom lib imports
    sys.path.append("./gdrive/My Drive/Colab Notebooks/DeepFRET-Model")
    plt.style.use("default")

import lib.model
import lib.plotting
import lib.ml
import lib.utils

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Name of model output", required=True, type = str)
parser.add_argument(
    "-e",
    "--exclude-alex",
    help="Whether to exclude ALEX from training data",
    required=True,
    type = lib.utils.str2bool
)
args = vars(parser.parse_args())

def main(
    running_on_google_colab,
    datadir,
    rootdir,
    outdir,
    percent_of_data,
    regression,
    dataname,
    tag,
    train,
    new_model,
    callback_timeout,
    epochs,
    batch_size,
    model_function,
    use_fret_for_training,
    exclude_alex_fret,
):

    gpu_available = tf.test.is_gpu_available()

    if new_model:
        print("**Training new model**")
    else:
        print("**Training most recent model**")

    rootdir = Path(rootdir)
    if running_on_google_colab:
        rootdir = "./gdrive/My Drive/Colab Notebooks/DeepFRET-Model"

    rootdir = Path(rootdir)
    outdir = rootdir.joinpath(outdir).expanduser()
    datadir = rootdir.joinpath(datadir).expanduser()

    X, labels = lib.utils.load_npz_data(
        top_percentage=percent_of_data,
        path=datadir,
        set_names=("X_" + dataname, "y_" + dataname),
    )
    n_classes = len(np.unique(labels))

    if not regression:
        # Use labels as classification target
        y = lib.ml.class_to_one_hot(labels, num_classes=n_classes)
        y = lib.ml.smoothe_one_hot_labels(y, amount=0.05)
    else:
        # Use E_true column as regression target
        y = np.expand_dims(X[..., 3], axis=-1)

    if use_fret_for_training:
        # Use E_raw column as input
        X = np.expand_dims(X[..., 4], axis=-1)
        X = X.clip(2, -2)
    else:
        X = X[..., 0:2] if exclude_alex_fret else X[..., 0:3]
        X = lib.utils.sample_max_normalize_3d(X)

    print("X: ", X.shape)
    print("y: ", y.shape)

    print("Splitting dataset...")
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    model_name = "{}_best_model.h5".format(dataname)

    model = lib.model.get_model(
        n_features=X.shape[-1],
        n_classes=n_classes,
        train=train,
        new_model=new_model,
        model_name=model_name,
        model_path=outdir,
        gpu=gpu_available,
        tag=tag,
        regression=regression,
        model_function=model_function,
    )

    if tag is not None:
        dataname += "_" + tag
        model_name = model_name.replace("best_model", tag + "_best_model")

    if train:
        callbacks = lib.ml.generate_callbacks(
            patience=callback_timeout, outdir=outdir, name=dataname
        )
        model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )
        try:
            lib.plotting.plot_losses(
                logpath=outdir, outdir=outdir, name=dataname
            )
        except IndexError:
            pass

        # Convert final model to GPU
        if gpu_available:
            print("Converted model from GPU to CPU-compatible")
            cpu_model = model_function(
                gpu=False,
                n_features=X.shape[-1],
                regression=regression,
                n_classes=n_classes,
            )
            lib.ml.gpu_model_to_cpu(
                trained_gpu_model=model,
                untrained_cpu_model=cpu_model,
                outdir=outdir,
                modelname=model_name,
            )

    print("Evaluating...")
    y_pred = model.predict(X_val)

    if not regression:
        lib.plotting.plot_confusion_matrices(
            y_target=y_val,
            y_pred=y_pred,
            y_is_binary=False,
            targets_to_binary=[4, 5, 6, 7, 8],
            outdir=outdir,
            name=dataname,
        )


if __name__ == "__main__":
    # In order to run this on Google Colab, everything must be placed
    # according to "~/Google Drive/Colab Notebooks/DeepFRET/"
    main(
        running_on_google_colab=running_on_google_colab,
        regression=False,
        train=True,
        new_model=True,
        rootdir=".",
        datadir="data",
        outdir="output",
        dataname="sim",
        tag=args["name"],
        percent_of_data=100,
        batch_size=32,
        epochs=100,
        callback_timeout=5,
        model_function=lib.model.create_deepconvlstm_model,
        use_fret_for_training=False,
        exclude_alex_fret=args["exclude_alex"],
    )
