import os
import matplotlib.ticker
import numpy as np
import pandas as pd
import mlxtend.evaluate
from matplotlib import pyplot as plt
import lib.ml
import lib.utils

# For predictions
labels_binary = ("non-smFRET", "smFRET")
colors_binary = ("red", "green")

# Labels
labels_full = (
    "Bleached",
    "Aggregate",
    "Noisy",
    "Scrambled",
    "Static",
    "Dynamic",
)

# Colors for each label
colors_full = ("darkgrey", "red", "royalblue", "purple", "orange", "green")

# DD, DA, AA, E, S for smFRET plots
fret_plot_colors = ("seagreen", "salmon", "firebrick", "orange", "purple")

target_vals = [4, 5]
max_target_val = 5
dynamic_val = 5

merge_cols = [5, 6, 7, 8]
keep_cols = [0, 1, 2, 3, 4]


def plot_losses(logpath, outdir, name, show_only=False):
    """Plots training and validation loss"""
    stats = pd.read_csv(
        os.path.join(str(logpath), name + "_training.log")
    ).values
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    axes = axes.ravel()

    epoch = stats[:, 0]
    train_acc = stats[:, 1]
    train_loss = stats[:, 2]
    val_acc = stats[:, 3]
    val_loss = stats[:, 4]

    best_loss = np.argmin(val_loss)

    axes[0].plot(epoch, train_loss, "o-", label="train loss", color="salmon")
    axes[0].plot(epoch, val_loss, "o-", label="val loss", color="lightblue")
    axes[0].axvline(best_loss, color="black", ls="--", alpha=0.5)

    axes[1].plot(
        epoch,
        train_acc,
        "o-",
        label="train acc (best: {:.4f})".format(train_acc.max()),
        color="salmon",
    )
    axes[1].plot(
        epoch,
        val_acc,
        "o-",
        label="val acc (best: {:.4f})".format(val_acc.max()),
        color="lightblue",
    )
    axes[1].axvline(best_loss, color="black", ls="--", alpha=0.5)

    axes[0].legend(loc="lower left")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    if show_only:
        plt.show()
    else:
        plt.savefig(os.path.join(str(outdir), name + "_loss.pdf"))
        plt.close()


def _plot_confusion_matrix_mlxtend(
    conf_mat,
    hide_spines=False,
    hide_ticks=False,
    cmap=None,
    colorbar=False,
    show_absolute=True,
    show_normed=False,
):
    """
    A modified version of mlxtend.plotting.plot_confusion_matrix
    -----------

    Plot a confusion matrix via matplotlib.
    Parameters
    -----------
    conf_mat : array-like, shape = [n_classes, n_classes]
        Confusion matrix from evaluate.confusion matrix.
    hide_spines : bool (default: False)
        Hides axis spines if True.
    hide_ticks : bool (default: False)
        Hides axis ticks if True
    figsize : tuple (default: (2.5, 2.5))
        Height and width of the figure
    cmap : matplotlib colormap (default: `None`)
        Uses matplotlib.pyplot.cm.Blues if `None`
    colorbar : bool (default: False)
        Shows a colorbar if True
    show_absolute : bool (default: True)
        Shows absolute confusion matrix coefficients if True.
        At least one of  `show_absolute` or `show_normed`
        must be True.
    show_normed : bool (default: False)
        Shows normed confusion matrix coefficients if True.
        The normed confusion matrix coefficients give the
        proportion of training examples per class that are
        assigned the correct label.
        At least one of  `show_absolute` or `show_normed`
        must be True.
    Returns
    -----------
    fig, ax : matplotlib.pyplot subplot objects
        Figure and axis elements of the subplot.
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/
    """
    if not (show_absolute or show_normed):
        raise AssertionError("Both show_absolute and show_normed are False")

    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype("float") / total_samples

    scale = 0.8 if len(conf_mat) > 2 else 1.3
    figsize = (len(conf_mat) * scale, len(conf_mat) * scale)

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues

    matshow = ax.matshow(normed_conf_mat, cmap=cmap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                n = matplotlib.ticker.EngFormatter(places=1).format_data(
                    conf_mat[i, j]
                )
                if float(n) < 1000:
                    n = str(int(float(n)))
                cell_text += n
                if show_normed:
                    cell_text += "\n" + "("
                    cell_text += format(normed_conf_mat[i, j], ".2f") + ")"
            else:
                cell_text += format(normed_conf_mat[i, j], ".2f")
            ax.text(
                x=j,
                y=i,
                s=cell_text,
                va="center",
                ha="center",
                color="white" if normed_conf_mat[i, j] > 0.5 else "black",
            )

    if hide_spines:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    plt.xlabel("predicted label")
    plt.ylabel("true label")
    return fig, ax


def plot_predictions(
    X, model, outdir, name, nrows, ncols, y_val=None, y_pred=None
):
    """Plots a number of predictions for quick inspection"""
    if y_pred is None:
        y_pred = model.predict(X)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(nrows * 2, ncols * 2)
    )
    axes = axes.ravel()

    clrs = ("darkgrey", "salmon", "seagreen", "darkorange", "royalblue", "cyan")
    for i, ax in enumerate(axes):
        xi_val = X[i, :, :]
        yi_prd = y_pred[i, :, :]

        ax.plot(xi_val[:, 0], color="darkgreen", alpha=0.30)
        ax.plot(xi_val[:, 1], color="darkred", alpha=0.30)
        # Plot y_pred as lines
        for j, c in zip(range(len(clrs)), clrs):
            ax.plot(yi_prd[:, j], color=c, lw=2)

        yi_val = y_val[i, :, :] if y_val is not None else y_pred[i, :, :]
        plot_category(yi_val, colors=clrs, alpha=0.30, ax=ax)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_ylim(-0.15, 1.15)
        ax.set_xlim(0, len(xi_val))

    if outdir is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(str(outdir), name + ".pdf"))
        plt.close()


def plot_confusion_matrices(
    y_target,
    y_pred,
    name="",
    outdir=None,
    targets_to_binary=None,
    y_is_binary=False,
    ticks_binary=None,
    ticks_multi=None,
    show_abs=False,
):
    """
    Plots multiclass and binary confusion matrices for smFRET classification
    *Very* hard-coded section, so make sure 0, 1, 2,.. labels match the strings!
    """
    axis = 2 if len(y_target.shape) == 3 else 1
    mkwargs = dict(show_normed=True, show_absolute=show_abs, colorbar=True)

    if y_is_binary:
        matrix = mlxtend.evaluate.confusion_matrix(
            y_target=y_target.argmax(axis=axis).ravel(),
            y_predicted=y_pred.argmax(axis=axis).ravel(),
        )

        fig, ax = _plot_confusion_matrix_mlxtend(matrix, **mkwargs)
        l = (
            ticks_binary
            if ticks_binary is not None
            else [""] + list(labels_binary)
        )
        ax.set_yticklabels(l)
        ax.set_xticklabels(l, rotation=90)
        plt.tight_layout()
        if outdir is not None:
            plt.savefig(
                os.path.join(str(outdir), name + "_binary_confusion_matrix.pdf")
            )
            plt.close()
    else:
        y_target, y_pred = [
            y.argmax(axis=axis).ravel() for y in (y_target, y_pred)
        ]

        matrix = mlxtend.evaluate.confusion_matrix(
            y_target=y_target, y_predicted=y_pred
        )

        l = ticks_multi if ticks_multi is not None else [""] + list(labels_full)
        fig, ax = _plot_confusion_matrix_mlxtend(matrix, **mkwargs)
        ax.set_yticklabels(l)
        ax.set_xticklabels(l, rotation=90)
        plt.tight_layout()
        if outdir is not None:
            plt.savefig(os.path.join(str(outdir), name + "_confusion_matrix.pdf"))
            plt.close()

        if (
            targets_to_binary is not None
        ):  # Converts smFRET classification to a binary problem
            y_target_b, y_pred_b = [
                lib.ml.labels_to_binary(
                    y, one_hot=False, to_ones=targets_to_binary
                ).ravel()
                for y in (y_target, y_pred)
            ]

            matrix = mlxtend.evaluate.confusion_matrix(
                y_target=y_target_b, y_predicted=y_pred_b
            )
            l = (
                ticks_binary
                if ticks_binary is not None
                else [""] + list(labels_binary)
            )
            fig, ax = _plot_confusion_matrix_mlxtend(matrix, **mkwargs)
            ax.set_yticklabels(l)
            ax.set_xticklabels(l, rotation=90)
            plt.tight_layout()
            if outdir is not None:
                plt.savefig(
                    os.path.join(str(outdir), name + "_binary_confusion_matrix.pdf")
                )
                plt.close()


def plot_category(y, ax, colors=None, alpha=0.2):
    """
    Plots a color for every class segment in a timeseries

    Parameters
    ----------
    y_:
        One-hot coded or categorical labels
    ax:
        Ax for plotting
    colors:
        Colors to cycle through
    """
    if colors is None:
        colors = colors_full

    y_ = y.argmax(axis=1) if len(y.shape) != 1 else y
    if len(colors) < len(set(y_)):
        raise ValueError("Must have at least a color for each class")

    adjs, lns = lib.utils.count_adjacent_values(y_)
    position = range(len(y_))
    for idx, ln in zip(adjs, lns):
        label = y_[idx]
        ax.axvspan(
            xmin=position[idx],
            xmax=position[idx] + ln,
            alpha=alpha,
            facecolor=colors[label],
        )


def plot_trace_label_distribution(X, y, method="multi"):
    """
    Plots the distribution of labels over time
    """
    if method == "binary":
        pal = ["#F19E9B", "#A4CC9E"]
        lbs = ["Non-usable", "Usable"]
        labels = [0, 1]
        y = lib.ml.labels_to_binary(y, one_hot=False, to_ones=(4, 5, 6, 7, 8))
    elif method == "multi":
        if len(np.unique(y)) == 2:
            pal = ["#F19E9B", "seagreen"]
            lbs = ["Non-usable", "Usable"]
            labels = [0, 1]
        else:
            pal = [
                "darkgrey",
                "red",
                "royalblue",
                "mediumvioletred",
                "orange",
                "lightgreen",
                "springgreen",
                "limegreen",
                "green",
            ]

            lbs = [
                "Bleached",
                "Aggregate",
                "Noisy",
                "Scramble",
                "1-state",
                "2-state",
                "3-state",
                "4-state",
                "5-state",
            ]

            labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]

            if len(np.unique(y)) == 6:
                pal = pal[0:6]
                lbs = lbs[0:6]
                labels = labels[0:6]
                lbs[-1] = "Dynamic"
                lbs[-2] = "Static"
    else:
        raise ValueError

    label_count = []
    for label in labels:
        masked = np.ma.masked_not_equal(y, label, copy=True)
        mc = masked.count(axis=0).T[0]
        label_count.append(mc)

    x = range(1, len(label_count[0]) + 1)
    y = label_count

    y_ = np.reshape(y, newshape=(len(labels), -1))

    fig, ax = plt.subplots()
    ax.stackplot(x, y, labels=lbs, colors=pal, edgecolor="black", alpha=0.5)
    ax.set_xlim(1, X.shape[1])
    ax.set_ylim(0, y_[:, 0].sum())
    ax.legend(loc="lower right")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cumulative frame count")
    plt.tight_layout()
    return fig, ax


def _align_yaxis(ax1, ax2, v1=0, v2=0):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny + dy, maxy + dy)


def plot_smfret_trace(x, axes, custom_cmap=None, legend=True, align_y=True):
    """
    Parameters
    ----------
    signals:
        DD, DA, AA, signals from smFRET measurements
    axes:
        Axes on which to return on (must have 6 axes in total to find the plots)

    Returns
    -------
    Axes with plots on
    """
    DD = x[:, 0]
    DA = x[:, 1]
    AA = x[:, 2]

    E = DA / (DD + DA)
    S = (DD + DA) / (DD + DA + AA)

    try:
        axes = axes.ravel()
    except AttributeError:
        pass

    if len(axes) < 4:
        raise ValueError("Not enough axes for all plots")

    t = np.arange(len(DD))
    bg = np.zeros(len(DD))

    ALPHA = 0.2

    cmap = "seagreen", "salmon", "firebrick", "orange", "purple"
    if custom_cmap is not None:
        if len(custom_cmap) != len(cmap):
            raise ValueError("Custom cmap must contain exactly 5 colors")
        cmap = custom_cmap

    axes[0].plot(t, DD, color=cmap[0], lw=1.5, label="DD")
    axes[0].plot(t, bg, color="black", ls="--")
    axes[0].set_ylabel("DD")

    ax_acc = axes[0].twinx()
    ax_acc.plot(t, DA, color=cmap[1], label="DA")
    ax_acc.set_ylabel("DA")

    axes[1].plot(t, AA, color=cmap[2], alpha=1, label="AA")
    axes[1].plot(t, bg, color="black", ls="--", alpha=1)
    axes[1].set_ylabel("AA")

    axes[2].plot(t, E, color=cmap[3], label="E")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_ylabel("E")

    axes[3].plot(t, S, color=cmap[4], label="S")
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].axhline(0.5, color=cmap[4], alpha=ALPHA, ls=":")
    axes[3].set_ylabel("S")

    for a in axes:
        a.set_xlim(0, t.max())
        if legend:
            a.legend(loc="upper right")
        if a != axes[-1]:
            a.set_xticks([])
    if legend:
        ax_acc.legend(loc="lower right")
    if align_y:
        ax_acc.set_ylim(-0.15, 1.15)
        axes[0].set_ylim(-0.15, 1.15)
        _align_yaxis(axes[0], ax_acc)

    return axes, ax_acc


def plot_trace_and_preds(
    xi,
    yi,
    tracename,
    target_values,
    smfret_axes,
    detect_bleach=True,
    clrs=None,
    outdir=None,
    binary=False,
    y_line=False,
    y_shade=True,
    noticks=False,
    yi_true=None,
    shade_as_groundtruth=False,
    bleach_skip_threshold=0.5,
):
    """Plots a single trace from a set of X_rw and y_pred"""
    plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.85)

    smfret_axes, ax_acc = lib.plotting.plot_smfret_trace(
        xi, axes=smfret_axes, legend=False, align_y=False
    )

    if detect_bleach:
        bleach = lib.ml.find_bleach(
            yi[:, 0], threshold=bleach_skip_threshold, window=15
        )
        if bleach is not None:
            for ax in smfret_axes:
                ax.axvspan(bleach, len(xi), color="lightgrey", alpha=0.5)

    if noticks:
        plt.subplots_adjust(wspace=0, hspace=0)
        for ax in smfret_axes:
            ax.set_yticks([])
        ax_acc.set_yticks([])
        smfret_axes[-1].set_xticks([])
        smfret_axes[-1].set_ylabel("label")

    if clrs is not None:
        clrs = colors_binary if binary else colors_full

    if y_line:
        p, confidence = lib.ml.seq_probabilities(
            yi,
            bleach_skip_threshold=bleach_skip_threshold,
            target_values=target_values,
        )

        for i in range(yi.shape[-1]):
            # plot_trace_and_preds predicted probabilities
            smfret_axes[-1].plot(
                yi[:, i],
                color=clrs[i],
                alpha=0.6,
                label="{:.2f} %".format(p[i] * 100) if p[i] != 0 else None,
            )

        smfret_axes[-1].annotate(
            s="confidence: {} %".format(round(p[target_vals].sum() * 100, 0)),
            xy=(0.02, 0.8),
            xycoords="axes fraction",
            fontweight="bold",
        )
        smfret_axes[-1].legend(loc="upper right", ncol=2)
        smfret_axes[-1].set_ylim(-0.15, 1.15)
        smfret_axes[-1].set_ylabel("class")

    if y_shade:
        if shade_as_groundtruth:
            lib.plotting.plot_category(
                y=yi_true, ax=smfret_axes[-1], alpha=0.4, colors=clrs
            )
        else:
            lib.plotting.plot_category(
                y=yi, ax=smfret_axes[-1], alpha=0.4, colors=clrs
            )

    if outdir is not None:
        plt.suptitle(tracename)
        path = os.path.expanduser(
            os.path.join(str(outdir), str(tracename) + ".pdf")
        )
        plt.savefig(path)
        plt.close()
    else:
        return smfret_axes, ax_acc
