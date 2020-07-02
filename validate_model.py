import numpy as np
import sklearn.model_selection
import tensorflow.keras.models
import matplotlib.pyplot as plt
import lib.plotting
import lib.ml
import lib.utils

REGRESSION = False

model = tensorflow.keras.models.load_model(
    "output/sim_experimental_best_model.h5")

X, labels = lib.utils.load_npz_data(
    top_percentage=100,
    path="./data",
    set_names=("X_test", "y_test"),
)
print(X.shape)
print("Contains labels: ", np.unique(labels))

if REGRESSION:
    # Use E_true column as regression target
    y = np.expand_dims(X[..., 4], axis = -1)

else:
    # Use labels as classification target
    set_y = set(labels.ravel())
    y = lib.ml.class_to_one_hot(labels, num_classes=len(set_y))
    y = lib.ml.smoothe_one_hot_labels(y, amount=0.05)

print("X: ", X.shape)
print("y: ", y.shape)
print("Splitting dataset...")
_, X_val, _, y_val = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2, random_state=1
)

if REGRESSION:
    E_true = X_val[..., [0, 1, 2]]
    E_pred = model.predict(np.expand_dims(X_val[..., 4], axis = -1))

    fig, ax = plt.subplots(nrows = 5, ncols = 2, figsize = (15, 10))

    for i in range(5):
        ax[i, 0].plot(X_val[i, :, 0], color = "green", label = "D")
        ax[i, 0].plot(X_val[i, :, 1], color = "red", label = "A")

        ax[i, 1].plot(E_true[i], label = "FRET", color = "grey")
        ax[i, 1].plot(E_pred[i], label = "FRET PRED", color = "red")
        ax[i, 1].plot(y_val[i], label = "FRET TRUE", color = "orange")
        ax[i, 1].set_ylim(0, 1)

        ax[i, 0].legend(loc = "upper right")
        ax[i, 1].legend(loc = "upper right")
    plt.tight_layout()
else:
    y_pred = model.predict(X_val, verbose = True)
    lib.plotting.plot_confusion_matrices(
        y_target=y_val,
        y_pred=y_pred,
        y_is_binary=False,
    )

plt.show()