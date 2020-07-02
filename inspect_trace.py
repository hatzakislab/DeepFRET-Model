import matplotlib.pyplot as plt
import tensorflow.keras

import lib.algorithms
import lib.plotting
from lib.ml import _merge_hmm_labels

trace_len = 300

traces = lib.algorithms.generate_traces(
    n_traces=1,
    scramble_decouple_prob = 0.95,
    aggregation_prob=0,
    scramble_prob=1,
    D_lifetime=200,
    A_lifetime=None,
    noise = 0.05,
    trace_length=trace_len,
    return_matrix=False,
    run_headless_parallel=True,
)


X = traces[["D-Dexc-rw", "A-Dexc-rw", "A-Aexc-rw"]].values.reshape(
    -1, trace_len, 3
)

labels = traces["label"]
labels = _merge_hmm_labels(labels)

print("class: {}".format(traces["label"][0]))

y = traces[["label"]].values.reshape(-1, trace_len, 1)
y = tensorflow.keras.utils.to_categorical(y, num_classes=6)

fig, axes = plt.subplots(nrows=5)
lib.plotting.plot_trace_and_preds(
    xi=X[0],
    yi=y[0],
    tracename="",
    smfret_axes=axes,
    detect_bleach = False,
    clrs=lib.plotting.colors_full,
    target_values=lib.plotting.target_vals,
)

plt.show()
