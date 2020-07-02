from tensorflow import keras
from tensorflow.python.keras import Input, backend as K
from tensorflow.python.keras.layers import (Activation, BatchNormalization, Bidirectional, Conv1D, CuDNNLSTM, Dense,
                                            Dropout, LSTM, Lambda, MaxPooling1D, TimeDistributed, add)


class VariableRepeatVector:
    """
    Tidies up the call to a lambda function by integrating it in a
    layer-like wrapper

    The two usages are identical:
    decoded = VariableRepeatVector()([inputs, encoded])
    decoded = Lambda(variable_repeat)([inputs, encoded])
    """

    @staticmethod
    def variable_repeat(x):
        # matrix with ones, shaped as (batch, steps, 1)
        step_matrix = K.ones_like(x[0][:, :, :1])
        # latent vars, shaped as (batch, 1, latent_dim)
        latent_matrix = K.expand_dims(x[1], axis=1)
        return K.batch_dot(step_matrix, latent_matrix)

    def __call__(self, x):
        return Lambda(self.variable_repeat)(x)


class ResidualConv1D:
    """
    ResidualConv1D for use with best performing classifier
    """

    def __init__(self, filters, kernel_size, pool=False):
        self.pool = pool
        self.kernel_size = kernel_size
        self.params = {
            "padding": "same",
            "kernel_initializer": "he_uniform",
            "strides": 1,
            "filters": filters,
        }

    def build(self, x):

        res = x
        if self.pool:
            x = MaxPooling1D(1, padding="same")(x)
            res = Conv1D(kernel_size=1, **self.params)(res)

        out = Conv1D(kernel_size=1, **self.params)(x)

        out = BatchNormalization()(out)
        out = Activation("relu")(out)
        out = Conv1D(kernel_size=self.kernel_size, **self.params)(out)

        out = BatchNormalization()(out)
        out = Activation("relu")(out)
        out = Conv1D(kernel_size=self.kernel_size, **self.params)(out)

        out = add([res, out])

        return out

    def __call__(self, x):
        return self.build(x)


def create_lstm_model(gpu, n_features, regression):
    LSTM_ = CuDNNLSTM if gpu else LSTM

    inputs = Input(shape=(None, n_features))

    x = Bidirectional(LSTM_(units=128, return_sequences=True))(inputs)
    x = Bidirectional(LSTM_(units=64, return_sequences=True))(x)
    final = x

    if regression:
        outputs = TimeDistributed(Dense(1, activation=None))(final)
        metric = "mse"
        loss = "mse"
    else:
        outputs = TimeDistributed(Dense(6, activation="softmax"))(final)
        metric = "accuracy"
        loss = "categorical_crossentropy"

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer="adam", metrics=[metric])
    return model


def create_deepconvlstm_model(gpu, n_features, n_classes, regression):
    """
    Creates Keras model that resulted in the best performing classifier so far
    """

    LSTM_ = CuDNNLSTM if gpu else LSTM

    inputs = Input(shape=(None, n_features))

    x = Conv1D(
        filters=32,
        kernel_size=16,
        padding="same",
        kernel_initializer="he_uniform",
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # residual net part
    x = ResidualConv1D(filters=32, kernel_size=65, pool=True)(x)  # 65
    x = ResidualConv1D(filters=32, kernel_size=65)(x)
    x = ResidualConv1D(filters=32, kernel_size=65)(x)

    x = ResidualConv1D(filters=64, kernel_size=33, pool=True)(x)  # 33
    x = ResidualConv1D(filters=64, kernel_size=33)(x)
    x = ResidualConv1D(filters=64, kernel_size=33)(x)

    x = ResidualConv1D(filters=128, kernel_size=15, pool=True)(x)  # 15
    x = ResidualConv1D(filters=128, kernel_size=15)(x)
    x = ResidualConv1D(filters=128, kernel_size=15)(x)

    x = ResidualConv1D(filters=256, kernel_size=7, pool=True)(x)  # 7
    x = ResidualConv1D(filters=256, kernel_size=7)(x)
    x = ResidualConv1D(filters=256, kernel_size=7)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(1, padding="same")(x)
    x = Bidirectional(LSTM_(16, return_sequences=True))(x)
    x = Dropout(rate=0.4)(x)

    if regression:
        outputs = Dense(1, activation=None)(x)
        metric = "mse"
        loss = "mse"
    else:
        outputs = Dense(n_classes, activation="softmax")(x)
        metric = "accuracy"
        loss = "categorical_crossentropy"

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer="adam", metrics=[metric])
    return model

def get_model(
    n_features,
    n_classes,
    train,
    new_model,
    model_name,
    model_path,
    model_function,
    gpu,
    regression,
    print_summary=True,
    tag=None,
):
    """Loader for model"""
    if train:
        if new_model:
            print("Created new model.")
            model = model_function(
                gpu=gpu,
                n_features=n_features,
                n_classes = n_classes,
                regression=regression,
            )
        else:
            try:
                if tag is not None:
                    model_name = model_name.replace(
                        "best_model", tag + "_best_model"
                    )
                model = keras.models.load_model(
                    str(model_path.joinpath(model_name))
                )
            except OSError:
                print("No model found. Created new model.")
                model = model_function(
                    gpu=gpu,
                    n_features=n_features,
                    n_classes = n_classes,
                    regression=regression,
                )
    else:
        print("Loading model from file..")
        model = keras.models.load_model(str(model_path.joinpath(model_name)))

    if print_summary:
        model.summary()
    return model
