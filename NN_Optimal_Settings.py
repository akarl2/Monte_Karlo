import tensorflow as tf

# Custom R² metric for regression
def r_squared(y_true, y_pred):
    total_error = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    residual_error = tf.reduce_sum(tf.square(y_true - y_pred))
    r2 = 1 - (residual_error / total_error)
    return r2

# Dictionary for loss functions, activations, and metrics
LOSS_METRICS_DICT = {
    "BinaryCrossEntropy": {
        "activation": tf.keras.activations.sigmoid,
        "loss": tf.keras.losses.BinaryCrossentropy(),
        "metrics": [tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    },
    "CategoricalCrossEntropy": {
        "activation": tf.keras.activations.softmax,
        "loss": tf.keras.losses.CategoricalCrossentropy(),
        "metrics": [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    },
    "SparseCategoricalCrossEntropy": {
        "activation": tf.keras.activations.softmax,
        "loss": tf.keras.losses.SparseCategoricalCrossentropy(),
        "metrics": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    },
    "MeanSquaredError": {
        "activation": tf.keras.activations.linear,
        "loss": tf.keras.losses.MeanSquaredError(),
        "metrics": [
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            r_squared
        ]
    }
}


