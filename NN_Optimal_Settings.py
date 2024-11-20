import tensorflow as tf

# Custom R² metric for regression
def r_squared(y_true, y_pred):
    total_error = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    residual_error = tf.reduce_sum(tf.square(y_true - y_pred))
    r2 = 1 - (residual_error / total_error)
    return r2

# Dictionary for loss functions, activations, and actual metric functions
LOSS_METRICS_DICT = {
    "Binary Cross-Entropy": {
        "activation": tf.keras.activations.sigmoid,
        "loss": tf.keras.losses.BinaryCrossentropy(),
        "metrics": [tf.keras.metrics.BinaryAccuracy()]  # Using the actual metric class
    },
    "Categorical Cross-Entropy": {
        "activation": tf.keras.activations.softmax,
        "loss": tf.keras.losses.CategoricalCrossentropy(),
        "metrics": [tf.keras.metrics.CategoricalAccuracy()]  # Using the actual metric class
    },
    "Sparse Categorical Cross-Entropy": {
        "activation": tf.keras.activations.softmax,
        "loss": tf.keras.losses.SparseCategoricalCrossentropy(),
        "metrics": [tf.keras.metrics.SparseCategoricalAccuracy()]  # Using the actual metric class
    },
    "Mean Squared Error": {
        "activation": tf.keras.activations.linear,
        "loss": tf.keras.losses.MeanSquaredError(),
        "metrics": [
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.R2Score()
        ]
    }
}




