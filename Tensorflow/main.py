import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 28, 28), dtype=tf.uint8)
    ])
    def predict(self, x):
        x = tf.expand_dims(x, axis=3)
        x = tf.cast(x, tf.float32)
        x = x / 255.0
        return self.call(x)

# Create an instance of the model


def load_data(path):
    from mnist import MNIST
    mndata = MNIST(path, return_type='numpy')

    x_train, y_train = mndata.load_training()
    x_test, y_test = mndata.load_testing()

    x_train = x_train.reshape((-1, 28, 28))
    x_test = x_test.reshape((-1, 28, 28))

    return (x_train[:1000], y_train[:1000]), (x_test[:100], y_test[:100])


def log_progress(fp, message, end=False, success=False, metrics={}):
    import pathlib
    import json
    from datetime import datetime
    with pathlib.Path(fp).open('a') as f:
        if not end:
            f.write(json.dumps(
                {"event": "STARTED",
                 "ts": datetime.now().isoformat(),
                 "description": message
                 }))
        elif success:
            f.write(json.dumps(
                {"event": "SUCCESS",
                 "ts": datetime.now().isoformat(),
                 "metrics": metrics,
                 "description": message
                 }))
        else:
            f.write(json.dumps(
                {"event": "ERROR",
                 "ts": datetime.now().isoformat(),
                 "description": message
                 }))
        f.write('\n')


def train(dataset_path, workspace, progress_log_path):
    import pathlib
    workspace = pathlib.Path(workspace)

    model = MyModel()
    (x_train, y_train), (x_test, y_test) = load_data(dataset_path)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')

    EPOCHS = 10

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        message = (
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )

        log_progress(progress_log_path, message)

        print(message)

    log_progress(progress_log_path, 'Succeeded', end=True, success=True)

    export_dir = workspace / 'export/1'
    export_dir = export_dir.as_posix()
    tf.saved_model.save(model, export_dir, {'predict': model.predict})


if __name__ == "__main__":
    import os
    train(os.getenv('DATASET') or '/data/dataset',
          os.getenv('WORKSPACE') or '/data/workspace', 
          os.getenv('PROGRESS_LOG_PATH') or '/data/progress/log.jsonl')
