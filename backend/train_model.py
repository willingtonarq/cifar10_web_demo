from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cifar10_dense_model.keras"
DATASET_PATH = Path.home() / ".keras" / "datasets" / "cifar-10-python.tar.gz"
DATASET_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"


def ensure_dataset() -> None:
    if DATASET_PATH.exists() and DATASET_PATH.stat().st_size > 150_000_000:
        return
    if DATASET_PATH.exists():
        DATASET_PATH.unlink()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando CIFAR-10 en {DATASET_PATH}...")
    urlretrieve(DATASET_URL, DATASET_PATH)


def main() -> None:
    tf.keras.utils.set_random_seed(42)

    ensure_dataset()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(32, 32, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.MeanSquaredError(name="mse")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=40,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )
    test_loss, test_accuracy, test_mse = model.evaluate(x_test, y_test, verbose=0)
    model.save(MODEL_PATH)

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    print(f"Épocas ejecutadas: {len(history.history['loss'])}")
    print(f"Mejor época: {best_epoch}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Modelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
