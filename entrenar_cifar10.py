"""Entrenamiento local de la red densa CIFAR-10.

Equivalente al notebook incompleto, sin Google Colab.
El modelo queda en backend/cifar10_dense_model.keras
y las figuras en resultados/.
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "backend" / "cifar10_dense_model.keras"
RESULTS_DIR = ROOT / "resultados"

# El enunciado permite hasta 40 épocas y un batch de 128.
# EarlyStopping puede cortar antes si val_loss deja de mejorar.
MAX_EPOCHS = 40
BATCH_SIZE = 128

CLASS_NAMES = [
    "Avión", "Automóvil", "Ave", "Gato", "Ciervo",
    "Perro", "Rana", "Caballo", "Barco", "Camión",
]


def load_data():
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train_full = tf.keras.utils.to_categorical(y_train_full.squeeze(), 10)
    y_test_onehot = tf.keras.utils.to_categorical(y_test.squeeze(), 10)

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(x_train_full))
    val_idx = indices[:5000]
    train_idx = indices[5000:]

    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]
    x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]

    print(f"Entrenamiento: {x_train.shape[0]} imágenes")
    print(f"Validación:    {x_val.shape[0]} imágenes")
    print(f"Prueba:        {x_test.shape[0]} imágenes")
    print("Forma de cada imagen:", x_train.shape[1:])
    return (x_train, y_train), (x_val, y_val), (x_test, y_test, y_test_onehot)


def save_sample_grid(x_train, y_train):
    fig = plt.figure(figsize=(10, 5))
    for i in range(15):
        plt.subplot(3, 5, i + 1)
        plt.imshow(x_train[i])
        plt.title(CLASS_NAMES[int(np.argmax(y_train[i]))], fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    path = RESULTS_DIR / "muestras_entrenamiento.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("Muestras guardadas en", path)


def build_model():
    tf.keras.utils.set_random_seed(42)

    # ReLU en las capas ocultas: no se satura como sigmoid y entrena mejor esta red.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.MeanSquaredError(name="mse"),
        ],
    )
    model.summary()
    return model


def train(model, x_train, y_train, x_val, y_val):
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
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed_seconds = time.time() - start_time
    epochs_used = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

    print(f"\nÉpocas ejecutadas: {epochs_used} de {MAX_EPOCHS} máximas")
    print(f"Época con menor val_loss: {best_epoch}")
    print(f"Tiempo de entrenamiento: {elapsed_seconds / 60:.2f} minutos")
    print(f"Menor pérdida de validación registrada: {min(history.history['val_loss']):.4f}")
    return history, epochs_used


def save_training_curves(history, epochs_used):
    epochs_range = range(1, epochs_used + 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    axes[0].plot(epochs_range, history.history["loss"], marker="o", label="Entrenamiento")
    axes[0].plot(epochs_range, history.history["val_loss"], marker="o", label="Validación")
    axes[0].set_title("Pérdida (categorical cross-entropy)")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Pérdida")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs_range, history.history["mse"], marker="o", label="Entrenamiento")
    axes[1].plot(epochs_range, history.history["val_mse"], marker="o", label="Validación")
    axes[1].set_title("Error cuadrático medio (MSE)")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs_range, history.history["accuracy"], marker="o", label="Entrenamiento")
    axes[2].plot(epochs_range, history.history["val_accuracy"], marker="o", label="Validación")
    axes[2].set_title("Accuracy")
    axes[2].set_xlabel("Época")
    axes[2].set_ylabel("Accuracy")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    path = RESULTS_DIR / "curvas_entrenamiento.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("Curvas guardadas en", path)


def verify_predictions(model, x_test, y_test):
    """Muestra 15 imágenes de prueba con la clase real y la predicha."""
    n_show = 15
    sample = x_test[:n_show]
    true_idx = y_test[:n_show].squeeze()
    probabilities = model.predict(sample, verbose=0)
    pred_idx = np.argmax(probabilities, axis=1)

    correct = int(np.sum(pred_idx == true_idx))
    print(f"Aciertos en la muestra visual: {correct}/{n_show}")
    for i in range(n_show):
        mark = "ok" if pred_idx[i] == true_idx[i] else "fallo"
        print(
            f"  {i + 1:2d}. real={CLASS_NAMES[int(true_idx[i])]:<10} "
            f"pred={CLASS_NAMES[int(pred_idx[i])]:<10} "
            f"conf={probabilities[i, pred_idx[i]]:.2%}  {mark}"
        )

    fig = plt.figure(figsize=(10, 5.5))
    for i in range(n_show):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(sample[i])
        match = pred_idx[i] == true_idx[i]
        ax.set_title(
            f"R: {CLASS_NAMES[int(true_idx[i])]}\nP: {CLASS_NAMES[int(pred_idx[i])]}",
            fontsize=8,
            color="green" if match else "crimson",
        )
        ax.axis("off")
    plt.tight_layout()
    path = RESULTS_DIR / "predicciones_prueba.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("Predicciones guardadas en", path)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print("TensorFlow:", tf.__version__)
    print("Dispositivos:", tf.config.list_physical_devices())

    (x_train, y_train), (x_val, y_val), (x_test, y_test, y_test_onehot) = load_data()
    save_sample_grid(x_train, y_train)

    model = build_model()
    history, epochs_used = train(model, x_train, y_train, x_val, y_val)
    save_training_curves(history, epochs_used)

    test_loss, test_accuracy, test_mse = model.evaluate(x_test, y_test_onehot, verbose=0)
    print(f"Test loss (cross-entropy): {test_loss:.4f}")
    print(f"Test accuracy:             {test_accuracy:.2%}")
    print(f"Test MSE:                  {test_mse:.4f}")

    verify_predictions(model, x_test, y_test)

    model.save(MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / (1024 ** 2)
    print("Modelo guardado en", MODEL_PATH)
    print(f"Tamaño: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
