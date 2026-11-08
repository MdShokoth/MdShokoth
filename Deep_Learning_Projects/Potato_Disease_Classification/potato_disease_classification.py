# -*- coding: utf-8 -*-
"""Potato disease classification using TensorFlow and Keras.

Update DATASET_DIR to the root directory containing the class-specific
PlantVillage image folders before running this script.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS = 3
EPOCHS = 50
DATASET_DIR = "/content/drive/MyDrive/Colab Notebooks/Potato_disease/training/PlantVillage"


def get_dataset_partitions_tf(
    ds, train_split=0.8, val_split=0.1, test_split=0.1,
    shuffle=True, shuffle_size=10000
):
    if not np.isclose(train_split + val_split + test_split, 1.0):
        raise ValueError("Train, validation, and test splits must sum to 1.")

    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size + val_size)
    return train_ds, val_ds, test_ds


def build_model(n_classes):
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.Rescaling(1.0 / 255),
    ])

    model = models.Sequential([
        resize_and_rescale,
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def predict_image(model, image, class_names):
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities)) * 100
    return class_names[predicted_index], round(confidence, 2)


def main():
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}. "
            "Update DATASET_DIR to your PlantVillage dataset location."
        )

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        seed=123,
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
    )
    class_names = dataset.class_names
    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])
    train_ds = train_ds.map(
        lambda images, labels: (data_augmentation(images, training=True), labels)
    ).prefetch(autotune)

    model = build_model(len(class_names))
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.tight_layout()
    plt.show()

    for images, labels in test_ds.take(1):
        for i in range(min(9, len(images))):
            predicted_class, confidence = predict_image(
                model, images[i].numpy(), class_names
            )
            actual_class = class_names[int(labels[i])]
            print(
                f"Actual: {actual_class} | Predicted: {predicted_class} "
                f"| Confidence: {confidence:.2f}%"
            )

    os.makedirs("models", exist_ok=True)
    model.save("models/potato_disease_classifier.keras")
    model.save("models/potato_disease_classifier.h5")


if __name__ == "__main__":
    main()
