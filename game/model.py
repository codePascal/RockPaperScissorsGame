#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rock-Paper-Scissors model

Module implements the image classifier for the game.
"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import keras

from pathlib import Path

# Hyper-parameters
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 5
TRAIN_SPLIT = 0.8


class RockPaperScissorModel:
    """RockPaperScissorModel

    Class implements a CNN-based image classifier for the dataset
    `rock_paper_scissors` using TensorFlow and Keras.

    See Also:
        https://laurencemoroney.com/datasets.html#rock-paper-scissors-dataset
        https://www.tensorflow.org/datasets/catalog/rock_paper_scissors
    """
    classes = {
        0: 'rock',
        1: 'paper',
        2: 'scissors'
    }
    log_dir = Path(__file__).parent.parent.joinpath('logs')
    model_path = Path(__file__).parent.parent.joinpath('out', 'rpsModel.keras')

    def __init__(self):
        """Create an instance of RockPaperScissorsModel.
        """
        self._img_size = (300, 300)
        self._tf_name = 'rock_paper_scissors'

    def info(self):
        """Load dataset info."""
        _, ds_info = tfds.load(self._tf_name, with_info=True)
        return ds_info

    def train_dataset(self):
        """Load train-validation dataset with given split."""
        train_pct = f'{100 * TRAIN_SPLIT}%'
        ds_train, ds_val = tfds.load(
            self._tf_name,
            split=[f'train[:{train_pct}]', f'train[{train_pct}:]'],
            as_supervised=True,
            with_info=False
        )
        return ds_train, ds_val

    def test_dataset(self):
        """Load test dataset."""
        ds_test = tfds.load(
            self._tf_name,
            split='test',
            as_supervised=True,
            with_info=False
        )
        return ds_test

    def train(self):
        """Train the model using train-validation and evaluate using test."""
        # Create the dataset splits
        ds_train, ds_val = self.train_dataset()
        ds_train = (
            ds_train
            .repeat()
            .shuffle(1024)
            .map(self.preprocess, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )
        ds_val = (
            ds_val
            .map(self.preprocess, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )

        # Create game
        model = self.create_model()

        # Initialize callback for training, keep default 'logs' directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        tensorboard = keras.callbacks.TensorBoard()

        # Configure fitting with proper output
        ds_info = self.info()
        train_size = ds_info.splits['train'].num_examples
        steps_per_echo = int(TRAIN_SPLIT * train_size // BATCH_SIZE)
        validation_steps = int((1 - TRAIN_SPLIT) * train_size // BATCH_SIZE)

        # Fit the game using train and validation data
        model.fit(
            ds_train,
            verbose=1,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_echo,
            validation_data=ds_val,
            validation_steps=validation_steps,
            callbacks=[tensorboard]
        )

        # Evaluate the game using test data
        ds_test = self.test_dataset()
        ds_test = (
            ds_test
            .map(self.preprocess, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )
        model.evaluate(ds_test)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(self.model_path)

    def test_sample(self):
        """Get test sample from test dataset."""
        ds_test = self.test_dataset()
        sample = next(iter(ds_test))
        img = sample[0].numpy()
        label = self.classes[sample[1].numpy()]
        return img, label

    def load(self):
        """Load compiled model."""
        return keras.models.load_model(self.model_path)

    def preprocess(self, image: np.ndarray, label: str = None):
        """Convert and resize image for model input."""
        image = tf.image.resize(image, self._img_size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        if label is None:
            return image
        return image, label

    @staticmethod
    def create_model():
        """Create transfer game with MobileNet as backbone."""
        base = keras.applications.MobileNetV2(
            input_shape=(300, 300, 3),
            include_top=False,
            weights='imagenet'
        )
        base.trainable = False

        model = keras.Sequential([
            base,
            keras.layers.GlobalAvgPool2D(),
            keras.layers.Dense(3, activation='softmax')
        ])
        model.summary()
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    @staticmethod
    def infer(model, img):
        """Infer an image with the model"""
        prediction = model.predict(tf.reshape(img, (-1, 300, 300, 3)))[0]
        predicted_label = np.argmax(prediction)
        probability = prediction[predicted_label]
        return RockPaperScissorModel.classes[predicted_label], probability
