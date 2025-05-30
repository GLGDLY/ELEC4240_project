import os
import tensorflow as tf
import datetime
from model import UNet
import matplotlib.pyplot as plt
import numpy as np
import glob
from data_loader import SegmentationDataGenerator


class TrainingPipeline:
    def __init__(
        self,
        input_size=(256, 256, 3),
        batch_size=32,
        learning_rate=0.001,
        checkpoint_dir="checkpoints",
        logs_dir="logs",
    ):

        self.input_size = input_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # Initialize model
        self.model = UNet(input_size=input_size)

        # Define loss and optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.metrics = {
            "loss": tf.keras.metrics.Mean(name="loss"),
            "accuracy": tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            "iou": tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1]),
        }

    def compile_model(self):
        """Compile the model with loss and metrics"""
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.dice_loss,
            metrics=["accuracy", self.iou_metric],
        )

    def dice_loss(self, y_true, y_pred):
        """Custom Dice loss for better segmentation"""
        smooth = 1e-6
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return 1 - (2.0 * intersection + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
        )

    def iou_metric(self, y_true, y_pred):
        """IoU (Intersection over Union) metric"""
        threshold = 0.5
        y_pred = tf.cast(y_pred > threshold, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        return intersection / (union + tf.keras.backend.epsilon())

    def get_callbacks(self):
        """Define training callbacks"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        callbacks = [
            # save best weights
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    self.checkpoint_dir, "model_{epoch:02d}_{val_loss:.2f}.h5"
                ),
                save_best_only=True,
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
            ),
            # prevent overfitting
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=5, min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.logs_dir, timestamp), histogram_freq=1
            ),
        ]

        return callbacks

    def train(self, train_dataset, val_dataset, epochs=50):
        self.compile_model()

        callbacks = self.get_callbacks()

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
        )

        self.model.save_weights("final_model.h5")
        return history

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_history.png")
        plt.show()

    def predict_and_visualize(self, test_dataset, num_samples=5):
        """Visualize predictions on test data"""
        plt.figure(figsize=(15, 5 * num_samples))

        for i, (image, mask) in enumerate(test_dataset.take(num_samples)):
            prediction = self.model.predict(image)

            plt.subplot(num_samples, 3, i * 3 + 1)
            plt.imshow(tf.keras.preprocessing.image.array_to_img(image[0]))
            plt.title("Input Image")
            plt.axis("off")

            plt.subplot(num_samples, 3, i * 3 + 2)
            plt.imshow(tf.keras.preprocessing.image.array_to_img(mask[0]), cmap="gray")
            plt.title("True Mask")
            plt.axis("off")

            plt.subplot(num_samples, 3, i * 3 + 3)
            plt.imshow(
                tf.keras.preprocessing.image.array_to_img(prediction[0]), cmap="gray"
            )
            plt.title("Predicted Mask")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig("predictions.png")
        plt.show()


if __name__ == "__main__":
    import pathlib
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    print("GPU Devices:", tf.config.list_physical_devices("GPU"))

    processed_data_dir = "./processed_data"

    image_paths = sorted(glob.glob(os.path.join(processed_data_dir, "images/*.jpg")))
    # mask_paths = sorted(glob.glob(os.path.join(processed_data_dir, "masks/*_mask.jpg")))

    # train_images, val_images, train_masks, val_masks = train_test_split(
    #     image_paths, mask_paths, test_size=0.2, random_state=42
    # )
    train_images, val_images = train_test_split(
        image_paths, test_size=0.2, random_state=42
    )

    batch_size = 8
    train_generator = SegmentationDataGenerator(
        image_paths=train_images,
        # mask_paths=train_masks,
        batch_size=batch_size,
    )

    val_generator = SegmentationDataGenerator(
        image_paths=val_images,
        # mask_paths=val_masks,
        batch_size=batch_size,
    )

    pipeline = TrainingPipeline(
        input_size=(256, 256, 3), batch_size=batch_size, learning_rate=0.01
    )

    train_dataset = train_generator.get_dataset(training=True)
    val_dataset = val_generator.get_dataset(training=False)

    history = pipeline.train(
        train_dataset=train_dataset, val_dataset=val_dataset, epochs=20
    )

    pipeline.plot_training_history(history)

    pipeline.predict_and_visualize(val_dataset, num_samples=5)
