import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pathlib
from datetime import datetime
from typing import List, Literal

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data_loader import InpaintingDataGenerator, prepare_kfold_data
from model import (
    Discriminator,
    Generator,
    GeneratorType,
    discriminator_loss,
    generator_loss,
)

log_dir = "logs/"

# TODO: try freezing discriminator first? after the l1 loss is stable, then start training the discriminator?


class Trainer:
    def __init__(
        self,
        generator_type: GeneratorType,
        model_suffix: str = "",
        output_dir: str = "./models",
        save_best_method: Literal["l1", "gan", "total"] = "total",
    ):
        self.generator = Generator(generator_type=generator_type)
        self.discriminator = Discriminator()

        self.model_suffix = (
            model_suffix + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        self.save_best_method = save_best_method
        self.output_dir = output_dir
        self.best_generator_path = (
            self.output_dir + f"/best_generator_{self.model_suffix}.h5"
        )
        self.best_discriminator_path = (
            self.output_dir + f"/best_discriminator_{self.model_suffix}.h5"
        )
        self.last_generator_path = (
            self.output_dir + f"/last_generator_{self.model_suffix}.h5"
        )
        self.last_discriminator_path = (
            self.output_dir + f"/last_discriminator_{self.model_suffix}.h5"
        )

        generator_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            2e-4, decay_steps=100000, decay_rate=0.96, staircase=True
        )
        discriminator_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            5e-5, decay_steps=100000, decay_rate=0.96, staircase=True
        )
        self.generator_optimizer = tf.keras.optimizers.Adam(
            generator_scheduler, beta_1=0.5
        )
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            discriminator_scheduler, beta_1=0.5
        )

        self.generator.compile(
            optimizer=self.generator_optimizer,
            loss=generator_loss,
        )
        self.discriminator.compile(
            optimizer=self.discriminator_optimizer,
            loss=discriminator_loss,
        )

        self.summary_writer = tf.summary.create_file_writer(
            log_dir + "fit/" + self.model_suffix
        )

    @tf.function
    def __train_step(self, masked_image, in_mask, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator([masked_image, in_mask], training=True)

            noise_level = 0.05 * tf.random.uniform([])
            masked_image = (
                masked_image + tf.random.normal(tf.shape(masked_image)) * noise_level
            )
            target = target + tf.random.normal(tf.shape(target)) * noise_level

            disc_real_output = self.discriminator([masked_image, target], training=True)
            disc_generated_output = self.discriminator(
                [masked_image, gen_output], training=True
            )

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
                disc_generated_output, gen_output, target, in_mask
            )
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

    @tf.function
    def __val(self, masked_image, target, in_mask):
        gen_output = self.generator([masked_image, in_mask], training=False)

        disc_real_output = self.discriminator([masked_image, target], training=False)
        disc_generated_output = self.discriminator(
            [masked_image, gen_output], training=False
        )

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target, in_mask
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

    def __fit(self, epoch, train_ds, val_ds):
        # train
        print("Training...")
        best_loss = float("inf")

        # train
        t_gen_total_loss, t_gen_gan_loss, t_gen_l1_loss, t_disc_loss = 0, 0, 0, 0
        for (masked_image, in_mask), target in tqdm(train_ds):
            ################### DEBUG START ###################
            # print(masked_image.shape, in_mask.shape, target.shape)
            # Image.fromarray((masked_image[0].numpy()).astype(np.uint8)).show()
            # Image.fromarray((target[0].numpy()).astype(np.uint8)).show()
            # exit()
            ################### DEBUG END ###################

            _gen_total_loss, _gen_gan_loss, _gen_l1_loss, _disc_loss = (
                self.__train_step(masked_image, in_mask, target)
            )
            t_gen_total_loss += _gen_total_loss
            t_gen_gan_loss += _gen_gan_loss
            t_gen_l1_loss += _gen_l1_loss
            t_disc_loss += _disc_loss
        t_gen_total_loss /= len(train_ds)
        t_gen_gan_loss /= len(train_ds)
        t_gen_l1_loss /= len(train_ds)
        t_disc_loss /= len(train_ds)

        # val
        print("Validating...")
        gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = 0, 0, 0, 0
        for (masked_image, in_mask), target in tqdm(val_ds):
            _gen_total_loss, _gen_gan_loss, _gen_l1_loss, _disc_loss = self.__val(
                masked_image, target, in_mask
            )
            gen_total_loss += _gen_total_loss
            gen_gan_loss += _gen_gan_loss
            gen_l1_loss += _gen_l1_loss
            disc_loss += _disc_loss
        gen_total_loss /= len(test_ds)
        gen_gan_loss /= len(test_ds)
        gen_l1_loss /= len(test_ds)
        disc_loss /= len(test_ds)

        if self.save_best_method == "l1":
            this_loss = gen_l1_loss
        elif self.save_best_method == "gan":
            this_loss = gen_gan_loss
        elif self.save_best_method == "total":
            this_loss = gen_total_loss
        if this_loss < best_loss:
            best_loss = this_loss
            self.generator.save(self.best_generator_path)
            self.discriminator.save(self.best_discriminator_path)
        self.generator.save(self.last_generator_path)
        self.discriminator.save(self.last_discriminator_path)

        print(
            f"[Train] gen_total_loss: {t_gen_total_loss:.4f}, gen_gan_loss: {t_gen_gan_loss:.4f}, gen_l1_loss: {t_gen_l1_loss:.4f}, disc_loss: {t_disc_loss:.4f}"
        )
        print(
            f"[Val] gen_total_loss: {gen_total_loss:.4f}, gen_gan_loss: {gen_gan_loss:.4f}, gen_l1_loss: {gen_l1_loss:.4f}, disc_loss: {disc_loss:.4f}"
        )

        # TODO: find out how tensorboard record by epoch instead of using step like this to brute force it
        with self.summary_writer.as_default(step=epoch):
            tf.summary.scalar("gen_total_loss/train", t_gen_total_loss)
            tf.summary.scalar("gen_gan_loss/train", t_gen_gan_loss)
            tf.summary.scalar("gen_l1_loss/train", t_gen_l1_loss)
            tf.summary.scalar("disc_loss/train", t_disc_loss)

            tf.summary.scalar("gen_total_loss/val", gen_total_loss)
            tf.summary.scalar("gen_gan_loss/val", gen_gan_loss)
            tf.summary.scalar("gen_l1_loss/val", gen_l1_loss)
            tf.summary.scalar("disc_loss/val", disc_loss)

    def train(
        self,
        kfold_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        epochs: int,
    ):
        train_ds, val_ds = kfold_ds
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.__fit(epoch, train_ds, val_ds)

        print("Training finished.")

        # test
        print("Testing on best model...")
        self.generator.load_weights(self.best_generator_path)
        self.discriminator.load_weights(self.best_discriminator_path)

        gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = 0, 0, 0, 0
        for (masked_image, in_mask), target in tqdm(test_ds):
            _gen_total_loss, _gen_gan_loss, _gen_l1_loss, _disc_loss = self.__val(
                masked_image, target, in_mask
            )
            gen_total_loss += _gen_total_loss
            gen_gan_loss += _gen_gan_loss
            gen_l1_loss += _gen_l1_loss
            disc_loss += _disc_loss
        gen_total_loss /= len(test_ds)
        gen_gan_loss /= len(test_ds)
        gen_l1_loss /= len(test_ds)
        disc_loss /= len(test_ds)

        print(f"gen_total_loss: {gen_total_loss}")
        print(f"gen_gan_loss: {gen_gan_loss}")
        print(f"gen_l1_loss: {gen_l1_loss}")
        print(f"disc_loss: {disc_loss}")


if __name__ == "__main__":
    print(tf.config.list_physical_devices("GPU"))

    ds_dir = "./data/dataset"
    train_val_dir = ds_dir + "/train"
    test_dir = ds_dir + "/test"

    batch_size = 64
    kfold_ds = prepare_kfold_data(train_val_dir, batch_size=batch_size)

    test_data_root = pathlib.Path(test_dir)
    test_image_paths = [str(p) for p in test_data_root.glob("*.jpg")]
    test_image_paths = np.array(test_image_paths)
    test_ds = InpaintingDataGenerator(
        test_image_paths, batch_size=batch_size
    ).get_dataset(training=False)

    for i, fold_ds in enumerate(kfold_ds):
        print(f"Fold {i + 1}/{len(kfold_ds)}")
        trainer = Trainer(
            GeneratorType.PARTIAL_CONV,
            model_suffix=f"p_conv_fixmask_50lambda_with_smooth_0.05noise_500epochs_fold{i+1}",
            save_best_method="total",
        )
        trainer.train(fold_ds, test_ds, epochs=500)
