import argparse
import datetime
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses, metrics
from tensorflow_skyrmion_dataset import SKYRMION

parser = argparse.ArgumentParser()
parser.add_argument("--activation", default="relu", type=str, help="Activation type")
parser.add_argument("--augment", default=None, type=str, choices=["cutmix", "mixup"], help="Augmentation type")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Dataloader workers.")
parser.add_argument("--decay", default=None, type=str, choices=["linear", "exponential", "cosine", "piecewise"], help="Decay type")
parser.add_argument("--depth", default=56, type=int, help="Model depth")
parser.add_argument("--dropout", default=0., type=float, help="Dropout")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
parser.add_argument("--label_smoothing", default=0., type=float, help="Label smoothing.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("--learning_rate_final", default=0.001, type=float, help="Final learning rate")
parser.add_argument("--model", default="model5", type=str, choices=["model5", "resnet"], help="Model of choice")
parser.add_argument("--optimizer", default="SGD", type=str, choices=["SGD", "Adam"], help="Optimizer type")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--stochastic_depth", default=0., type=float, help="Stochastic depth")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--weight_decay", default=0.004, type=float, help="Weight decay")
parser.add_argument("--width", default=1, type=int, help="Model width")

class TensorBoardCallback(callbacks.TensorBoard):
    def __init__(self, log_dir):
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            logs["learning_rate"] = self.model.optimizer.learning_rate.numpy()
        super().on_epoch_end(epoch, logs)

def main(args: argparse.Namespace) -> None:
    # Set random seed
    tf.random.set_seed(args.seed)
    
    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    skyrmion = SKYRMION()
    # skyrmion = SKYRMION(path="data/train/cutmix_data.npz")

    # Data processing functions
    def process_element(example):
        image, label = example["image"], example["label"]
        return image, tf.one_hot(label, len(SKYRMION.LABELS))

    train = skyrmion.train.map(process_element)
    dev = skyrmion.dev.map(process_element)
    test = skyrmion.test.map(process_element)

    # Augmentations
    def rotate90(image):
        if tf.random.uniform(()) < 0.5:
            return tf.image.rot90(image)
        return image

    augmentations = []
    if args.augment:
        augmentations.append(lambda image: tf.image.random_flip_left_right(image))
        augmentations.append(rotate90)

    if augmentations:
        def augment(image, label):
            for aug in augmentations:
                image = aug(image)
            return image, label
        train = train.map(augment)

    train = train.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    dev = dev.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test = test.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # Model selection
    if args.model == "model5":
        from models import Model5
        model = Model5(args)
    elif args.model == "resnet":
        from models import ResNet
        model = ResNet(args)
    else:
        raise ValueError(f"Unknown model '{args.model}'")

    # Learning rate schedule
    def get_schedule(args):
        total_steps = args.epochs * len(train)
        if args.decay == "linear":
            return optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.learning_rate,
                decay_steps=total_steps,
                end_learning_rate=args.learning_rate_final
            )
        elif args.decay == "exponential":
            decay_rate = (args.learning_rate_final / args.learning_rate) ** (1 / args.epochs)
            return optimizers.schedules.ExponentialDecay(
                initial_learning_rate=args.learning_rate,
                decay_steps=len(train),
                decay_rate=decay_rate
            )
        elif args.decay == "cosine":
            return optimizers.schedules.CosineDecay(
                initial_learning_rate=args.learning_rate,
                decay_steps=total_steps
            )
        elif args.decay == "piecewise":
            return optimizers.schedules.PiecewiseConstantDecay(
                boundaries=[int(0.5 * total_steps), int(0.75 * total_steps)],
                values=[args.learning_rate, args.learning_rate / 10, args.learning_rate / 100]
            )
        else:
            return args.learning_rate

    # Optimizer
    if args.optimizer == "SGD":
        optimizer = optimizers.SGD(
            learning_rate=get_schedule(args),
            momentum=0.9,
            nesterov=True
        )
    elif args.optimizer == "Adam":
        optimizer = optimizers.AdamW(
            learning_rate=get_schedule(args),
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer '{args.optimizer}'")

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[metrics.CategoricalAccuracy()]
    )

    # Callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

    # Fit the model
    model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tensorboard_callback])

    # Save test set predictions
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "skyrmion_test.txt"), "w", encoding="utf-8") as predictions_file:
        for probs in model.predict(test):
            predictions_file.write(f"{np.argmax(probs)}\n")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
