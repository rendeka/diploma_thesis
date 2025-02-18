#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import sys
import time
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from pathlib import Path
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch
import torchvision.transforms.v2 as v2

from skyrmion_dataset import SKYRMION
from tensorboard_callback import TorchTensorBoardCallback

parser = argparse.ArgumentParser()
parser.add_argument("--activation", default="relu", type=str, help="Activation type (see keras.activations.__dict__ to see all options)")
parser.add_argument("--augment", default=None, type=str, choices=["cutmix", "mixup"], nargs="+", help="Augmentation type")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--bias_regularizer", default=1e-5, type=float, help="Parameter for L2 regularization of bias in convolutional kernel")
parser.add_argument("--conv_type", default="standard", type=str, choices=["standard", "ds"], help="Convolution type ('ds' stands for depthwise-separable)")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Dataloader workers")
parser.add_argument("--decay", default=None, type=str, choices=["linear", "exponential", "cosine", "piecewise"], help="Decay type")
parser.add_argument("--depth", default=3, type=int, help="Model depth (use default=56 for ResNet)")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--filters", default=8, type=int, help="Number of filters in the first convolutional layer")
parser.add_argument("--ffm", default=False, type=bool, help="If True, filters and feature maps will be saved. Check 'log_filters_and_features' function")
parser.add_argument("--head", default="softmax", type=str, choices=["softmax", "sigmoid"], help="Activation function for the classification head)")
parser.add_argument("--kernel_regularizer", default=1e-4, type=float, help="Parameter for L2 regularization of convolutional kernel")
parser.add_argument("--kernel_size", default=3, type=int, help="Kernel size")
parser.add_argument("--label_smoothing", default=0., type=float, help="Label smoothing")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("--learning_rate_final", default=0.001, type=float, help="Final learning rate")
parser.add_argument("--logdir_suffix", default=None, type=str, help="Creates subdirectory 'logs_{logdir_suffix}/' in the 'logs/' directory")
parser.add_argument("--model", default="model5", type=str, choices=["model5", "resnet", "cbam"], help="Model of choice")
parser.add_argument("--optimizer", default="SGD", type=str, choices=["SGD", "Adam"], help="Optimizer type")
parser.add_argument("--padding", default="same", type=str, choices=["same", "valid"], help="Padding in convolutional layers")
parser.add_argument("--pooling", default="max", type=str, choices=["max", "average"], help="Pooling type")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--save_model", default=False, type=bool, help="If True, trained model will be saved in'saved_models/' directory")
parser.add_argument("--stochastic_depth", default=0., type=float, help="Stochastic depth")
parser.add_argument("--stride", default=1, type=int, help="Stride in convolutional layers")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use")
parser.add_argument("--weight_decay", default=None, type=float, help="Weight decay")
parser.add_argument("--width", default=1, type=int, help="Model width")

def main(args: argparse.Namespace) -> None:
    # Get path to the project base directory ()
    try:
        base_path = Path(os.environ["SKYRMION_BASE_PATH"])
    except KeyError as e:
        raise RuntimeError(
            "Environment variable SKYRMION_BASE_PATH is not set.\n"
            "Please set it using: export SKYRMION_BASE_PATH='/path/to/data' in your .bashrc"
        ) from e

    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Format arguments
    # Unnecessary args to have in the args_str 
    no_log_args = ("epochs", "model", "save_model", "ffm", "logdir_suffix", "logdir")
    # Taking only the descriptive subset of non-default arguments for args string
    log_args = {arg: value for arg, value in vars(args).items() 
                        if parser.get_default(arg) != vars(args)[arg] and arg not in no_log_args}
    args_str = ",".join(
        f"{arg[:3]}={'+'.join([v[0] for v in value]) if isinstance(value, list) else value}"
        for arg, value in sorted(log_args.items())
    )

    # Construct the log directory path
    base_log_dir = base_path / "logs" / f"{args.model}-{args.logdir_suffix}" if args.logdir_suffix is not None else f"{args.model}"
    args.logdir = str(base_log_dir / f"{timestamp}-{args_str}") # Must be converted to string in order to be serializable

    # Load tran/dev/test data
    skyrmion = SKYRMION(path=(base_path / "data" / "train" / "skyrmion_dataset").with_suffix(".npz"))

    # Load data for evaluating performance in phase transition
    skyrmion_transitions = {
        trans[:5]: SKYRMION(path=(base_path / "data" / "test" / trans).with_suffix(".npz"))
        for trans in ["fe_sk_transition", "sk_sp_transition"]
    }

    # Dataset to extract feature maps in tensorboard callback
    skyrmion_fm = SKYRMION(path=(base_path / "data" / "test" / "fm_dataset").with_suffix(".npz"))

    # train/dev/test dataset creation
    def process_element(example):
        image, label = torch.from_numpy(example["image"]), torch.tensor(example["label"], dtype=torch.int64)
        return image, torch.nn.functional.one_hot(label, len(SKYRMION.LABELS))

    train = skyrmion.train.transform(process_element)
    dev = skyrmion.dev.transform(process_element)
    test = skyrmion.test.transform(process_element)

    # # Rotating an image by 90 degrees with the chance 1/2
    # def rotate90(image):
    #     if np.random.rand() < 0.5:
    #         return v2.functional.rotate(image, angle=90)
    #     else:
    #         return image

    # # Augmentations
    # augmentations = []
    # if args.augment:
    #     augmentations.append(v2.RandomHorizontalFlip())
    #     augmentations.append(rotate90)
            
    #     # augmentations.append(v2.RandomCrop((SKYRMION.H, SKYRMION.W), padding=4, fill=127))
    # # if "autoaugment" in args.augment:
    # #     augmentations.append(v2.AutoAugment(v2.AutoAugmentPolicy.SKYRMION, fill=127))
    # # if randaugment := re.search(r"randaugment-(\d+)-(\d+)", args.augment):
    # #     n, m = map(int, randaugment.groups())
    # #     augmentations.append(v2.RandAugment(n, m, fill=127))
    # # if "cutout" in args.augment:
    # #     def cutout(image):
    # #         y, x = np.random.randint(SKYRMION.H), np.random.randint(SKYRMION.W)
    # #         image = image.clone()
    # #         image[:, max(0, y - 8):y + 8, max(0, x - 8):x + 8] = 127
    # #         return image
    # #     augmentations.append(v2.Lambda(cutout))
    #     augmentations = v2.Compose(augmentations)
    #     train = train.transform(lambda image, label: (augmentations(image.permute(2, 0, 1)).permute(1, 2, 0), label))
    if args.augment:
        batch_augmentations = []
        if "cutmix" in args.augment:
            batch_augmentations.append(v2.CutMix(num_classes=len(SKYRMION.LABELS)))
        if "mixup" in args.augment:
            batch_augmentations.append(v2.Compose([v2.ToDtype(torch.float32), v2.MixUp(num_classes=len(SKYRMION.LABELS))]))
        if batch_augmentations:
            # We are creating augmented images during training instead of creating larger dataset that already contains
            # augmented images. In this way, we can artifically increase the size of our dataset by increasing number
            # of epochs, because in each epoch new data are generated. I choose to increase the dataset (or rather number
            # of epochs) by factor of 10 * number_of_different_augmentations 

            # args.epochs *= 2 * len(batch_augmentations) # Changed epoch-scalling factor to 2
            print("currently, we are not increasing the number of epochs for the runs with augmentations")

            batch_augmentations = v2.RandomChoice(batch_augmentations) # Randomly picks one of the augemntations
            def augmented_collate_fn(examples):
                images, labels = torch.utils.data.default_collate(examples)
                if np.random.rand() < 0.1: # Do batch augmentation in 90% of the time
                    # #########
                    # SKYRMION.visualize_images(images.squeeze(), labels, row_size=4)
                    # #########
                    return images, labels
                images, labels = batch_augmentations(images.permute(0, 3, 1, 2), torch.argmax(labels, dim=1))
                # #########
                # SKYRMION.visualize_images(images.squeeze(), labels, row_size=4)
                # #########                           
                return images.permute(0, 2, 3, 1), labels

    train = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True, collate_fn=augmented_collate_fn if args.augment else None,
        num_workers=args.dataloader_workers, persistent_workers=args.dataloader_workers > 0)
    dev = torch.utils.data.DataLoader(dev, batch_size=args.batch_size)
    test = torch.utils.data.DataLoader(test, batch_size=args.batch_size)

    # Decay schedule
    def get_schedule(args):
        total_steps = args.epochs * len(train) // args.batch_size
        training_batches = args.epochs * len(train)

        if args.decay is None:
            return args.learning_rate

        elif args.decay == "linear":
            return keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.learning_rate,
                decay_steps=total_steps,
                end_learning_rate=args.learning_rate_final,
                power=1.0
            )

        elif args.decay == "exponential":
            rate_calculated = np.exp(np.log(args.learning_rate_final / args.learning_rate) / args.epochs)
            return keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=args.learning_rate,
                decay_steps=len(train) / args.batch_size,
                decay_rate=rate_calculated,
                staircase=False
            )

        elif args.decay == "cosine":
            return keras.optimizers.schedules.CosineDecay(args.learning_rate, training_batches)
            # return keras.optimizers.schedules.CosineDecay(
            #     initial_learning_rate=args.learning_rate,
            #     decay_steps=total_steps,
            #     alpha=args.learning_rate_final / args.learning_rate
            # )

        elif args.decay == "piecewise":
            return keras.optimizers.schedules.PiecewiseConstantDecay(
                [int(0.5 * training_batches), int(0.75 * training_batches)],
                [args.learning_rate, args.learning_rate / 10, args.learning_rate / 100])

        else:
            raise ValueError("Uknown decay '{}'".format(args.decay))
        
    # Optimizer
    if args.optimizer == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=get_schedule(args), weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    elif args.optimizer.startswith("Adam"):
        # beta2, epsilon = map(float, args.optimizer.split(":")[1:]) # Let's use default values
        optimizer = keras.optimizers.AdamW(learning_rate=get_schedule(args), weight_decay=args.weight_decay, clipnorm=1.0)#, beta_2=beta2, epsilon=epsilon)
    else:
        raise ValueError("Uknown optimizer '{}'".format(args.optimizer))
    optimizer.exclude_from_weight_decay(var_names=["bias"])

    # Model
    if args.model == "model5":
        from models import Model5
        model = Model5(args)
    elif args.model == "resnet":
        from models import ResNet
        model = ResNet(args)
    elif args.model == "cbam":
        from models import ModelCBAM
        model = ModelCBAM(args)
    else:
        raise ValueError("Uknown model '{}'".format(args.model))
    
    model.summary()
    model = model.to(device)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[keras.metrics.CategoricalAccuracy("accuracy")],
    )

    tb_callback = TorchTensorBoardCallback(args, transition_datasets=skyrmion_transitions, fm_dataset=skyrmion_fm)

    model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback])

    if args.save_model:
        model.save((base_path / "saved_models" / f"{args.model}-{timestamp}-{args_str}").with_suffix(".keras"))

    if skyrmion.test.__len__() > 0:
        with open(args.logdir / "skyrmion_test.txt", "w", encoding="utf-8") as predictions_file:
            for probs in model.predict(skyrmion.test.data["images"], batch_size=args.batch_size):
                print(np.argmax(probs), file=predictions_file)
    else:
        print(f"No test data provided. File 'skyrmion_test.txt' won't be created.") 

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)