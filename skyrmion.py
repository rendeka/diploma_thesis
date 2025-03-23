#!/usr/bin/env python3
import argparse
import datetime
import os
# import re
# import sys
# import time
# import matplotlib.pyplot as plt
# from typing import Optional, Dict, List
from pathlib import Path
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch

from skyrmion_dataset import SKYRMION
from tensorboard_callback import TorchTensorBoardCallback
from augmentation import choose_augmentation
import models

parser = argparse.ArgumentParser()
parser.add_argument("--activation", default="relu", type=str, help="Activation type (see keras.activations.__dict__ to see all options)")
parser.add_argument("--alpha_dropout", default=False, type=bool, help="True to use alpha dropout (iff selu activations is used)")
parser.add_argument("--augment", default="tailored", type=str, choices=["None", "cutmix", "mixup", "adaptive", "tailored"], nargs="+", help="Augmentation type")
parser.add_argument("--batch_norm", default=True, type=bool, help="True to use batch normalization")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--bias_regularizer", default=1e-5, type=float, help="Parameter for L2 regularization of bias in convolutional kernel")
parser.add_argument("--conv_type", default="standard", type=str, choices=["standard", "ds"], help="Convolution type ('ds' stands for depthwise-separable)")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Dataloader workers")
parser.add_argument("--decay", default=None, type=str, choices=["None", "linear", "exponential", "cosine", "piecewise"], help="Decay type")
parser.add_argument("--depth", default=3, type=int, help="Model depth (use default=56 for ResNet)")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate for dense layers or pixelwise")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--fag", default="GAP", type=str, choices=["GAP", "Flatten", "SE"], nargs="+", help="feature-aggregation: going from CONV to DENSE layer")
parser.add_argument("--filters", default=8, type=int, help="Number of filters in the first convolutional layer")
parser.add_argument("--ffm", default=False, type=bool, help="If True, filters and feature maps will be saved. Check 'log_filters_and_features' function")
parser.add_argument("--grad_cam", default=False, type=bool, help="True to generage grad-CAM images in callback")
parser.add_argument("--head", default="softmax", type=str, help="Activation function for the classification head (use any valid keras activation)")
parser.add_argument("--kernel_regularizer", default=1e-4, type=float, help="Parameter for L2 regularization of convolutional kernel")
parser.add_argument("--kernel_size", default=3, type=int, help="Kernel size")
parser.add_argument("--label_smoothing", default=0., type=float, help="Label smoothing")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("--learning_rate_final", default=0.001, type=float, help="Final learning rate")
parser.add_argument("--logdir_suffix", default=None, type=str, help="Creates subdirectory 'logs_{logdir_suffix}/' in the 'logs/' directory")
parser.add_argument("--loss", default="CCE", type=str, choices=["CCE", "KLD", "MSE"], help="Loss function")
parser.add_argument("--model", default="model5", type=str, choices=["model5", "resnet", "cbam", "ffn"], help="Model of choice")
parser.add_argument("--optimizer", default="SGD", type=str, choices=["SGD", "Adam", "AdamW"], help="Optimizer type")
parser.add_argument("--padding", default ="same", type=str, choices=["same", "valid"], help="Padding in convolutional layers")
parser.add_argument("--pooling", default="max", type=str, choices=["max", "average", "no"], help="Pooling type") # TRY None
parser.add_argument("--save_model", default=False, type=bool, help="If True, trained model will be saved in'saved_models/' directory")
parser.add_argument("--scope", default="sub", type=str, choices=["", "sub"])
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--spatial_dropout", default=0.0, type=float, help="Spatial dropout rate for feature maps")
parser.add_argument("--stochastic_depth", default=0., type=float, help="Stochastic depth")
parser.add_argument("--stride", default=1, type=int, help="Stride in convolutional layers")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use")
parser.add_argument("--trans_probs", default=False, type=bool, help="Whether to create transitional probs images")
parser.add_argument("--weight_decay", default=0.004, type=float, help="Weight decay")
parser.add_argument("--width", default=1, type=int, help="Model width")

def replace_none_strings(obj):
    """Replace 'None' with None in args."""
    if isinstance(obj, str) and obj.lower() == "none":
        return None
    elif isinstance(obj, list):
        return [replace_none_strings(item) for item in obj]
    elif isinstance(obj, dict): 
        return {key: replace_none_strings(value) for key, value in obj.items()}
    return obj 

def main(args: argparse.Namespace) -> None:
    # get path to the project base directory (necessary for running on metacentrum)
    try:
        base_path = Path(os.environ["SKYRMION_BASE_PATH"])
    except KeyError as e:
        raise RuntimeError(
            "Environment variable SKYRMION_BASE_PATH is not set.\n"
            "Please set it using: export SKYRMION_BASE_PATH='/path/to/data' in your .bashrc"
        ) from e

    # check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Format arguments
    # Unnecessary args to have in the args_str 
    no_log_args = ("epochs", "model", "save_model", "ffm", "logdir_suffix", "logdir", "scope", "trans_probs")
    # Taking only the descriptive subset of non-default arguments for args string
    log_args = {arg: value for arg, value in vars(args).items() 
                        if parser.get_default(arg) != vars(args)[arg] and arg not in no_log_args}
    args_str = ",".join(
        f"{arg[:3]}={'+'.join([str(v)[0] for v in value]) if isinstance(value, list) else str(value)}"
        for arg, value in sorted(log_args.items())
    )

    args_str = args_str if args_str else "default"

    # Construct the log directory path
    log_subdir = f"{args.scope}-" if args.scope else ""
    log_subdir += f"{args.model}-{args.logdir_suffix}" if args.logdir_suffix else f"{args.model}"
    base_log_dir = base_path / "logs" / log_subdir
   
    args.logdir = str(base_log_dir / f"{timestamp}-{args_str}") # Must be converted to string in order to be serializable

    # Load tran/dev/test data
    skyrmion = SKYRMION(path=(base_path / "data" / "train" / (f"skyrmion_dataset_{args.scope}" if args.scope else "skyrmion_dataset")).with_suffix(".npz"))

    # Load data for evaluating performance in phase transition
    skyrmion_transitions = {
        trans[:5]: SKYRMION(path=(base_path / "data" / "test" / trans).with_suffix(".npz"))
        for trans in [f"{base}_{args.scope}" if args.scope else base for base in ["fe_sk_transition", "sk_sp_transition"]]
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

    def augmented_collate_fn(examples, augment=args.augment):
        images, labels = torch.utils.data.default_collate(examples)
        if np.random.rand() < 0.1: # Do batch augmentation in 90% of the time
            return images, labels

        batch_aug, labels = choose_augmentation(labels, augment)

        images, labels = batch_aug(images.permute(0, 3, 1, 2), labels)
        # #########
        # SKYRMION.visualize_images(images.squeeze(), labels, row_size=args.batch_size, base_size=8)
        # #########                           
        return images.permute(0, 2, 3, 1), labels

    train = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True, collate_fn=augmented_collate_fn,
        num_workers=args.dataloader_workers, persistent_workers=args.dataloader_workers > 0)
    dev = torch.utils.data.DataLoader(dev, batch_size=args.batch_size)
    test = torch.utils.data.DataLoader(test, batch_size=args.batch_size)

    # learning rate decay schedule
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

        elif args.decay == "piecewise":
            return keras.optimizers.schedules.PiecewiseConstantDecay(
                [int(0.5 * training_batches), int(0.75 * training_batches)],
                [args.learning_rate, args.learning_rate / 10, args.learning_rate / 100])

        else:
            raise ValueError("Uknown decay '{}'".format(args.decay))
        
    # Optimizer
    if args.optimizer == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=get_schedule(args), weight_decay=args.weight_decay, momentum=0.9, nesterov=True, clipnorm=1.0)
    elif args.optimizer.startswith("Adam"):
        # beta2, epsilon = map(float, args.optimizer.split(":")[1:]) # Let's use default values instead
        optimizer = keras.optimizers.AdamW(learning_rate=get_schedule(args), weight_decay=args.weight_decay, clipnorm=1.0)#, beta_2=beta2, epsilon=epsilon)
    else:
        raise ValueError("Uknown optimizer '{}'".format(args.optimizer))
    optimizer.exclude_from_weight_decay(var_names=["bias"])

    # Model
    if args.model == "model5":
        model = models.Model5(args)
    elif args.model == "resnet":
        model = models.ResNet(args)
    elif args.model == "cbam":
        model = models.ModelCBAM(args)
    elif args.model == "ffn":
        model = models.ModelFFN(args)
    else:
        raise ValueError("Uknown model '{}'".format(args.model))
    
    model.summary()
    model = model.to(device)

    # Loss
    if args.loss == "CCE":
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
    elif args.loss == "KLD":
        loss = keras.losses.KLDivergence()
        if args.label_smoothing != parser.get_default("label_smoothing"):
            raise Warning("Label smoothing is not implemented for KL-divergence loss.")
    elif args.loss == "MSE":
        loss = keras.losses.MeanSquaredError()

    model.compile(
        optimizer=optimizer,
        loss=loss,
    metrics=[
        # keras.metrics.KLDivergence(name="kl_div"),
        keras.metrics.CategoricalAccuracy("accuracy")
        # keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
        # keras.metrics.MeanSquaredError(name="mse_accuracy")        
        ])

    tb_callback = TorchTensorBoardCallback(args, transition_datasets=skyrmion_transitions, fm_dataset=skyrmion_fm)

    model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback])

    if args.save_model:
        save_dir = base_path / "saved_models" / ("sub" if args.scope else "full")
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save((save_dir / f"{args.model}-{timestamp}-{args_str}").with_suffix(".keras"))

    if skyrmion.test.__len__() > 0:
        with open((Path(args.logdir) / "skyrmion_test").with_suffix(".txt"), "w", encoding="utf-8") as predictions_file:
            for probs in model.predict(skyrmion.test.data["images"], batch_size=args.batch_size):
                print(np.argmax(probs), file=predictions_file)
    else:
        print(f"No test data provided. File 'skyrmion_test.txt' won't be created.") 

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    args = replace_none_strings(vars(args))
    main(argparse.Namespace(**args))