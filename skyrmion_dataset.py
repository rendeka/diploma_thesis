#!/usr/bin/env python3
import sys
from typing import Any, Callable, Sequence, TextIO, TypedDict, Optional, Union
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import torch


class SKYRMION:
    H: int = 200
    W: int = 200
    C: int = 1
    LABELS: list[str] = ["ferromagnet", "skyrmion", "spiral"]

    Element = TypedDict("Element", {"image": np.ndarray, "label": np.ndarray})
    Elements = TypedDict("Elements", {"images": np.ndarray, "labels": np.ndarray})

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data: "SKYRMION.Elements") -> None:
            self._data = data
            self._data["labels"] = self._data["labels"].ravel()

        @property
        def data(self) -> "SKYRMION.Elements":
            return self._data

        def __len__(self) -> int:
            return len(self._data["images"])

        def __getitem__(self, index: int) -> "SKYRMION.Element":
            return {key.removesuffix("s"): value[index] for key, value in self._data.items()}
        
        def to_torch(self):
            images = self._data["images"]
            labels = self._data["labels"]

            images, labels = torch.from_numpy(images), torch.from_numpy(labels).long().squeeze()
            return images, torch.nn.functional.one_hot(labels, len(SKYRMION.LABELS))


        def transform(self, transform: Callable[["SKYRMION.Element"], Any]) -> "SKYRMION.TransformedDataset":
            return SKYRMION.TransformedDataset(self, transform)
        

    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset: torch.utils.data.Dataset, transform: Callable[..., Any]) -> None:
            self._dataset = dataset
            self._transform = transform

        def __len__(self) -> int:
            return len(self._dataset)

        def __getitem__(self, index: int) -> Any:
            item = self._dataset[index]
            return self._transform(*item) if isinstance(item, tuple) else self._transform(item)

        def transform(self, transform: Callable[..., Any]) -> "SKYRMION.TransformedDataset":
            return SKYRMION.TransformedDataset(self, transform)

    def __init__(self, path: Union[Path, str], size: dict[str, int] = {}) -> None:
        
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            print("SKYRMION dataset not found...", file=sys.stderr)

        skyrmion = np.load(path, allow_pickle=True)
        if "fm_dataset" in str(path): # Feature map dataset
            self.LABELS = ["Unique ID of an image (originated from datasetMC.npz)"]
            setattr(self, "dataset", self.Dataset(skyrmion["arr_0"].item()))

        elif "transition" in str(path):
            self.LABELS = ["Set of non-negative integers ordering images in the phase transition"]
            dataset = skyrmion["arr_0"].item()
            for D_value, data in dataset.items():
                setattr(self, f"{path.stem}-{D_value}", self.Dataset(data))

        else:
            for dataset in ["train", "dev", "test"]:
                data = {key[len(dataset) + 1:]: skyrmion[key][:size.get(dataset, None)]
                        for key in skyrmion if key.startswith(dataset)}
                setattr(self, dataset, self.Dataset(data))

    @staticmethod
    def visualize_images(images: np.ndarray, labels: np.ndarray, row_size: int = 1, base_size:int = 3) -> None:
        """
        Visualize a grid of images with their corresponding labels.

        Args:
            images (np.ndarray): Array of images to visualize.
            labels (np.ndarray): Array of labels corresponding to the images.
            row_size (int): Number of images per row in the grid.
        """

        Nsamp = np.arange(len(images))
        n_rows = math.ceil(len(Nsamp) / row_size)

        fig, axs = plt.subplots(
            n_rows, row_size, 
            figsize=(base_size * row_size, base_size * n_rows), 
            sharey=True
        )

        # Flatten axs for simpler indexing, in case of a single row
        axs = axs.ravel() if len(Nsamp) > 1 else [axs]

        for n, num_sample in enumerate(Nsamp):
            axs[n].imshow(images[num_sample], vmin=0.0, vmax=1.0, cmap="RdBu")

            axs[n].set_xlim((0.0, 200.0))
            axs[n].set_ylim((0.0, 200.0))

            axs[n].text(
                100, 185, f"{np.round(np.array(labels[num_sample]), 2)}",
                color="black", size=12, ha="center", va="top",
                bbox=dict(facecolor="yellow", edgecolor="black", alpha=0.7),
            )

            axs[n].axis("off")  # Hide axes for cleaner visualization

        # Hide any unused subplots
        for n in range(len(Nsamp), len(axs)):
            axs[n].axis("off")

        fig.subplots_adjust(wspace=0.03)
        plt.show()

    train: Dataset
    dev: Dataset
    test: Dataset

    # Same goes for 
    # fe_sk_transition_{D_value}: Dataset
    # sk_sp_transition_{D_value}: Dataset

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: Dataset, predictions: Sequence[int]) -> float:
        gold = gold_dataset.data["labels"]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct = sum(gold[i] == predictions[i] for i in range(len(gold)))
        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        predictions = [int(line) for line in predictions_file]
        return SKYRMION.evaluate(gold_dataset, predictions)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = SKYRMION.evaluate_file(getattr(SKYRMION(), args.dataset), predictions_file)
        print("SKYRMION accuracy: {:.2f}%".format(accuracy))