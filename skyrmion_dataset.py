import os
import sys
from typing import Any, Callable, Sequence, TextIO, TypedDict

import numpy as np
import torch


class SKYRMION:
    H: int = 200
    W: int = 200
    C: int = 1
    # LABELS: list[str] = ["ferromagnet", "skyrmion", "spiral"]
    LABELS: list[int] = [0, 1, 2]

    Element = TypedDict("Element", {"image": np.ndarray, "label": np.ndarray})
    Elements = TypedDict("Elements", {"images": np.ndarray, "labels": np.ndarray})

    _PATH = "data/train/skyrmion_dataset.npz"

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

    def __init__(self, size: dict[str, int] = {}) -> None:
        # path = os.path.basename(self._URL)
        path = self._PATH
        if not os.path.exists(path):
            print("SKYRMION dataset not found...", file=sys.stderr)


        skyrmion = np.load(path)
        for dataset in ["train", "dev", "test"]:
            data = {key[len(dataset) + 1:]: skyrmion[key][:size.get(dataset, None)]
                    for key in skyrmion if key.startswith(dataset)}
            setattr(self, dataset, self.Dataset(data))

    train: Dataset
    dev: Dataset
    test: Dataset

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