import os
import sys
from typing import Any, Callable, Sequence, TextIO, TypedDict
import numpy as np
import tensorflow as tf


class SKYRMION:
    H: int = 200
    W: int = 200
    C: int = 1
    LABELS: list[int] = [0, 1, 2]

    Element = TypedDict("Element", {"image": np.ndarray, "label": np.ndarray})
    Elements = TypedDict("Elements", {"images": np.ndarray, "labels": np.ndarray})

    def __init__(self, path: str = "data/train/skyrmion_dataset.npz", size: dict[str, int] = {}) -> None:
        if not os.path.exists(path):
            print("SKYRMION dataset not found...", file=sys.stderr)
            sys.exit(1)

        # Load the dataset
        skyrmion = np.load(path)
        for dataset in ["train", "dev", "test"]:
            data = {key[len(dataset) + 1:]: skyrmion[key][:size.get(dataset, None)]
                    for key in skyrmion if key.startswith(dataset)}
            setattr(self, dataset, self.create_dataset(data))

    def create_dataset(self, data: "SKYRMION.Elements") -> tf.data.Dataset:
        """Creates a TensorFlow dataset from the given data."""
        images = data["images"]
        labels = data["labels"].ravel()

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        return dataset.map(self.process_element, num_parallel_calls=tf.data.AUTOTUNE)

    @staticmethod
    def process_element(image: np.ndarray, label: int) -> tuple:
        """Processes each dataset element into the required format."""
        image = tf.convert_to_tensor(image, dtype=tf.float64)
        label = tf.one_hot(label, depth=len(SKYRMION.LABELS))
        return image, label

    @staticmethod
    def evaluate(gold_dataset: tf.data.Dataset, predictions: Sequence[int]) -> float:
        """Evaluates accuracy by comparing predictions with gold labels."""
        gold_labels = np.concatenate([label.numpy() for _, label in gold_dataset])
        if len(predictions) != len(gold_labels):
            raise RuntimeError(f"The predictions are of different size than gold data: {len(predictions)} vs {len(gold_labels)}")

        correct = sum(int(np.argmax(gold_labels[i]) == predictions[i]) for i in range(len(gold_labels)))
        return 100 * correct / len(gold_labels)

    @staticmethod
    def evaluate_file(gold_dataset: tf.data.Dataset, predictions_file: TextIO) -> float:
        """Evaluates accuracy by reading predictions from a file."""
        predictions = [int(line.strip()) for line in predictions_file]
        return SKYRMION.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            dataset = SKYRMION()
            accuracy = SKYRMION.evaluate_file(getattr(dataset, args.dataset), predictions_file)
        print(f"SKYRMION accuracy: {accuracy:.2f}%")
