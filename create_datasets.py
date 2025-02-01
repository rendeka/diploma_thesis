import numpy as np

#from augmentation_functions import get_cutmix_samples, get_mixup_samples # Not going to augment the sdata here (for now)

def reshape_images(images, reshape_dims=(200, 200, 1)):
    """Creates additional dimension for the 2D image."""
    return np.array([image.reshape(reshape_dims) for image in images])

def separate_dataset(images, labels):
    """Divides the data and the corresponding labels into 3 separate datasets, each one containing just a single phase"""
    phases = ["ferromagnet", "skyrmion", "spiral"]
    separated_dataset = {}

    for label, phase in enumerate(phases): # 0-ferromagnet, 1-skyrmion, 2-spiral
        idxs = labels == label 
        separated_dataset[phase] = (images[idxs], labels[idxs])
    
    return separated_dataset

def shuffle_dataset(images, labels, random_seed=42):
    """Suffles the samples in the dataset"""
    np.random.seed(random_seed)
    indices = np.random.permutation(len(images))
    return images[indices], labels[indices]


def split_dataset(images, labels, train_ratio, dev_ratio, random_seed=42):
    """Splits the data into train, dev, and test sets."""

    images, labels = shuffle_dataset(images, labels, random_seed)
    
    train_stop = int(len(images) * train_ratio)
    dev_stop = train_stop + int(len(images) * dev_ratio)

    splitted_dataset = {
        "train": {"images": images[:train_stop], "labels": labels[:train_stop]},
        "dev": {"images": images[train_stop:dev_stop], "labels": labels[train_stop:dev_stop]},
        "test": {"images": images[dev_stop:], "labels": labels[dev_stop:]},
    }
    
    return splitted_dataset

def train_dev_test_split(images, labels, train_ratio=0.8, dev_ratio=0.1):
    """
    Splits data and labels into train, dev, and test sets by aggregating results 
    from all classes in the dataset.
    """
    # Initialize combined dataset structure using dictionary comprehensions
    combined_splits = {split: {"images": [], "labels": []} for split in ["train", "dev", "test"]}
    
    # Iterate over separated data and combine splits
    for subset_images, subset_labels in separate_dataset(images, labels).values():
        splits = split_dataset(subset_images, subset_labels, train_ratio, dev_ratio)
        for split in combined_splits: # train, dev, test
            for key in combined_splits[split]: # images, labels
                combined_splits[split][key].extend(splits[split][key])
    
    return combined_splits


def save_dataset(filename, train, dev, test):
    """Saves the dataset into a `.npz` file."""
    np.savez(
        filename,
        train_images=train["images"], train_labels=train["labels"],
        dev_images=dev["images"], dev_labels=dev["labels"],
        test_images=test["images"], test_labels=test["labels"]
    )


if __name__ == "__main__":

    # Create training dataset called skyrmion_dataset.npz from sup_data_npz
    sup_data = np.load("data/train/sup_data.npz")
    train, dev, test = train_dev_test_split(reshape_images(sup_data["data"]), sup_data["labels"], train_ratio=0.8, dev_ratio=0.2).values()
    save_dataset("data/train/skyrmion_dataset.npz", train, dev, test)