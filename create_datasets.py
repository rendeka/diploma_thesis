import numpy as np
from augmentation_functions import get_cutmix_samples, get_mixup_samples

data_original = np.load("data/train/sup_data.npz")

print(data_original)