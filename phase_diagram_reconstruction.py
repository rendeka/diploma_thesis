# import torch
import numpy as np

import keras
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

dataset = np.load("data/train/datasetMC.npz")

data = dataset["data"]
V = dataset["values"]

# Old (non-relevant) toy model <- need to be replaced

# This does no longer work (I would have to save the model like this: 
#   model.save("saved_models/model_original.keras", save_format="keras"))
# model = tf.keras.models.load_model(Path("saved_models") / "model_original")

model = keras.layers.TFSMLayer("saved_models/model_original", call_endpoint="serving_default")

# preds = model.predict(data)
preds = model(data).numpy()
r, g, b = preds[:, :3].T

norm = Normalize(vmin=0, vmax=1)
r, g, b = norm(r), norm(g), norm(b)
colors = np.vstack([r, g, b]).T


fig, ax = plt.subplots(figsize=(11,8))

m = ax.scatter(*V.T, c=colors, marker="o", s=60)

ax.set_title("Topological number", size=16)
ax.set_xlabel("D", size=16)
ax.set_ylabel("B", size=16)


cbar = fig.colorbar(m)

plt.show()