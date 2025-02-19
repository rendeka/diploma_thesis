import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
from pathlib import Path
from skyrmion_dataset import SKYRMION
import keras
import torch

class TorchTensorBoardCallback(keras.callbacks.Callback):
    def __init__(self, args:argparse.Namespace, transition_datasets: Optional[Dict[str, SKYRMION]]=None, 
                 fm_dataset: Optional[SKYRMION.Dataset]=None, device: str="cuda"):
        self.args = args
        self._writers = {}
        self._transition_datasets = transition_datasets
        self.fm_dataset = fm_dataset
        self.device = device
        self._logged_epochs = set()  # To track logged epochs

    def writer(self, writer):
        if writer not in self._writers:
            import torch.utils.tensorboard
            self._writers[writer] = torch.utils.tensorboard.SummaryWriter(Path(self.args.logdir) / writer)
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        if logs:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()

    def evaluate_phase_transition(self, group_size: int = 5) -> Dict[str, Dict[str, float]]:
        """Evaluate how smoothly the model's predictions transition across ranks."""
        if self._transition_datasets is None:
            return None
        
        performance_metrics = {} # Storing the overall the performance results
        metric_types = ["out-of-order", "in-group-variance", "transition-smoothness"]

        model = self.model
        model.eval()
        
        for trans_type, skyrmion_trans_dataset in self._transition_datasets.items():
            performance_metrics[trans_type] = { 
                metric_type: 0.0 for metric_type in metric_types # Initialize metrics to zero
                }
            
            transition_attributes = [attr for attr in skyrmion_trans_dataset.__dict__ if "transition" in attr]
            
            for attr in transition_attributes:
                ordered_dataset = getattr(skyrmion_trans_dataset, attr)[:]
            
                images = torch.tensor(ordered_dataset["image"]).float().to(self.device)
                
                # labels are now ranks ordering images through phase transitions
                labels = np.array(ordered_dataset["label"])
                
                with torch.no_grad():
                    preds = model(images).cpu().numpy()

                ## Evaluate out-of-order metric

                if trans_type == "fe_sk":
                    ord_idx = 0 # Ascending order with increasing ferromagnet probability
                elif trans_type == "sk_sp":
                    ord_idx = 1 # Ascending order with increasing skyrmion probability
                else:
                    raise ValueError(f"Unknown phase transition type: '{trans_type}'. Transition dataset name must start with either 'fe_sk' or 'sk_sp'.")                
                
                preds_ordering = np.argsort(preds[:, ord_idx])

                # Transform preds_ordering to the target ordering format: 
                # [0, 1, 2, ..., n-1, n] -> [0, ..., 0, 1, ..., 1, m, ..., m]

                # Note that group_size = 5 corresponds to the fact that there are always 5 
                # different (but for our purpose equivalent) pictures for the 
                # simulated with same set of parameters B and D.

                preds_ordering = np.ceil((preds_ordering + 1) / group_size).astype(int) - 1

                # Using Euclidean distance as out-of-order metric for given B-value (TODO: consider some other metric??)
                ordering_metric = np.linalg.norm(preds_ordering - labels)

                # Adding out of order metrics (TODO: add it in a different manner??)
                performance_metrics[trans_type]["out-of-order"] += ordering_metric

                ## Evaluate in-group-variace metric
                # TODO: ?? the .sum() in the end is summing across 3 categories and all ranks in the same way, which might not be optimal
                preds_var = np.array([np.var(preds[preds_ordering == r], axis=0) 
                                     for r in np.unique(preds_ordering)]).sum()
                
                performance_metrics[trans_type]["in-group-variance"] += preds_var

                ## Evaluate transition-smoothness metric
                # Compute mean prediction per rank
                mean_preds = np.array([preds[preds_ordering == r].mean(axis=0) for r in np.unique(labels)])

                # Compute transition smoothness across ranks
                diffs = np.diff(mean_preds, axis=0)  # Difference between consecutive ranks
                transition_smoothness = np.mean(np.abs(diffs))  # Average absolute change

                performance_metrics[trans_type]["transition-smoothness"] += transition_smoothness

        return performance_metrics
    
    def log_filters_and_features(self, epoch):
        """Logs convolutional filters and feature maps to TensorBoard at given milestones of training."""
        if not self.model:
            return

        total_epochs = self.args.epochs
        log_milestones = {int(0.6 * total_epochs), total_epochs}
        
        if epoch not in log_milestones or epoch in self._logged_epochs:
            return  # Skip if it's not a logging epoch or already logged

        self._logged_epochs.add(epoch)

        writer = self.writer("filters_features")

        images = torch.tensor(self.fm_dataset.dataset[:]["image"]).float().unsqueeze(1).unsqueeze(-1).to(self.device)
        labels = np.array(self.fm_dataset.dataset[:]["label"])

        # Retrieve the first convolutional layer and filters
        filters = []
        conv_layers = []
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.SeparableConv2D):
                filters.append(layer.get_weights()[0])
                conv_layers.append(layer)

        num_rows = len(filters) # Same for plotting feature maps
        num_cols = min(8, filters[0].shape[-1])

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 3 * num_rows))
        axes = np.array(axes).ravel()
        fig.subplots_adjust(hspace=0.01, wspace=0.1)

        for i, (row, col) in enumerate(np.ndindex(num_rows, num_cols)):
            ax = axes[i]
            ax.imshow(np.mean(filters[row][..., col], axis=-1), cmap="gray")  # Averaging across channels
            ax.axis("off")

        writer.add_figure("conv_filters", fig, epoch)

        # Log feature maps
        num_cols += 1 # +1 for the original input image

        if len(filters) != len(conv_layers):
            raise ValueError(f"Unexpected mismatch: {len(filters)} filter levels vs {len(conv_layers)} layers")

        for image, label in zip(images, labels):
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 3 * num_rows))
            axes = np.array(axes).ravel()
            fig.subplots_adjust(hspace=0.1, wspace=0.1)
            for row, conv_layer in enumerate(conv_layers): 
                feature_extractor = keras.Model(inputs=self.model.input, outputs=conv_layer.output)
                feature_maps = feature_extractor(image).cpu().detach().numpy()[0] # Shape: (H, W, num_filters), and num filter increases
                for col in range(num_cols):
                    ax = axes[row * num_cols + col]
                    if col == 0:
                        if row == 0:
                            ax.imshow(image.squeeze(0).squeeze(-1).cpu().detach().numpy(), cmap="RdBu")
                    else:
                        ax.imshow(feature_maps[..., col - 1], cmap="RdBu") # Plotting just first 'num_col' fearure maps
                    ax.axis("off")
            writer.add_figure(f"feature_map_image_{label}", fig, epoch)

        writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            if isinstance(getattr(self.model, "optimizer", None), keras.optimizers.Optimizer):
                logs = logs | {"learning_rate": keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)}
            self.add_logs("train", {k: v for k, v in logs.items() if not k.startswith("val_")}, epoch + 1)
            self.add_logs("val", {k[4:]: v for k, v in logs.items() if k.startswith("val_")}, epoch + 1)

            # Log phase transition evaluation if applicable
            phase_transition_scores = self.evaluate_phase_transition()
            if phase_transition_scores is not None:
                for trans_type, metrics in phase_transition_scores.items():
                    for metric, score in metrics.items():
                        metric_category = Path(trans_type) / metric
                        self.writer(metric_category).add_scalar(metric, score, epoch + 1)
                        self.writer(metric_category).flush()

            if self.fm_dataset is not None and self.args.ffm:
                self.log_filters_and_features(epoch + 1)