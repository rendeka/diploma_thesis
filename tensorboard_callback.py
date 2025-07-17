import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
from pathlib import Path
from skyrmion_dataset import SKYRMION
import keras
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import gc

def get_callbacks(args, skyrmion_transitions, skyrmion_fm, test_dataset):
    callbacks = []
    
    tb_callback = TorchTensorBoardCallback(
        args, 
        transition_datasets=skyrmion_transitions, 
        fm_dataset=skyrmion_fm,
        test_dataset=test_dataset
    )
    callbacks.append(tb_callback)
    
    if args.decay == "plateau":
        reduce_on_plateau = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=15,
            verbose=1,
            min_lr=1e-8
        )
        callbacks.append(reduce_on_plateau)

    if args.early_stopping:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=40,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

    return callbacks


class ClassifierOutputSoftTarget:
    def __init__(self, label):
        self.label = torch.tensor(label).float()

    def __call__(self, model_output):
        return torch.sum(model_output * self.label.to(model_output.device))

class TorchTensorBoardCallback(keras.callbacks.Callback):
    def __init__(self, args:argparse.Namespace, 
                 transition_datasets: Optional[Dict[str, SKYRMION]]=None, 
                 fm_dataset: Optional[SKYRMION.Dataset]=None,
                 test_dataset: Optional[SKYRMION.Dataset]=None, 
                 device: str="cuda"):
        self.args = args
        self._writers = {}
        self._transition_datasets = transition_datasets
        self.fm_dataset = fm_dataset
        self.test_dataset = test_dataset
        self.device = device
        self._logged_epochs = set()  # To track logged epochs

        if self.args.scope == "sub":
                self.group_size = 20
        elif self.args.scope == "full":
            self.group_size = 5
        else:
            self.group_size = 5
            raise ValueError(f"args.scope == {self.args.scope}, does not have assigned group size, group_size == 5 will be used")

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

    def evaluate_phase_transition(self) -> Dict[str, Dict[str, float]]:
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
                
                # make predictions in batches
                dataset = TensorDataset(images)
                dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)

                all_preds = []
                with torch.no_grad():
                    for batch in dataloader:
                        batch_images = batch[0].to(self.device)
                        preds = model(batch_images).cpu().numpy()
                        all_preds.append(preds)

                preds = np.concatenate(all_preds, axis=0)

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

                preds_ordering = np.ceil((preds_ordering + 1) / self.group_size).astype(int) - 1

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
    
    def log_phase_transition_probs(self) -> Dict[str, Dict[str, float]]:
        """Evaluate how smoothly the model's predictions transition across ranks."""
        if self._transition_datasets is None:
            return None

        model = self.model
        model.eval()

        writer = self.writer("phase_trans_probs")
        
        for trans_type, skyrmion_trans_dataset in self._transition_datasets.items():
            
            transition_attributes = [attr for attr in skyrmion_trans_dataset.__dict__ if "transition" in attr]
            
            for attr in transition_attributes:
                D = float(attr.split("-")[-1])
                ordered_dataset = getattr(skyrmion_trans_dataset, attr)[:]
            
                images = torch.tensor(ordered_dataset["image"]).float().to(self.device)
                
                # labels are now ranks ordering images through phase transitions
                b_values = np.array(ordered_dataset["b_value"])
                b_unique = np.unique(b_values)
                num_groups = len(b_unique)

                # make predictions in batches
                dataloader = DataLoader(TensorDataset(images), batch_size=self.args.batch_size, shuffle=False)

                preds = []
                with torch.no_grad():
                    for batch in dataloader:
                        batch_images = batch[0].to(self.device)
                        preds.extend(model(batch_images).cpu().numpy())

                preds = np.array(preds).reshape(num_groups, self.group_size, -1)
                b_values = b_values.reshape(num_groups, self.group_size)

                mean_preds = preds.mean(axis=1)
                var_preds = preds.var(axis=1)

                labels = ["fe", "sk", "sp"]
                colors = ['r', 'g', 'b']

                fig = plt.figure(figsize=(14, 8))
                gs = fig.add_gridspec(2, num_groups, height_ratios=[1, 1])

                ax1 = fig.add_subplot(gs[0, :num_groups // 2])
                ax2 = fig.add_subplot(gs[0, num_groups // 2:])

                axes = [fig.add_subplot(gs[1, i]) for i in range(num_groups)]

                fig.suptitle(f"Transition: {trans_type.replace('_', '-')}, D: {D}")

                for i, (color, label) in enumerate(zip(colors, labels)):
                    ax1.plot(b_unique, mean_preds[:, i], marker='o', linestyle='-', color=color, label=label)

                ax1.set_title("Average Probabilities")
                ax1.set_xlabel("B")
                ax1.set_ylabel("Mean Prediction")
                ax1.grid(True)
                ax1.legend()

                for i, (color, label) in enumerate(zip(colors, labels)):
                    ax2.plot(b_unique, var_preds[:, i], marker='o', linestyle='-', color=color, label=label)

                ax2.set_title("Variances")
                ax2.set_xlabel("B")
                ax2.set_ylabel("Variance of Prediction")
                ax2.grid(True)
                ax2.legend()

                sample_images = images[::self.group_size]
                for i in range(num_groups):
                    axes[i].imshow(sample_images[i].cpu().numpy(), vmin=0.0, vmax=1.0, cmap="RdBu")
                    axes[i].axis('off')

                plt.tight_layout(rect=[0, 0, 1, 0.95])

                # writer.add_figure(str(Path('transition probabilities') / f"{trans_type.replace('_', '-')}" / f"D: {D}"), fig) # old version 
                # new version
                base_path = f"tran-{Path(self.args.logdir).name[18:]}"
                writer.add_figure(str(Path(base_path) / f"D: {D}"), fig) 
        writer.flush()             

        return None
    
    def evaluate_test(self, n_worst:int = 8):
        "Evaluates the accuracy on the test set, and outputs `n_worst` images and predictions"

        model = self.model
        model.eval()

        # dataset = self.test_dataset.data[:]

        # images = torch.Tensor(dataset["image"]).float().to(self.device)
        # labels = torch.nn.functional.one_hot(torch.Tensor(dataset["label"], dtype=torch.long), num_classes=len(SKYRMION.LABELS))

        # dataloader = DataLoader(TensorDataset(images), batch_size=self.args.batch_size, shuffle=False)

        dataloader = self.test_dataset

        images, labels, preds = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                batch_images = batch[0].to(self.device)
                preds.extend(model(batch_images).cpu().numpy())
                images.extend(batch[0])
                labels.extend(batch[1])

        preds = np.array(preds)
        images = np.array(images)
        labels = np.array(labels)

        errors = np.abs(preds - labels).sum(axis=1)
        total_err = np.sum(errors)
        worst_idcs = np.argsort(errors)[-n_worst:]

        worst_samples = {
            "images": images[worst_idcs],
            "labels": labels[worst_idcs],
            "preds": preds[worst_idcs]
        }

        return total_err, worst_samples

    def log_test_results(self, epoch):
        total_err, samples = self.evaluate_test()

        writer = self.writer("test_error")
        writer.add_scalar("Test Absolute error", total_err, epoch)
        writer.flush()
        
        # Log worst images with predictions
        if self.args.grad_cam:
            fig = self.get_gradcam(samples)
        else:
            fig = SKYRMION.visualize_images(samples["images"].squeeze(-1), labels=samples["preds"], row_size=4, base_size=3, show_images=False)
        if fig:
            writer.add_figure("Worst Predictions", fig, epoch)
            writer.flush()
    
    def log_filters_and_features(self, epoch):
        """Logs convolutional filters and feature maps to TensorBoard at given milestones of training."""
        if self.model.__class__.__name__ == "ModelFFN":
            return

        # total_epochs = self.args.epochs
        # log_milestones = {int(0.6 * total_epochs), total_epochs}
        
        # if epoch not in log_milestones or epoch in self._logged_epochs:
        #     return  # Skip if it's not a logging epoch or already logged

        self._logged_epochs.add(epoch)

        writer = self.writer("filters_features")

        images = torch.tensor(self.fm_dataset.dataset[:]["image"]).float().unsqueeze(1).unsqueeze(-1).to(self.device)
        labels = np.array(self.fm_dataset.dataset[:]["label"])

        # Retrieve convolutional layers and filters
        # def get_all_conv_layers(model):
        #     conv_layers = []
        #     filters = []

        #     def recurse_layers(layer):
        #         if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
        #             conv_layers.append(layer)
        #             weights = layer.get_weights()
        #             if weights:
        #                 filters.append(weights[0])
        #         elif hasattr(layer, 'layers'):
        #             for sublayer in layer.layers:
        #                 recurse_layers(sublayer)

        #     recurse_layers(model)
        #     return filters, conv_layers
        
        # filters, conv_layers = get_all_conv_layers(self.model)

        filters = []
        conv_layers = []

        for layer in self.model.layers:
            # Conv2D or SeparableConv2D
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
                conv_layers.append(layer)
                filters.append(layer.get_weights()[0])
            
            # Sequential - padding + conv
            elif isinstance(layer, keras.models.Sequential):
                if layer.layers and isinstance(layer.layers[-1], (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
                    conv_layers.append(layer)
                    filters.append(layer.layers[-1].get_weights()[0])

        num_rows = len(filters) # Same for plotting feature maps
        num_cols = min(4, filters[0].shape[-1])

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 3 * num_rows))
        axes = np.array(axes).ravel()
        fig.subplots_adjust(hspace=0.01, wspace=0.1)

        for i, (row, col) in enumerate(np.ndindex(num_rows, num_cols)):
            ax = axes[i]
            ax.imshow(np.mean(filters[row][..., col], axis=-1), cmap="gray")  # Averaging across channels
            # ax.imshow(filters[row][0, ..., col], cmap="gray") 
            ax.axis("off")

        writer.add_figure("conv_filters", fig, epoch)

        # Log feature maps
        # num_rows += 1 # +1 for the original input image

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
                    if row == 0:
                        if col == 0:
                            ax.imshow(image.squeeze(0).squeeze(-1).cpu().detach().numpy(), cmap="RdBu")
                    else:
                        ax.imshow(feature_maps[..., col - 1], cmap="RdBu") # Plotting just first 'num_col' fearure maps
                    ax.axis("off")
            writer.add_figure(f"feature_map_image_{label}", fig, epoch)

        writer.flush()

    def get_gradcam(self, data, row_size=4):

        class NCHWModelWrapper(torch.nn.Module):
            def __init__(self, keras_model):
                super().__init__()
                self.keras_model = keras_model
            
            def forward(self,x):
                x_nchw = x.permute(0, 2, 3, 1).contiguous()
                return self.keras_model(x_nchw)

        if not self.model:
            return
        
        torch.cuda.empty_cache()
        gc.collect()
                
        # model = NCHWModelWrapper(self.model)

        # self.model.eval()
        # model.eval()

        images = data["images"]
        labels = data["labels"]

        # Convert to torch tensor
        images_tensor = torch.tensor(images).float().to(self.device)
        labels_tensor = torch.tensor(labels).float()

        # Prepare targets
        targets = [ClassifierOutputSoftTarget(label.tolist()) for label in labels_tensor]
        # targets = [ClassifierOutputTarget(label.argmax()) for label in labels_tensor]

        # Find last convolutional layer
        target_layer = None
        # for layer in reversed(self.model.layers):
        for layer in reversed(self.model.layers):
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
                target_layer = layer
                break
            elif isinstance(layer, keras.models.Sequential):
                if layer.layers and isinstance(layer.layers[-1], (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
                    target_layer = layer
                    break

        if target_layer is None:
            raise ValueError("No convolutional layer found for Grad-CAM.")
        
        model_wrapped = NCHWModelWrapper(self.model)

        grayscale_cams = []

        for img, target in zip(images_tensor.permute(0, 3, 1, 2), targets):
            with GradCAM(model=model_wrapped, target_layers=[target_layer]) as cam:
                grayscale_cams.append(cam(input_tensor=img.unsqueeze(0), targets=[target]))

        # Prepare images for visualization (convert to RGB and normalize per image)
        img_rgb = np.repeat(images[..., np.newaxis], 3, axis=-1)  # shape: (N, H, W, 3)
        img_rgb = img_rgb / (np.max(img_rgb, axis=(1, 2), keepdims=True) + 1e-8)  # normalize

        # Plotting
        N = len(images)
        cols = min(row_size, N)
        rows = (N + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

        if rows == 1:
            axs = np.expand_dims(axs, axis=0)
        if cols == 1:
            axs = np.expand_dims(axs, axis=1)


        for i in range(rows * cols):
            ax = axs[i // cols, i % cols]
            if i >= N:
                ax.axis("off")
                continue

            grayscale_cam = grayscale_cams[i]
            if grayscale_cam.ndim == 3:
                if grayscale_cam.shape[-1] == 1:
                    grayscale_cam = grayscale_cam.squeeze(-1)
                if grayscale_cam.shape[0] == 1:
                    grayscale_cam = grayscale_cam.squeeze(0)

            cam_image = show_cam_on_image(img_rgb[i].squeeze(2), grayscale_cam, image_weight=0.8, use_rgb=True)

            ax.text(
                100, 15, f"{np.round(data['preds'][i], 2)}",
                color="black", size=12, ha="center", va="top",
                bbox=dict(facecolor="yellow", edgecolor="black", alpha=0.7),
            )
            ax.imshow(cam_image)
            ax.axis("off")

        fig.tight_layout()
        return fig

    def on_epoch_end(self, epoch, logs=None):
        self.last_epoch = epoch + 1
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
            
    def on_train_end(self, logs=None):

        if self.args.trans_probs:
            self.log_phase_transition_probs()

        if self.test_dataset is not None:
            self.log_test_results(self.last_epoch)

        if self.fm_dataset is not None and self.args.ffm:
            self.log_filters_and_features(self.last_epoch)