import torch
import torch.nn.functional as F

class WeightedSoftCrossEntropy(torch.nn.Module):
    def __init__(self, class_weights=[1, 2, 0.5]):
        super().__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=1e-7, max=1.0)
        log_preds = torch.log(y_pred)
        weights = self.class_weights.to(y_pred.device)
        loss = -y_true * log_preds * weights
        return loss.sum(dim=1).mean()

    def get_config(self):
        return {
            "class_weights": self.class_weights.cpu().numpy().tolist()
        }

    @classmethod
    def from_config(cls, config):
        return cls(class_weights=config.get("class_weights", [1, 1.5, 0.75]))