import torch
import torch.nn.functional as F
from typing import Dict, Optional
from torchmetrics import Accuracy
from composer.models import ComposerModel
from dbrx_llm.models.pytorch import SentimentClassifier


class ComposerSentimentClassifier(ComposerModel):
    """A sentiment classifier model compatible with MosaicML's Composer framework.

    This class wraps a PyTorch sentiment classifier model to make it compatible with
    Composer's training framework, providing methods for forward pass, loss calculation,
    and metric tracking.
    """

    def __init__(self, model_name: str, num_labels: int = 2) -> None:
        """Initialize the composer sentiment classifier.

        Args:
            model_name (str): Name of the pretrained model to use.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
        """
        super(ComposerSentimentClassifier, self).__init__()
        self.model = SentimentClassifier(model_name, num_labels)
        self.train_accuracy = Accuracy(task="binary", average="micro")
        self.val_accuracy = Accuracy(task="binary", average="micro")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a forward pass through the model.

        Args:
            batch (Dict[str, torch.Tensor]): Input batch containing 'input_ids' and 'attention_mask'.

        Returns:
            torch.Tensor: Model predictions.
        """
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        return self.model(input_ids, attention_mask)

    def loss(self, outputs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate the loss for the current batch.

        Args:
            outputs (torch.Tensor): Model predictions.
            batch (Dict[str, torch.Tensor]): Input batch containing 'label'.

        Returns:
            torch.Tensor: Calculated loss value.
        """
        labels = batch["label"]
        return F.cross_entropy(outputs, labels)

    def eval_forward(
        self, batch: Dict[str, torch.Tensor], outputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform a forward pass during evaluation.

        Args:
            batch (Dict[str, torch.Tensor]): Input batch.
            outputs (Optional[torch.Tensor], optional): Pre-computed outputs. Defaults to None.

        Returns:
            torch.Tensor: Predicted class indices.
        """
        if outputs is not None:
            return outputs.argmax(dim=1)
        outputs = self.forward(batch)
        return outputs.argmax(dim=1)

    def update_metric(
        self, batch: Dict[str, torch.Tensor], outputs: torch.Tensor, metric: Accuracy
    ) -> None:
        """Update the specified metric with the current batch results.

        Args:
            batch (Dict[str, torch.Tensor]): Input batch containing 'label'.
            outputs (torch.Tensor): Model predictions.
            metric (Accuracy): Metric to update.
        """
        labels = batch["label"]
        metric.update(outputs, labels)

    def get_metrics(self, is_train: bool = False) -> Dict[str, Accuracy]:
        """Get the appropriate metrics for the current phase.

        Args:
            is_train (bool, optional): Whether to return training metrics. Defaults to False.

        Returns:
            Dict[str, Accuracy]: Dictionary of metrics for the current phase.
        """
        device = next(self.parameters()).device
        return (
            {"Accuracy": self.train_accuracy.to(device)}
            if is_train
            else {"Accuracy": self.val_accuracy.to(device)}
        )
