import torch
import torch.nn as nn
from transformers import AutoModel


class SentimentClassifier(nn.Module):
    """A sentiment classification model based on a pretrained transformer.

    This model uses a pretrained transformer model (e.g., BERT) as a base and adds
    a classification head on top for sentiment classification tasks.
    """

    def __init__(self, model_name: str, num_labels: int = 2) -> None:
        """Initialize the sentiment classifier.

        Args:
            model_name (str): Name of the pretrained model to use.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
        """
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Perform a forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.

        Returns:
            torch.Tensor: Classification logits.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])
