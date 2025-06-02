import mlflow
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Union
import numpy as np


class CustomModel(mlflow.pyfunc.PythonModel):
    """A custom MLflow model wrapper for PyTorch sentiment classification models.

    This class wraps a PyTorch model to make it compatible with MLflow's Python model interface,
    handling the conversion of inputs to tensors and outputs to numpy arrays.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the custom model wrapper.

        Args:
            model (nn.Module): The PyTorch model to wrap.
        """
        super().__init__()
        self.model = model
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Make predictions using the wrapped model.

        Args:
            context (mlflow.pyfunc.PythonModelContext): MLflow model context.
            model_input (Dict[str, Any]): Input data containing 'input_ids' and 'attention_mask'.
            params (Optional[Dict[str, Any]], optional): Additional parameters. Defaults to None.

        Returns:
            np.ndarray: Model predictions as a numpy array.
        """
        input_ids, attention_mask = (
            model_input["input_ids"],
            model_input["attention_mask"],
        )
        input_ids = torch.LongTensor(input_ids).to(self.device)
        attention_mask = torch.LongTensor(attention_mask).to(self.device)
        output = self.model(input_ids, attention_mask)
        return output.cpu().numpy()
