import torch
from pytorch_forecasting.metrics import QuantileLoss

class WeightedQuantileLoss(QuantileLoss):
    def __init__(self, quantiles=[0.1, 0.5, 0.9], use_exponential_weights=False, **kwargs) -> None:
        super().__init__(quantiles=quantiles, **kwargs)
        self.use_exponential_weights: bool = use_exponential_weights
        # Explicitly register the quantiles tensor buffer to ensure it's available
        self.register_buffer(
            "quantiles_tensor", 
            torch.tensor(self.quantiles, dtype=torch.float)
        )

    def loss(self, y_pred, target) -> torch.Tensor:
        """
        y_pred: (batch_size, decoder_length, num_quantiles)
        target: (batch_size, decoder_length, 1)
        """
        decoder_length: int = y_pred.size(1)

        # Generate weights: either linear or exponential
        if self.use_exponential_weights:
            weights: torch.Tensor = torch.exp(torch.linspace(0, 1, decoder_length, device=y_pred.device))
        else:
            weights: torch.Tensor = torch.arange(1, decoder_length + 1, device=y_pred.device).float()

        weights = weights / weights.sum()  # Normalize to sum to 1
        weights = weights.view(1, -1, 1)  # Reshape to (1, decoder_length, 1)

        # Ensure target has a third dimension, then expand it
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        
        # Expand target to match y_pred shape
        target: torch.Tensor = target.expand_as(y_pred)

        # Compute quantile loss
        errors: torch.Tensor = target - y_pred
        losses: torch.Tensor = torch.max(
            self.quantiles_tensor.view(1, 1, -1) * errors,
            (self.quantiles_tensor.view(1, 1, -1) - 1) * errors
        )

        # Apply time-dependent weights
        weighted_losses: torch.Tensor = losses * weights

        return weighted_losses.mean()
