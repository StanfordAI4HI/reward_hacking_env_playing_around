import torch
import torch.nn.functional as F
from torch.distributions import Beta


class UnclampedBeta:
    """
    Version of Beta that doesn't clamp the inputs to be in a small range.
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        low: float = 0.0,
        high: float = 1.0,
    ):
        self.inputs = inputs
        # Stabilize input parameters (possibly coming from a linear layer).
        self.inputs = F.softplus(self.inputs) + 1.0  # type: ignore
        self.low = low
        self.high = high
        alpha, beta = torch.chunk(self.inputs, 2, dim=-1)  # type: ignore
        # Note: concentration0==beta, concentration1=alpha (!)
        self.dist = Beta(concentration1=alpha, concentration0=beta)

