import torch

from utils.action_distributions import UnclampedBeta


class GlucoseBeta(UnclampedBeta):
    def __init__(
        self,
        inputs: torch.Tensor,
        low: float = 0,
        high: float = 0.1,
    ):
        # can get low and high from the action space
        # action_space = model.action_space
        # assert(isinstance(action_space, Box))

        self.low = 0
        self.high = 0.1

        self.low_t  = torch.as_tensor(self.low)
        self.high_t = torch.as_tensor(self.high)

        super().__init__(inputs, self.low, self.high)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(x.clamp(self.low + 1e-3, self.high - 1e-3))

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy().sum(-1)

    def kl(self, other: "GlucoseBeta") -> torch.Tensor:
        return torch.distributions.kl_divergence(self.dist, other.dist).sum(-1)


    def sample_action(self, deterministic: bool = False):
        """
        Sample an action from the GlucoseBeta distribution and return both
        the action in [low, high] and its log probability.

        Args:
            deterministic (bool): If True, use the mean of the Beta distribution
                                instead of sampling.

        Returns:
            (action, logp): tuple of tensors
                action: sampled (or mean) action in [low, high]
                logp: log probability of that action under the distribution
        """
        # 1. sample in [0, 1]
        if deterministic:
            u = self.dist.mean
        else:
            u = self.dist.sample()  # or .rsample() if you need reparameterized gradients

        # 2. map to [low, high]
        scale = self.high_t - self.low_t
        action = self.low_t + u * scale

        # 3. compute log-probability with change of variables
        # log p(a) = log p(u) - log |d a / d u| = log p(u) - log(scale)
        logp = self.dist.log_prob(u) - torch.log(scale)
        logp = logp.sum(dim=-1)  # sum over action dims

        return action, logp