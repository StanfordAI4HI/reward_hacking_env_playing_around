import logging
from typing import List, Optional, TypedDict, cast

import torch
from gymnasium.spaces import utils
from torch import nn


class GlucoseModelConfig(TypedDict, total=False):
    hidden_size: int
    num_layers: int
    action_scale: float
    use_cgm_for_obs: bool
    use_subcutaneous_glucose_obs: bool


def normalize_obs(obs):
    obs[..., 0] = (obs[..., 0] - 100) / 100
    obs[..., 1] = obs[..., 1] * 10
    return obs


class GlucoseModel(nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__()

        assert model_config["vf_share_layers"] is True
        custom_model_config = self.model_config
        
        self.hidden_size = custom_model_config.get("hidden_size", 64)
        self.num_layers = custom_model_config.get("num_layers", 3)
        self.action_scale = custom_model_config.get("action_scale", 10)
        lstm_input_size = 2
        

        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
        )
        self.action_head = nn.Linear(self.hidden_size, num_outputs)
        self.value_head = nn.Linear(self.hidden_size, 1)

        # Since we scale the action outputs by the action_scale, initialize the weights
        # and biases of the action head to be small to prevent large initial
        # action outputs.
        self.action_head.weight.data.mul_(1 / self.action_scale)
        self.action_head.bias.data.mul_(1 / self.action_scale)
        self.in_dim = self.hidden_size
    

    def forward(self, observations, state):
        obs = observations.permute(0, 2, 1).clone()
        # Normalize observations
        normalize_obs(obs)
        lstm_out, _ = self.lstm(obs)
        self._backbone_out = lstm_out[:, -1, :]
        # self._backbone_out = self.backbone(obs[:, :, :].mean(axis=1))
        acs = self.action_head(self._backbone_out) * self.action_scale
        return acs, state

    def value_function(self):
        return self.value_head(self._backbone_out).squeeze(-1)

