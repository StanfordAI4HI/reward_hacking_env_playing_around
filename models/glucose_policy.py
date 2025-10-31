import logging
from typing import List, Optional, TypedDict, cast

import torch
from gymnasium.spaces import utils
from torch import nn
from utils.glucose_action_dist import GlucoseBeta


def normalize_obs(obs):
    obs[..., 0] = (obs[..., 0] - 100) / 100
    obs[..., 1] = obs[..., 1] * 10
    return obs


class GlucosePolicy(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        assert model_config["vf_share_layers"] is True
        custom_model_config = model_config
        
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
        self.action_head = nn.Linear(self.hidden_size, 2) #num_outputs = 2; alpha and beta for the beta distribution
        self.value_head = nn.Linear(self.hidden_size, 1)

        # Since we scale the action outputs by the action_scale, initialize the weights
        # and biases of the action head to be small to prevent large initial
        # action outputs.
        self.action_head.weight.data.mul_(1 / self.action_scale)
        self.action_head.bias.data.mul_(1 / self.action_scale)
        self.in_dim = self.hidden_size
    

    def forward_eval(self, observations, state=None):
        obs = observations.float().permute(0, 2, 1).clone()
        # Normalize observations
        normalize_obs(obs)
        # Ensure obs matches the dtype of LSTM parameters
        obs = obs.to(dtype=next(self.lstm.parameters()).dtype)
        lstm_out, _ = self.lstm(obs)
        self._backbone_out = lstm_out[:, -1, :]
        # self._backbone_out = self.backbone(obs[:, :, :].mean(axis=1))
        logits = self.action_head(self._backbone_out) * self.action_scale
        values = self.value_head(self._backbone_out)
        # print ("logits shape: ", logits.shape)

        '''
        pufferl uses the following lines of code to sample actions:

        logits, value = self.policy.forward_eval(o_device, state)
        action, logprob, _ = pufferlib.pytorch.sample_logits(logits)

        (from /next/u/stephhk/miniconda3/envs/rl/lib/python3.10/site-packages/pufferlib/pufferl.py )

        but GlucosePolicy is not compatible with this; the action returned by GlucosePolicy.forward_eval is already sampled from our custom GlucoseBeta distribution. 
        '''
        dist = GlucoseBeta(logits, self)
        action,_= dist.sample_action()
        action = action.squeeze(-1).float()
        action = action.detach().cpu().numpy()
        print ("logits: ", logits)
        print ("action: ", action)
        return action, values

    def forward(self, observations, state=None):
        logits, values = self.forward_eval(observations, state)
        return logits, state

    def value_function(self):
        return self.value_head(self._backbone_out).squeeze(-1)

