import numpy as np
import torch

class Policy(torch.nn.Module):
    """
    Policy network that builds architecture from ppo_config.
    
    Args:
        env: PufferLib vectorized environment
        ppo_config: Dict containing:
            - model["fcnet_hiddens"]: List of hidden layer sizes (e.g., [128, 128])
            - model["fcnet_activation"]: Activation function name (e.g., "relu", "tanh")
    """
    def __init__(self, env, ppo_config):
        super().__init__()
        
        # Get architecture specs from config
        model_config = ppo_config["model"]
        fcnet_hiddens = model_config["fcnet_hiddens"]
        activation_name = model_config["fcnet_activation"].lower()
        custom_action_dist = model_config.get("custom_action_dist", None)
        
        # Map activation name to PyTorch activation
        activation_map = {
            "relu": torch.nn.ReLU,
            "tanh": torch.nn.Tanh,
            "leaky_relu": torch.nn.LeakyReLU,
            "elu": torch.nn.ELU,
        }
        activation_fn = activation_map.get(activation_name, torch.nn.ReLU)
        
        # Build shared network (for feature extraction)
        layers = []
        input_size = env.single_observation_space.shape[0]*env.single_observation_space.shape[1]*env.single_observation_space.shape[2]
        print (f"Input size: {input_size}")
        print ("env.single_observation_space.shape: ", env.single_observation_space.shape)
        for hidden_size in fcnet_hiddens:
            layers.append(torch.nn.Linear(input_size, hidden_size))
            layers.append(activation_fn())
            input_size = hidden_size

        self.net = torch.nn.Sequential(*layers).float()

        # Action and value heads
        # Check if discrete or continuous action space
        if hasattr(env.single_action_space, 'n'):
            # Discrete action space
            action_size = env.single_action_space.n
        else:
            # Continuous action space (Box)
            action_size = np.prod(env.single_action_space.shape)
        
        self.action_head = torch.nn.Linear(input_size, action_size)
        self.value_head = torch.nn.Linear(input_size, 1)

    def forward_eval(self, observations, state=None):
        # print (f"observations shape in forward_eval: {observations.float().view(observations.size(0), -1).shape}")
        hidden = self.net(observations.float().view(observations.size(0), -1))
        logits = self.action_head(hidden)
        values = self.value_head(hidden)        
        return logits, values

    # We use this to work around a major torch perf issue
    def forward(self, observations, state=None):
        return self.forward_eval(observations.float().view(observations.size(0), -1), state)
