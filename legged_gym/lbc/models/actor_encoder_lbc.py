import os

import torch
import torch.nn as nn
from rsl_rl.modules.models.rnn_state_encoder import build_rnn_state_encoder
from rsl_rl.modules.models.simple_cnn import SimpleCNN
from torch.distributions import Normal

PROPRIO_SIZE = 48


class CNNRNN(nn.Module):
    def __init__(
        self,
        image_size,
        cnn_out_size,
        rnn_hidden_size,
        rnn_out_size,
        rnn_layers=2,
        no_rnn=False,
    ):
        super().__init__()

        self.image_size = image_size

        # initialize encoder
        class observationSpace:
            def __init__(self):
                self.spaces = {"depth": torch.zeros(image_size)}

        self.cnn = SimpleCNN(observationSpace(), cnn_out_size)

        self.rnn_layers = rnn_layers
        self.rnn = build_rnn_state_encoder(
            cnn_out_size + PROPRIO_SIZE,
            rnn_hidden_size,
            rnn_type="lstm",
            num_layers=rnn_layers,
        )
        self.rnn_linear = nn.Linear(rnn_hidden_size, rnn_out_size)
        self.rnn_hidden_size = rnn_hidden_size
        self.hidden_state = None

        # Add support for NOT using RNN
        if no_rnn:
            input_size = cnn_out_size + PROPRIO_SIZE
            hidden_sizes = rnn_layers * [rnn_hidden_size]
            self.mlp = construct_mlp_base(input_size, hidden_sizes)
        else:
            self.mlp = None

    def forward(self, observations, masks):
        if isinstance(observations, torch.Tensor):
            observations = observations[:, :-1]
            prop, _, images = (
                observations[:, :48],
                observations[:, 48 : 48 + 187],
                observations[:, 48 + 187 :].view(
                    -1, self.image_size[0], self.image_size[1], 1
                ),
            )
        elif isinstance(observations, dict):
            prop = observations["proprioception"]
            images = observations["tilted_image"]
        else:
            raise RuntimeError(f"Obs type {type(observations)} invalid!")
        image_enc = self.cnn({"depth": images})
        rnn_input = torch.cat([image_enc, prop], dim=-1)
        if self.mlp is None:
            masks = masks.cuda()
            out, self.hidden_state = self.rnn.forward(rnn_input, self.hidden_state, masks)
            self.hidden_state = self.hidden_state.detach()
        else:
            out = self.mlp(rnn_input)

        return self.rnn_linear(out)


class VisionEncoder(nn.Module):
    def __init__(
        self,
        image_size=[180, 320, 1],
        cnn_output_size=32,
        rnn_hidden_size=64,
        rnn_out_size=32,
    ):
        super(VisionEncoder, self).__init__()
        no_rnn = os.environ.get("ISAAC_NO_RNN", "False") == "True"
        self.encoder = CNNRNN(
            image_size, cnn_output_size, rnn_hidden_size, rnn_out_size, no_rnn=no_rnn
        )
        print(f"(Student) ENCODER MLP: {self.encoder}")

    def forward(self, obs):
        """Takes full observations, then will throw out the depth map itself"""
        if isinstance(obs, dict):
            masks = torch.tensor([True], dtype=torch.bool, device="cuda")
        else:
            masks = obs[:, -1].bool()
        return self.encoder(obs, masks=masks)


class Actor(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        image_size=[180, 320, 1],
        cnn_output_size=32,
        rnn_hidden_size=64,
        rnn_out_size=32,
        train_type="lbc",  # standard, priv, lbc
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        """
        image: [H, W, C]
        """
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(Actor, self).__init__()

        activation = get_activation(activation)
        self.train_type = train_type

        mlp_input_dim_a = num_actor_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        print(f"(Student) Actor MLP: {self.actor}")
        print("(Student) Train Type: ", self.train_type)

        # self.encoder_input_rows, self.encoder_input_cols, _ = encoder_input_size

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def parse_observations(self, observations):
        return (
            observations[:, :48],
            observations[:, 48 : 48 + 187],
            observations[:, 48 + 187 :],
        )

    def act(self, proprio, enc_depth_map, **kwargs):
        if os.environ["ISAAC_BLIND"] != "True":
            observations = torch.cat((proprio, enc_depth_map), dim=-1)
        else:
            observations = proprio
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, proprio, enc_depth_map):
        if os.environ["ISAAC_BLIND"] != "True":
            observations = torch.cat((proprio, enc_depth_map), dim=-1)
        else:
            observations = proprio
        actions_mean = self.actor(observations)
        return actions_mean


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


def construct_mlp_base(input_size, hidden_sizes):
    layers = []
    prev_size = input_size
    for out_size in hidden_sizes:
        layers.append(
            nn.Linear(int(prev_size), int(out_size))
        )
        layers.append(nn.ReLU())
        prev_size = out_size
    mlp = nn.Sequential(*layers) if len(layers) > 1 else layers[0]

    return mlp
