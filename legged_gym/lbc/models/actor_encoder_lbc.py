from matplotlib import image
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from rsl_rl.modules.models.simple_cnn import SimpleCNN
from rsl_rl.modules.models.rnn_state_encoder import build_rnn_state_encoder

PROPRIO_SIZE = 48


class CNNRNN(nn.Module):
    def __init__(
        self, image_size, cnn_out_size, rnn_hidden_size, rnn_out_size, rnn_layers=2
    ):
        super().__init__()

        self.image_size = image_size

        # initialize encoder
        class observationSpace:
            def __init__(self):
                self.spaces = {"depth": torch.zeros(image_size)}

        print("CNN OUT SIZE: ", cnn_out_size)
        print("IM SIZE: ", image_size)

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

    def forward(self, observations, masks):
        prop, _, images = (
            observations[:, :48],
            observations[:, 48 : 48 + 187],
            observations[:, 48 + 187 :].view(
                -1, self.image_size[0], self.image_size[1], 1
            ),
        )
        image_enc = self.cnn({"depth": images})
        rnn_input = torch.cat([image_enc, prop], dim=-1)
        masks = masks.cuda()
        out, self.hidden_state = self.rnn.forward(rnn_input, self.hidden_state, masks)
        self.hidden_state = self.hidden_state.detach()

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
        self.encoder = CNNRNN(
            image_size, cnn_output_size, rnn_hidden_size, rnn_out_size
        )
        print(f"(Student) ENCODER MLP: {self.encoder}")

    def forward(self, obs):
        """Takes full observations, then will throw out the depth map itself"""
        return self.encoder(obs, masks=torch.tensor([True]))


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

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

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

    # def _trans_dm(self, depth_map):
    #     """
    #         Turn 1d dm to 2d
    #     """
    #     envs, _ = depth_map.shape
    #     depth_map = depth_map.view(envs, self.encoder_input_rows, self.encoder_input_cols, 1)

    #     dm_dict = {"depth": depth_map}
    #     return dm_dict

    def parse_observations(self, observations):
        return (
            observations[:, :48],
            observations[:, 48 : 48 + 187],
            observations[:, 48 + 187 :],
        )

    def act(self, proprio, enc_depth_map, **kwargs):
        observations = torch.cat((proprio, enc_depth_map), dim=-1)

        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, proprio, enc_depth_map):
        observations = torch.cat((proprio, enc_depth_map), dim=-1)

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
