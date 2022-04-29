import torch
import torch.nn as nn
import torch.optim as optim

from legged_gym.lbc.models.actor_encoder_lbc import ActorEncoder
from legged_gym.lbc.algorithms.lbc_storage import LbcStorage


class LBC:
    actor_enc: ActorEncoder

    def __init__(
        self,
        actor_enc,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1e-3,
        schedule="fixed",
        device="cpu",
    ):

        self.device = device

        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_enc = actor_enc
        self.actor_enc.to(self.device)
        # self.storage = self.init_storage()  # Currently bc using LSTM we don't need storage buffer
        self.optimizer = optim.Adam(
            self.actor_enc.encoder.parameters(), lr=learning_rate
        )

        # LBC parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

    def init_storage(self):
        self.storage = (
            LbcStorage()
        )  # TODO write the LBC Storage with add im, add dm, clear functions.

    def test_mode(self):
        self.actor_enc.test()

    def train_mode(self):
        self.actor_enc.train()

    def act(self, obs):
        """
            image_slice: env * H * W
            dm_slice: env * dm_size
        """
        # if self.actor_critic.is_recurrent:
        #     self.transition.hidden_states = self.actor_critic.get_hidden_states()

        # Compute the actions and values
        actions = self.actor_enc.act(obs).detach()
        # need to record obs and critic_obs before env.step()
        # self.storage.add_pair(image_slice, dm_slice)   #TODO if we ever actually need an image buffer we'll use this
        return actions

    def process_env_step(self, rewards, dones, infos):
        self.actor_enc.reset(dones)

    def update(self, obs):
        # if self.actor_critic.is_recurrent:
        #     generator = self.storage.reccurent_mini_batch_generator(
        #         self.num_mini_batches, self.num_learning_epochs
        #     )
        # else:
        #     generator = self.storage.mini_batch_generator(
        #         self.num_mini_batches, self.num_learning_epochs
        #     )
        
        # image_batch = self.storage.image_batch() #TODO if we need storage use this
        # depth_batch = self.storage.dm_batch()
        
        prop, image, depth = obs[:, :48], obs[:, 48:48+187], obs[:, 48+187:]

        self.optimizer.zero_grad()
        pred = self.actor_enc.encode(obs) #although we give it the whole obs, it throws out the depth map
        loss = torch.nn.functional.mse_loss(pred, depth)
        loss.backward()
        self.optimizer.step()
        
        self.storage.clear()

        return loss # , reconstruction_loss
