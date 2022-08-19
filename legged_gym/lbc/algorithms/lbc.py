import torch
import torch.nn as nn
import torch.optim as optim

from legged_gym.lbc.models.actor_encoder_lbc import Actor, VisionEncoder
from legged_gym.lbc.algorithms.lbc_storage import LbcStorage
from rsl_rl.modules.actor_critic import DmEncoder

import itertools
# import pydot


class LBC:
    actor: Actor
    vision_encoder: VisionEncoder
    dm_encoder: DmEncoder

    def __init__(
        self,
        actor,
        vision_encoder,
        dm_encoder,
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
        self.actor = actor
        self.actor.to(self.device)
        
        self.vision_encoder = vision_encoder
        self.vision_encoder.to(self.device)
        
        self.dm_encoder = dm_encoder
        self.dm_encoder.to(self.device)
        
        # self.storage = self.init_storage()  # Currently bc using LSTM we don't need storage buffer
        self.optimizer = optim.Adam(
            self.vision_encoder.parameters(), lr=learning_rate
        )

        # LBC parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

    def init_storage(self):
        self.storage = (
            LbcStorage()
        )  # TODO write the LBC Storage with add im, add dm, clear functions.

    def test_mode(self):
        #TODO might need to set diff nns to train and test
        self.actor.eval()
        self.vision_encoder.eval()
        self.dm_encoder.eval()

    def train_mode(self):
        self.actor.eval()
        self.vision_encoder.train()
        self.dm_encoder.eval()

    def act_and_sl(self, obs):
        """
            Parameters:
                obs: (envs, prop+depth+image)
            Returns:
                actions:
                loss: the supervised loss for this set of obs
        """
        # if self.actor_critic.is_recurrent:
        #     self.transition.hidden_states = self.actor_critic.get_hidden_states()

        # Compute the actions and values
        prop, depth, image = obs[:, :48], obs[:, 48:48+187], obs[:, 48+187:]
        student_enc_dm = self.vision_encoder(obs)
        # student_enc_dm = torch.ones_like(student_enc_dm, device=self.device, requires_grad=True)

        actions = self.actor.act(prop, student_enc_dm.detach()).detach() #I think i need the inner detach bc we currrently don't want to update the student encoder based on the actions / reward

        with torch.no_grad():
            teacher_enc_dm = self.dm_encoder(depth)

        # actions = self.actor.act(prop, teacher_enc_dm.detach()).detach() # TODO: rm, this is just a baseline
        
        # print("Encoded shapes: ", student_enc_dm.shape, teacher_enc_dm.shape)
        loss = torch.nn.functional.mse_loss(student_enc_dm, teacher_enc_dm)


        # need to record obs and critic_obs before env.step()
        # self.storage.add_pair(image_slice, dm_slice)   #TODO if we ever actually need an image buffer we'll use this


        return actions, loss

    def process_env_step(self, rewards, dones, infos):
        self.actor.reset(dones)

    def update(self, loss, debug_it=None):
        """Performs the supervised loss update"""
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

        if debug_it is not None:
            dot = make_dot(loss, params=dict(itertools.chain(self.dm_encoder.named_parameters(), self.vision_encoder.named_parameters())))
            dot.render(format='png', outfile=f"graph{debug_it}.png")



        self.optimizer.zero_grad()

        # print("backward call!")
        loss.backward()
        self.optimizer.step()
        
        # self.storage.clear()

        # return loss # , reconstruction_loss
    
    def act_inference(self, obs):
        prop, depth, image = obs[:, :48], obs[:, 48:48+187], obs[:, 48+187:]
        student_enc_dm = self.vision_encoder(obs)
        # student_enc_dm = torch.ones_like(student_enc_dm, device=self.device, requires_grad=True)

        actions = self.actor.act(prop, student_enc_dm.detach()).detach() #I think i need the inner detach bc we currrently don't want to update the student encoder based on the actions / reward

        return actions