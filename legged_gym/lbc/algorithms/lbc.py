import numpy as np
import torch
import torch.optim as optim
from legged_gym.lbc.algorithms.lbc_storage import LbcStorage
from legged_gym.lbc.models.actor_encoder_lbc import Actor, VisionEncoder
from rsl_rl.modules.actor_critic import DmEncoder
from rsl_rl.modules.models.kin_policy import NavPolicy

NAV_INTERVAL = 10


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
        kin_nav_policy=None,
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

        self.optimizer = optim.Adam(self.vision_encoder.parameters(), lr=learning_rate)

        # LBC parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

        # Nav-specific
        if kin_nav_policy is not None:
            print(f"Loading the navigation policy: {kin_nav_policy}")
            self.kin_nav_policy = NavPolicy(kin_nav_policy, device="cuda")
            self.kin_nav_policy.reset()
        self.poll_count = 0
        self.lin_vel, self.ang_vel = torch.zeros(2)

    def init_storage(self):
        self.storage = (
            LbcStorage()
        )  # TODO write the LBC Storage with add im, add dm, clear functions.

    def test_mode(self):
        # TODO might need to set diff nns to train and test
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

        # Compute the actions and values
        prop, depth, image = obs[:, :48], obs[:, 48 : 48 + 187], obs[:, 48 + 187 :]
        student_enc_dm = self.vision_encoder(obs)

        actions = self.actor.act(
            prop, student_enc_dm.detach()
        ).detach()  # I think i need the inner detach bc we currrently don't want to update the student encoder based on the actions / reward

        with torch.no_grad():
            teacher_enc_dm = self.dm_encoder(depth)

        loss = torch.nn.functional.mse_loss(student_enc_dm, teacher_enc_dm)

        return actions, loss

    def process_env_step(self, rewards, dones, infos):
        self.actor.reset(dones)

    def update(self, loss, debug_it=None):
        """Performs the supervised loss update"""
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

    def act_inference(self, obs):
        if self.kin_nav_policy is not None:
            obs = self.update_cmds(obs)

        if isinstance(obs, torch.Tensor):
            prop = obs[:, :48]
        elif isinstance(obs, dict):
            prop = obs["proprioception"]
        else:
            raise RuntimeError(f"Obs type {type(obs)} invalid!")

        student_enc_dm = self.vision_encoder(obs)

        with torch.no_grad():
            actions = self.actor.act(prop, student_enc_dm)

        return actions

    def update_cmds(self, obs):
        if (self.poll_count - 1) % NAV_INTERVAL == 0:
            rho_theta = torch.tensor(obs["rho_theta"], dtype=torch.float32)
            level_depth = obs["level_image"].squeeze(0)
            level_depth = torch.clip(-level_depth, 0, 10.0) / 10.0
            kin_obs = {
                "depth": level_depth,
                "pointgoal_with_gps_compass": rho_theta,
            }

            actions = self.kin_nav_policy.act(kin_obs)
            self.lin_vel, self.ang_vel = actions
            self.lin_vel = (self.lin_vel + 1.0) / 2.0
            rt = rho_theta.cpu().numpy().tolist()
            rt[1] = np.rad2deg(rt[1])
            print("rho_theta:", rt)
            print(
                "kin_actions",
                self.lin_vel.item() * 0.25,
                np.rad2deg(self.ang_vel.item() * np.deg2rad(30)),
            )
        self.poll_count += 1

        obs["proprioception"][0, 9] *= self.lin_vel.item() * 0.25
        obs["proprioception"][0, 10] *= self.ang_vel.item() * np.deg2rad(30)
        obs["proprioception"][0, 11] = 0  # no hor vel

        return obs
