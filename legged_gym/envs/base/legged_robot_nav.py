# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
from legged_gym.envs.base.legged_robot import LeggedRobot
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from torchvision.utils import save_image


class LeggedRobotNav(LeggedRobot):
    def __init__(
        self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # print("in step")
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.extras["torque"] = self.torques  # gymtorch.unwrap_tensor(self.torques)
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        # print("calling post phys step")
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        self.compute_eval()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # print("in post phys calling reset idx")
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
    
    def save_im(self, im, path, height, width):
        trans_im = im.detach().clone()
        # print(trans_im)
        trans_im = -1 / trans_im
        trans_im = trans_im / torch.max(trans_im)
        # print("transformed im", trans_im)
        # print("image size: ", trans_im.shape)
        save_image(
            trans_im.view((height, width, 1)).permute(2, 0, 1).float(),
            path,
        )
        # if self.count == 500:
        #     exit()

    def compute_observations(self):
        """ Computes observations
        """
        # print("lin vel shape: ", (self.base_lin_vel * self.obs_scales.lin_vel).shape)
        # print("ang vel shape: ", (self.base_ang_vel * self.obs_scales.ang_vel).shape)
        # print("gravity shape: ", (self.projected_gravity).shape)
        # print("commands shape: ", (self.commands[:, :3] * self.commands_scale).shape)
        # print(
        #     "dof pos shape: ",
        #     ((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos).shape,
        # )
        # print("dof vel shape: ", (self.dof_vel * self.obs_scales.dof_vel).shape)
        # print("actions shape: ", (self.actions).shape)
        # exit()
        self.obs_buf = torch.cat(
            (
                torch.tensor([[-1]]).cuda(),
                torch.tensor([[-1]]).cuda()
                # @Naoki put rho theta here
            ),
            dim=-1,
        )
        self.count += 1

        if self.cfg.env.train_type == "lbc":

            width, height = self.cfg.env.camera_res
            image_buf = torch.zeros(self.cfg.env.num_envs * 2, height, width).to(self.device)

            self.gym.start_access_image_tensors(self.sim)
            for i in range(self.num_envs):
                if self.cfg.env.camera_type == "d":
                    im = self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        self.envs[i],
                        self.camera_handles[2 * i],
                        gymapi.IMAGE_DEPTH,
                    )
                    im = gymtorch.wrap_tensor(im)
                    # print(im)
                    image_buf[2 * i] = im
                    # print("im shape: ", im.shape)

                    im2 = self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        self.envs[i],
                        self.camera_handles[2 * i + 1],
                        gymapi.IMAGE_DEPTH,
                    )
                    im2 = gymtorch.wrap_tensor(im2)
                    # print(im)
                    image_buf[2 * i + 1] = im2
                    # print("im shape: ", im.shape)

                    if self.cfg.env.save_im:
                        path = f"images/dim/{i}_{self.count}_down.png"
                        self.save_im(im, path, height, width)
                        path = f"images/dim/{i}_{self.count}_up.png"
                        self.save_im(im2, path, height, width)
                        
                        if self.count == 50:
                            exit()
                else:
                    raise NotImplementedError("rgb not implemented for two cams, just mimic the one camera approach if you need this.")

            self.gym.end_access_image_tensors(self.sim)
            self.obs_buf = torch.cat((self.obs_buf, image_buf.view(self.cfg.env.num_envs, -1)), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec
