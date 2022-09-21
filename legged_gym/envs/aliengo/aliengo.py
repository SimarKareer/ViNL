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

import os
from time import time

# from torch.tensor import Tensor
from typing import Dict, Tuple

import numpy as np
import torch
from isaacgym.torch_utils import *

from isaacgym import gymapi, gymtorch, gymutil
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import LeggedRobot

from .mixed_terrains.aliengo_rough_config import AliengoRoughCfg


class Aliengo(LeggedRobot):
    cfg: AliengoRoughCfg

    def __init__(
        self, cfg, sim_params, physics_engine, sim_device, headless, record=False
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.camera_handles = []

        print("ALIENGO INIT")

        if cfg.env.train_type == "lbc":
            print("INITIALIZING CAMERAS")
            for i in range(self.num_envs):
                # TODO Add camera sensors here?
                camera_props = gymapi.CameraProperties()
                # print("FOV: ", camera_props.horizontal_fov)
                # camera_props.horizontal_fov = 75.0
                # 1280 x 720
                width, height = cfg.env.camera_res
                camera_props.width = width
                camera_props.height = height
                camera_props.enable_tensors = True
                # print("envs[i]", self.envs[i])
                # print("len envs: ", len(self.envs))
                camera_handle = self.gym.create_camera_sensor(
                    self.envs[i], camera_props
                )
                # print("cam handle: ", camera_handle)
                self.camera_handles.append(camera_handle)

                local_transform = gymapi.Transform()
                # local_transform.p = gymapi.Vec3(75.0, 75.0, 30.0)
                # local_transform.r = gymapi.Quat.from_euler_zyx(0, 3.14 / 2, 3.14)
                local_transform.p = gymapi.Vec3(0.35, 0.0, 0.0)
                local_transform.r = gymapi.Quat.from_euler_zyx(0.0, 3.14 / 6, 0.0)

                body_handle = self.gym.find_actor_rigid_body_handle(
                    self.envs[i], self.actor_handles[i], "base"
                )

                self.gym.attach_camera_to_body(
                    camera_handle,  # camera_handle,
                    self.envs[i],
                    body_handle,
                    local_transform,
                    gymapi.FOLLOW_TRANSFORM,
                )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.0
        self.sea_cell_state_per_env[:, env_ids] = 0.0

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(
            self.num_envs * self.num_actions,
            1,
            2,
            device=self.device,
            requires_grad=False,
        )
        self.sea_hidden_state = torch.zeros(
            2,
            self.num_envs * self.num_actions,
            8,
            device=self.device,
            requires_grad=False,
        )
        self.sea_cell_state = torch.zeros(
            2,
            self.num_envs * self.num_actions,
            8,
            device=self.device,
            requires_grad=False,
        )
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(
            2, self.num_envs, self.num_actions, 8
        )
        self.sea_cell_state_per_env = self.sea_cell_state.view(
            2, self.num_envs, self.num_actions, 8
        )

    def _compute_torques(self, actions):
        # Choose between pd controller and actuator network
        # if self.cfg.control.use_actuator_network:
        #     with torch.inference_mode():
        #         self.sea_input[:, 0, 0] = (actions * self.cfg.control.action_scale + self.default_dof_pos - self.dof_pos).flatten()
        #         self.sea_input[:, 0, 1] = self.dof_vel.flatten()
        #         torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
        #     return torques
        # else:
        #     # pd controller
        return super()._compute_torques(actions)
