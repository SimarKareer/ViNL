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

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .mixed_terrains.aliengo_rough_config import AliengoRoughCfg


class Aliengo(LeggedRobot):
    cfg: AliengoRoughCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, record=False):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.camera_handles = []

        print("ALIENGO INIT")
        follow_cam, follow_trans = self.make_handle_trans((1920, 1080), 0, (1.0, -1.0, 0.0), (0.0, 0.0, 3*3.14/4))
        self.follow_cam = follow_cam
        body_handle = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], "base"
        )

        self.gym.attach_camera_to_body(
            follow_cam,  # camera_handle,
            self.envs[0],
            body_handle,
            follow_trans,
            gymapi.FOLLOW_POSITION,
        )


        if cfg.env.train_type == "lbc":
            # print("INITIALIZING 2 CAMERAS")
            for i in range(self.num_envs):
                
                res = cfg.env.camera_res
                cam1, trans1 = self.make_handle_trans(res, i, (0.35, 0.0, 0.0), (0.0, 3.14/6, 0))
                
                self.camera_handles.append(cam1)


                body_handle = self.gym.find_actor_rigid_body_handle(
                    self.envs[i], self.actor_handles[i], "base"
                )

                self.gym.attach_camera_to_body(
                    cam1,  # camera_handle,
                    self.envs[i],
                    body_handle,
                    trans1,
                    gymapi.FOLLOW_TRANSFORM,
                )

            # self.gym.set_camera_transform(camera_handle, self.envs[i], local_transform)
        # if record:
        #     camera_props = gymapi.CameraProperties()
        #     width, height = cfg.env.camera_res
        #     camera_props.width = 128
        #     camera_props.height = 128
        #     # camera_props.enable_tensors = True
        #     camera_handle = self.gym.create_camera_sensor(
        #         self.envs[0], camera_props
        #     )
        #     print("CAM HANDLE: ", camera_handle)
        #     self.camera_handles.append(camera_handle)

        #     local_transform = gymapi.Transform()
        #     local_transform.p = gymapi.Vec3(0.35, 0.0, 0.0)
        #     local_transform.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0)
        # # load actuator network
        # if self.cfg.control.use_actuator_network:
        #     actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        #     self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)

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
