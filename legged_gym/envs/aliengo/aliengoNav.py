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

import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil

from legged_gym.envs.base.legged_robot_nav import LeggedRobotNav

from .mixed_terrains.aliengo_rough_config import AliengoRoughCfg


class AliengoNav(LeggedRobotNav):
    cfg: AliengoRoughCfg

    def make_handle_trans(self, cfg, angle, env_num, hfov=None):
        camera_props = gymapi.CameraProperties()
        if hfov is not None:
            camera_props.horizontal_fov = hfov
        # 1280 x 720
        width, height = cfg.env.camera_res
        camera_props.width = width
        camera_props.height = height
        camera_props.enable_tensors = True
        # print("envs[i]", self.envs[i])
        # print("len envs: ", len(self.envs))
        camera_handle = self.gym.create_camera_sensor(
            self.envs[env_num], camera_props
        )
        # print("cam handle: ", camera_handle)

        local_transform = gymapi.Transform()
        # local_transform.p = gymapi.Vec3(75.0, 75.0, 30.0)
        # local_transform.r = gymapi.Quat.from_euler_zyx(0, 3.14 / 2, 3.14)
        local_transform.p = gymapi.Vec3(0.35, 0.0, 0.0)
        local_transform.r = gymapi.Quat.from_euler_zyx(0.0, angle, 0.0)

        return camera_handle, local_transform

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, record=False):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.camera_handles = []

        if cfg.env.train_type == "lbc":
            for i in range(self.num_envs):

                cam1, trans1 = self.make_handle_trans(cfg, np.deg2rad(30), i)
                cam2, trans2 = self.make_handle_trans(cfg, np.deg2rad(-15), i, hfov=70)
                
                self.camera_handles.append(cam1)
                self.camera_handles.append(cam2)

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

                self.gym.attach_camera_to_body(
                    cam2,  # camera_handle,
                    self.envs[i],
                    body_handle,
                    trans2,
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
        return super()._compute_torques(actions)
