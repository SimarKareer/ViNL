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

import cv2
import numpy as np
import torch

from isaacgym import gymapi, gymtorch
from legged_gym.envs.base.legged_robot_nav import LeggedRobotNav

from .aliengo import AlienGoCameraMixin
from .mixed_terrains.aliengo_rough_config import AliengoRoughCfg


class AliengoNav(AlienGoCameraMixin, LeggedRobotNav):
    cfg: AliengoRoughCfg

    def __init__(
        self, cfg, sim_params, physics_engine, sim_device, headless, record=False
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.camera_handles = []

        width, height = cfg.env.camera_res
        trans = (0.35, 0.0, 0.0)
        if cfg.env.train_type == "lbc":
            for env_idx in range(self.num_envs):
                cam1, trans1 = self.make_handle_trans(
                    width, height, env_idx, trans, (0.0, np.deg2rad(30), 0.0)
                )
                cam2, trans2 = self.make_handle_trans(
                    width, height, env_idx, trans, (0.0, np.deg2rad(-15), 0.0), hfov=70
                )

                self.camera_handles.append(cam1)
                self.camera_handles.append(cam2)

                body_handle = self.gym.find_actor_rigid_body_handle(
                    self.envs[env_idx], self.actor_handles[env_idx], "base"
                )

                for c, t in ([cam1, trans1], [cam2, trans2]):
                    self.gym.attach_camera_to_body(
                        c,  # camera_handle,
                        self.envs[env_idx],
                        body_handle,
                        t,
                        gymapi.FOLLOW_TRANSFORM,
                    )

        self.init_aux_cameras()
        self.floating_cam_moved = False

    def step(self, actions):
        ret = super().step(actions)

        if self.follow_cam is not None:
            self.gym.start_access_image_tensors(self.sim)
            image = gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim,
                    self.envs[0],
                    self.follow_cam,
                    gymapi.IMAGE_COLOR,
                )
            )
            self.gym.end_access_image_tensors(self.sim)
            img = cv2.cvtColor(image.cpu().numpy(), cv2.COLOR_RGB2BGR)
            cv2.imshow("Follow camera", img)
            cv2.waitKey(1)

        if not self.floating_cam_moved:
            x0, y0, x1, y1 = [
                float(i) for i in os.environ["isaac_bounds"].split("_")
            ]
            midpoint = (np.array([x0, y0]) + np.array([x1, y1])) / 2
            camera_target = gymapi.Vec3(*midpoint, 0)
            camera_position = camera_target + gymapi.Vec3(1, 1, 15)
            if not self.headless:
                self.gym.viewer_camera_look_at(
                    self.viewer, None, camera_position, camera_target
                )
            if self.floating_cam is not None:
                self.gym.set_camera_location(
                    self.floating_cam, self.envs[0], camera_position, camera_target
                )
            self.floating_cam_moved = True

        if self.floating_cam is not None:
            self.gym.start_access_image_tensors(self.sim)
            image = gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim,
                    self.envs[0],
                    self.floating_cam,
                    gymapi.IMAGE_COLOR,
                )
            )
            self.gym.end_access_image_tensors(self.sim)
            img = cv2.cvtColor(image.cpu().numpy(), cv2.COLOR_RGB2BGR)
            cv2.imshow("Floating camera", img)
            cv2.waitKey(1)

        return ret

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
