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

from legged_gym.envs import AliengoRoughCfg, AliengoRoughCfgPPO
import torch


class AliengoFlatCfg(AliengoRoughCfg):
    class viewer:
        pos = [5, 5, 3]  # [m]
        lookat = [3.0, 0, 1.0]  # [m]
    
    class env(AliengoRoughCfg.env):
        num_observations = 48
        num_envs=1024


    class terrain(AliengoRoughCfg.terrain):
        mesh_type = "plane"
        measure_heights = False

    class asset(AliengoRoughCfg.asset):
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(AliengoRoughCfg.rewards):
        max_contact_force = 350.0

        class scales(AliengoRoughCfg.rewards.scales):
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 2.0
            base_height = -1.0  # I added this reward to get the robot a high higher up
            # feet_contact_forces = -0.01
    class commands(AliengoRoughCfg.commands):
        heading_command = False
        resampling_time = 4.0

        class ranges(AliengoRoughCfg.commands.ranges):
            lin_vel_x = [0.7, 1.0]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [0.0, 0.0]  # min max [rad/s]
            heading = [0, 0]

    class domain_rand(AliengoRoughCfg.domain_rand):
        friction_range = [
            0.0,
            1.5,
        ]  # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
    
    class control(AliengoRoughCfg.control):
        action_scale = 0.25

class AliengoFlatCfgPPO(AliengoRoughCfgPPO):
    class policy(AliengoRoughCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(AliengoRoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(AliengoRoughCfgPPO.runner):
        num_test_envs = 1
        run_name = ""
        experiment_name = "flat_Aliengo"
        load_run = -1
        max_iterations = 300

        def baseline_policy(self, obs):
            # print(obs)
            return torch.zeros(12).view(1, 12)

        # eval_baseline = True
