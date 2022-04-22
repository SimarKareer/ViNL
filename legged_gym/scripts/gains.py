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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(train_cfg.runner.num_test_envs, 50)

    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # records
    # actions_record = np.zeros((1000, 12))
    # torque_record = np.zeros((1000, 12))
    # obs_record = np.zeros((1000, 48))

    obs_pickle = pickle.load(open("./pickles/obs.p", "rb"))

    env_cfg.control.action_scale = 1

    print(obs_pickle)
    print(obs_pickle.shape)
    actual = np.zeros((1000, 12))
    pickle_actions = np.zeros((1000, 12))
    for i in range(300):#obs_pickle.shape[0]):
        actions = obs_pickle[i, 12:24].copy()
        pickle_actions[i] = actions

        # actions_record[i] = actions.detach().cpu()[0]
        # obs_record[i] = obs.detach().cpu()[0]
        # torque_record[i] = env._compute_torques(actions).detach().cpu()[0]

        # print(obs.shape)
        # if i == 999:
        #     pickle.dump(actions_record, open("actions.p", "wb"))
        #     pickle.dump(obs_record, open("obs.p", "wb"))
        #     pickle.dump(torque_record, open("torque.p", "wb"))
        #     break

        obs, _, rews, dones, infos = env.step(torch.tensor(actions).view(1, -1).float())
        # print("obs.shape", obs.shape)
        actual[i] = obs[0, 12:24].detach().cpu().numpy()

    for i in [0, 1, 2]:
        plt.plot(actual[:, i], label="actual")
        plt.plot(pickle_actions[:, i], label="pickle")
        plt.legend()
        plt.show()



if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
