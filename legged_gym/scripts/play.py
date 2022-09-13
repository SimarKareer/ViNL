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

import isaacgym

import numpy as np
import torch

from legged_gym.envs import *
from legged_gym.utils import Logger, export_policy_as_jit, get_args, task_registry


def play(args):
    # EXTREEEEME H-H-H-H-H-HACK!
    if args.seed is None:
        args.seed = 1
    os.environ["ISAAC_SEED"] = str(args.seed)
    os.environ["ISAAC_EPISODE_ID"] = str(args.episode_id)
    os.environ["ISAAC_MAP"] = args.map

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.terrain.map_path = args.map
    # override some parameters for testing
    env_cfg.env.num_envs = min(train_cfg.runner.num_test_envs, 50)

    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg, record=True
    )
    env.reset()
    obs = env.get_observations()
    # print("obs shape: ", obs.shape)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 100  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # obs_actions_motor = []
    # actions_record = np.zeros((1000, 12))
    # torque_record = np.zeros((1000, 12))
    # obs_record = np.zeros((1000, 48))

    for i in range(10 * int(env.max_episode_length)):
        if train_cfg.runner.eval_baseline:
            actions = train_cfg.runner.baseline_policy(obs)
        else:
            actions = policy(obs)

        # actions_record[i] = actions.detach().cpu()[0]
        # obs_record[i] = obs.detach().cpu()[0]
        # torque_record[i] = env._compute_torques(actions).detach().cpu()[0]

        # # print(obs.shape)
        # if i == 999:
        #     pickle.dump(actions_record, open("actions.p", "wb"))
        #     pickle.dump(obs_record, open("obs.p", "wb"))
        #     pickle.dump(torque_record, open("torque.p", "wb"))
        #     break

        obs, _, rews, dones, infos = env.step(actions.detach())

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    # LEGGED_GYM_ROOT_DIR,
                    "/home/simar/Projects/isaacVL/localDev/legged_gym",
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                # filename = "/home/simar/Projects/isaacVL/localDev/legged_gym/logs/obs_aliengo/exported/frames/1.png"
                # print(filename)
                # print("env viewer", env.viewer)
                # print(env.camera_handles[0])
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                # env.gym.write_camera_image_to_file(env.sim, env.camera_handles[0], gymapi.IMAGE_COLOR, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item()
                    * env.cfg.control.action_scale,
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
