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
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import quaternion
import torch
from isaacgym.torch_utils import (
    get_euler_xyz,
    quat_apply,
    quat_rotate_inverse,
    torch_rand_float,
)
from torchvision.utils import save_image

from isaacgym import gymapi, gymtorch, gymutil
from legged_gym.envs.aliengo.mixed_terrains.aliengo_nav_config import AliengoNavCfg
from legged_gym.envs.base.legged_robot import LeggedRobot

SHOW = False
PRINT_RT = False
SUCCESS_RADIUS = 0.3235


def wrap_heading(heading):
    return (heading + np.pi) % (2 * np.pi) - np.pi


def quat_to_yaw(quat):
    original_euler = quaternion.as_euler_angles(quat)
    euler_angles = np.array(
        [
            (np.random.rand() - 0.5) * np.pi / 9.0 + original_euler[0],
            (np.random.rand() - 0.5) * np.pi / 9.0 + original_euler[1],
            (np.random.rand() - 0.5) * np.pi / 9.0 + original_euler[2],
        ]
    )
    return euler_angles[2]


class LeggedRobotNav(LeggedRobot):
    def __init__(
        self, cfg: AliengoNavCfg, sim_params, physics_engine, sim_device, headless
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        episode = [float(i) for i in os.environ["isaac_episode"].split("_")]
        self.start_pos = np.array(episode[:2])
        self.goal_xy = np.array(episode[2:])
        default_pose = gymapi.Transform()
        default_pose.p.x, default_pose.p.y = 0.0, 0.0
        default_pose.p.z = 0.0
        start_sphere_geom = gymutil.WireframeSphereGeometry(
            0.15, 20, 20, default_pose, color=(0, 1, 0)
        )
        goal_sphere_geom = gymutil.WireframeSphereGeometry(
            0.15, 20, 20, default_pose, color=(1, 0, 0)
        )
        self.gym.set_light_parameters(
            self.sim,
            0,
            gymapi.Vec3(0, 0, 0),
            gymapi.Vec3(0, 0, 0),
            gymapi.Vec3(0, 0, 0),
        )
        self.gym.set_light_parameters(
            self.sim,
            1,
            gymapi.Vec3(1, 1, 1),
            gymapi.Vec3(1, 1, 1),
            gymapi.Vec3(0, 0, 1),
        )
        default_pose = gymapi.Transform()
        for pose, geom in zip(
            [self.start_pos, self.goal_xy],
            [start_sphere_geom, goal_sphere_geom],
        ):
            default_pose.p.x, default_pose.p.y = pose
            gymutil.draw_lines(
                geom,
                self.gym,
                self.viewer,
                self.envs[0],
                default_pose,
            )
        self.success = False
        self.eval_metrics = defaultdict(float)
        self.prev_xy = self.start_pos
        self.num_steps = 0
        self.feet_collided = False
        self.use_dm = cfg.env.use_dm
        if self.use_dm:
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "!!!!!!!!!!!PRIVILEGED DEPTH IN USE!!!!!!!!!!!\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

    def reset_idx(self, env_ids):
        if torch.numel(env_ids) == 0:
            return
        if self.num_steps > 1:
            self.eval_metrics["feet_collisions_per_meter"] = (
                self.eval_metrics["feet_collisions"]
                / self.eval_metrics["dist_traveled"]
            )
            self.eval_metrics["num_steps"] = self.num_steps
            if "success" not in self.eval_metrics:
                self.eval_metrics["success"] = 0.0

            str_data = ""
            for k, v in self.eval_metrics.items():
                str_data += f"{k}: {v:.3f}\n"
            print("Nav episode final stats:\n", str_data)

            if os.environ["ISAAC_WRITE"] == "True":
                eval_dir = os.environ["ISAAC_EVAL_DIR"]
                os.makedirs(eval_dir, exist_ok=True)
                map_name = os.environ["ISAAC_MAP_NAME"]
                episode_id = os.environ["ISAAC_EPISODE_ID"]
                seed = os.environ["ISAAC_SEED"]
                filename = (
                    f"{eval_dir}/{map_name}_{episode_id}_{seed}"
                    f"_{os.environ['ISAAC_NUM_COMPLETED_EPS']}.json"
                )
                json_dict = {**self.eval_metrics}
                json_dict["map_name"] = map_name
                json_dict["episode_id"] = episode_id
                json_dict["seed"] = seed
                json_dict["attempt"] = os.environ["ISAAC_NUM_COMPLETED_EPS"]
                with open(filename, "w") as f:
                    json.dump(json_dict, f)

            os.environ[
                "ISAAC_NUM_COMPLETED_EPS"
            ] = f"{int(os.environ['ISAAC_NUM_COMPLETED_EPS']) + 1}"

        self.num_steps = 0
        super().reset_idx(env_ids)
        self.success = False
        self.eval_metrics = defaultdict(float)
        self.prev_xy = self.start_pos
        if int(os.environ["ISAAC_NUM_COMPLETED_EPS"]) >= 5:
            quit()

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[0, :2] = torch.tensor(
            self.start_pos, device="cuda", dtype=torch.float32
        )
        # base velocities; start at 0-ish for nav
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.000001, 0.00001, (len(env_ids), 6), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    @property
    def curr_xy(self):
        return self.root_states[0, :2].cpu().numpy()

    def get_rho_theta(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        rho = np.linalg.norm(self.goal_xy - self.curr_xy)
        dx, dy = self.goal_xy - self.curr_xy
        theta = wrap_heading(np.arctan2(dy, dx) - yaw)
        return torch.tensor([rho, theta], device="cuda", dtype=torch.float32)

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.commands[0, :3] = torch.tensor([1.0, 0.0, 0.0], device="cuda")
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.extras["torque"] = self.torques
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        # Rewards are calculated here in the post_physics_step.
        self.feet_collided = False
        self.post_physics_step()
        if self.feet_collided:
            self.eval_metrics["feet_collisions"] += 1

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        for k, v in self.obs_buf.items():
            if not isinstance(v, bool):
                self.obs_buf[k] = torch.clip(v, -clip_obs, clip_obs)

        if self.obs_buf["rho_theta"][0] <= SUCCESS_RADIUS:
            self.success = True

        self.eval_metrics["dist_traveled"] += np.linalg.norm(
            self.curr_xy - self.prev_xy
        )
        self.prev_xy = self.curr_xy

        self.num_steps += 1

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def check_termination(self):
        super().check_termination()
        if self.success:
            self.reset_buf[0] = True
            self.eval_metrics["success"] = 1.0
            self.success = False
        # Must kill the robot if its roll or pitch is too high
        q = self.root_states[:, 2:6]
        # NOTE: Not actually quit sure if roll and pitch be flipped
        _, roll, pitch = get_euler_xyz(q)
        roll, pitch = wrap_heading(roll), wrap_heading(pitch)
        if (max(abs(roll), abs(pitch))) > np.deg2rad(120):
            self.reset_buf[0] = True

    def post_physics_step(self):
        """check terminations, compute observations and rewards
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
        self.reset_idx(env_ids)
        # in some cases a simulation step might be required
        # to refresh some obs (for example body positions)
        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def save_im(self, im, path, height, width):
        trans_im = -1 / im
        trans_im = trans_im / torch.max(trans_im)
        save_image(
            trans_im.view((height, width, 1)).permute(2, 0, 1).float(),
            path,
        )

    def compute_observations(self):
        self.count += 1
        # commands_scale seems to be tensor([2.0000, 2.0000, 0.2500])
        dummy_cmd = (
            torch.ones_like(self.commands[:, :3], device="cuda") * self.commands_scale
        )
        prop = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                dummy_cmd,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        rho_theta = self.get_rho_theta()
        level_image, tilted_image = self.get_images()
        self.obs_buf = {
            "proprioception": prop,
            "level_image": level_image,
            "rho_theta": rho_theta,
            "reset": self.num_steps == 0,
        }

        if self.use_dm:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self._get_heights(),
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            self.obs_buf["depth_map"] = heights
        else:
            self.obs_buf["tilted_image"] = tilted_image

        level_image, tilted_image = self.get_images()

        if PRINT_RT:
            rt = rho_theta.cpu().numpy().tolist()
            rt[1] = np.rad2deg(rt[1])
            print("xy:", self.curr_xy, "\trho_theta:", rt)
        if SHOW:
            imgs = [
                np.uint8(np.clip(-i[0, :, :, 0].cpu().numpy(), 0, 10) / 10 * 255)
                for i in (level_image, tilted_image)
            ]
            cv2.imshow("", np.hstack(imgs))
            cv2.waitKey(1)

    def get_images(self):
        i = 0
        assert self.cfg.env.camera_type == "d" and self.num_envs == 1
        self.gym.start_access_image_tensors(self.sim)
        images = [
            gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim,
                    self.envs[i],
                    self.camera_handles[2 * i + cam_id],
                    gymapi.IMAGE_DEPTH,
                )
            )
            .unsqueeze(0)
            .unsqueeze(-1)
            for cam_id in range(2)
        ]
        self.gym.end_access_image_tensors(self.sim)
        tilted_image, level_image = images

        return level_image, tilted_image

    def _reward_feet_stumble(self):
        rew = super()._reward_feet_stumble()
        self.feet_collided = rew[0].item() or self.feet_collided
        return rew

    def _reward_feet_step(self):
        rew = super()._reward_feet_step()
        self.feet_collided = rew[0].item() or self.feet_collided
        return rew
