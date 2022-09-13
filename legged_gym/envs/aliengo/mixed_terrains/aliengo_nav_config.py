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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

"""
changes from a1 to aliengo
- pd gains
- starting height
- target height?
- action scale
"""


class AliengoNavCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_actions = 12
        num_observations = 235
        num_proprio_obs = 48
        save_im = False
        camera_res = [320, 180]
        camera_type = "d"  # rgb
        num_privileged_obs = None  # 187
        train_type = "lbc"  # standard, priv, lbc
        episode_length_s = 125  # episode length in seconds

    class terrain(LeggedRobotCfg.terrain):
        # terrain_proportions = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        mesh_type = "trimesh"
        map_path = None  # gets overwritten in play.py

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.38]  # x,y,z [m]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.1,  # [rad]
            "RL_hip_joint": 0.1,  # [rad]
            "FR_hip_joint": -0.1,  # [rad]
            "RR_hip_joint": -0.1,  # [rad]
            "FL_thigh_joint": 0.8,  # [rad]
            "RL_thigh_joint": 1.0,  # [rad]
            "FR_thigh_joint": 0.8,  # [rad]
            "RR_thigh_joint": 1.0,  # [rad]
            "FL_calf_joint": -1.5,  # [rad]
            "RL_calf_joint": -1.5,  # [rad]
            "FR_calf_joint": -1.5,  # [rad]
            "RR_calf_joint": -1.5,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        # stiffness = {'joint': 20.}  # [N*m/rad]
        stiffness = {"joint": 40.0}  # [N*m/rad]
        # damping = {'joint': 0.5}     # [N*m*s/rad]
        damping = {"joint": 2.0}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo/urdf/aliengo.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = []
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5.0, 5.0]

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.5
        max_contact_force = 500.0
        only_positive_rewards = True

        class scales(LeggedRobotCfg.rewards.scales):
            feet_step = -1.0
            feet_stumble = -1.0

    class evals(LeggedRobotCfg.evals):
        feet_stumble = True
        feet_step = True
        crash_freq = True
        any_contacts = True

    class commands(LeggedRobotCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.7, 1.0]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class noise(LeggedRobotCfg.noise):
        add_noise = False


class AliengoNavCfgAlg(LeggedRobotCfgPPO):
    class obsSize(LeggedRobotCfgPPO.obsSize):
        encoder_hidden_dims = [128, 64, 32]
        cnn_out_size = 32
        num_dm_encoder_obs = 187

    class runner(LeggedRobotCfgPPO.runner):
        alg = "lbc"
        run_name = "debug"
        # run_name = ""
        experiment_name = "lbc_aliengo"
        load_run = -1
        max_iterations = 10000  # number of policy updates
        num_test_envs = 1

        resume = True
        resume_path = "weights/Sep11_23-48-28_debug_model_10000_16.232642258265987.pt"

        teacher_policy = "weights/Sep11_21-26-00_ObsEncDM_model_1150_19.086456518173218.pt"
        kin_nav_policy = "weights/VISUAL_LOCOMOTION_aliengo_kinematic_habitat_camera_up_2hz_57deg_camera_noise_sd_2_ckpt.88.pth"
        # kin_nav_policy = "weights/VISUAL_LOCOMOTION_aliengo_kinematic_habitat_camera_up_2hz_15deg_camera_noise_sd_2_ckpt.96.pth"
        # kin_nav_policy = "weights/VISUAL_LOCOMOTION_aliengo_kinematic_habitat_camera_up_2hz_57deg_camera_noise_sd_1_ckpt.2.pth"

    class lbc(LeggedRobotCfgPPO.lbc):
        batch_size = 10
