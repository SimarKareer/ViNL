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
import glob
import os
import os.path as osp
from itertools import permutations

import cv2
import imageio
import numpy as np
from numpy.random import choice

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import tqdm

OBS_DIST_THRESH = 0.5
EUCLID_THRESH = 5.0


def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = y_ - y
    x_pad = x_ - x
    return np.pad(
        a,
        ((y_pad // 2, y_pad // 2 + y_pad % 2), (x_pad // 2, x_pad // 2 + x_pad % 2)),
        mode="constant",
    )


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [
            np.sum(cfg.terrain_proportions[: i + 1])
            for i in range(len(cfg.terrain_proportions))
        ]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.terrain_start = None
        if cfg.map_path:
            im = np.array(imageio.v2.imread(cfg.map_path))
            im = im[:, :, 3]
            scaled_im = im.repeat(3, axis=0).repeat(3, axis=1)
            self.height_field_raw = to_shape(scaled_im, (900, 900))

        elif cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            if cfg.map_path:
                hscale, vscale = 0.4, 4
            else:
                hscale, vscale = 1, 1
            (
                self.vertices,
                self.triangles,
            ) = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale * hscale,
                self.cfg.vertical_scale * vscale,
                self.cfg.slope_treshold,
            )
            if cfg.map_path:
                self.set_start_goal()
                # Add small blocks on the ground
                self.add_blocks()

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        num_cols = (
            self.cfg.tot_cols if hasattr(self.cfg, "tot_cols") else self.cfg.num_cols
        )
        num_rows = (
            self.cfg.tot_rows if hasattr(self.cfg, "tot_rows") else self.cfg.num_rows
        )
        for j in range(num_cols):
            for i in range(num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop("type")
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 1.0 * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.0
            )
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.0
            )
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-0.05,
                max_height=0.05,
                step=0.005,
                downsampled_scale=0.2,
            )
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=0.31, step_height=step_height, platform_size=3.0
            )
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.0
            rectangle_max_size = 2.0
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.0,
            )
        elif choice < self.proportions[5]:
            num_rectangles = int(200 * difficulty)
            rectangle_min_size = 2
            rectangle_max_size = 5
            terrain_utils.discrete_obstacles_terrain_cells(
                terrain,
                float(os.environ["ISAAC_BLOCK_MIN_HEIGHT"]),
                float(os.environ["ISAAC_BLOCK_MAX_HEIGHT"]),
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.0,
            )
        elif choice < self.proportions[6]:
            terrain_utils.stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.0,
                platform_size=4.0,
            )
        elif choice < self.proportions[7]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.0)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.0)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2.0 - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2.0 + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 1) / terrain.horizontal_scale)
        env_origin_z = (
            np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        )
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def set_start_goal(self):
        episode_id = int(os.environ["ISAAC_EPISODE_ID"])
        scene = os.path.basename(self.cfg.map_path).split(".")[0]
        os.environ["ISAAC_MAP_NAME"] = scene
        if episode_id == -10:
            start, goal = self.generate_episode()
        else:
            if episode_id == -1:
                # Select a random episode
                episode_id = np.random.randint(10)
            matching_files = glob.glob(f"episodes/{scene}_{episode_id}_*png")
            assert (
                len(matching_files) <= 1
            ), f"Too many episode matches: {matching_files}"
            assert (
                matching_files
            ), f"Episode id {episode_id} for scene {scene} not found!"
            start_goal = [
                float(i)
                for i in osp.basename(matching_files[0])[: -len(".png")].split("_")[2:6]
            ]
            start = np.array(start_goal[:2])
            goal = np.array(start_goal[2:4])

        self.terrain_start = start.copy()
        self.terrain_goal = goal.copy()

        # Convert coordinates to be proper terrain coordinates ("global")
        start, goal = [
            np.array([-(89.9 * 250 / 900) + i[0], -(89.9 * 250 / 900) + i[1]])
            for i in [start, goal]
        ]

        os.environ["isaac_episode"] = "_".join([str(i) for i in [*start, *goal]])

    def generate_episode(self):
        x0, x1, y0, y1 = self.get_terrain_bounds()
        print("Map bounds: ", x0, x1, y0, y1)
        done = False
        start, goal = None, None
        while not done:
            start, goal = [
                np.array([np.random.uniform(x0, x1), np.random.uniform(y0, y1)])
                for _ in range(2)
            ]
            done = self.validate_start_goal(start, goal)

        # Get start and goal in image coordinates
        img = cv2.imread(self.cfg.map_path, cv2.IMREAD_UNCHANGED)
        u0, v0, w, h = cv2.boundingRect(img[..., 3])
        start_im, goal_im = [
            (
                int(map_range(x, x0, x1, v0, v0 + h)),
                int(map_range(y, y0, y1, u0, u0 + w)),
            )
            for y, x in [start, goal]
        ]

        img = cv2.cvtColor(img[..., -1], cv2.COLOR_GRAY2BGR)
        for (cx, cy), color in zip([start_im, goal_im], [(0, 255, 0), (0, 0, 255)]):
            cv2.circle(img, (cx, cy), 3, color, -1)
        os.makedirs("episodes", exist_ok=True)
        scene = os.path.basename(self.cfg.map_path).split(".")[0]
        cv2.imwrite(
            f"episodes/{scene}_"
            f"{'_'.join([f'{i:.2f}' for i in [*start, *goal]])}.png",
            img,
        )
        return start, goal

    def validate_start_goal(self, start, goal):
        # Check the Euclidean distance
        if np.linalg.norm(goal - start) < EUCLID_THRESH:
            return False

        # Check if there are any obstacles near the start and goal
        obs_xy_vertices = self.vertices[self.vertices[:, 2] > 0.8][:, :2]
        for pt in [start, goal]:
            for v in obs_xy_vertices:
                if np.linalg.norm(pt - v) < OBS_DIST_THRESH:
                    return False
        return True

    def get_terrain_bounds(self):
        x0 = np.amin([i[0] for i in self.vertices if i[2] > 0.1])
        x1 = np.amax([i[0] for i in self.vertices if i[2] > 0.1])
        y0 = np.amin([i[1] for i in self.vertices if i[2] > 0.1])
        y1 = np.amax([i[1] for i in self.vertices if i[2] > 0.1])
        return x0, x1, y0, y1

    def add_blocks(self):
        BLOCKS_PER_AREA = 1.0
        DIST_THRESH = 0.75
        SPAWN_OBS_THRESH = 1.5
        POTENTIAL_DIMS = [(0.15, 0.15), (0.15, 0.3), (0.3, 0.15)]
        min_block_height = float(os.environ["ISAAC_BLOCK_MIN_HEIGHT"])
        max_block_height = float(os.environ["ISAAC_BLOCK_MAX_HEIGHT"])

        x0, x1, y0, y1 = self.get_terrain_bounds()
        area = (x1 - x0) * (y1 - y0)
        num_blocks = int(area * BLOCKS_PER_AREA)
        # A block is an x, y, s1, s2, and h
        blocks = []
        np.random.seed(int(os.environ["ISAAC_SEED"]))
        print(f"Generating {num_blocks} obstacles..")
        for _ in tqdm.trange(num_blocks):
            success = False
            while not success:
                s1, s2 = POTENTIAL_DIMS[np.random.randint(3)]
                x = np.random.rand() * (x1 - x0) + x0
                y = np.random.rand() * (y1 - y0) + y0
                if (
                    np.linalg.norm(np.array([x, y]) - self.terrain_start)
                    < SPAWN_OBS_THRESH
                ):
                    continue
                if (
                    np.linalg.norm(np.array([x, y]) - self.terrain_goal)
                    < SPAWN_OBS_THRESH
                ):
                    continue
                block_height = np.random.uniform(min_block_height, max_block_height)
                new_block = (x, y, s1, s2, block_height)
                if blocks:
                    blocks_arr = np.array(blocks)[:, :2]
                    new_block_arr = np.array(new_block)[:2]
                    diff = blocks_arr - new_block_arr
                    if min(np.linalg.norm(diff, axis=1)) < DIST_THRESH:
                        continue
                blocks.append(new_block)
                success = True

        for block in blocks:
            self.add_block(*block)

    def add_block(self, x0, y0, s1, s2, h):
        # A rectangular prism has 8 vertices
        new_vertices = [
            (x0, y0, 0.0),
            (x0 + s1, y0, 0.0),
            (x0, y0 + s2, 0.0),
            (x0 + s1, y0 + s2, 0.0),
            (x0, y0, h),
            (x0 + s1, y0, h),
            (x0, y0 + s2, h),
            (x0 + s1, y0 + s2, h),
        ]
        # Spam every possible combination
        new_triangles = list(permutations(range(8), 3))
        self.triangles = np.concatenate(
            [
                self.triangles,
                np.array(new_triangles, dtype=np.uint32) + self.vertices.shape[0],
            ]
        )
        self.vertices = np.concatenate(
            [self.vertices, np.array(new_vertices, dtype=np.float32)]
        )


def gap_terrain(terrain, gap_size, platform_size=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[
        center_x - x2 : center_x + x2, center_y - y2 : center_y + y2
    ] = -1000
    terrain.height_field_raw[
        center_x - x1 : center_x + x1, center_y - y1 : center_y + y1
    ] = 0


def pit_terrain(terrain, depth, platform_size=1.0):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
