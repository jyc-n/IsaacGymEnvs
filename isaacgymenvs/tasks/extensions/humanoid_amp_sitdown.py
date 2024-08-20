# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE..

from enum import Enum
import numpy as np
import torch
import os

from gym import spaces

from isaacgym import gymapi
from isaacgym import gymtorch

from isaacgymenvs.tasks.amp.humanoid_amp_base import HumanoidAMPBase, dof_to_obs
from isaacgymenvs.tasks.amp.utils_amp import gym_util
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib

from isaacgymenvs.utils.torch_jit_utils import (
    quat_mul,
    to_torch,
    calc_heading_quat,
    calc_heading_quat_inv,
    quat_to_tan_norm,
    my_quat_rotate,
)


NUM_AMP_OBS_PER_STEP = (
    13 + 52 + 28 + 12
)  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]


class HumanoidAMPSitdown(HumanoidAMPBase):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    # same as base
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        # self.cfg = cfg # This exists in the base class

        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidAMPSitdown.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert self._num_amp_obs_steps >= 2

        self._tar_speed = cfg["env"]["tarSpeed"]
        self._tar_change_steps_min = cfg["env"]["tarChangeStepsMin"]
        self._tar_change_steps_max = cfg["env"]["tarChangeStepsMax"]
        self._tar_dist_max = cfg["env"]["tarDistMax"]

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(
            config=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        motion_file = cfg["env"].get("motion_file", "amp_humanoid_walk.npy")
        motion_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../assets/amp/motions/" + motion_file,
        )
        self._load_motion(motion_file_path)

        self.num_amp_obs = self._num_amp_obs_steps * NUM_AMP_OBS_PER_STEP

        self._amp_obs_space = spaces.Box(
            np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf
        )

        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP),
            device=self.device,
            dtype=torch.float,
        )
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        self._amp_obs_demo_buf = None

        # TODO: new, for target position
        self._tar_change_steps = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )
        self._prev_root_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )
        # target position (x, y, 0) only
        self._tar_pos = torch.zeros(
            [self.num_envs, 2], device=self.device, dtype=torch.float
        )

        # actor id offset
        offset = 1
        # if not self.headless:
        #     offset = self._build_marker_state_tensors(offset)
        offset = self._build_object_state_tensors(offset)

        return

    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 2
        return obs_size

    def _set_humanoid_col_filter(self):
        self._humanoid_actor_col_filter = 1
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        # Note: object will collide with humanoid if (object_filter XOR humanoid_filter) = 1
        # TODO: the handle are also ordered. should be the same order as the actor ids
        # if not self.headless:
        #     self._build_marker(env_id, env_ptr, col_filter=1)
        self._build_object(env_id, env_ptr, col_filter=1)

        return

    def _build_marker(self, env_id, env_ptr, col_filter):
        col_group = env_id
        segmentation_id = 0
        default_pose = gymapi.Transform()

        marker_handle = self.gym.create_actor(
            env_ptr,
            self._marker_asset,
            default_pose,
            "marker",
            col_group,
            col_filter,
            segmentation_id,
        )

        self.gym.set_rigid_body_color(
            env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0)
        )
        self._marker_handles.append(marker_handle)

        return

    def _build_object(self, env_id, env_ptr, col_filter):
        col_group = env_id
        segmentation_id = 0

        # scale = 1
        default_pose = gymapi.Transform()
        # default_pose.p = gymapi.Vec3(0, 0, 0.05 * scale / 2.0)

        object_handle = self.gym.create_actor(
            env_ptr,
            self._object_asset,
            default_pose,
            "object",
            col_group,
            col_filter,
            segmentation_id,
        )

        # self.gym.set_actor_scale(env_ptr, object_handle, scale)
        self.gym.set_rigid_body_color(
            env_ptr,
            object_handle,
            0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(0.357, 0.675, 0.769),
        )
        self._object_handles.append(object_handle)

        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        if not self.headless:
            self._marker_handles = []
            self._load_marker_asset()

        self._object_handles = []
        self._load_object_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_marker_asset(self):
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../assets/mjcf/",
        )
        asset_file = "location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        return

    def _load_object_asset(self):
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../assets/urdf/",
        )
        asset_file = "cube.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        # self._object_asset = self.gym.load_asset(
        #     self.sim, asset_root, asset_file, asset_options
        # )
        self._object_asset = self.gym.create_box(self.sim, 1.0, 1.0, 1.0, asset_options)

        return

    def _build_marker_state_tensors(self, offset):
        self._marker_idx = offset
        num_actors = self._root_states.shape[0] // self.num_envs
        
        self._marker_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., self._marker_idx, :]
        self._marker_pos = self._marker_states[..., :3]

        self._marker_actor_ids = self._humanoid_actor_ids + offset

        # assert self._marker_pos.storage().data_ptr() == self._marker_states.storage().data_ptr()
        # assert self._marker_pos.storage().data_ptr() == self._root_states.storage().data_ptr()

        return offset + 1

    def _build_object_state_tensors(self, offset):
        self._object_idx = offset
        num_actors = self._root_states.shape[0] // self.num_envs
        
        self._object_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., self._object_idx, :]
        self._object_pos = self._object_states[..., :3]

        self._object_actor_ids = self._humanoid_actor_ids + offset

        # assert self._object_pos.storage().data_ptr() == self._object_states.storage().data_ptr()
        # assert self._object_pos.storage().data_ptr() == self._root_states.storage().data_ptr()

        return offset + 1

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)

        # self._update_task()
        return

    # task-specific update
    def _update_task(self):
        reset_task_mask = self.progress_buf >= self._tar_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            print("do reset!!!")
            self._reset_task(rest_env_ids)

        # self._update_object()
        # self._update_marker()
        return

    def _reset_task(self, env_ids):
        print("reset task")
        n = len(env_ids)

        # randomly generate a new target location near the humanoid
        char_root_pos = self._humanoid_root_states[env_ids, 0:2]
        rand_pos = self._tar_dist_max * (
            2.0 * torch.rand([n, 2], device=self.device) - 1.0
        )
        self._tar_pos[env_ids] = char_root_pos + rand_pos
        # self._object_states[env_ids, 0:2] = self._tar_pos[env_ids]

        change_steps = torch.randint(
            low=self._tar_change_steps_min,
            high=self._tar_change_steps_max,
            size=(n,),
            device=self.device,
            dtype=torch.int64,
        )
        self._tar_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps

        self._update_object(env_ids)
        # self._update_marker(env_ids)

        return

    # same as base
    def post_physics_step(self):
        super().post_physics_step()

        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        return

    def render(self):
        super().render()

        # task specific rendering
        # if self.viewer:
        #     self._draw_task()
        # self._update_object()
        # self._update_marker()

        return

    # Add lines connecting humanoid root to target
    def _update_debug_viz(self):
        # print("-------------------")
        # print("before change")
        # tmp_root = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        # print(tmp_root)
        # print(self._root_states)

        # self._update_object()
        # self._update_marker()

        # print("after change")
        # print(tmp_root)
        # print(self._root_states)

        color = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self._object_pos
        # ends = self._marker_pos

        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(
                self.viewer, env_ptr, curr_verts.shape[0], curr_verts, color
            )

        return

    def _update_object(self, env_ids):
        self._object_pos[..., 0:2] = self._tar_pos
        self._object_pos[..., 2] = 0.0

        print("object")
        print(self._object_pos)

        object_actor_ids_int32 = self._object_actor_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(object_actor_ids_int32),
            len(object_actor_ids_int32),
        )
        return

    def _update_marker(self, env_ids):
        self._marker_pos[..., 0:2] = self._tar_pos
        self._marker_pos[..., 2] = 0.5

        print("marker")
        print(self._marker_pos)

        marker_actor_ids_int32 = self._marker_actor_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(marker_actor_ids_int32),
            len(marker_actor_ids_int32),
        )
        return

    # same as base
    def get_num_amp_obs(self):
        return self.num_amp_obs

    # same as base
    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    # same as base
    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)

    # same as base
    def fetch_amp_obs_demo(self, num_samples):
        dt = self.dt
        motion_ids = self._motion_lib.sample_motions(num_samples)

        if self._amp_obs_demo_buf is None:
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert self._amp_obs_demo_buf.shape[0] == num_samples

        motion_times0 = self._motion_lib.sample_time(motion_ids)
        motion_ids = np.tile(
            np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps]
        )
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -dt * np.arange(0, self._num_amp_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
            self._motion_lib.get_motion_state(motion_ids, motion_times)
        )
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(
            root_states, dof_pos, dof_vel, key_pos, self._local_root_obs
        )
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)

        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())
        return amp_obs_demo_flat

    # same as base
    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros(
            (num_samples, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP),
            device=self.device,
            dtype=torch.float,
        )
        return

    # same as base
    def _load_motion(self, motion_file):
        self._motion_lib = MotionLib(
            motion_file=motion_file,
            num_dofs=self.num_dof,
            key_body_ids=self._key_body_ids.cpu().numpy(),
            device=self.device,
        )
        return

    # same as base
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._init_amp_obs(env_ids)
        self._reset_task(env_ids)
        return

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids[env_ids].to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    # same as base
    def _reset_actors(self, env_ids):
        print("actors reset")
        if self._state_init == HumanoidAMPSitdown.StateInit.Default:
            self._reset_default(env_ids)
        elif (
            self._state_init == HumanoidAMPSitdown.StateInit.Start
            or self._state_init == HumanoidAMPSitdown.StateInit.Random
        ):
            self._reset_ref_state_init(env_ids)
        elif self._state_init == HumanoidAMPSitdown.StateInit.Hybrid:
            self._reset_hybrid_state_init(env_ids)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(
                str(self._state_init)
            )

        # self.progress_buf[env_ids] = 0
        # self.reset_buf[env_ids] = 0
        # self._terminate_buf[env_ids] = 0

        return

    # same as base
    def _reset_default(self, env_ids):
        print("reset to default")
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._initial_humanoid_root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self._reset_default_env_ids = env_ids
        return

    # same as base
    def _reset_ref_state_init(self, env_ids):
        print("reset to ref state")
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)

        if (
            self._state_init == HumanoidAMPSitdown.StateInit.Random
            or self._state_init == HumanoidAMPSitdown.StateInit.Hybrid
        ):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif self._state_init == HumanoidAMPSitdown.StateInit.Start:
            motion_times = np.zeros(num_envs)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(
                str(self._state_init)
            )

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
            self._motion_lib.get_motion_state(motion_ids, motion_times)
        )

        self._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
        )

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    # same as base
    def _reset_hybrid_state_init(self, env_ids):
        print("reset hybrid")
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(
            np.array([self._hybrid_init_prob] * num_envs), device=self.device
        )
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if len(ref_reset_ids) > 0:
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if len(default_reset_ids) > 0:
            self._reset_default(default_reset_ids)

        return

    # same as base
    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if len(self._reset_default_env_ids) > 0:
            self._init_amp_obs_default(self._reset_default_env_ids)

        if len(self._reset_ref_env_ids) > 0:
            self._init_amp_obs_ref(
                self._reset_ref_env_ids,
                self._reset_ref_motion_ids,
                self._reset_ref_motion_times,
            )
        return

    # same as base
    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    # same as base
    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = np.tile(
            np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps - 1]
        )
        motion_times = np.expand_dims(motion_times, axis=-1)
        time_steps = -dt * (np.arange(0, self._num_amp_obs_steps - 1) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
            self._motion_lib.get_motion_state(motion_ids, motion_times)
        )
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(
            root_states, dof_pos, dof_vel, key_pos, self._local_root_obs
        )
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(
            self._hist_amp_obs_buf[env_ids].shape
        )
        return

    # same as base
    def _set_env_state(
        self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel
    ):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel

        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        env_ids_int32 = self._humanoid_actor_ids[env_ids].to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        return

    # same as base
    def _update_hist_amp_obs(self, env_ids=None):
        if env_ids is None:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return

    # same as base
    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if env_ids is None:
            # TODO: why this is duplicated?
            self._curr_amp_obs_buf[:] = build_amp_observations(
                self._humanoid_root_states,
                self._dof_pos,
                self._dof_vel,
                key_body_pos,
                self._local_root_obs,
            )
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(
                self._humanoid_root_states[env_ids],
                self._dof_pos[env_ids],
                self._dof_vel[env_ids],
                key_body_pos[env_ids],
                self._local_root_obs,
            )
        return

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            tar_pos = self._tar_pos
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_pos = self._tar_pos[env_ids]

        obs = compute_location_observations(root_states, tar_pos)
        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        self.rew_buf[:] = compute_location_reward(
            root_pos,
            self._prev_root_pos,
            root_rot,
            self._tar_pos,
            self._tar_speed,
            self.dt,
        )
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_location_observations(root_states, tar_pos):
    # type: (Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos3d = torch.cat([tar_pos, torch.zeros_like(tar_pos[..., 0:1])], dim=-1)
    heading_rot = calc_heading_quat_inv(root_rot)

    local_tar_pos = my_quat_rotate(heading_rot, tar_pos3d - root_pos)
    local_tar_pos = local_tar_pos[..., 0:2]

    obs = local_tar_pos
    return obs


# same as base
@torch.jit.script
def build_amp_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )

    dof_obs = dof_to_obs(dof_pos)

    obs = torch.cat(
        (
            root_h,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_location_reward(root_pos, prev_root_pos, root_rot, tar_pos, tar_speed, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    dist_threshold = 0.5

    pos_err_scale = 0.5
    vel_err_scale = 4.0

    pos_reward_w = 0.5
    vel_reward_w = 0.4
    face_reward_w = 0.1

    pos_diff = tar_pos - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    tar_dir = tar_pos - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    heading_rot = calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = my_quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    dist_mask = pos_err < dist_threshold
    facing_reward[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    reward = (
        pos_reward_w * pos_reward
        + vel_reward_w * vel_reward
        + face_reward_w * facing_reward
    )

    return reward
