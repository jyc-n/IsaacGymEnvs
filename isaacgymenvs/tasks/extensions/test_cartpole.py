import os
import torch
import numpy as np
from isaacgym import gymtorch, gymapi, gymutil

from ..base.vec_task import VecTask


class TestCartpole(VecTask):
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
        self.cfg = cfg

        # from .yaml file
        self.reset_dist = self.cfg["env"]["resetDist"]
        self.max_push_effort = self.cfg["env"]["maxEffort"]

        self.max_episode_length = 500

        # 4 observations: cart position, cart velocity, pole angle, pole angular velocity
        # 1 action: direction of the fixed force
        # https://gymnasium.farama.org/environments/classic_control/cart_pole/
        self.cfg["env"]["numObservations"] = 4
        self.cfg["env"]["numActions"] = 1

        super().__init__(
            config=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # global tensors for computing observations
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # dof_state is (1024 x 2) -> view as (512 x 2 x 2)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

    # ----
    # Mandatory functions
    # ----
    def create_sim(self):
        # implement sim set up and environment creation here
        #    - set up-axis
        self.up_axis = self.cfg["sim"]["up_axis"]
        #    - call super().create_sim with device args (see docstring)
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        #    - create ground plane
        self._create_ground_plane()
        #    - set up environments
        self._create_envs(
            self.num_envs,
            self.cfg["env"]["envSpacing"],
            int(np.sqrt(self.num_envs)),
        )

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        pass

    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self._compute_observations()
        self._compute_reward()

    # reset if in a bad state
    def reset_idx(self, env_ids):
        positions = 0.2 * (
            torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5
        )
        velocities = 0.5 * (
            torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5
        )

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    # ----
    # Helper functions
    # ----

    # initialization helpers
    def _create_ground_plane(self):
        # configure the ground plane
        plane_params = gymapi.PlaneParams()
        # z-up plane plane
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)

        # create the ground plane
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define environment size
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        # load asset based on config
        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.cfg["env"]["asset"]["assetRoot"],
            )
            asset_file = self.cfg["env"]["asset"]["assetFileName"]
        else:
            raise Exception("Need to specify asset in .yaml file")

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        # this is used because the cart slides along a fixed rail
        asset_options.fix_base_link = True
        cartpole_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(cartpole_asset)

        # default initial transform
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 2.0)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # z-up, identity quaternion

        self.cartpole_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create single env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            cartpole_handle = self.gym.create_actor(
                env_ptr, cartpole_asset, pose, "cartpole", i, 1, 0
            )

            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
            # drive mode enables effort control
            dof_props["driveMode"][0] = (
                gymapi.DOF_MODE_EFFORT
            )  # The DOF will respond to effort (force or torque) commands
            dof_props["driveMode"][1] = gymapi.DOF_MODE_NONE
            dof_props["stiffness"][:] = 0.0
            dof_props["damping"][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)

            self.envs.append(env_ptr)
            self.cartpole_handles.append(cartpole_handle)

            print(f"create {i}")

    # compute reward
    def _compute_reward(self):
        # retrieve observations
        cart_pos = self.obs_buf[:, 0]
        cart_vel = self.obs_buf[:, 1]
        pole_angle = self.obs_buf[:, 2]
        pole_vel = self.obs_buf[:, 3]

        self.rew_buf[:], self.reset_buf[:] = compute_cartpole_reward(
            pole_angle,
            pole_vel,
            cart_pos,
            cart_vel,
            self.reset_dist,
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length,
        )

    # update observation
    def _compute_observations(self):
        # this will retrieve from all envs
        env_ids = np.arange(self.num_envs)

        # update dof_state_tensor
        self.gym.refresh_dof_state_tensor(self.sim)

        # obs_buf is a (num_envs x num_obs) tensor
        #       cart_pos, cart_vel, pole_angle, pole_vel
        # squeeze doesn't have effect here, no extra dimensions
        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()

        return self.obs_buf


# need to jit compile so that the reward is computed on GPU
@torch.jit.script
def compute_cartpole_reward(
    pole_angle,
    pole_vel,
    cart_pos,
    cart_vel,
    reset_dist,
    reset_buf,
    progress_buf,
    max_episode_length,
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward: (num_envs)
    # combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = (
        1.0
        - pole_angle * pole_angle
        - 0.01 * torch.abs(cart_vel)
        - 0.005 * torch.abs(pole_vel)
    )

    # if too far or fall over, reset and penalize
    reward = torch.where(
        torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward
    )
    reward = torch.where(
        torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward
    )

    reset = torch.where(
        torch.abs(cart_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf
    )
    reset = torch.where(
        torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reset_buf), reset
    )

    # reset episode if reaches max episode length
    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset
    )

    return reward, reset
