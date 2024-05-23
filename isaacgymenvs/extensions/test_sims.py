from isaacgym import gymapi
from isaacgym import gymutil

gym = gymapi.acquire_gym()

args = gymutil.parse_arguments()

sim_params = gymapi.SimParams()
sim = gym.create_sim(
    # args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params
    args.compute_device_id, args.graphics_device_id, gymapi.SIM_FLEX, sim_params
)

sim_params.up_axis = gymapi.UP_AXIS_Y
sim_params.gravity = gymapi.Vec3(0, -9.8, 0)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 1, 0)  # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

# load ball asset
asset_root = "../../assets"
asset_file = "urdf/ball.urdf"
asset = gym.load_asset(sim, asset_root, asset_file)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

envs = []
actor_handles = []

# create one env
lower = gymapi.Vec3(0, 0, 0)
upper = gymapi.Vec3(1, 1, 1)
env = gym.create_env(sim, lower, upper, 1)
envs.append(env)

# create 5 balls
for i in range(5):
  # actor transform
  pose = gymapi.Transform()
  pose.p = gymapi.Vec3(0, 5 + 2 * i, 0)

  actor_handle = gym.create_actor(env, asset, pose, f"ball{i}", 0, 0)
  actor_handles.append(actor_handle)

while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)


gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
