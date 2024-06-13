from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

gym = gymapi.acquire_gym()

args = gymutil.parse_arguments()

# Tensor API only supports PhysX
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Y
sim_params.gravity = gymapi.Vec3(0, -9.8, 0)
# Use GPU pipeline
sim_params.use_gpu_pipeline = True
sim_params.physx.use_gpu = True

sim = gym.create_sim(
    args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params
)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 1, 0)  # y-up
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
camera_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, camera_props)

envs = []
actor_handles = []

# create one env
spacing = 5
lower = gymapi.Vec3(-spacing, 0, spacing)
upper = gymapi.Vec3(-spacing, spacing, spacing)

# create multiple envs
for e in range(3):
   
  env = gym.create_env(sim, lower, upper, 2)
  envs.append(env)

  # create 5 balls
  ref_pos = gymapi.Vec3(spacing * (e + 1), 0, 0)
  for i in range(5):
    # actor transform
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.1 * i, 2 + i, 0) + ref_pos

    actor_handle = gym.create_actor(env, asset, pose, f"env{e}_ball{i}", e, 0)
    actor_handles.append(actor_handle)

    color = gymapi.Vec3(e * 0.2, 0, 0)
    gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

# Initialize the internal data structures used by the tensor API
gym.prepare_sim(sim)

# Get root tensor descriptor
_root_tensor = gym.acquire_actor_root_state_tensor(sim)
# Slice root tensor for more convenient access
root_tensor = gymtorch.wrap_tensor(_root_tensor)
root_positions = root_tensor[:, 0:3]
root_orientations = root_tensor[:, 3:7]
root_linvels = root_tensor[:, 7:10]
root_angvels = root_tensor[:, 10:13]

# Save root tensor
saved_root_tensor = root_tensor.clone()

step = 0
while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh the state tensors
    gym.refresh_actor_root_state_tensor(sim)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)


gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
