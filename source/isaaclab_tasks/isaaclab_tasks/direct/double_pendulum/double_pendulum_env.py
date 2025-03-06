from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.double_pendulum import DOUBLE_PENDULUM_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.noise import GaussianNoiseCfg, gaussian_noise, NoiseModelWithAdditiveBiasCfg

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names= "Link2"),
            "mass_distribution_params": (0.2, 0.5),
            "operation": "add",
        },
    )


@configclass
class DoublePendulumEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2 #"Decimation factor for rendering" # try to reduce this to see how its working
    episode_length_s = 15.0
    possible_agents = ["pendulum"]
    
    observation_spaces = {"pendulum": 4}
    state_space = -1

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = DOUBLE_PENDULUM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    shoulder_dof_name = "Joint1"
    elbow_dof_name = "Joint2"
    robot_type : str =  "acrobot" #"acrobot" #"double_pendulum" #"pendubot"
    
    if robot_type == "acrobot":
        action_spaces = {"pendulum": 1}
    elif robot_type == "pendubot":
        action_spaces = {"pendulum": 1}
    elif robot_type == "double_pendulum":
        action_spaces = {"pendulum": 2}
    
    # action_noise_model: dict = {
    #     "pendulum": GaussianNoiseCfg(mean=0.0, std=0.1, operation="add")
    # }
    
    # observation_noise_model: dict = {
    #     "pendulum": GaussianNoiseCfg(mean=0.0, std=0.1, operation="add")
    # }
    # action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #   noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
    #   bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    # )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)
    
    # events
    events: EventCfg = EventCfg()

    # reset
    initial_shoulder_angle_range = [-1.0, 1.0]  # the range in which the pole angle is sampled from on reset [rad]
    initial_elbow_angle_range = [-1.0, 1.0]  # the range in which the pendulum angle is sampled from on reset [rad]

    # action scales
    torque_scale : float = 10.0
    
    # define goal state, here it is for swing up.
    goal_state: list[float] = [-math.pi, 0.0, 0.0, 0.0]

    # reward scales
    reward_state_weights: list[float] = [1.0, 2.0, 0.1, 0.1]
    # Weights for [shoulder_angle, elbow_angle, shoulder_vel, elbow_vel]
    reward_action_weight: float = 0.1
    reward_scale: float = 1.0
    time_penalty: float = 0.01 # penalty for each time step
    bonus_reward: float = 20.0 # bonus reward for reaching the state early
    
    # normalisation for observation velocities
    max_velocity: float = 5.0 # Used to scale the joint velocities in observation
    success_threshold: float = 0.1 # To end the episode early if the goal is reached.

    sustain_steps: int = 10  # number of steps to sustain the goal state
    
    
class DoublePendulumEnv(DirectMARLEnv):
    cfg: DoublePendulumEnvCfg

    def __init__(self, cfg: DoublePendulumEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._shoulder_dof_idx, _ = self.robot.find_joints(self.cfg.shoulder_dof_name)
        self._elbow_dof_idx, _ = self.robot.find_joints(self.cfg.elbow_dof_name)
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        # Initialize a counter for consecutive stable steps
        self._stable_count = torch.zeros((self.joint_pos.shape[0],), dtype=torch.int32, device=self.joint_pos.device)
        
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        torque = self.actions["pendulum"] * self.cfg.torque_scale
        # Apply torque based on robot type.
        if self.cfg.robot_type == "acrobot":
            # Actuate only the second joint (elbow)
            self.robot.set_joint_effort_target(
                torque, joint_ids=self._elbow_dof_idx)
        elif self.cfg.robot_type == "pendubot":
            # Actuate only the first joint (shoulder)
            self.robot.set_joint_effort_target(
                torque, joint_ids=self._shoulder_dof_idx)
        elif self.cfg.robot_type == "double_pendulum":
            self.robot.set_joint_effort_target(torque)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        shoulder_joint_pos = normalize_angle(self.joint_pos[:, self._shoulder_dof_idx[0]].unsqueeze(dim=1))
        elbow_joint_pos = normalize_angle(self.joint_pos[:, self._elbow_dof_idx[0]].unsqueeze(dim=1))
        
        # Scale Joint velocities to normalise them
        shoulder_vel = self.joint_vel[:, self._shoulder_dof_idx[0]].unsqueeze(dim=1) / self.cfg.max_velocity
        elbow_vel = self.joint_vel[:, self._elbow_dof_idx[0]].unsqueeze(dim=1) / self.cfg.max_velocity
        
        # Optionally Clip velocities within [-1, 1]
        shoulder_vel = torch.clamp(shoulder_vel, -1.0, 1.0)
        elbow_vel = torch.clamp(elbow_vel, -1.0, 1.0)
        observations = torch.cat((shoulder_joint_pos, elbow_joint_pos, shoulder_vel, elbow_vel), dim=-1)
        
        return {"pendulum": observations}

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # goal = torch.tensor(self.cfg.goal_state, device=self.robot.data.joint_pos.device)
        
        goal_tensor = torch.tensor(self.cfg.goal_state, device=self.robot.data.joint_pos.device, dtype=torch.float32)
        goal_tensor[:2] = normalize_angle(goal_tensor[:2])
        
        state = self._get_observations()["pendulum"]
        error = state - goal_tensor
        
        # Create diagonal matrix Q from reward_state_weights
        Q = torch.diag(torch.tensor(self.cfg.reward_state_weights, device=state.device, dtype=torch.float32))
        
        # Compute the quadratic state cost: error^T * Q * error
        state_cost = torch.einsum("bi, ij, bj->b", error, Q, error)
        
        # Compute control cost 
        action = self.actions["pendulum"]
        action_cost = self.cfg.reward_action_weight * torch.sum(action**2, dim=-1)
        total_cost = state_cost + action_cost
        
        # base reward is negative cost minus a time penalty
        base_reward = - (self.cfg.reward_scale * total_cost) - self.cfg.time_penalty
        
        # Compute error norm to check stability
        error_norm = torch.norm(error, dim=-1)
        
        # If the state error is below the threshold, add the bonus reward
        bonus = torch.where(
            torch.norm(state - goal_tensor, dim=-1) < self.cfg.success_threshold,
            torch.tensor(self.cfg.bonus_reward, device=state.device),
            torch.tensor(0.0, device=state.device)
        )
        
        stable_condition = error_norm < self.cfg.success_threshold
        self._stable_count = torch.where(stable_condition, self._stable_count + 1, torch.zeros_like(self._stable_count))
        
        # For instance, partial rewards for being "closer" to the target angle:
        angle_dist = torch.abs(error[:, :2])  # just the angles
        angle_shaping = 1.0 - torch.tanh(angle_dist)  # example shaping
        base_reward += angle_shaping.sum(dim=-1) * 0.01  # scale as needed
            
        reward = base_reward + bonus
        
        return {"pendulum": reward}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        # Calculate State Error 
        state = self._get_observations()["pendulum"]
        goal = torch.tensor(self.cfg.goal_state, device=state.device, dtype=torch.float32)
        error_norm = torch.norm(state - goal, dim=-1)
        
        # Early termination threshold, TODO: experiment with this value
        success_threshold = self.cfg.success_threshold
        success = error_norm < success_threshold
        
        # Early termination if the pendulum has been stable for enough consecutive steps
        sustained_stability = self._stable_count >= self.cfg.sustain_steps

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        dones = {"pendulum": time_out}
        time_outs = {"pendulum": time_out}
        return dones, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids]

        joint_pos[:, self._shoulder_dof_idx] += sample_uniform(
            self.cfg.initial_shoulder_angle_range[0] * math.pi,
            self.cfg.initial_shoulder_angle_range[1] * math.pi,
            joint_pos[:, self._shoulder_dof_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self._elbow_dof_idx] += sample_uniform(
            self.cfg.initial_elbow_angle_range[0] * math.pi,
            self.cfg.initial_elbow_angle_range[1] * math.pi,
            joint_pos[:, self._elbow_dof_idx].shape,
            joint_pos.device,
        )
        
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # Update the root state based on environment origins
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset the stable counter upon environment reset
        self._stable_count[env_ids] = 0


@torch.jit.script
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi
