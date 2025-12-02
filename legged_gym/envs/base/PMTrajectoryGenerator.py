# SPDX-FileCopyrightText: Copyright (c) 2023, HUAWEI TECHNOLOGIES. All rights reserved.
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

import numpy as np
import time
from typing import Any, Mapping, Sequence, Tuple
from legged_gym.utils.isaacgym_utils import get_euler_xyz, coordinate_rotation, to_torch, quat_apply
import torch

_TROT_PHASE_OFFSET = [torch.pi,torch.pi,0,0,0,torch.pi]
_WALK_PHASE_OFFSET = [0, torch.pi / 3, 2 * torch.pi / 3, torch.pi, 4 * torch.pi / 3, 5 * torch.pi / 3]
_BOUND_PHASE_OFFSET = [0, 0, torch.pi, torch.pi, 0, 0]


class PMTrajectoryGenerator:
    """
    Class for generating foot target trajectories for a quadruped robot.
    """

    #-----------------------Initialization-----------------------#
    def __init__(
        self,
        robot: Any,
        clock: Any,
        device: torch.device,
        num_envs: int,
        param: Any,
        task_name='el_mini_test',
    ):
        """
        Initialize the PMTrajectoryGenerator.

        Args:
            robot: The robot object.
            clock: The clock object for timing.
            device: The device to run the calculations on.
            num_envs: The number of parallel environments.
            param: The parameters for trajectory generation.
            task_name: The name of the task.

        Raises:
            Exception: If an invalid task_name is provided.
        """

        # Import robot parameters based on task_name
        if 'el_mini_test' in task_name:
            from legged_gym.envs.el_mini.test.el_mini_real.robot_utils import (INIT_MOTOR_ANGLES, UPPER_LEG_LENGTH, LOWER_LEG_LENGTH,
                                                                HIP_LENGTH, HIP_POSITION, COM_OFFSET, HIP_OFFSETS)
        # elif 'lite' in task_name:
        #     from legged_gym.envs.lite3.lite3_real.robot_utils import (INIT_MOTOR_ANGLES, UPPER_LEG_LENGTH, LOWER_LEG_LENGTH,
        #                                                               HIP_LENGTH, HIP_POSITION, COM_OFFSET, HIP_OFFSETS)
        # elif 'go' in task_name:
        #     from legged_gym.envs.go2.go2_real.robot_utils import (INIT_MOTOR_ANGLES, UPPER_LEG_LENGTH, LOWER_LEG_LENGTH,
        #                                                           HIP_LENGTH, HIP_POSITION, COM_OFFSET, HIP_OFFSETS)
        else:
            raise Exception("")

        # Store robot parameters
        self.INIT_MOTOR_ANGLES = INIT_MOTOR_ANGLES
        self.UPPER_LEG_LENGTH = UPPER_LEG_LENGTH
        self.LOWER_LEG_LENGTH = LOWER_LEG_LENGTH
        self.HIP_LENGTH = HIP_LENGTH
        self.HIP_POSITION = HIP_POSITION
        self.COM_OFFSET = COM_OFFSET
        self.HIP_OFFSETS = HIP_OFFSETS

        self.hip_offset = 0.034  # 髋关节偏移量
        self.knee_offset = 0.0535  # 膝关节偏移量

        self.robot = robot
        self.clock = clock
        self.gait_type = param.gait_type
        self.base_frequency = param.base_frequency
        self.max_clearance = param.max_clearance
        self.body_height = param.body_height
        self.duty_factor = param.duty_factor
        self.max_horizontal_offset = param.max_horizontal_offset
        self.max_y_offset = param.max_y_offset
        self.offset_y_foot_to_hip = param.offset_y_foot_to_hip
        self.train_mode = param.train_mode

        self.device = device
        self.num_envs = num_envs

        # 初始化mirror_coe
        self.mirror_coe = torch.ones(6,device=self.device)  # 创建一 = [0, torch.pi, torch.pi, 0,torch.pi,0]个长度为6的全1张量
        # self.mirror_coe = torch.ones(6)  # 创建一个长度为6的全1张量
        self.mirror_coe[[3, 4, 5]] = -1  # 设置第0, 1和5号腿的mirror_coe为-1

        # 初始化offset_coe
        self.offset_coe = torch.ones(6,device=self.device)  # 创建一个长度为6的全1张量
        # self.offset_coe = torch.ones(6)  # 创建一个长度为6的全1张量
        self.offset_coe[[0,3]] = -1  # 

        # Set initial phase based on gait type
        if self.gait_type == 'trot':
            self.initial_phase = to_torch(_TROT_PHASE_OFFSET, device=self.device)
        elif self.gait_type == 'walk':
            self.initial_phase = to_torch(_WALK_PHASE_OFFSET, device=self.device)
        elif self.gait_type == 'bound':
            self.initial_phase = to_torch(_BOUND_PHASE_OFFSET, device=self.device)

        self.default_joint_position = torch.zeros(self.num_envs, 18, dtype=torch.float, device=self.device)

        # Initial the joint positions and joint angles
        self.foot_target_position_in_hip_frame = torch.zeros(self.num_envs, 18, dtype=torch.float, device=self.device)
        # self.foot_target_position_in_base_frame = torch.zeros(self.num_envs, 18, dtype=torch.float, device=self.device)
        self.target_joint_angles = torch.zeros(self.num_envs, 18, dtype=torch.float, device=self.device)

        self.is_swing = torch.zeros((self.num_envs, 6), dtype=torch.bool, device=self.device)
        self.clip_workspace_tensor = torch.ones((1, 1, 3), device=self.device, dtype=torch.float) * 0.15

        self.l_hip_sign = torch.tensor([1, -1, 1, -1, 1, -1], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
        self.base_frequency_tensor = torch.tensor(self.base_frequency, dtype=torch.float,
                                                  device=self.device).repeat(self.num_envs, 1)

        self.initial_phase = self.initial_phase.repeat(self.num_envs, 1)
        self.phi = self.initial_phase.clone()
        self.swing_phi = torch.zeros_like(self.phi)
        self.delta_phi = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device)
        self.cos_phi = torch.cos(self.phi)
        self.sin_phi = torch.sin(self.phi)
        self.reset_time = torch.tensor(self.clock(), dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
        self.time_since_reset = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)

        self.foot_trajectory = torch.zeros((self.num_envs, 6 ,3), dtype=torch.float, device=self.device)
        self.foot_trajectory_z = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device) - self.body_height

        self.com_offset = to_torch(COM_OFFSET, device=self.device)
        self.hip_offsets = to_torch(HIP_OFFSETS, device=self.device)
        self.hip_position = to_torch(HIP_POSITION, device=self.device)

        self.f_up = self.gen_func(param.z_updown_height_func[0])
        self.f_down = self.gen_func(param.z_updown_height_func[1])
    def gen_func(self, func_name):
        """
        Generate a lambda function based on the provided function name.

        Args:
            func_name (str): Name of the function to generate.

        Returns:
            function: Lambda function based on the provided function name.

        Raises:
            NotImplementedError: If the provided function name is not supported.
        """
        if func_name == 'cubic_up':
            return lambda x: -16 * x**3 + 12 * x**2
        elif func_name == 'cubic_down':
            return lambda x: 16 * x**3 - 36 * x**2 + 24 * x - 4
        elif func_name == 'linear_down':
            return lambda x: 2.0 - 2.0 * x
        elif func_name == 'sin':
            return lambda x: torch.sin(torch.pi * x)
        else:
            raise NotImplementedError("PMTG z height")

    def reset(self, index_list):
        """
        Reset the specified indices in the motion planner.

        Args:
            index_list (list): A list of indices to reset.
        """

        self.initial_phase[index_list, 0], self.initial_phase[index_list, 1] = self.initial_phase[index_list, 1], self.initial_phase[index_list, 0]
        self.initial_phase[index_list, 2], self.initial_phase[index_list, 3] = self.initial_phase[index_list, 3], self.initial_phase[index_list, 2]
        self.phi[index_list] = self.initial_phase[index_list]
        self.swing_phi[index_list] = 0
        self.delta_phi[index_list] = 0
        self.cos_phi[index_list] = torch.cos(self.phi[index_list])
        self.sin_phi[index_list] = torch.sin(self.phi[index_list])
        self.is_swing[index_list] = False
        self.reset_time[index_list] = self.clock()
        self.time_since_reset[index_list] = 0


    def set_base_frequency(self, desired_frequency: float):
        """
        Set the desired base frequency for trajectory generation.

        Args:
            desired_frequency: The desired base frequency.
        """
        self.base_frequency = desired_frequency

    def get_base_frequency(self) -> float:
        """
        Get the current base frequency.

        Returns:
            The current base frequency.
        """
        return self.base_frequency

    def set_gait_type(self, desired_gait_type: str):
        """
        Set the desired gait type for trajectory generation.

        Args:
            desired_gait_type: The desired gait type.
        """
        self.gait_type = desired_gait_type

    def get_gait_type(self) -> str:
        """
        Get the current gait type.

        Returns:
            The current gait type.
        """
        return self.gait_type

    def get_delta_phi(self) -> Sequence[float]:
        """
        Get the delta_phi values.

        Returns:
            A sequence of delta_phi values.
        """
        return self.delta_phi

    def get_cos_phi(self) -> Sequence[float]:
        """
        Get the cos_phi values.

        Returns:
            A sequence of cos_phi values.
        """
        return self.cos_phi

    def get_sin_phi(self) -> Sequence[float]:
        """
        Get the sin_phi values.

        Returns:
            A sequence representing the sin_phi values.
        """
        return self.sin_phi

    #-----------------------observation-----------------------#
    def update_observation(self):
        """
        Update the last action and parameters of the Central Pattern Generator (CPG) and return the observation.

        Returns:
            observation (torch.Tensor): The updated observation containing delta_phi, cos_phi, sin_phi, and base_frequency.

        """
        base_frequency = self.base_frequency_tensor
        observation = torch.cat((self.delta_phi, self.cos_phi, self.sin_phi, base_frequency), dim=1)

        return observation
    

    #-----------------------action-----------------------#
    def get_action(self, delta_phi, residual_xyz, residual_angle, base_orientation, **kwargs):
        """
        compute the position in base frame, given the base orientation.

        Args:
          delta_phi: phase variable.
          residual_xyz: residual in horizontal hip reference frame.
          residual_angle: residual in joint space.
base_orientation: quaternion (w,x,y,z) of the base link.

        Returns:
            target_joint_angles: joint angle of for leg (FL,FR,RL,RR)
        """
        delta_phi, residual_angle = delta_phi.to(self.device), residual_angle.to(self.device)
        if torch.isnan(residual_angle).any():
            print("NaN detected in residual_angle")
        self.gen_foot_target_position_in_horizontal_hip_frame(delta_phi, residual_xyz, **kwargs)
        # self.foot_target_position_in_base_frame = self.transform_to_base_frame(self.foot_target_position_in_hip_frame,
        #                                                                        base_orientation)
        # self.target_joint_angles = self.get_target_joint_angles(self.foot_target_position_in_base_frame)
        self.target_joint_angles = self.get_target_joint_angles(self.foot_target_position_in_hip_frame)
        # Flatten target_joint_angles to match the expected shape
        self.target_joint_angles = self.target_joint_angles.view(self.num_envs, -1)
        self.target_joint_angles += residual_angle
        return self.target_joint_angles


    def gen_foot_target_position_in_horizontal_hip_frame(self, delta_phi: Sequence[float], residual_xyz: Sequence[float],
                                                         **kwargs) -> Sequence[float]:
        """
        Compute the foot target positions in the horizontal hip reference frame.

        Args:
            delta_phi: A sequence of floats representing the phase variable.
            residual_xyz: A sequence of floats representing the residual in the horizontal hip reference frame.

        Returns:
            A sequence of floats representing the foot target positions in the horizontal hip reference frame.
        """
        self.foot_target_position_in_hip_frame = residual_xyz.reshape(-1, 6, 3)
        self.gen_foot_trajectory_axis_z(delta_phi, self.clock())
        self.foot_target_position_in_hip_frame += self.foot_trajectory

        return self.foot_target_position_in_hip_frame
    
    def gen_foot_trajectory_axis_z(self, delta_phi: Sequence[float], t: float) -> Sequence[float]:
        """
        Generate the foot trajectory along the z-axis.

        Args:
            delta_phi: A sequence of floats representing the change in phase for each leg.
            t: The current time.

        Returns:
            A sequence of floats representing the foot trajectory along the z-axis.
        """
        # duty_factor = 支撑阶段时间 / 总步态周期时间

        self.time_since_reset = self.clock() - self.reset_time
        self.phi = ((self.initial_phase + 2 * torch.pi * self.base_frequency * self.time_since_reset + delta_phi) / torch.pi) % 2 * torch.pi

        self.delta_phi = delta_phi
        self.cos_phi = torch.cos(self.phi)
        self.sin_phi = torch.sin(self.phi)

        k3 = (self.phi / (2 * torch.pi)) < self.duty_factor
        self.is_swing = ~k3.clone()
        self.swing_phi = (self.phi / (2 * torch.pi) - self.duty_factor) / (1 - self.duty_factor)  # [0,1)
        # f_up 抬升部分轨迹函数     f_down 降落部分轨迹函数
        factor = torch.where(self.swing_phi < 0.5, self.f_up(self.swing_phi), self.f_down(self.swing_phi))
        self.foot_trajectory[:, :, 2] = factor * (self.is_swing * self.max_clearance) - self.body_height  # max_clearance 最大抬升高度
        self.foot_trajectory[:,:,1] = 0.7*self.max_y_offset* factor*self.is_swing + self.offset_y_foot_to_hip
        self.foot_trajectory[:,[3,4,5],1]*= -1
        self.foot_trajectory[:, :, 0] = -self.max_horizontal_offset * torch.sin(self.swing_phi * 2 * torch.pi) * self.is_swing

    def get_target_joint_angles(self, target_position_in_base_frame):
        """
        Compute the joint angles given the foot target positions in the base frame.

        Args:
            target_position_in_base_frame: A tensor representing the foot target positions in the base frame.

        Returns:
            A tensor representing the joint angles for each leg.
        """
        foot_position = target_position_in_base_frame - self.hip_offsets
        # print("foot_position", foot_position[0])
        joint_angles = self.foot_position_in_hip_frame_to_joint_angle(foot_position)
        # print("joint_angles", joint_angles[0])
        if torch.isnan(foot_position).any():
            print("NaN detected in foot_position")
        if torch.isnan(joint_angles).any():
            print("NaN detected in joint_angles")

        return joint_angles

    #-----------------------inverse kinematics-----------------------#
    def foot_position_in_hip_frame_to_joint_angle(self, foot_position):
        """
        Compute the motor angles for one leg using inverse kinematics (IK).

        Args:
            foot_position: A tensor representing the foot positions in the hip frame.

        Returns:
            A tensor representing the motor angles for one leg.
        """
        # self.l2 = self.UPPER_LEG_LENGTH
        # self.l1 = self.LOWER_LEG_LENGTH
        # self.l0 = self.HIP_LENGTH

        self.l0 = 0.15
        self.l1 = 0.13
        self.l2 = 0.232

        # p的形状为[n, 6, 3]
        pos = foot_position.clone()
        pos[:,:,2] += self.hip_offset  # 偏移量
        q00 = torch.atan2(pos[:, :, 0], self.mirror_coe*pos[:, :, 1])
        K0 = torch.sqrt(torch.square(pos[:, :, 0]) + torch.square(pos[:, :, 1]))
        q0 = q00 - self.offset_coe*torch.asin(self.knee_offset/K0)
        K = torch.sqrt(torch.square(K0) - self.knee_offset**2) - self.l0 # 计算K的值

        beta = torch.atan2(pos[:, :, 2], K)

        temp = (torch.square(K) + torch.square(pos[:, :, 2]) + self.l1 ** 2 - self.l2 ** 2) / \
               (2 * self.l1 * torch.sqrt(torch.square(K) + torch.square(pos[:, :, 2])))
        fai = torch.acos(self._limit(temp))

        q1 = beta + fai
        temp = (torch.square(K) + torch.square(pos[:, :, 2]) - self.l1 ** 2 - self.l2 ** 2) / \
               (2 * self.l1 * self.l2)
        q2 = -torch.acos(self._limit(temp))
        q1 = torch.pi / 2 - q1
        q2 += torch.pi

        q_limited = self._limit_angle(torch.stack([q0, q1, q2], dim=2))  # 输出形状为[n, 6, 3]
        return q_limited

    def _limit(self, value, upper=1.0, lower=-1.0):
        return torch.clamp(value, min=lower, max=upper)

    def _limit_angle(self, q):
        q[:, :, 1] = torch.clamp(q[:, :, 1], min=-0.5233, max=3.14)  # 限制 q1 在 [-0.5233, 3.1]
        q[:, :, 2] = torch.clamp(q[:, :, 2], min=-0.6978, max=3.14)  # 限制 q2 在 [-0.6978, 3.14]
        return q

    
    #-----------------------base_frame-----------------------#

    def transform_to_base_frame(self, position, quaternion):
        """
        Compute the position in the base frame, given the base orientation.

        Args:
            position: A tensor representing the point position.
            quaternion: A tensor representing the quaternion (x, y, z, w) of the base link.

        Returns:
            A tensor representing the position in the base frame.
        """
        if torch.isnan(quaternion).any():
            print("NaN detected in quaternion")
        if torch.isnan(self.foot_target_position_in_hip_frame).any():
            print("NaN detected in foot_target_position_in_hip_frame")
        rpy = get_euler_xyz(quaternion)
        rpy[:, 2] = 0
        R = torch.matmul(coordinate_rotation(0, rpy[:, 0]), coordinate_rotation(1, rpy[:, 1]))
        rotated_position = torch.matmul(R, position.transpose(1, 2)).transpose(1, 2)
        rotated_position = rotated_position + self.hip_position.unsqueeze(0)

        return rotated_position

    def quat_apply_feet_positions(self, quat, positions):
        """
        Apply quaternion rotation to the foot positions.

        Args:
            quat: A tensor representing the quaternion (x, y, z, w).
            positions: A tensor representing the foot positions.

        Returns:
            A tensor representing the foot positions after applying the quaternion rotation.
        """
        quat *= torch.tensor([-1, -1, -1, 1]).to(self.device)
        num_feet = positions.shape[1]
        quat = quat.repeat(1, num_feet).reshape(-1, num_feet)
        quat_pos = quat_apply(quat, positions.reshape(-1, 3))

        return quat_pos.view(positions.shape)

    



if __name__ == "__main__":
    num_envs = 4096
    device = torch.device("cuda:0")

    pmtg = PMTrajectoryGenerator(robot=None, num_envs=num_envs, device=device, gait_type='trot')

    while True:
        residual_xyz = torch.zeros(num_envs, 18, dtype=torch.float, device=device)
        residual_angle = torch.zeros(num_envs, 18, dtype=torch.float, device=device)
        delta_phi = torch.zeros(num_envs, 6, dtype=torch.float, device=device)
        base_quat = torch.tensor([0, 0, 0, 1], dtype=torch.float, device=device).repeat(num_envs, 1)

        observation = pmtg.update_observation()
        print("observation: \n", observation)

        joint_angles = pmtg.get_action(delta_phi, residual_xyz, residual_angle, base_quat)
        print("joint_angles: \n", joint_angles)
        time.sleep(0.001)
