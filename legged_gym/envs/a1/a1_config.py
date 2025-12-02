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
# Copyright (c) 2023, HUAWEI TECHNOLOGIES

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class A1Cfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.36]  # x,y,z [m]
        default_joint_angles = {
            # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.,
            'RL_hip_joint': 0.,
            'FR_hip_joint': 0.,
            'RR_hip_joint': 0.,

            'FL_thigh_joint': 0.9,
            'RL_thigh_joint': 0.9,
            'FR_thigh_joint': 0.9,
            'RR_thigh_joint': 0.9,

            'FL_calf_joint': -1.8,
            'RL_calf_joint': -1.8,
            'FR_calf_joint': -1.8,
            'RR_calf_joint': -1.8,
        }

    class env(LeggedRobotCfg.env):
        num_envs = 1000
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_observation_history = 40
        episode_length_s = 20  # episode length in seconds
        curriculum_factor = 0.8

    class pmtg:
        gait_type = 'walk'
        duty_factor = 0.5
        base_frequency = 1.25
        max_clearance = 0.10
        body_height = 0.28
        consider_foothold = False
        z_updown_height_func = ["cubic_up", "cubic_down"]
        max_horizontal_offset = 0.0
        train_mode = True

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_torch_vel_estimator = False
        use_actuator_network = False

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'  # a1_unitree.urdf
        name = "a1"
        foot_name = "foot"
        shoulder_name = "shoulder"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "shoulder"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        restitution_mean = 0.5
        restitution_offset_range = [-0.1, 0.1]
        compliance = 0.5


class A1RoughCfgPPO(LeggedRobotCfgPPO):

    class policy:
        init_noise_std = 0.1
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'a1'
        max_iterations = 1000 # number of policy updates
        resume = False
        resume_path = 'legged_gym/logs/a1'  # updated from load_run and ckpt
        load_run = ''  # -1 = last run
        checkpoint = -1  # -1 = last saved model
