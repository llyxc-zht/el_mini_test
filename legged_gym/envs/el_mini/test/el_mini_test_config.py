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

class EL_MINI_TEST_Cfg( LeggedRobotCfg ):
    class env:
        num_envs = 4096
        num_observations = 272 -187
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 18
        num_policy_outputs = 24
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class pmtg:
        gait_type = 'trot'
        duty_factor = 0.5
        base_frequency = 2
        max_clearance = 0.07
        body_height = 0.17
        consider_foothold = True
        z_updown_height_func = ["cubic_up", "cubic_down"]
        max_horizontal_offset = 0.07
        max_y_offset = 0.05
        offset_y_foot_to_hip = 0.25
        train_mode = True

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1,0.1,0.35,0.25,0.2]#[0.25, 0.25, 0, 0.25, 0.25]
        # trimesh only:
        slope_treshold = 0.6 # slopes above this threshold will be corrected to vertical surfaces

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.25] # x,y,z [m]
        default_joint_angles = {
                # 左前腿 (LF)
                "LF_HAA": 0.0,    # 左前髋关节外展/内收角度
                "LF_HFE": 0.3,    # 左前髋关节屈曲/伸展角度
                "LF_KFE": 0.5,   # 左前膝关节屈曲/伸展角度

                # 左中腿 (LM)
                "LM_HAA": 0.0,    # 左中髋关节外展/内收角度
                "LM_HFE": 0.3,    # 左中髋关节屈曲/伸展角度
                "LM_KFE": 0.5,   # 左中膝关节屈曲/伸展角度

                # 左后腿 (LB)
                "LB_HAA": 0.0,    # 左后髋关节外展/内收角度
                "LB_HFE": 0.3,   # 左后髋关节屈曲/伸展角度
                "LB_KFE": 0.5,    # 左后膝关节屈曲/伸展角度

                # 右前腿 (RF)
                "RF_HAA": 0.0,   # 右前髋关节外展/内收角度
                "RF_HFE": 0.3,    # 右前髋关节屈曲/伸展角度
                "RF_KFE": 0.5,   # 右前膝关节屈曲/伸展角度

                # 右中腿 (RM)
                "RM_HAA": 0.0,   # 右中髋关节外展/内收角度
                "RM_HFE": 0.3,    # 右中髋关节屈曲/伸展角度
                "RM_KFE": 0.5,   # 右中膝关节屈曲/伸展角度

                # 右后腿 (RB)
                "RB_HAA": 0.0,   # 右后髋关节外展/内收角度
                "RB_HFE": 0.3,   # 右后髋关节屈曲/伸展角度
                "RB_KFE": 0.5,    # 右后膝关节屈曲/伸展角度
            }

    class control( LeggedRobotCfg.control ):
        control_type = 'D' # P: position, V: velocity, T: torques, D:compute delays、frictions、disturbances...
        # PD Drive parameters:
        stiffness = {'HAA': 40., 'HFE': 40., 'KFE': 40.}  # [N*m/rad]
        damping = {'HAA': 0.8, 'HFE': 0.8, 'KFE': 0.8}     # [N*m*s/rad]
        # stiffness = {'HAA': 60., 'HFE': 60., 'KFE': 60.}  # [N*m/rad]
        # damping = {'HAA': 0.8, 'HFE': 0.8, 'KFE': 0.8}     # [N*m*s/rad]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"
        class motor_dynamics:
            motor_tau_delay = 0.1
            J=[0.01, 0.01, 0.01]  # 各关节初始转动惯量
            B=[0.1, 0.1, 0.1]        # 初始粘性阻尼系数
            Tc=[0.05, 0.05, 0.05] # 初始库伦摩擦力矩
            Ts=[0.02, 0.02, 0.02] # 初始Stribeck摩擦系数
            omega_s: float = 0.1,  # Stribeck速度阈值
            gravity_flag = 'CONSTANT',  # 是否启用重力扰动,F=False,T=True,SIN=正弦扰动,CONSTANT=常数扰动
            gravity_amp: float = 0.5,    # 重力扰动振幅
            gravity_freq: float = 0.2,   # 重力扰动频率
            dist_flag = 'CONSTANT',     # 是否启用外部扰动,F=False,T=True,SIN=正弦扰动,CONSTANT=常数扰动
            dist_amp: float = 0.3,       # 外部扰动振幅
            dist_freq: float = 1.0       # 外部扰动频率
    
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        gamepad_commands = False
        class ranges:
            lin_vel_x = [-0.8, 0.8] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.8, 0.8]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/el_mini/urdf/el_mini.urdf"
        name = "el_mini"
        foot_name = "FOOT"
        shoulder_name = "shoulder"
        penalize_contacts_on = ["SHANK","THIGH"]
        terminate_after_contacts_on = ["BASE","HIP", "KNEE", "SHOULDER"]  #plane
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_base_mass1500 = True
        added_mass_range = [-5., 5.]
    
    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
  
    class rewards:
        class scales:

            # plane
            lin_vel_z = -1
            ang_vel_xy = -1
            orientation = -1
            torques = -0.00001
            dof_vel = -0.00001
            dof_acc = -0.000001
            action_rate =-1.e-6
            collision = -1.5
            termination = -1.5
            dof_pos_limits = -1
            dof_vel_limits = -0.5
            torque_limits = -0.001
            tracking_lin_vel = 6
            tracking_ang_vel = 2.5
            feet_air_time = 0.5 #1
            stumble = 0 
            stand_still = -0.5
            feet_contact_forces = 0#-0.5#-1
            leg_swing_control = 0
            base_height = -1

            # # trimesh
            # lin_vel_z = -1
            # ang_vel_xy = -1
            # orientation = 0 #-1
            # torques = -0.001 #-0.00001
            # dof_vel = -0.0005
            # dof_acc = -0.000001
            # action_rate =-1.e-6
            # collision = -1.5
            # termination = -1.5
            # dof_pos_limits = -1
            # dof_vel_limits = -1
            # torque_limits = 0
            # tracking_lin_vel = 6
            # tracking_ang_vel = 2
            # feet_air_time = 0
            # stumble = -2
            # high_acceleration = 0
            # stand_still = -1
            # feet_contact_forces = 0
            # leg_swing_control = 0
            # base_height = -2


        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.2
        max_contact_force = 300. # forces above this value are penalized
        still_all = False
    
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.


    
class EL_MINI_TEST_PPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.1
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'relu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class runner( LeggedRobotCfgPPO.runner ):
        max_iterations = 1500

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'cpo'#'cpo'
        run_name = ''
        # load and resume
        resume = False         
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt


