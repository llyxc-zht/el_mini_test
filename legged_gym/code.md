# Legged_gym 代码阅读
## Legged_robot.py
### 环境创建
#### __init__
初始化一些变量，引入cfg。
**self._init_buffers()**  可以添加一些需要的变量
#### create_sim
创建环境。可以选择环境，在**cfg.terrain**中。一般有地形使用**trimesh**，平地使用**plane**
#### _init_height_points
terrain中参数设定的量测范围measured_points_y、measured_points_x，初始化由机器人周围矩形范围内点在环境坐标系下的张量
### 创建机器人
#### _create_envs
创建一个环境，该函数的主要功能是加载URDF文件，创建训练环境，为机器人增加物理属性，增加机器人自由度和形状属性等
1. loads the robot URDF/MJCF asset,
2. For each environment
    1. creates the environment, 
    2. calls DOF and Rigid shape properties callbacks,
    3. create actor with these properties and add them to the env
3. Store indices of different bodies of the robot
#### _get_env_origins
设置机器人在环境中生成是的相对原点
将绝对坐标系设置在self.env_origins[:]
#### _get_heights
计算高度有关的函数

#### _process_dof_props
该函数在creat_envs函数中调用，主要功能是储存、修改、随机每个环境的自由度属性，输入参数是每个自由度的属性，输出是修改后的自由度属性

### explore
#### push_robots
训练中给机器人一个推力，从而保证训练模型的鲁棒性

#### _update_terrain_curriculum和update_command_curriculum
设置curriculum，这两个函数可以使得地形难度和速度逐渐升高

#### _get_noise_scale_vec
加入噪声干扰

#### check_termination
保护机器人不作出危险动作，比如contact_forces。作出危险动作会进行重置（调用函数reset_idx），并惩罚

### process推动训练进行
#### step
将action应用到仿真的机器人上进行训练的函数。  
**self.cfg.control.decimation**表示仿真和实际机器人运行的差值，平衡仿真和实际机器人的运算时间和效率问题
#### post_physics_step
计算观测值和奖励，判断一个智能体应不应该被终结
**self.episode_length_buf**时间步数，表示每个机器人经过了多少step

#### _compute_torques
计算力矩，有三种方法，代表三种不同的输入

#### compute_observations
计算观测值。**self.obs_buf**是你想要观测的值。通常来说这些观测值是用来计算奖励的

### reward奖励
#### _prepare_reward_function
初始化时使用的，将奖励函数导入进去。导入的规则在config中r**rewards类**

## legged_gym_config.py
### env
1. num_env 环境数量
2. num_observations 观测值数量
3. num_actions 机器人可以动的关节
4. env_spacing 机器人之间的距离
5. episode_length_s 机器人生存时间
### terrain
与地形相关的参数
### commands
1. range中是一些范围
2. resampling_time 切换命令的时间，一般不要太短也不要太长，10左右即可
### init_state
与重置有关，一般只考虑pos的z坐标

### asset
与URDF文件有关
1. file URDF文件的位置
2. penalize_contacts_on 惩罚
3. foot_name 脚的位置
4. terminate_after_contacts_on 终止条件，一般是躯干位置
5. self_collisions 身体自碰撞开关

### rewards
奖励的权值在这里设置


## 自主开发
### 修改URDF文件
#### legged_gym_config.py中的修改
1. **asset**中修改URDF文件位置。  
   一般在legged_gym根文件夹的resources文件中。可以自己新建一个文件夹
   继续设置**keypoint**、**feetname**等等
2. **contorl**中设置**siffness、damping**。这些与自己的URDF文件要对应上，给的值通常要大概测算一下
3. **init_state**中设置各项初始化和重置位置
4. **runner**中的**run_name、experiment_name**可以进行一下修改
#### env/__init__.py中注册
1. 引入大类、config和PPO算法
2. **task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )**注册任务
