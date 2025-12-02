# legged_gym架构
## legged_gym
### envs
各种机器人环境，其中的**base**中包含最基础的类**legged_robot.py和legged_robot_config.py**。**base**中的**PMTrajectoryGenerator.py**的作用是利用pmtg模块，覆写机器人的奖励惩罚模块、观测值获取模块和动作输出模块。
### el_mini
六足机器人的文件夹
## scripts
主要包含**train.py和play.py**这两个文件夹。运行时终端可附加参数，参数在**helper.py**中有定义。