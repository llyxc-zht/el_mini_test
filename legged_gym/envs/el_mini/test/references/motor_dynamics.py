import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

class MotorDynamics(nn.Module):
    """
    电机动力学模型类：包含Stribeck摩擦模型、控制延迟补偿
    基于文档中的动力学方程实现
    """
    def __init__(self, 
                 joint_num: int,  # 关节数量
                 dt: float = 0.005,  # 仿真步长（需与LeggedGym配置一致）
                 tau_delay: float = 0.01,  # 控制延迟时间常数（初始值，可辨识）
                 J = None,  # 转动惯量（初始值，可辨识）
                 B = None,  # 粘性阻尼系数（初始值，可辨识）
                 Tc = None,  # 库伦摩擦力矩（初始值，可辨识）
                 Ts = None,  # Stribeck摩擦系数（初始值，可辨识）
                 omega_s: float = 0.1,  # Stribeck速度阈值（文档默认建议值）
                 # --- 新增：正弦扰动参数 ---
                 gravity_amp: float = 0.5,    # 重力扰动振幅
                 gravity_freq: float = 0.2,   # 重力扰动频率
                 dist_amp: float = 0.3,       # 外部扰动振幅
                 dist_freq: float = 1.0       # 外部扰动频率
                 ):
        super().__init__()
        self.joint_num = joint_num
        self.dt = dt
        self.omega_s = omega_s  # 区分高低速摩擦的阈值
        
        # 初始化可辨识参数（注册为nn.Parameter便于后续优化）
        self.J = nn.Parameter(torch.ones(joint_num) * 0.01 if J is None else J)  # 转动惯量
        self.B = nn.Parameter(torch.ones(joint_num) * 0.1 if B is None else B)   # 粘性阻尼系数
        self.Tc = nn.Parameter(torch.ones(joint_num) * 0.05 if Tc is None else Tc) # 库伦摩擦力矩
        self.Ts = nn.Parameter(torch.ones(joint_num) * 0.02 if Ts is None else Ts) # Stribeck摩擦系数
        self.tau_delay = nn.Parameter(torch.tensor(tau_delay))  # 控制延迟时间常数
        
        # 控制延迟一阶滞后模型的状态（存储上一时刻的指令力矩）
        self.prev_cmd_torque = torch.zeros(1, self.joint_num)

        # --- 新增：内部时间和扰动参数 ---
        self.time = 0.005
        self.gravity_amp = gravity_amp
        self.gravity_freq = gravity_freq
        self.dist_amp = dist_amp
        self.dist_freq = dist_freq
    
    def compute_torque_feedfoeward(self,qdd,qd):
            """
            计算关节加速度和速度对应的力矩前馈项：J*qdd + B*qd
            输入：qdd - 关节角加速度 (batch_size, joint_num)
                qd - 关节角速度 (batch_size, joint_num)
            输出：torque_ff - 力矩前馈项 (与输入维度一致)
            """
            qdd = qdd.float()
            qd = qd.float()
            
            # 默认使用广播机制处理批量数据
            torque_ff = self.J * qdd + self.B * qd
            
            return torque_ff
            
    def compute_torque_stribeck_friction(self, omega):
        """
        计算Stribeck摩擦力矩（分段函数，含库伦摩擦+粘性摩擦+Stribeck效应）
        输入：omega - 关节角速度 (batch_size, joint_num)
        输出：Tf - 摩擦力矩 (与输入维度一致)
        """
        # 确保输入维度兼容
        omega = omega.float()
        
        # 分段计算：低速段（|omega| <= omega_s）和高速段（|omega| > omega_s）
        low_speed_mask = torch.abs(omega) <= self.omega_s
        high_speed_mask = ~low_speed_mask
        
        # 初始化摩擦力矩
        Tf = torch.zeros_like(omega)
        
        # 低速段：Stribeck效应主导 Tf = Ts * sign(omega) * exp(-|omega|/omega_s) + B*omega
        # 注意：这里使用 Ts (Stribeck摩擦系数) 而不是 Tc
        Tf[low_speed_mask] = (self.Ts * torch.sign(omega[low_speed_mask]) * 
                            torch.exp(-torch.abs(omega[low_speed_mask])/self.omega_s) +
                            self.B * omega[low_speed_mask])
        
        # 高速段：库伦摩擦+粘性摩擦 Tf = Tc * sign(omega) + B*omega
        Tf[high_speed_mask] = (self.Tc * torch.sign(omega[high_speed_mask]) +
                            self.B * omega[high_speed_mask])
        
        return Tf
    
    def compute_torque_gravity(self,time):
        """
        计算重力补偿力矩，此处修改为正弦扰动。
        输出：torque_gravity - 正弦形式的重力扰动力矩 (joint_num,)
        """
        # 计算当前时刻的扰动力矩值
        gravity_torque_scalar = self.gravity_amp * torch.sin(2 * torch.pi * self.gravity_freq * time)
        
        # 将标量值扩展到所有关节
        torque_gravity = torch.full((self.joint_num,), gravity_torque_scalar, device=self.J.device)
        return torque_gravity
         
    def compute_torque_disturbance(self,time):
        """
        计算外部扰动力矩，此处修改为正弦扰动。
        输出：torque_disturbance - 正弦形式的外部扰动力矩 (joint_num,)
        """
        # 计算当前时刻的扰动力矩值
        disturbance_torque_scalar = self.dist_amp * torch.sin(2 * torch.pi * self.dist_freq * time)

        # 将标量值扩展到所有关节
        torque_disturbance = torch.full((self.joint_num,), disturbance_torque_scalar, device=self.J.device)
        return torque_disturbance

    def compute_torque_delay_compensate(self, cmd_torque,time):
        """
        一阶滞后模型补偿控制延迟：tau_delay * d(T_cmd_delay)/dt + T_cmd_delay = T_cmd
        离散化（欧拉法）：T_cmd_delay(k) = (dt/(tau_delay + dt)) * T_cmd(k) + (tau_delay/(tau_delay + dt)) * T_cmd_delay(k-1)
        输入：cmd_torque - 当前指令力矩 (batch_size, joint_num)
        输出：T_cmd_delay - 延迟后的指令力矩（与输入维度一致）
        """
        cmd_torque = cmd_torque.float()
        
        # 离散化计算延迟后力矩
        alpha = time / (self.tau_delay + time)
        T_cmd_delay = alpha * cmd_torque + (1 - alpha) * self.prev_cmd_torque
        
        # 更新上一时刻状态
        self.prev_cmd_torque = T_cmd_delay.detach()  # 避免计算图累积
        
        return T_cmd_delay