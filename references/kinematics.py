import math
import numpy as np
import torch


class Kinematic:
    def __init__(self, leg_length,device='cuda'):
        self.l0 = leg_length[0]
        self.l1 = leg_length[1]
        self.l2 = leg_length[2]
        self.hip_offset = leg_length[3]
        self.knee_offset = leg_length[4]
        self.device = device
        # 初始化mirror_coe
        self.mirror_coe = torch.ones(6,device=self.device)  # 创建一个长度为6的全1张量
        # self.mirror_coe = torch.ones(6)  # 创建一个长度为6的全1张量
        self.mirror_coe[[3, 4, 5]] = -1  # 设置第0, 1和5号腿的mirror_coe为-1

        # 初始化offset_coe
        self.offset_coe = torch.ones(6,device=self.device)  # 创建一个长度为6的全1张量
        # self.offset_coe = torch.ones(6)  # 创建一个长度为6的全1张量
        self.offset_coe[[0,3]] = -1  # 

    def forward_kinematic(self, q):
        # q的形状为[n, 6, 3]
        theta0 = q[:, :, 0]
        theta1 = q[:, :, 1]
        theta2 = q[:, :, 2]

        l = self.l0 + self.l1 * torch.sin(theta1) + self.l2 * torch.sin(theta2 - theta1)
        x = l * torch.sin(theta0)+self.offset_coe*self.knee_offset*torch.cos(theta0)
        y = self.mirror_coe* (l * torch.cos(theta0)-self.offset_coe*self.knee_offset*torch.sin(theta0))
        z = self.l1 * torch.cos(theta1) - self.l2 * torch.cos(theta2 - theta1) - self.hip_offset

        return torch.stack([x, y, z], dim=2)  # 输出形状为[n, 6, 3]

    def inverse_kinematic(self, p):
        # p的形状为[n, 6, 3]
        pos = p.clone()
        pos[:,:,2] += self.hip_offset  # 偏移量
        q00 = torch.atan2(pos[:, :, 0], self.mirror_coe*pos[:, :, 1])
        K0 = torch.sqrt(torch.square(pos[:, :, 0]) + torch.square(pos[:, :, 1]))
        q0 = q00 - self.offset_coe*torch.asin(self.knee_offset/K0)
        K = torch.sqrt(torch.square(K0) - self.knee_offset**2) - self.l0 # 计算K的值
        # K = torch.sqrt(torch.square(pos[:, :, 0]) + torch.square(pos[:, :, 1])) - self.l0
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


if __name__ == "__main__":
    # 5 个参数表示 3 个连杆长度、髋关节和膝盖偏移
    device = 'cuda'
    leg_length = [0.15, 0.13, 0.232, 0.034, 0.0535]
    kinematic = Kinematic(leg_length,device=device)

    # 示例电机角度数据，形状为 (2, 6, 3)，表示 2 个机器人，每个机器人有 6 条腿，每条腿有 3 个关节
    q = torch.tensor([[[0.0, 0.3,0.5],[0.0, 0.3,0.5],[0.0, 0.3,0.5],[0.0, 0.3,0.5],[0.0, 0.3,0.5],[0.0, 0.3,0.5]]], dtype=torch.float32, device=device)
    p = torch.tensor([[[-0.0535,  0.0000, -0.1700],
        [ 0.0535,  0.0000, -0.1700],
        [-0.0462,  0.0157, -0.1376],
        [-0.1532, -0.0157, -0.1376],
        [-0.0462, -0.0157, -0.1376],
        [ 0.0535,  0.0000, -0.1700]]],device=device)
    # 调用前向运动学函数计算脚的位置
    foot_position = kinematic.forward_kinematic(q)
    print("Foot position:\n", foot_position)
    q_inverse = kinematic.inverse_kinematic(p)
    print("Inverse kinematics result:\n", q_inverse)
