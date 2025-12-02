import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -------------------------- 1. 修复PD控制器：添加偏差限幅+注释说明 --------------------------
def pd_controller(desired, actual, Kp, Kd, prev_error=0, dt=0.001, error_limit=5.0):
    """
    修复版PD控制器：添加偏差限幅，避免微分项冲击
    :param error_limit: 偏差限幅（超过此值则截断，防止补偿过大）
    """
    # 1. 计算当前偏差，并限幅（避免阶跃时偏差突变）
    current_error = desired - actual
    current_error = np.clip(current_error, -error_limit, error_limit)  # 偏差限制在±5N·m
    
    # 2. 计算比例项（安全范围，不超限）
    proportional_term = Kp * current_error
    proportional_term = np.clip(proportional_term, -10.0, 10.0)  # 比例项限幅±10N·m
    
    # 3. 计算微分项（用小步长平滑，避免突变）
    derivative_term = Kd * (current_error - prev_error) / dt
    derivative_term = np.clip(derivative_term, -5.0, 5.0)  # 微分项限幅±5N·m
    
    # 4. 总补偿力矩（再次限幅，确保物理合理性）
    pd_compensate = proportional_term + derivative_term
    pd_compensate = np.clip(pd_compensate, -15.0, 15.0)  # 总补偿限幅±15N·m（适配6N·m输入）
    
    return pd_compensate, current_error

# 一阶滞后模型（保持不变）
def first_order_lag_vectorized(tau_in_array, tau_delay, dt, init_output=0.0):
    n_steps = len(tau_in_array)
    tau_out_array = np.zeros_like(tau_in_array)
    alpha = tau_delay / (tau_delay + dt) if (tau_delay + dt) != 0 else 0.9
    tau_out_array[0] = alpha * init_output + (1 - alpha) * tau_in_array[0]
    for k in range(1, n_steps):
        tau_out_array[k] = alpha * tau_out_array[k-1] + (1 - alpha) * tau_in_array[k]
    return tau_out_array

# -------------------------- 2. 关键：降低Kp，适配Kd=0.1 --------------------------
tau_delay = 0.08  # 一阶滞后时间常数（不变）
dt = 0.005        # 仿真步长（不变）
sim_steps = 5000  # 总步数（5000×0.005=25秒，不变）
t = np.arange(sim_steps) * dt  # 时间序列

# 修复后的PD参数（Kp从15→3，Kd=0.1不变，适配当前场景）
Kp = 3.0    # 关键修改：降低比例系数，避免震荡溢出
Kd = 0.1    # 保持你的需求值
error_limit = 5.0  # 偏差限幅（与PD控制器一致）

# -------------------------- 3. 构造输入力矩（不变） --------------------------
tau_in_array = np.zeros(sim_steps)
for i in range(sim_steps):
    time = t[i]
    if time < 3.0:
        tau_in_array[i] = 0.0
    elif 3.0 <= time < 8.0:
        freq = 0.5 + (5 - 0.5) * (time - 3.0) / 5.0
        tau_in_array[i] = 2 * np.sin(2 * np.pi * freq * time)
    elif 8.0 <= time < 13.0:
        tau_in_array[i] = 6.0
    elif 13.0 <= time < 18.0:
        tau_in_array[i] = 2.0
    else:
        tau_in_array[i] = 0.0

# -------------------------- 4. 核心仿真：添加力矩限幅（关键修复） --------------------------
tau_out_lag = np.zeros(sim_steps)
pd_compensate_array = np.zeros(sim_steps)
prev_error = 0.0  # 初始化偏差为0，避免初始冲击

# 电机物理力矩上限（根据实际场景设置，这里设为±10N·m，覆盖输入+补偿）
max_torque = 10.0  

for k in range(sim_steps):
    desired_torque = tau_in_array[k]
    actual_torque = tau_out_lag[k-1] if k > 0 else 0.0  # 第一步实际输出为0
    
    # 1. PD补偿（调用修复版控制器）
    pd_compensate, current_error = pd_controller(
        desired=desired_torque,
        actual=actual_torque,
        Kp=Kp,
        Kd=Kd,
        prev_error=prev_error,
        dt=dt,
        error_limit=error_limit
    )
    pd_compensate_array[k] = pd_compensate
    prev_error = current_error  # 更新偏差
    
    # 2. 补偿后的输入力矩（添加物理限幅，关键！）
    compensated_input = desired_torque + pd_compensate
    compensated_input = np.clip(compensated_input, -max_torque, max_torque)  # 限制在±10N·m
    
    # 3. 一阶滞后计算（确保输入在合理范围）
    alpha = tau_delay / (tau_delay + dt) if (tau_delay + dt) != 0 else 0.9
    if k == 0:
        tau_out_lag[k] = alpha * 0.0 + (1 - alpha) * compensated_input
    else:
        tau_out_lag[k] = alpha * tau_out_lag[k-1] + (1 - alpha) * compensated_input
    # 输出力矩也限幅（避免滞后模型输出异常）
    tau_out_lag[k] = np.clip(tau_out_lag[k], -max_torque, max_torque)

# -------------------------- 5. 绘图：避免无穷大值导致的矩阵错误 --------------------------
# 先清理异常值（防止残留inf/nan）
tau_out_lag = np.nan_to_num(tau_out_lag, nan=0.0, posinf=max_torque, neginf=-max_torque)
pd_compensate_array = np.nan_to_num(pd_compensate_array, nan=0.0, posinf=15.0, neginf=-15.0)
tracking_error = tau_in_array - tau_out_lag
tracking_error = np.nan_to_num(tracking_error, nan=0.0, posinf=5.0, neginf=-5.0)

# 正常绘图（不变）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.figure(figsize=(14, 10))

ax1 = plt.subplot(3, 1, 1)
ax1.plot(t, tau_in_array, label='Desired Input Torque', linestyle='--', color='#1f77b4', linewidth=1.5)
ax1.plot(t, tau_out_lag, label='Actual Output Torque (PD + 1st-Order Lag)', color='#2ca02c', linewidth=2)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Torque (N·m)', fontsize=12)
ax1.set_title(f'Tracking Effect (Kp={Kp}, Kd={Kd}, τ_delay={tau_delay}s)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 25)
ax1.set_ylim(-12, 12)  # 适配限幅后的力矩范围

ax2 = plt.subplot(3, 1, 2)
ax2.plot(t, pd_compensate_array, label='PD Compensation Torque', color='#d62728', linewidth=1.5)
ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('PD Compensation Torque (N·m)', fontsize=12)
ax2.set_title('PD Controller Compensation Torque', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 25)
ax2.set_ylim(-18, 18)  # 适配补偿力矩限幅

ax3 = plt.subplot(3, 1, 3)
ax3.plot(t, tracking_error, label='Tracking Error (Desired - Actual)', color='#9467bd', linewidth=1.5)
ax3.axhline(y=0, color='red', linestyle=':', alpha=0.7)
ax3.fill_between(t, tracking_error, 0, where=(tracking_error >= 0), color='green', alpha=0.1, label='Positive Error')
ax3.fill_between(t, tracking_error, 0, where=(tracking_error < 0), color='red', alpha=0.1, label='Negative Error')
ax3.set_xlabel('Time (s)', fontsize=12)
ax3.set_ylabel('Tracking Error (N·m)', fontsize=12)
ax3.set_title(f'Tracking Error (Max Error: {np.max(np.abs(tracking_error)):.3f}N·m)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 25)
ax3.set_ylim(-6, 6)

plt.tight_layout()
plt.savefig('fixed_pd_lag_simulation.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- 6. 量化输出（不变） --------------------------
print("=== 修复后PD控制跟踪效果 ===")
print(f"PD参数：Kp={Kp}, Kd={Kd}")
max_error = np.max(np.abs(tracking_error))
rms_error = np.sqrt(np.mean(tracking_error**2))
sin_mask = (t >= 3.0) & (t <= 8.0)
step1_mask = (t >= 8.0) & (t <= 13.0)
step2_mask = (t >= 13.0) & (t <= 18.0)

print(f"整体最大跟踪误差：{max_error:.3f}N·m")
print(f"整体均方根误差（RMS）：{rms_error:.3f}N·m")
print(f"正弦扫频阶段最大误差：{np.max(np.abs(tracking_error[sin_mask])):.3f}N·m")
print(f"6N·m阶跃阶段最大误差：{np.max(np.abs(tracking_error[step1_mask])):.3f}N·m")