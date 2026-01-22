import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import env.ikfastpy.LH_ikFast as LH_ikFast
import matplotlib.pyplot as plt

physics_client = bc.BulletClient(connection_mode=p.DIRECT)  # 无GUI模式
lh_kin = LH_ikFast.PyKinematics()

robot_id = physics_client.loadURDF(
    "./env/linghou_description/urdf/LingHouUrdf3.urdf",
    [0.7, 0.6, 0.005],
    [0.0, 0.0, 1.0, 0.0],
    useFixedBase=1,
)

actual_base_pos, actual_base_orn = physics_client.getBasePositionAndOrientation(
    robot_id
)
base_rot = np.array(physics_client.getMatrixFromQuaternion(actual_base_orn)).reshape(
    3, 3
)

print("=" * 60)
print("检查坐标系对齐 - 多姿态统计")
print("=" * 60)

# 生成100个随机姿态
np.random.seed(42)
n_samples = 100
pos_errors = []
rot_errors = []

for i in range(n_samples):
    # 随机关节角度
    angles = np.random.uniform([-2, -2, -3, -3, -2, -6], [2, 1, 0, 3, 2, 6])

    # PyBullet FK
    for j, angle in enumerate(angles):
        physics_client.resetJointState(robot_id, j, angle)
    physics_client.stepSimulation()

    ee_state = physics_client.getLinkState(robot_id, 7)
    pybullet_pos_world = np.array(ee_state[0])
    pybullet_orn_world = np.array(ee_state[1])

    pybullet_pos_base = base_rot.T @ (pybullet_pos_world - np.array(actual_base_pos))
    pybullet_rot_base = base_rot.T @ np.array(
        physics_client.getMatrixFromQuaternion(pybullet_orn_world)
    ).reshape(3, 3)

    # IKFast FK
    fk_result = lh_kin.forward(angles.tolist())
    ikfast_pose = np.reshape(fk_result, [3, 4])
    ikfast_pos = ikfast_pose[:, -1]
    ikfast_rot = ikfast_pose[:, :3]

    # 计算误差
    pos_error = np.linalg.norm(pybullet_pos_base - ikfast_pos)
    rot_error = np.linalg.norm(pybullet_rot_base - ikfast_rot, "fro")

    pos_errors.append(pos_error)
    rot_errors.append(rot_error)

pos_errors = np.array(pos_errors)
rot_errors = np.array(rot_errors)

print(f"\n位置误差统计 (米):")
print(f"  平均: {pos_errors.mean():.6f} ({pos_errors.mean()*1000:.2f} mm)")
print(f"  中位数: {np.median(pos_errors):.6f} ({np.median(pos_errors)*1000:.2f} mm)")
print(f"  最大: {pos_errors.max():.6f} ({pos_errors.max()*1000:.2f} mm)")
print(f"  最小: {pos_errors.min():.6f} ({pos_errors.min()*1000:.2f} mm)")
print(f"  标准差: {pos_errors.std():.6f} ({pos_errors.std()*1000:.2f} mm)")

print(f"\n旋转误差统计:")
print(f"  平均: {rot_errors.mean():.6f}")
print(f"  中位数: {np.median(rot_errors):.6f}")
print(f"  最大: {rot_errors.max():.6f}")
print(f"  最小: {rot_errors.min():.6f}")
print(f"  标准差: {rot_errors.std():.6f}")

print(f"\n判断:")
if pos_errors.mean() < 0.001:
    print("✓✓ 坐标系完美对齐 (平均位置误差 <1mm)")
    print("  建议: 直接使用,无需标定")
elif pos_errors.mean() < 0.01:
    print("✓ 坐标系基本对齐 (平均位置误差 <1cm)")
    print("  建议: 可以使用,但建议进行标定以提高精度")
elif pos_errors.mean() < 0.05:
    print("⚠ 坐标系有偏差 (平均位置误差 1-5cm)")
    print("  建议: 需要标定修正")
else:
    print("✗ 坐标系严重不对齐 (平均位置误差 >5cm)")
    print("  建议: 检查 URDF 或重新生成 IKFast")

# 绘制误差分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(pos_errors * 1000, bins=30, edgecolor="black")
plt.xlabel("位置误差 (毫米)")
plt.ylabel("频数")
plt.title("位置误差分布")
plt.axvline(
    pos_errors.mean() * 1000,
    color="r",
    linestyle="--",
    label=f"平均: {pos_errors.mean()*1000:.2f}mm",
)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(rot_errors, bins=30, edgecolor="black")
plt.xlabel("旋转误差 (Frobenius范数)")
plt.ylabel("频数")
plt.title("旋转误差分布")
plt.axvline(
    rot_errors.mean(), color="r", linestyle="--", label=f"平均: {rot_errors.mean():.4f}"
)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("coordinate_alignment_check.png", dpi=150)
print(f"\n误差分布图已保存到: coordinate_alignment_check.png")
