# 用于机器人坐标标定
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as p
import pybullet_utils.bullet_client as bc
import env.ikfastpy.LH_ikFast as LH_ikFast

physics_client = bc.BulletClient(connection_mode=p.GUI)
lh_kin = LH_ikFast.PyKinematics()

robot_id = physics_client.loadURDF(
    "/home/xilifeng/bridge_construction/env/linghou_description/urdf/LingHouUrdf3.urdf",
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

print(f"修改后的基座位置: {actual_base_pos}")

# 收集多组数据
samples = []
np.random.seed(42)
for _ in range(100):
    # 随机关节角度
    angles = np.random.uniform([-2, -2, -3, -3, -2, -6], [2, 1, 0, 3, 2, 6])

    # PyBullet FK
    for i, angle in enumerate(angles):
        physics_client.resetJointState(robot_id, i, angle)
    physics_client.stepSimulation()

    ee_state = physics_client.getLinkState(robot_id, 7)
    pybullet_pos = base_rot.T @ (np.array(ee_state[0]) - np.array(actual_base_pos))

    # IKFast FK
    fk_result = lh_kin.forward(angles.tolist())
    ikfast_pos = np.reshape(fk_result, [3, 4])[:, -1]

    samples.append((pybullet_pos, ikfast_pos))

# 使用最小二乘法求解变换
pybullet_points = np.array([s[0] for s in samples])
ikfast_points = np.array([s[1] for s in samples])

# 求解 ikfast = R @ pybullet + t
centroid_pb = pybullet_points.mean(axis=0)
centroid_ik = ikfast_points.mean(axis=0)

H = (pybullet_points - centroid_pb).T @ (ikfast_points - centroid_ik)
U, S, Vt = np.linalg.svd(H)
R = Vt.T @ U.T
t = centroid_ik - R @ centroid_pb

print("\n修改 URDF 后的新标定结果:")
print(f"旋转矩阵 R:")
print(R)
print(f"\n平移向量 t: {t}")

# 验证
errors = []
for pb, ik in samples:
    transformed = R @ pb + t
    error = np.linalg.norm(transformed - ik)
    errors.append(error)

print(f"\n平均误差: {np.mean(errors):.6f} 米")
print(f"最大误差: {np.max(errors):.6f} 米")

print("\n请将以下代码复制到 robots.py 的 LHRobot.run_ik 方法中:")
print("```python")
print(f"R_calib = np.array([")
for row in R:
    print(f"    {list(row)},")
print(f"])")
print(f"t_calib = np.array({list(t)})")
print("```")
