import numpy as np


def transform_point(T_4x4, p3):
    p = np.array([p3[0], p3[1], p3[2], 1.0], dtype=np.float64)
    out = T_4x4 @ p
    return out[:3]


def rotmat_to_rpy_zyx(R):
    """
    旋转矩阵 -> RPY(roll, pitch, yaw)，单位 rad
    使用 ZYX 欧拉顺序
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def camera_point_to_robot_pose(
    T_base_cam,
    p_cam_mm,
    fixed_rpy=(3.1415926, 0.0, 0.0),
    z_offset_mm=0.0
):
    """
    相机点 -> 机器人目标位姿 [x,y,z,rx,ry,rz]
    固定末端姿态 fixed_rpy 可按抓手方向调整
    """
    p_base = transform_point(T_base_cam, p_cam_mm)
    x, y, z = p_base
    z = z + z_offset_mm
    rx, ry, rz = fixed_rpy
    return np.array([x, y, z, rx, ry, rz], dtype=np.float64)