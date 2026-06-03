import cv2
import numpy as np


def load_intrinsics(path="data/camera_intrinsics.npz"):
    d = np.load(path)
    K = d["K"]
    dist = d["dist"]
    return K, dist


def load_table_plane(path="data/table_plane_from_pnp.npz"):
    d = np.load(path)
    n = d["n"].reshape(3).astype(np.float64)
    dd = float(d["d"])
    return n, dd


def pixel_to_camera_on_plane(u, v, K, dist, z_cam_mm):
    """
    将像素点反投影到相机坐标系平面 Z=z_cam_mm（单位 mm）
    返回: np.array([X, Y, Z])
    """
    pts = np.array([[[float(u), float(v)]]], dtype=np.float32)
    und = cv2.undistortPoints(pts, K, dist)  # 归一化坐标 (x', y')
    x_n, y_n = und[0, 0, 0], und[0, 0, 1]

    X = x_n * z_cam_mm
    Y = y_n * z_cam_mm
    Z = float(z_cam_mm)
    return np.array([X, Y, Z], dtype=np.float64)


def pixel_to_camera_on_tilted_table(u, v, K, dist, n, d):
    """
    像素 -> 相机系3D（与平面 n^T X + d = 0 求交）
    """
    pts = np.array([[[float(u), float(v)]]], dtype=np.float32)
    und = cv2.undistortPoints(pts, K, dist)
    x_n, y_n = float(und[0, 0, 0]), float(und[0, 0, 1])

    ray = np.array([x_n, y_n, 1.0], dtype=np.float64)  # 相机原点出发射线方向
    denom = float(n @ ray)
    if abs(denom) < 1e-9:
        raise RuntimeError("射线与桌面近平行，无法求交")

    t = -d / denom
    if t <= 0:
        raise RuntimeError("交点在相机后方，请检查平面法向方向/标定结果")

    p_cam = t * ray
    return p_cam.astype(np.float64)