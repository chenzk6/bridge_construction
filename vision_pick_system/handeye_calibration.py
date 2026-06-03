import numpy as np
from pathlib import Path
import cv2
from camera_capture import CameraCapture
from pixel_to_camera import load_intrinsics, load_table_plane, pixel_to_camera_on_tilted_table


def estimate_rigid_transform(camera_points_mm, robot_points_mm):
    """
    根据 N 组 3D 点对应关系估计 T_base_cam (4x4)
    camera_points_mm: Nx3 相机坐标系点
    robot_points_mm:  Nx3 机器人基坐标系点
    """
    A = np.asarray(camera_points_mm, dtype=np.float64)
    B = np.asarray(robot_points_mm, dtype=np.float64)
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("输入点形状必须为 Nx3，且两组点数量一致")
    if A.shape[0] < 3:
        raise ValueError("至少需要 3 对点")

    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    AA = A - ca
    BB = B - cb

    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def save_transform(T, path="data/T_base_cam.npy"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, T)
    print(f"saved: {path}")


def load_transform(path="data/T_base_cam.npy"):
    return np.load(path)


def _chessboard_inner_corners(gray, board_size=(7, 7)):
    """
    检测棋盘，返回 4 个内角的像素坐标和所有角点。
    内角 = 棋盘内侧 2x2 格子的 4 个角点（不包括最外边框）。
    
    例如 7x7 棋盘（49 个角点）：
    - 最外侧有 28 个角点（边框）
    - 内侧 5x5 有 25 个角点
    - 内侧 2x2 = 4 个关键角点，在第 [1,1], [1,5], [5,1], [5,5]（0-indexed）
    
    返回: (corners_4, all_corners) 或 (None, None)
      corners_4: 4x2 数组，顺序为 [左上, 右上, 左下, 右下]
      all_corners: Nx2 所有角点
    """
    bw, bh = board_size
    found, corners = cv2.findChessboardCorners(gray, board_size)
    if not found:
        return None, None

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term).reshape(-1, 2)

    # 4 个内角坐标（跳过最外框，取倒数第二圈）
    # [1,1] 左上, [1, bw-2] 右上, [bh-2, 1] 左下, [bh-2, bw-2] 右下
    idx_tl = 1 * bw + 1              # top-left
    idx_tr = 1 * bw + (bw - 2)       # top-right
    idx_bl = (bh - 2) * bw + 1       # bottom-left
    idx_br = (bh - 2) * bw + (bw - 2)  # bottom-right

    corners_4 = np.array([
        corners[idx_tl],  # 左上
        corners[idx_tr],  # 右上
        corners[idx_bl],  # 左下
        corners[idx_br],  # 右下
    ], dtype=np.float32)

    return corners_4, corners


def chessboard_inner_corners_3d(board_size=(7, 7), square_size_mm=25.0):
    """
    棋盘坐标系下的 4 个内角 3D 坐标（z=0）。
    """
    bw, bh = board_size
    # 内框位置（倒数第二圈）
    x1, y1 = 1 * square_size_mm, 1 * square_size_mm
    x2 = (bw - 2) * square_size_mm
    y2 = (bh - 2) * square_size_mm

    return np.array([
        [x1, y1, 0.0],  # 左上
        [x2, y1, 0.0],  # 右上
        [x1, y2, 0.0],  # 左下
        [x2, y2, 0.0],  # 右下
    ], dtype=np.float64)


def _save_pairs_npz(save_path, camera_points_mm, robot_points_mm):
    A = np.asarray(camera_points_mm, dtype=np.float64)
    B = np.asarray(robot_points_mm, dtype=np.float64)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, camera_points_mm=A, robot_points_mm=B)
    print(f"saved: {save_path}, total pairs={len(A)}")


def collect_handeye_pairs_inner_corners(
    camera_ip="192.168.1.88",
    board_size=(7, 7),
    square_size_mm=25.0,
    intr_path="data/camera_intrinsics.npz",
    plane_path="data/table_plane_from_pnp.npz",
    save_path="data/handeye_pairs_inner_corners.npz",
    num_samples=12,
    window_size=(1080, 720),
):
    """
    采集棋盘 4 个内角点对。
    每次采样时直接输入 4 个角的机器人坐标。
    """
    K, dist = load_intrinsics(intr_path)
    n, d = load_table_plane(plane_path)

    camera_points_mm = []
    robot_points_mm = []

    # 可选：若已有历史文件则继续追加
    if Path(save_path).exists():
        old = np.load(save_path)
        camera_points_mm = list(old["camera_points_mm"])
        robot_points_mm = list(old["robot_points_mm"])
        print(f"检测到已有数据: {save_path}, existing pairs={len(camera_points_mm)}")

    with CameraCapture(camera_ip=camera_ip) as cam:
        cv2.namedWindow("handeye_inner_corners_collect", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("handeye_inner_corners_collect", window_size[0], window_size[1])

        i = len(camera_points_mm) // 4
        while i < num_samples:
            frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners_4, all_corners = _chessboard_inner_corners(gray, board_size)

            vis = frame.copy()
            if corners_4 is not None:
                for p in all_corners:
                    cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 180, 0), -1)

                colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0)]
                labels = ["TL", "TR", "BL", "BR"]
                for p, color, label in zip(corners_4, colors, labels):
                    u, v = float(p[0]), float(p[1])
                    cv2.circle(vis, (int(u), int(v)), 10, color, -1)

                # 左上角统一显示像素坐标（字体更大）
                cv2.putText(
                    vis, f"sample {i+1}/{num_samples}  press c to capture",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
                )

                y0 = 80
                dy = 40
                for k, (label, p, color) in enumerate(zip(labels, corners_4, colors)):
                    u, v = float(p[0]), float(p[1])
                    cv2.putText(
                        vis, f"{label}: ({u:.1f}, {v:.1f})",
                        (20, y0 + k * dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2
                    )
            else:
                cv2.putText(vis, "chessboard not found",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            show = cv2.resize(vis, window_size, interpolation=cv2.INTER_AREA)
            cv2.imshow("handeye_inner_corners_collect", show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key != ord("c"):
                continue
            if corners_4 is None:
                print("未检测到棋盘，跳过")
                continue

            p_cam_list = []
            for u, v in corners_4:
                p_cam_list.append(pixel_to_camera_on_tilted_table(u, v, K, dist, n, d))

            print(f"\n[{i+1}] 当前像素坐标:")
            for name, (u, v) in zip(["TL", "TR", "BL", "BR"], corners_4):
                print(f"  {name}: ({float(u):.2f}, {float(v):.2f})")

            print(f"[{i+1}] 依次输入 4 个角的机器人坐标 (TL, TR, BL, BR):")
            robot_points_this_sample = []
            corner_names = ["TL(左上)", "TR(右上)", "BL(左下)", "BR(右下)"]

            valid = True
            for corner_name in corner_names:
                raw = input(f"  {corner_name} x y z(mm): ").strip()
                try:
                    x, y, z = [float(t) for t in raw.split()]
                    robot_points_this_sample.append(np.array([x, y, z], dtype=np.float64))
                except Exception:
                    print(f"  {corner_name} 输入错误，本次采样作废")
                    valid = False
                    break

            if not valid or len(robot_points_this_sample) != 4:
                print("本次采样已取消\n")
                continue

            camera_points_mm.extend(p_cam_list)
            robot_points_mm.extend(robot_points_this_sample)
            i += 1

            # 关键：每次采样完成后立刻保存
            _save_pairs_npz(save_path, camera_points_mm, robot_points_mm)
            print(f"✓ 已记录 {i}/{num_samples}\n")

    cv2.destroyAllWindows()
    return np.asarray(camera_points_mm), np.asarray(robot_points_mm)


def fit_handeye_from_pairs(
    pairs_path="data/handeye_pairs_inner_corners.npz",
    save_path="data/T_base_cam.npy",
):
    """拟合手眼变换"""
    d = np.load(pairs_path)
    A = d["camera_points_mm"]
    B = d["robot_points_mm"]
    T = estimate_rigid_transform(A, B)
    save_transform(T, save_path)

    R = T[:3, :3]
    t = T[:3, 3]
    pred = (R @ A.T).T + t
    err = np.linalg.norm(pred - B, axis=1)
    print(f"pairs={len(A)}, mean_err={err.mean():.3f} mm, max_err={err.max():.3f} mm")
    return T


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"

    if mode == "collect":
        collect_handeye_pairs_inner_corners(num_samples=12)
    elif mode == "fit":
        fit_handeye_from_pairs()
    else:
        print("用法: python3 handeye_calibration.py [collect|fit]")