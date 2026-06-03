import cv2
import numpy as np
from pathlib import Path
from camera_capture import CameraCapture


def collect_chessboard_images(
    output_dir="data/calib_images",
    camera_ip="192.168.1.88",
    board_size=(7, 7),
    interval_key="s",
    window_size=(1280, 720),  # 窗口大小
):
    """
    按键采集标定板图像：
    - s: 保存当前帧
    - q: 退出
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with CameraCapture(camera_ip=camera_ip) as cam:
        cv2.namedWindow("collect_calib", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("collect_calib", window_size[0], window_size[1])

        idx = 0
        while True:
            frame = cam.read()
            show = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, board_size)
            if found:
                cv2.drawChessboardCorners(show, board_size, corners, found)

            cv2.putText(show, f"[{interval_key}] save  [q] quit  saved={idx}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            win_w, win_h = window_size
            h, w = show.shape[:2]
            scale = min(win_w / w, win_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            disp = cv2.resize(show, (new_w, new_h), interpolation=cv2.INTER_AREA)

            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            y0 = (win_h - new_h) // 2
            x0 = (win_w - new_w) // 2
            canvas[y0:y0+new_h, x0:x0+new_w] = disp

            cv2.imshow("collect_calib", canvas)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(interval_key):
                img_path = out / f"calib_{idx:03d}.png"
                cv2.imwrite(str(img_path), frame)
                idx += 1
                print(f"saved: {img_path}")
            elif k == ord("q"):
                break
    cv2.destroyAllWindows()
    

def calibrate_camera_from_images(
    image_dir="data/calib_images",
    board_size=(7, 7),
    square_size_mm=25.0,
    save_path="data/camera_intrinsics.npz"
):
    """
    相机内参标定，单位 mm（棋盘格边长）。
    """
    image_dir = Path(image_dir)
    imgs = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    if not imgs:
        raise RuntimeError(f"未找到标定图像: {image_dir}")

    # 世界坐标（棋盘平面 Z=0）
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    grid = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size_mm

    obj_points = []
    img_points = []
    img_shape = None

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for p in imgs:
        img = cv2.imread(str(p))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, board_size)
        if not found:
            continue
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        obj_points.append(objp)
        img_points.append(corners2)

    if len(obj_points) < 5:
        raise RuntimeError("有效标定图像太少，至少需要 5 张可检测到棋盘格的图像")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, K=K, dist=dist, reproj_error=ret)
    print(f"标定完成: reproj_error={ret:.4f}")
    print(f"K=\n{K}")
    print(f"dist=\n{dist.ravel()}")
    print(f"saved: {save_path}")
    return K, dist, ret


def estimate_table_z_with_pnp(
    intrinsics_path="data/camera_intrinsics.npz",
    image_path=None,
    camera_ip="192.168.1.88",
    board_size=(7, 7),
    square_size_mm=25.0,
    save_path="data/table_plane_from_pnp.npz",
):
    """
    用棋盘格PnP估计桌面在相机系下的Z（mm）
    - 若 image_path 为 None，则实时采一帧
    - 返回 table_z_mm（角点Z的中位数）
    """
    d = np.load(intrinsics_path)
    K, dist = d["K"], d["dist"]

    if image_path is None:
        with CameraCapture(camera_ip=camera_ip) as cam:
            frame = cam.read()
    else:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"读取图像失败: {image_path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, board_size)
    if not found:
        raise RuntimeError("未检测到棋盘格角点，请更换姿态/光照后重试")

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    grid = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size_mm  # mm, 棋盘平面Z=0

    ok, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP 失败")

    R, _ = cv2.Rodrigues(rvec)

    # 角点从棋盘坐标系 -> 相机坐标系
    Xc = (R @ objp.T + tvec).T  # Nx3
    z_vals = Xc[:, 2]
    table_z_mm = float(np.median(z_vals))

    # 新增：保存桌面平面方程 n^T X + d = 0（相机坐标系）
    n = R[:, 2].reshape(3)              # 棋盘平面法向量在相机系中的方向
    n = n / np.linalg.norm(n)
    p0 = tvec.reshape(3)                # 棋盘原点在相机系中的坐标
    d_plane = -float(n @ p0)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        rvec=rvec, tvec=tvec, R=R,
        z_vals=z_vals, table_z_mm=table_z_mm,
        n=n, d=d_plane
    )

    print(f"table_z_mm (median) = {table_z_mm:.3f} mm")
    print(f"plane: n={n}, d={d_plane:.6f}")
    print(f"z range = [{z_vals.min():.3f}, {z_vals.max():.3f}] mm")
    print(f"saved: {save_path}")
    return table_z_mm


if __name__ == "__main__":
    # 1) 先采图：python camera_calibration.py collect
    # 2) 再标定：python camera_calibration.py calibrate
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "calibrate"
    if mode == "collect":
        collect_chessboard_images()
    elif mode == "table_z":
        estimate_table_z_with_pnp()
    else:
        calibrate_camera_from_images()