import argparse
from pathlib import Path

import cv2
import numpy as np

from camera_capture import CameraCapture
from pixel_to_camera import load_intrinsics


ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


def make_aruco_detector(dict_name):
    if dict_name not in ARUCO_DICT_MAP:
        raise ValueError(f"aruco_dict must be one of {list(ARUCO_DICT_MAP.keys())}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[dict_name])
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def detect_marker_center(frame_bgr, detector, marker_id=None, marker_ids=None):
    corners, ids, _ = detector.detectMarkers(frame_bgr)
    if ids is None or len(ids) == 0:
        return None, None, None

    ids_flat = ids.flatten().tolist()
    pick_idx = None
    if marker_ids is not None:
        for i, mid in enumerate(ids_flat):
            if int(mid) in marker_ids:
                pick_idx = i
                break
    elif marker_id is None:
        pick_idx = 0
    else:
        for i, mid in enumerate(ids_flat):
            if int(mid) == int(marker_id):
                pick_idx = i
                break

    if pick_idx is None:
        return None, ids_flat, None

    marker_corners = corners[pick_idx][0].astype(np.float32)
    center_uv = marker_corners.mean(axis=0)
    return center_uv, ids_flat, marker_corners


def detect_chessboard_corner(frame_bgr, board_size=(7, 7), corner_row=None, corner_col=None):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, board_size)
    if not found:
        return None, None, None

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term).reshape(-1, 2)

    bw, bh = board_size
    if corner_row is None:
        corner_row = bh // 2
    if corner_col is None:
        corner_col = bw // 2
    if not (0 <= corner_row < bh and 0 <= corner_col < bw):
        raise ValueError(f"corner_row/col out of range, board_size={board_size}")

    idx = int(corner_row) * bw + int(corner_col)
    selected_uv = corners[idx].astype(np.float32)
    return selected_uv, corners, (int(corner_row), int(corner_col))


def undistort_pixel_points(points_uv, K, dist):
    pts = np.asarray(points_uv, dtype=np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1, 2).astype(np.float64)


def collect_xy_samples(
    camera_ip="192.168.1.88",
    intr_path="data/camera_intrinsics.npz",
    save_path="data/xy_plane_samples.npz",
    target_type="aruco",
    aruco_dict="DICT_4X4_50",
    aruco_id=None,
    aruco_ids=None,
    board_size=(7, 7),
    corner_row=None,
    corner_col=None,
    num_samples=12,
    window_size=(1080, 720),
):
    K, dist = load_intrinsics(intr_path)
    detector = make_aruco_detector(aruco_dict)

    pixel_points = []
    robot_xy = []
    robot_z = []

    if Path(save_path).exists():
        old = np.load(save_path)
        pixel_points = list(old["pixel_points_uv"])
        robot_xy = list(old["robot_points_xy"])
        if "robot_z_mm" in old:
            robot_z = list(old["robot_z_mm"])
        print(f"检测到已有数据: {save_path}, existing pairs={len(pixel_points)}")

    with CameraCapture(camera_ip=camera_ip) as cam:
        cv2.namedWindow("xy_plane_collect", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("xy_plane_collect", window_size[0], window_size[1])

        i = len(pixel_points)
        while i < num_samples:
            frame = cam.read()
            center_uv = None
            ids_seen = None
            marker_corners = None
            chess_corners = None
            corner_rc = None
            if target_type == "aruco":
                center_uv, ids_seen, marker_corners = detect_marker_center(
                    frame, detector, aruco_id, aruco_ids
                )
            else:
                center_uv, chess_corners, corner_rc = detect_chessboard_corner(
                    frame,
                    board_size=board_size,
                    corner_row=corner_row,
                    corner_col=corner_col,
                )

            vis = frame.copy()
            if marker_corners is not None or chess_corners is not None:
                if target_type == "aruco":
                    draw_pts = marker_corners.astype(np.int32)
                    cv2.polylines(vis, [draw_pts], True, (0, 255, 0), 2)
                    if aruco_ids is not None:
                        label = f"ids={sorted(list(aruco_ids))}"
                    else:
                        label = f"id={aruco_id}" if aruco_id is not None else "marker"
                else:
                    for p in chess_corners:
                        cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 180, 0), -1)
                    label = f"corner(r={corner_rc[0]}, c={corner_rc[1]})"

                cx, cy = center_uv.astype(int)
                cv2.circle(vis, (int(cx), int(cy)), 7, (0, 0, 255), -1)
                cv2.putText(
                    vis,
                    f"{label} center=({center_uv[0]:.1f}, {center_uv[1]:.1f})",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                status = "target found | press c to capture"
                status_color = (0, 255, 255)
            else:
                if target_type == "aruco" and ids_seen:
                    status = f"target id={aruco_id} not found, seen={ids_seen}"
                else:
                    status = "target not found"
                status_color = (0, 0, 255)

            cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(vis, f"sample {i+1}/{num_samples}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(vis, f"target={target_type}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(vis, "q: quit  c: capture", (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("xy_plane_collect", cv2.resize(vis, window_size, interpolation=cv2.INTER_AREA))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key != ord("c"):
                continue
            if center_uv is None:
                print("未检测到目标，跳过")
                continue

            raw = input(f"[{i+1}] 输入 robot x y [z] (mm): ").strip()
            try:
                vals = [float(v) for v in raw.split()]
            except Exception:
                print("输入格式错误，跳过")
                continue

            if len(vals) not in (2, 3):
                print("请输入 2 个或 3 个数字，例如: 320 -145 或 320 -145 25")
                continue

            pixel_points.append(np.array(center_uv, dtype=np.float64))
            robot_xy.append(np.array(vals[:2], dtype=np.float64))
            if len(vals) == 3:
                robot_z.append(float(vals[2]))
            i += 1
            save_xy_samples(save_path, pixel_points, robot_xy, robot_z)
            print(f"已记录 {i}/{num_samples}")

    cv2.destroyAllWindows()
    return np.asarray(pixel_points), np.asarray(robot_xy), np.asarray(robot_z, dtype=np.float64)


def save_xy_samples(save_path, pixel_points, robot_xy, robot_z):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    arr_z = np.asarray(robot_z, dtype=np.float64) if len(robot_z) > 0 else np.asarray([], dtype=np.float64)
    np.savez(
        save_path,
        pixel_points_uv=np.asarray(pixel_points, dtype=np.float64),
        robot_points_xy=np.asarray(robot_xy, dtype=np.float64),
        robot_z_mm=arr_z,
    )
    print(f"saved: {save_path}, total pairs={len(pixel_points)}")


def fit_xy_mapping(
    samples_path="data/xy_plane_samples.npz",
    intr_path="data/camera_intrinsics.npz",
    save_path="data/xy_plane_homography.npz",
):
    d = np.load(samples_path)
    pixel_uv = np.asarray(d["pixel_points_uv"], dtype=np.float64)
    robot_xy = np.asarray(d["robot_points_xy"], dtype=np.float64)
    if len(pixel_uv) < 4:
        raise RuntimeError("至少需要 4 个采样点才能拟合平面单应矩阵")

    K, dist = load_intrinsics(intr_path)
    und_uv = undistort_pixel_points(pixel_uv, K, dist)
    H, mask = cv2.findHomography(und_uv, robot_xy, method=0)
    if H is None:
        raise RuntimeError("拟合单应矩阵失败")

    pred_xy = map_pixels_to_robot_xy(pixel_uv, K, dist, H)
    err = np.linalg.norm(pred_xy - robot_xy, axis=1)

    z_mm = float(np.mean(d["robot_z_mm"])) if "robot_z_mm" in d and len(d["robot_z_mm"]) > 0 else 0.0
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        H_uv_to_robot_xy=H,
        mean_robot_z_mm=z_mm,
        intr_path=np.asarray(str(intr_path)),
        samples_path=np.asarray(str(samples_path)),
    )
    print(f"saved: {save_path}")
    print(f"pairs={len(pixel_uv)}, mean_err={err.mean():.3f} mm, max_err={err.max():.3f} mm")
    return H, err


def map_pixels_to_robot_xy(pixel_uv, K, dist, H):
    und_uv = undistort_pixel_points(pixel_uv, K, dist).reshape(-1, 1, 2).astype(np.float32)
    pred_xy = cv2.perspectiveTransform(und_uv, H).reshape(-1, 2)
    return pred_xy.astype(np.float64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("collect")
    p1.add_argument("--camera-ip", default="192.168.1.88")
    p1.add_argument("--intr", default="data/camera_intrinsics.npz")
    p1.add_argument("--save", default="data/xy_plane_samples.npz")
    p1.add_argument("--target", default="aruco", choices=["aruco", "chessboard"])
    p1.add_argument("--aruco-dict", default="DICT_4X4_50", choices=sorted(ARUCO_DICT_MAP.keys()))
    p1.add_argument("--aruco-id", type=int, default=None)
    p1.add_argument("--aruco-ids", type=str, default=None)  # 新增：逗号分隔
    p1.add_argument("--bw", type=int, default=7)
    p1.add_argument("--bh", type=int, default=7)
    p1.add_argument("--corner-row", type=int, default=None)
    p1.add_argument("--corner-col", type=int, default=None)
    p1.add_argument("--num", type=int, default=12)

    p2 = sub.add_parser("fit")
    p2.add_argument("--samples", default="data/xy_plane_samples.npz")
    p2.add_argument("--intr", default="data/camera_intrinsics.npz")
    p2.add_argument("--save", default="data/xy_plane_homography.npz")

    args = parser.parse_args()

    if args.cmd == "collect":
        aruco_ids = None
        if args.aruco_ids:
            aruco_ids = {int(v) for v in args.aruco_ids.split(",") if v.strip()}

        collect_xy_samples(
            camera_ip=args.camera_ip,  # 修复拼写
            intr_path=args.intr,
            save_path=args.save,
            target_type=args.target,
            aruco_dict=args.aruco_dict,
            aruco_id=args.aruco_id,
            aruco_ids=aruco_ids,
            board_size=(args.bw, args.bh),
            corner_row=args.corner_row,
            corner_col=args.corner_col,
            num_samples=args.num,
        )
    elif args.cmd == "fit": 
        fit_xy_mapping(
            samples_path=args.samples,
            intr_path=args.intr,
            save_path=args.save,
        )

# python3 xy_plane_calibration.py collect \
#   --target chessboard \
#   --bw 7 --bh 7 \
#   --corner-row 3 --corner-col 3 \
#   --num 12
