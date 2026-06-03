import argparse
from pathlib import Path
import numpy as np
import cv2

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


def euler_xyz_to_R(rx, ry, rz):
    """XYZ欧拉角(roll, pitch, yaw) -> 旋转矩阵，单位: 弧度"""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def chessboard_obj_points(board_size=(7, 7), square_size_mm=25.0):
    bw, bh = board_size
    objp = np.zeros((bw * bh, 3), np.float64)
    grid = np.mgrid[0:bw, 0:bh].T.reshape(-1, 2)
    objp[:, :2] = grid * float(square_size_mm)
    return objp


def _aruco_detector(dict_name):
    if dict_name not in ARUCO_DICT_MAP:
        raise ValueError(f"aruco_dict must be one of {list(ARUCO_DICT_MAP.keys())}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[dict_name])
    return cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())


def detect_chessboard_target_pose(frame_bgr, K, dist, board_size=(7, 7), square_size_mm=25.0):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, board_size)
    if not found:
        return None, None, None, {}

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term).reshape(-1, 2)

    objp = chessboard_obj_points(board_size, square_size_mm)
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, None, {}

    R_t2c, _ = cv2.Rodrigues(rvec)  # target -> cam
    t_t2c = tvec.reshape(3)
    return R_t2c, t_t2c, corners, {"target_name": "chessboard", "rvec": rvec, "obj_points": objp}


def detect_aruco_target_pose(frame_bgr, K, dist, detector, marker_length_mm, marker_id=None):
    corners, ids, _ = detector.detectMarkers(frame_bgr)
    if ids is None or len(ids) == 0:
        return None, None, None, {}

    ids_flat = ids.flatten().tolist()
    pick_idx = None
    if marker_id is None:
        pick_idx = 0
    else:
        for i, mid in enumerate(ids_flat):
            if int(mid) == int(marker_id):
                pick_idx = i
                break
    if pick_idx is None:
        return None, None, None, {"detected_ids": ids_flat}

    marker_length_m = float(marker_length_mm) / 1000.0
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_length_m, K, dist
    )
    rvec = rvecs[pick_idx, 0]
    tvec = tvecs[pick_idx, 0] * 1000.0  # m -> mm
    R_t2c, _ = cv2.Rodrigues(rvec)
    return R_t2c, tvec.reshape(3), corners[pick_idx][0], {
        "target_name": f"aruco id={ids_flat[pick_idx]}",
        "rvec": rvec.reshape(3, 1),
        "marker_id": int(ids_flat[pick_idx]),
        "detected_ids": ids_flat,
    }


def detect_target_pose(
    frame_bgr,
    K,
    dist,
    target_type="chessboard",
    board_size=(7, 7),
    square_size_mm=25.0,
    aruco_detector=None,
    marker_length_mm=40.0,
    marker_id=None,
):
    if target_type == "chessboard":
        return detect_chessboard_target_pose(
            frame_bgr, K, dist, board_size=board_size, square_size_mm=square_size_mm
        )
    if target_type == "aruco":
        if aruco_detector is None:
            raise ValueError("aruco target requires a detector")
        return detect_aruco_target_pose(
            frame_bgr,
            K,
            dist,
            detector=aruco_detector,
            marker_length_mm=marker_length_mm,
            marker_id=marker_id,
        )
    raise ValueError("target_type must be 'chessboard' or 'aruco'")


def collect_samples(
    camera_ip="192.168.1.88",
    intr_path="data/camera_intrinsics.npz",
    save_path="data/handeye_eye_to_hand_samples.npz",
    target_type="chessboard",
    board_size=(7, 7),
    square_size_mm=25.0,
    aruco_dict="DICT_4X4_50",
    marker_length_mm=40.0,
    marker_id=None,
    num_samples=20,
    angles_in_degree=False,
):
    K, dist = load_intrinsics(intr_path)
    detector = _aruco_detector(aruco_dict) if target_type == "aruco" else None

    R_base2gripper = []
    t_base2gripper = []
    R_target2cam = []
    t_target2cam = []

    with CameraCapture(camera_ip=camera_ip) as cam:
        cv2.namedWindow("collect_eye_to_hand", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("collect_eye_to_hand", 960, 540)

        i = 0
        while i < num_samples:
            frame = cam.read()
            R_t2c, t_t2c, corners, info = detect_target_pose(
                frame,
                K,
                dist,
                target_type=target_type,
                board_size=board_size,
                square_size_mm=square_size_mm,
                aruco_detector=detector,
                marker_length_mm=marker_length_mm,
                marker_id=marker_id,
            )

            vis = frame.copy()
            if corners is not None:
                if target_type == "chessboard":
                    for p in corners:
                        cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 200, 0), -1)
                else:
                    cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 255, 0), 2)
                    cxy = corners.mean(axis=0).astype(int)
                    label = info.get("target_name", "aruco")
                    cv2.putText(
                        vis,
                        label,
                        (int(cxy[0]) + 8, int(cxy[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                if info.get("rvec") is not None:
                    axis_len = square_size_mm * 2.0 if target_type == "chessboard" else marker_length_mm * 0.6
                    cv2.drawFrameAxes(vis, K, dist, info["rvec"], t_t2c.reshape(3, 1) / 1000.0, axis_len / 1000.0)
                cv2.putText(vis, "target found | press c to capture", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                msg = "target NOT found"
                if target_type == "aruco" and info.get("detected_ids"):
                    msg = f"target id={marker_id} not found, seen={info['detected_ids']}"
                cv2.putText(vis, msg, (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(vis, f"sample {i+1}/{num_samples}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(vis, f"target={target_type}", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("collect_eye_to_hand", cv2.resize(vis, (960, 540)))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key != ord("c"):
                continue
            if R_t2c is None:
                print("目标未检测到，跳过")
                continue

            raw = input(f"[{i+1}] 输入 base->gripper: x y z rx ry rz : ").strip()
            try:
                x, y, z, rx, ry, rz = [float(v) for v in raw.split()]
            except Exception:
                print("输入格式错误，跳过")
                continue

            if angles_in_degree:
                rx, ry, rz = np.deg2rad([rx, ry, rz])

            R_b2g = euler_xyz_to_R(rx, ry, rz)
            t_b2g = np.array([x, y, z], dtype=np.float64)

            R_base2gripper.append(R_b2g)
            t_base2gripper.append(t_b2g)
            R_target2cam.append(R_t2c)
            t_target2cam.append(t_t2c)
            i += 1
            print(f"已记录 {i}/{num_samples}")

    cv2.destroyAllWindows()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        R_base2gripper=np.asarray(R_base2gripper),
        t_base2gripper=np.asarray(t_base2gripper),
        R_target2cam=np.asarray(R_target2cam),
        t_target2cam=np.asarray(t_target2cam),
        target_type=np.asarray(target_type),
    )
    print(f"saved: {save_path}, pairs={len(R_base2gripper)}")


def fit_eye_to_hand(
    samples_path="data/handeye_eye_to_hand_samples.npz",
    save_path="data/T_base_cam.npy",
    method="TSAI",
):
    d = np.load(samples_path)

    R_b2g = [r for r in d["R_base2gripper"]]
    t_b2g = [t.reshape(3, 1) for t in d["t_base2gripper"]]
    R_t2c = [r for r in d["R_target2cam"]]
    t_t2c = [t.reshape(3, 1) for t in d["t_target2cam"]]

    method_map = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    if method not in method_map:
        raise ValueError(f"method must be one of {list(method_map.keys())}")

    # Eye-to-Hand技巧：
    # 将“base”视作calibrateHandEye里的“gripper”，输入R_base2gripper作为R_gripper2base
    # 输出R_cam2base, t_cam2base（cam -> base）
    R_c2b, t_c2b = cv2.calibrateHandEye(
        R_gripper2base=R_b2g,
        t_gripper2base=t_b2g,
        R_target2cam=R_t2c,
        t_target2cam=t_t2c,
        method=method_map[method],
    )

    T_c2b = make_T(R_c2b, t_c2b.reshape(3))
    T_b2c = inv_T(T_c2b)  # base -> cam（你项目里常用T_base_cam）

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, T_b2c)
    print(f"saved: {save_path}")
    print("T_base_cam =")
    print(T_b2c)

    # 简单一致性检查：gripper->target 应近似常量
    Ts = []
    for i in range(len(R_b2g)):
        T_b2g_i = make_T(R_b2g[i], d["t_base2gripper"][i])
        T_c2t_i = make_T(R_t2c[i], d["t_target2cam"][i])  # target->cam (cTt)
        T_b2t_i = T_b2c @ T_c2t_i
        T_g2t_i = inv_T(T_b2g_i) @ T_b2t_i
        Ts.append(T_g2t_i[:3, 3])
    Ts = np.asarray(Ts)
    print(f"consistency(g->t) std mm: {Ts.std(axis=0)} | norm std={np.linalg.norm(Ts.std(axis=0)):.3f}")

    return T_b2c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("collect")
    p1.add_argument("--camera-ip", default="192.168.1.88")
    p1.add_argument("--intr", default="data/camera_intrinsics.npz")
    p1.add_argument("--save", default="data/handeye_eye_to_hand_samples.npz")
    p1.add_argument("--num", type=int, default=20)
    p1.add_argument("--target", default="chessboard", choices=["chessboard", "aruco"])
    p1.add_argument("--bw", type=int, default=7)
    p1.add_argument("--bh", type=int, default=7)
    p1.add_argument("--square", type=float, default=25.0)
    p1.add_argument("--aruco-dict", default="DICT_4X4_50", choices=sorted(ARUCO_DICT_MAP.keys()))
    p1.add_argument("--marker-length", type=float, default=40.0, help="ArUco边长，单位mm")
    p1.add_argument("--aruco-id", type=int, default=None, help="只采集指定ArUco ID；不填则取当前第一个")
    p1.add_argument("--deg", action="store_true", help="输入rx ry rz为度")

    p2 = sub.add_parser("fit")
    p2.add_argument("--samples", default="data/handeye_eye_to_hand_samples.npz")
    p2.add_argument("--save", default="data/T_base_cam.npy")
    p2.add_argument("--method", default="TSAI",
                    choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"])

    args = parser.parse_args()

    if args.cmd == "collect":
        collect_samples(
            camera_ip=args.camera_ip,
            intr_path=args.intr,
            save_path=args.save,
            target_type=args.target,
            board_size=(args.bw, args.bh),
            square_size_mm=args.square,
            aruco_dict=args.aruco_dict,
            marker_length_mm=args.marker_length,
            marker_id=args.aruco_id,
            num_samples=args.num,
            angles_in_degree=args.deg,
        )
    elif args.cmd == "fit":
        fit_eye_to_hand(
            samples_path=args.samples,
            save_path=args.save,
            method=args.method,
        )
