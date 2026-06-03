import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from camera_capture import CameraCapture
from handeye_calibration import load_transform
from pixel_to_camera import load_intrinsics
from xy_plane_calibration import map_pixels_to_robot_xy


def cam_to_base(T_base_cam, p_cam):
    R = T_base_cam[:3, :3]
    t = T_base_cam[:3, 3]
    return R @ p_cam.reshape(3,) + t


def load_xy_mapping(path="data/xy_plane_homography.npz"):
    d = np.load(path)
    H = np.asarray(d["H_uv_to_robot_xy"], dtype=np.float64)
    z_mm = float(d["mean_robot_z_mm"]) if "mean_robot_z_mm" in d else 0.0
    return H, z_mm


def save_records(records, out_dir="data"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"aruco_blocks_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return str(path)
    


def save_initial_positions_and_yaws_py(
    records,
    out_dir="data",
    offset_x_mm= 710.0,#710.0,
    offset_y_mm=600.0,
    fixed_z_m=0.025,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"initial_positions_{ts}.py"

    # 按 id 排序，保证输出稳定
    rs = sorted(records, key=lambda r: int(r["id"]))

    positions = []
    yaws = []
    for r in rs:
        x_mm, y_mm, z_mm = r["center_base_mm"]
        x_m = (float(x_mm) + offset_x_mm) / 1000.0
        y_m = (float(y_mm) + offset_y_mm) / 1000.0
        z_m = float(fixed_z_m) if fixed_z_m is not None else float(z_mm) / 1000.0
        positions.append([round(x_m, 4), round(y_m, 4), round(z_m, 4)])
        yaws.append(round(float(r.get("yaw_rad", 0.0)), 6))

    lines = ["initial_positions = ["]
    for p in positions:
        lines.append(f"    [{p[0]}, {p[1]}, {p[2]}],")
    lines.append("]")
    lines.append("")
    lines.append("yaws = [")
    for y in yaws:
        lines.append(f"    {y},")
    lines.append("]")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return str(path)


def main():
    camera_ip = "192.168.1.88"
    intr_path = "data/camera_intrinsics.npz"
    extr_path = "data/T_base_cam.npy"
    xy_map_path = "data/xy_plane_homography.npz"

    # ArUco 实际边长（米），必须量准
    marker_length_m = 0.04  # 40mm

    # 每个id对应木块长度(mm)（推荐）
    length_map_mm = {
        1: 180.0,
        2: 200.0,
        3: 150.0,
    }

    # 读取内参和固定顶面XY映射
    K, dist = load_intrinsics(intr_path)
    H_xy, z_base_mm = load_xy_mapping(xy_map_path)
    T_base_cam = load_transform(extr_path)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    with CameraCapture(camera_ip=camera_ip) as cam:
        cv2.namedWindow("aruco_debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("aruco_debug", 960, 540)

        while True:
            frame = cam.read()
            vis = frame.copy()

            corners, ids, _ = detector.detectMarkers(frame)
            records = []

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_length_m, K, dist
                )

                for i, mid in enumerate(ids.flatten().tolist()):
                    tvec = tvecs[i, 0]
                    rvec = rvecs[i, 0]
                    c = corners[i][0]  # 4x2 角点 (TL,TR,BR,BL)
                    cxy = c.mean(axis=0)
                    xy_base_mm = map_pixels_to_robot_xy(
                        np.array([[float(cxy[0]), float(cxy[1])]], dtype=np.float64),
                        K,
                        dist,
                        H_xy,
                    )[0]

                    # 转换 rvec 到机械臂坐标系
                    R = T_base_cam[:3, :3]
                    rvec_base, _ = cv2.Rodrigues(R @ cv2.Rodrigues(rvec)[0])

                    # 从 rvec_base 提取 yaw（绕 z 轴的旋转）
                    R_base = cv2.Rodrigues(rvec_base)[0]
                    yaw_base = float(np.arctan2(R_base[1, 0], R_base[0, 0]))

                    rec = {
                        "id": int(mid),
                        "center_base_mm": [float(xy_base_mm[0]), float(xy_base_mm[1]), float(z_base_mm)],
                        "length_mm": float(length_map_mm.get(int(mid), 0.0)),
                        "yaw_rad": yaw_base,  # 现在是机械臂坐标系
                    }
                    records.append(rec)

                    cv2.drawFrameAxes(vis, K, dist, rvecs[i], tvecs[i], marker_length_m * 0.6)
                    txt = f"id={mid} X={xy_base_mm[0]:.1f} Y={xy_base_mm[1]:.1f} L={rec['length_mm']:.1f}"
                    cxy_i = cxy.astype(int)

                    cv2.putText(
                        vis, txt, (int(cxy_i[0]) + 6, int(cxy_i[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
                    )

            cv2.putText(vis, "s: save  q: quit", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("aruco_debug", vis)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                if records:
                    p_json = save_records(records, out_dir="data")
                    p_py = save_initial_positions_and_yaws_py(
                        records,
                        out_dir="data",
                        offset_x_mm=710.0,
                        offset_y_mm=600.0,
                        fixed_z_m=0.025,
                    )
                    print("saved:", p_json)
                    print("saved:", p_py)
                else:
                    print("no markers")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
