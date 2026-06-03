import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from camera_capture import CameraCapture
from detect_object import detect_object_pixel
from pixel_to_camera import (
    load_intrinsics,
    load_table_plane,
    pixel_to_camera_on_tilted_table,
)
from handeye_calibration import load_transform
from robot_control import RobotControl


def cam_to_base(T_base_cam, p_cam):
    R = T_base_cam[:3, :3]
    t = T_base_cam[:3, 3]
    return R @ p_cam.reshape(3,) + t


def detect_blocks_from_mask(mask, min_area=800):
    """从mask中提取多个木块，返回按位置排序的列表"""
    if mask is None:
        return []
    m = mask.copy()
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        (cx, cy), (w, h), angle = cv2.minAreaRect(c)
        theta = np.deg2rad(angle)
        if w < h:
            theta += np.pi / 2.0

        half = max(w, h) * 0.5
        dx, dy = np.cos(theta), np.sin(theta)
        e1 = (cx - half * dx, cy - half * dy)
        e2 = (cx + half * dx, cy + half * dy)

        box = cv2.boxPoints(((cx, cy), (w, h), angle))
        blocks.append(
            {
                "contour": c,
                "area": float(area),
                "center_px": (float(cx), float(cy)),
                "axis_p1_px": (float(e1[0]), float(e1[1])),
                "axis_p2_px": (float(e2[0]), float(e2[1])),
                "box_px": box.astype(np.float32),
            }
        )

    # 按位置排序：先从上到下，再从左到右
    blocks.sort(key=lambda x: (x["center_px"][1], x["center_px"][0]))
    return blocks


def block_to_robot_record(block_id, block, K, dist, n_plane, d_plane, T_base_cam, z_offset_mm=0.0):
    u, v = block["center_px"]
    u1, v1 = block["axis_p1_px"]
    u2, v2 = block["axis_p2_px"]

    p_cam = pixel_to_camera_on_tilted_table(u, v, K, dist, n_plane, d_plane)
    p1_cam = pixel_to_camera_on_tilted_table(u1, v1, K, dist, n_plane, d_plane)
    p2_cam = pixel_to_camera_on_tilted_table(u2, v2, K, dist, n_plane, d_plane)

    p_base = cam_to_base(T_base_cam, p_cam)
    p1_base = cam_to_base(T_base_cam, p1_cam)
    p2_base = cam_to_base(T_base_cam, p2_cam)

    v12 = p2_base - p1_base
    yaw = float(np.arctan2(v12[1], v12[0]))
    length_mm = float(np.linalg.norm(v12[:2]))

    pose_robot = np.array(
        [p_base[0], p_base[1], p_base[2] + z_offset_mm, np.pi, 0.0, yaw],
        dtype=np.float64,
    )

    return {
        "id": int(block_id),  # 当前帧标号
        "pixel": [float(u), float(v)],  # 像素坐标
        "center_base_mm": [float(p_base[0]), float(p_base[1]), float(p_base[2])],  # 机械臂坐标
        "axis_p1_base_mm": [float(p1_base[0]), float(p1_base[1]), float(p1_base[2])],
        "axis_p2_base_mm": [float(p2_base[0]), float(p2_base[1]), float(p2_base[2])],
        "length_mm": length_mm,
        "yaw_rad": yaw,
        "pose_robot_mm_rad": pose_robot.tolist(),
        "area_px2": float(block["area"]),
    }


def save_records(records, out_dir="data"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"blocks_{ts}.json"

    # 只保存: id, 机械臂坐标, 长度
    slim_records = []
    for r in records:
        slim_records.append(
            {
                "id": int(r["id"]),
                "center_base_mm": [float(r["center_base_mm"][0]), float(r["center_base_mm"][1]), float(r["center_base_mm"][2])],
                "length_mm": float(r["length_mm"]),
            }
        )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(slim_records, f, ensure_ascii=False, indent=2)
    return str(path)


def save_initial_positions_py(
    records,
    out_dir="data",
    offset_x_mm=800.0,
    offset_y_mm=510.0,
    fixed_z_m=0.025,  # 若想用实测z，改成 None
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"initial_positions_{ts}.py"

    # 按 id 排序，保证输出顺序稳定
    rs = sorted(records, key=lambda r: int(r["id"]))

    positions = []
    for r in rs:
        x_mm, y_mm, z_mm = r["center_base_mm"]
        x_m = (float(x_mm) + offset_x_mm) / 1000.0
        y_m = (float(y_mm) + offset_y_mm) / 1000.0
        z_m = float(fixed_z_m) if fixed_z_m is not None else float(z_mm) / 1000.0
        positions.append([round(x_m, 4), round(y_m, 4), round(z_m, 4)])

    lines = ["initial_positions = ["]
    for p in positions:
        lines.append(f"    [{p[0]}, {p[1]}, {p[2]}],")
    lines.append("]")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return str(path)


def main():
    intr_path = "data/camera_intrinsics.npz"
    plane_path = "data/table_plane_from_pnp.npz"
    extr_path = "data/T_base_cam.npy"
    camera_ip = "192.168.1.88"
    pick_z_offset_mm = 0.0
    view_size = (1080, 720)

    # 新增：长度阈值（15cm~25cm）
    min_len_mm = 150.0
    max_len_mm = 250.0

    K, dist = load_intrinsics(intr_path)
    n_plane, d_plane = load_table_plane(plane_path)
    T_base_cam = load_transform(extr_path)

    print("按键: [s] 保存当前所有木块坐标/姿态/长度  [p] 抓第1个  [q] 退出")

    with CameraCapture(camera_ip=camera_ip) as cam, RobotControl(dry_run=True) as robot:
        # 新增：窗口可缩放并设置初始大小
        cv2.namedWindow("vision_pick_debug", cv2.WINDOW_NORMAL)
        cv2.namedWindow("vision_pick_mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("vision_pick_debug", view_size[0], view_size[1])
        cv2.resizeWindow("vision_pick_mask", view_size[0], view_size[1])

        while True:
            frame = cam.read()
            _, debug, mask = detect_object_pixel(frame)

            blocks = detect_blocks_from_mask(mask, min_area=800)
            records = []

            for i, b in enumerate(blocks, start=1):
                rec = block_to_robot_record(
                    i, b, K, dist, n_plane, d_plane, T_base_cam, z_offset_mm=pick_z_offset_mm
                )

                # 新增：长度过滤（去掉过小/过大）
                if not (min_len_mm <= rec["length_mm"] <= max_len_mm):
                    continue

                records.append(rec)

                box = b["box_px"].astype(np.int32)
                cv2.polylines(debug, [box], True, (0, 255, 0), 2)
                c = np.array(b["center_px"]).astype(int)
                p1 = np.array(b["axis_p1_px"]).astype(int)
                p2 = np.array(b["axis_p2_px"]).astype(int)
                cv2.circle(debug, tuple(c), 4, (0, 0, 255), -1)
                cv2.line(debug, tuple(p1), tuple(p2), (255, 0, 0), 2)

                txt = f"id={i} L={rec['length_mm']:.1f}mm yaw={rec['yaw_rad']:.2f}"
                cv2.putText(debug, txt, (c[0] + 8, c[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            cv2.putText(debug, f"blocks={len(records)}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # 修改：显示前缩放
            debug_show = cv2.resize(debug, view_size, interpolation=cv2.INTER_AREA)
            mask_show = cv2.resize(mask, view_size, interpolation=cv2.INTER_NEAREST)
            cv2.imshow("vision_pick_debug", debug_show)
            cv2.imshow("vision_pick_mask", mask_show)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("s"):
                if not records:
                    print("当前无木块可保存")
                else:
                    p_json = save_records(records, out_dir="data")
                    p_py = save_initial_positions_py(
                        records,
                        out_dir="data",
                        offset_x_mm=710.0,
                        offset_y_mm=600.0,
                        fixed_z_m=0.025,
                    )
                    print(f"已保存: {p_json}")
                    print(f"已保存: {p_py}")
            elif k == ord("p"):
                if not records:
                    print("未检测到目标，无法抓取")
                else:
                    pose = np.array(records[0]["pose_robot_mm_rad"], dtype=np.float64)
                    print("抓取目标 #1:", pose)
                    robot.execute_pick(pose, approach_mm=60.0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()