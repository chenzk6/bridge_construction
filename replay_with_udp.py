import os
import pickle
import socket
import struct
import time

import numpy as np
from LCUS import SerialRelayController

NUM_JOINTS = 6
GRIPPER_KEYS = {"close_finger", "release_finger"}
SAFE_HOME_JOINTS = (0.0, -66.5, 33.7929, 0.0, 31.65, 134.45)
DEFAULT_DUPLICATE_TOL_DEG = 0.05
DEFAULT_MIN_POINT_SPACING_DEG = 1.0
DEFAULT_POINT_SEND_INTERVAL_S = 0.1;
DEFAULT_SEND_MODE = "buffered"  # "wait" / "timed" / "buffered"
PRECISION_STAGE_TAIL_STEPS = {
    "fetch_object": 3,
}

STAGE_NAME_MAP = {
    "fetch_object": "移动到抓取位置",
    "close_finger": "闭合夹爪",
    "change_pose": "移动到目标位置",
    "release_finger": "释放物体",
    "lift_up": "抬起夹爪",
    "move_back": "返回初始位置",
}


class RawUdpPLCClient:
    """与 bridge_construction/PLC.c 对齐的原始 UDP 客户端。"""

    CMD_TYPE_POWER = 0x01
    CMD_TYPE_MOTION = 0x02
    CMD_TYPE_STATUS = 0x03

    POWER_ON = 0x01
    POWER_OFF = 0x02

    MOTION_LINEAR_ABS = 0x01
    MOTION_PTP_ABS_ACP = 0x02
    MOTION_PTP_ABS_TCP = 0x03
    MOTION_PTP_BUFFERED_ACP = 0x04

    STATUS_COMPLETED = 0x01
    STATUS_MOVING = 0x02
    STATUS_ERROR = 0x03

    def __init__(self, plc_ip, plc_port, local_ip="0.0.0.0", local_port=0, timeout_s=1.0):
        self.plc_ip = plc_ip
        self.plc_port = plc_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((local_ip, local_port))
        self.sock.connect((plc_ip, plc_port))
        self.sock.settimeout(float(timeout_s))

        bound_ip, bound_port = self.sock.getsockname()
        self.local_ip = bound_ip
        self.local_port = bound_port

    def close(self):
        self.sock.close()

    def _send_command(self, payload, silent=False):
        self.sock.send(payload)
        if not silent:
            print(f"→ 发送 {len(payload)} 字节到 {self.plc_ip}:{self.plc_port}")
            print(f"  十六进制: {payload.hex(' ').upper()}")

    def _drain_socket(self):
        previous_timeout = self.sock.gettimeout()
        self.sock.settimeout(0.0)
        try:
            while True:
                self.sock.recv(2048)
        except (BlockingIOError, socket.timeout):
            pass
        finally:
            self.sock.settimeout(previous_timeout)

    def _pack_power_command(self, action):
        return bytes([self.CMD_TYPE_POWER, int(action) & 0xFF, 0x00])

    def _pack_motion_command(self, motion_type, *values):
        payload = struct.pack("<6f", *[float(v) for v in values])
        return bytes([self.CMD_TYPE_MOTION, int(motion_type) & 0xFF, 0x00]) + payload

    def power_on(self):
        print("\n【上电命令】")
        self._send_command(self._pack_power_command(self.POWER_ON))
        return True

    def power_off(self):
        print("\n【下电命令】")
        self._send_command(self._pack_power_command(self.POWER_OFF))
        return True

    def move_ptp_abs_acp(self, a1, a2, a3, a4, a5, a6, wait_complete=False, timeout=45.0):
        target = [float(v) for v in (a1, a2, a3, a4, a5, a6)]
        print(f"\n【PTP_ABS_ACP 运动】目标: {[round(v, 3) for v in target]}")
        self._send_command(self._pack_motion_command(self.MOTION_PTP_ABS_ACP, *target))

        if wait_complete:
            time.sleep(0.15)
            return self.wait_for_motion_complete(timeout=timeout)

        return True

    def move_ptp_buffered_acp(self, a1, a2, a3, a4, a5, a6, wait_complete=False, timeout=45.0):
        target = [float(v) for v in (a1, a2, a3, a4, a5, a6)]
        print(f"\n【PTP_BUFFERED_ACP 运动】目标: {[round(v, 3) for v in target]}")
        self._send_command(self._pack_motion_command(self.MOTION_PTP_BUFFERED_ACP, *target))

        if wait_complete:
            time.sleep(0.15)
            return self.wait_for_motion_complete(timeout=timeout)

        return True

    def move_joint_abs(self, a1, a2, a3, a4, a5, a6, wait_complete=False, timeout=45.0):
        """兼容旧调用名，底层改为 PTP_ABS_ACP。"""
        return self.move_ptp_abs_acp(a1, a2, a3, a4, a5, a6, wait_complete=wait_complete, timeout=timeout)

    def query_status(self, silent=False):
        if not silent:
            print("\n【查询状态】")

        self._drain_socket()
        self._send_command(bytes([self.CMD_TYPE_STATUS, 0x00, 0x00]), silent=silent)

        try:
            response = self.sock.recv(2048)
        except socket.timeout:
            if not silent:
                print("✗ UDP 状态查询超时")
            return None

        if len(response) < 50:
            if not silent:
                print(f"✗ UDP 状态包长度不足: {len(response)} 字节")
            return None

        motion_status = response[1]
        pose = struct.unpack("<ffffff", response[2:26])
        axes = struct.unpack("<ffffff", response[26:50])
        free_slot_count = int(response[50]) if len(response) >= 51 else None
        ready_for_next = bool(response[51]) if len(response) >= 52 else None

        if not silent:
            print(f"  运动状态: {self._parse_motion_status(motion_status)} (0x{motion_status:02X})")
            print(f"  当前TCP: {[round(v, 3) for v in pose]}")
            print(f"  当前关节: {[round(v, 3) for v in axes]}")
            if free_slot_count is not None:
                print(f"  双实例空槽数: {free_slot_count}")
            if ready_for_next is not None:
                print(f"  可发送下一点: {'是' if ready_for_next else '否'}")

        return {
            "motion_status": motion_status,
            "is_completed": motion_status == self.STATUS_COMPLETED,
            "is_moving": motion_status == self.STATUS_MOVING,
            "is_error": motion_status == self.STATUS_ERROR,
            "free_slot_count": free_slot_count,
            "ready_for_next": ready_for_next,
            "pose": pose,
            "axes": axes,
            "x": pose[0],
            "y": pose[1],
            "z": pose[2],
            "rx": pose[3],
            "ry": pose[4],
            "rz": pose[5],
            "a1": axes[0],
            "a2": axes[1],
            "a3": axes[2],
            "a4": axes[3],
            "a5": axes[4],
            "a6": axes[5],
        }

    def wait_until_buffer_ready(self, timeout=45.0, check_interval=0.05, require_pipeline_active=False):
        print(f"  [等待] 双实例可接收下一点（超时 {timeout} 秒）...")
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            status = self.query_status(silent=True)
            elapsed = time.time() - start_time

            if status is None:
                time.sleep(check_interval)
                continue

            if status["is_error"]:
                print(f"  ✗ PLC 返回错误状态 (耗时 {elapsed:.1f}s)")
                return False

            ready_for_next = status.get("ready_for_next")
            if ready_for_next is None:
                free_slot_count = status.get("free_slot_count")
                ready_for_next = free_slot_count is None or free_slot_count > 0

            if require_pipeline_active:
                free_slot_count = status.get("free_slot_count")
                pipeline_active = status["is_moving"]
                if free_slot_count is not None and free_slot_count < 2:
                    pipeline_active = True
                if not pipeline_active:
                    time.sleep(check_interval)
                    continue

            if ready_for_next:
                if elapsed >= 0.1:
                    print(f"  ✓ 检测到空槽可发送 (耗时 {elapsed:.1f}s)")
                return True

            time.sleep(check_interval)

        print(f"  ✗ 等待空槽超时 ({timeout}s)")
        return False

    def wait_until_buffer_consumed(self, previous_free_slot_count=None, timeout=8.0, check_interval=0.03):
        print(f"  [等待] PLC 接收并挂入缓存（超时 {timeout} 秒）...")
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            status = self.query_status(silent=True)
            elapsed = time.time() - start_time

            if status is None:
                time.sleep(check_interval)
                continue

            if status["is_error"]:
                print(f"  ✗ PLC 返回错误状态 (耗时 {elapsed:.1f}s)")
                return False

            free_slot_count = status.get("free_slot_count")
            if previous_free_slot_count is not None and free_slot_count is not None:
                if free_slot_count < previous_free_slot_count:
                    if elapsed >= 0.05:
                        print(f"  ✓ 已确认缓存占用变化 ({previous_free_slot_count} -> {free_slot_count})")
                    return True

            if status["is_moving"] and (free_slot_count is None or free_slot_count < 2):
                if elapsed >= 0.05:
                    print("  ✓ 已确认 PLC 开始消化缓存命令")
                return True

            time.sleep(check_interval)

        print(f"  ✗ 等待 PLC 确认接收超时 ({timeout}s)")
        return False

    def wait_for_motion_complete(self, timeout=45.0, check_interval=0.3):
        print(f"  [等待] 运动完成（超时 {timeout} 秒）...")
        start_time = time.time()
        motion_started = False
        check_count = 0

        while (time.time() - start_time) < timeout:
            check_count += 1
            status = self.query_status(silent=True)
            elapsed = time.time() - start_time

            if status is None:
                print(f"  [警告] 状态查询失败 (已等待 {elapsed:.1f}s)")
                time.sleep(check_interval)
                continue

            motion_status = status["motion_status"]

            if motion_status == self.STATUS_MOVING:
                if not motion_started:
                    print(f"  [等待] 机械臂开始运动... (已等待 {elapsed:.1f}s)")
                    motion_started = True
                elif check_count % 5 == 0:
                    print(f"  [等待] 运动中... (已等待 {elapsed:.1f}s)")
            elif motion_status == self.STATUS_COMPLETED:
                if motion_started or (elapsed >= 0.6 and check_count >= 2):
                    print(f"  ✓ 运动完成 (耗时 {elapsed:.1f}s)")
                    return True
            elif motion_status == self.STATUS_ERROR:
                print(f"  ✗ PLC 返回错误状态 (耗时 {elapsed:.1f}s)")
                return False
            else:
                print(f"  [等待] 未知状态 0x{motion_status:02X} (已等待 {elapsed:.1f}s)")

            time.sleep(check_interval)

        print(f"  ✗ 等待超时 ({timeout}s)")
        return False

    def _parse_motion_status(self, status_byte):
        if status_byte == self.STATUS_COMPLETED:
            return "运动完成"
        if status_byte == self.STATUS_MOVING:
            return "正在运动"
        if status_byte == self.STATUS_ERROR:
            return "运动错误"
        return "未知/空闲"


def load_trajectory_data(pkl_name):
    with open(pkl_name, "rb") as f:
        info = pickle.load(f)

    paths = info["paths"]
    meta = info["meta"]

    print("=" * 60)
    print("轨迹数据加载成功")
    print("=" * 60)
    print(f"元数据: {meta}")
    print(f"总路径数: {len(paths)}")
    print("=" * 60)

    return paths, meta


def gripper_action_name(motion_key):
    if motion_key == "close_finger":
        return "闭合"
    if motion_key == "release_finger":
        return "张开"
    return "未知"


def execute_gripper_action(
    relay_controller,
    motion_key,
    close_finger_uses_relay_on=True,
    action_delay_s=0.5,
):
    action = gripper_action_name(motion_key)

    if relay_controller is None:
        print(f"夹爪动作: {action}")
        input("请手动执行夹爪后按回车继续...")
        return True

    if motion_key == "close_finger":
        relay_controller.set_close_finger(close_finger_uses_relay_on=close_finger_uses_relay_on)
    elif motion_key == "release_finger":
        relay_controller.set_release_finger(close_finger_uses_relay_on=close_finger_uses_relay_on)
    else:
        print(f"[WARN] 未知夹爪动作: {motion_key}")
        return False

    print(f"夹爪动作已通过继电器执行: {action}")
    if action_delay_s > 0:
        time.sleep(action_delay_s)
    return True


def should_use_precision_send(cmd):
    if cmd.get("type") != "joint_motion":
        return False
    tail_steps = int(PRECISION_STAGE_TAIL_STEPS.get(cmd.get("motion_key"), 0))
    if tail_steps <= 0:
        return False
    total_steps = int(cmd.get("total_steps", 0))
    step_index = int(cmd.get("step", -1))
    if total_steps <= 0 or step_index < 0:
        return False
    return (total_steps - step_index) <= tail_steps


def transform_joint_angles(q):
    q_arr = np.asarray(q).reshape(-1)
    if q_arr.shape[0] < NUM_JOINTS:
        raise ValueError(f"关节维度不足: {q_arr.shape[0]} < {NUM_JOINTS}")

    angles_deg = np.rad2deg(q_arr[:NUM_JOINTS]).astype(float).tolist()
    angles_deg[1] -= 90.0
    angles_deg[4] = -angles_deg[4]
    angles_deg[5] =  angles_deg[5]-42.5
    return angles_deg


def angles_are_close(lhs, rhs, tol_deg=DEFAULT_DUPLICATE_TOL_DEG):
    if lhs is None or rhs is None:
        return False
    return all(abs(float(a) - float(b)) <= float(tol_deg) for a, b in zip(lhs, rhs))


def max_angle_diff(lhs, rhs):
    if lhs is None or rhs is None:
        return float("inf")
    return max(abs(float(a) - float(b)) for a, b in zip(lhs, rhs))


def display_all_trajectories(paths, meta):
    print("\n" + "=" * 80)
    print("轨迹详细信息")
    print("=" * 80)

    global_cmd_idx = 0

    for path_idx, total_path in enumerate(paths):
        if total_path is None:
            print(f"\n[路径 {path_idx}] 为空，跳过")
            continue

        print(f"\n{'=' * 80}")
        print(f"【路径 {path_idx}】(按 path 内部顺序)")
        print(f"{'=' * 80}")

        if not isinstance(total_path, dict):
            print(f"[WARN] path {path_idx} 不是 dict: {type(total_path)}")
            continue

        for motion_key, path_data in total_path.items():
            if path_data is None:
                print(f"  [{motion_key}] 数据为空")
                continue

            if motion_key in GRIPPER_KEYS:
                print(f"\n  ┌─ [{motion_key}] 动作: {gripper_action_name(motion_key)}")
                global_cmd_idx += 1
                continue

            print(f"\n  ┌─ [{motion_key}] 关节点数: {len(path_data)}")
            for step_idx, q in enumerate(path_data):
                try:
                    angles_deg = transform_joint_angles(q)
                except ValueError as exc:
                    print(f"  │  [WARN] 第 {step_idx + 1} 步 {exc}，已跳过")
                    continue

                angles_str = " | ".join(f"J{i + 1}: {a:8.2f}°" for i, a in enumerate(angles_deg))
                is_last = step_idx == len(path_data) - 1
                prefix = "  └─" if is_last else "  ├─"
                wait_mark = " [WAIT]" if is_last else ""
                print(
                    f"{prefix} [{global_cmd_idx:4d}] 步 {step_idx + 1:3d}/{len(path_data):3d}: "
                    f"{angles_str}{wait_mark}"
                )
                global_cmd_idx += 1

    print(f"\n{'=' * 80}")
    print(f"总命令数: {global_cmd_idx}")
    print(f"{'=' * 80}\n")
    return global_cmd_idx


def build_command_sequence(
    paths,
    duplicate_tol_deg=DEFAULT_DUPLICATE_TOL_DEG,
    min_point_spacing_deg=DEFAULT_MIN_POINT_SPACING_DEG,
):
    cmd_sequence = []
    skipped_duplicates = 0
    skipped_dense_points = 0
    last_joint_angles_deg = None

    for path_idx, total_path in enumerate(paths):
        if total_path is None:
            print(f"[WARN] path {path_idx} is None")
            continue

        if not isinstance(total_path, dict):
            print(f"[WARN] path {path_idx} 不是 dict: {type(total_path)}")
            continue

        for motion_key, path_data in total_path.items():
            if path_data is None:
                print(f"[WARN] path {path_idx} 动作为空: {motion_key}")
                continue

            if motion_key in GRIPPER_KEYS:
                cmd_sequence.append(
                    {
                        "type": "gripper",
                        "motion_key": motion_key,
                        "path_idx": path_idx,
                    }
                )
                continue

            for step_idx, q in enumerate(path_data):
                try:
                    angles_deg = transform_joint_angles(q)
                except ValueError:
                    print(f"[WARN] path={path_idx} {motion_key} step={step_idx} 维度不足，跳过")
                    continue

                # Keep continuity across stage boundaries so we don't resend the
                # first point of the next stage when it's identical to the last
                # point already reached in the previous stage.
                if angles_are_close(last_joint_angles_deg, angles_deg, tol_deg=duplicate_tol_deg):
                    skipped_duplicates += 1
                    continue

                if (
                    float(min_point_spacing_deg) > 0.0
                    and max_angle_diff(last_joint_angles_deg, angles_deg) < float(min_point_spacing_deg)
                ):
                    skipped_dense_points += 1
                    continue

                last_joint_angles_deg = angles_deg
                cmd_sequence.append(
                    {
                        "type": "joint_motion",
                        "motion_key": motion_key,
                        "path_idx": path_idx,
                        "step": step_idx,
                        "total_steps": len(path_data),
                        "angles_deg": angles_deg,
                    }
                )

    return cmd_sequence, skipped_duplicates, skipped_dense_points


def preview_mode(pkl_name):
    print("=" * 80)
    print("轨迹预览模式 (仅显示，不执行)")
    print("=" * 80)

    paths, meta = load_trajectory_data(pkl_name)
    display_all_trajectories(paths, meta)

    print("\n预览完成！")


def replay_with_udp(
    pkl_name,
    plc_ip="192.168.1.101",
    plc_port=10011,
    local_ip="0.0.0.0",
    local_port=10002,
    relay_port=None,
    relay_baudrate=9600,
    relay_timeout=1.0,
    relay_action_delay_s=0.5,
    close_finger_uses_relay_on=True,
    duplicate_tol_deg=DEFAULT_DUPLICATE_TOL_DEG,
    min_point_spacing_deg=DEFAULT_MIN_POINT_SPACING_DEG,
    point_send_interval_s=DEFAULT_POINT_SEND_INTERVAL_S,
    send_mode=DEFAULT_SEND_MODE,
):
    if send_mode not in ("timed", "wait", "buffered"):
        raise ValueError(f"未知发送模式: {send_mode}，仅支持 'timed' / 'wait' / 'buffered'")

    print("=" * 60)
    if send_mode == "wait":
        print("UDP真实机械臂轨迹复现（PTP_ABS_ACP，逐点等待完成）")
    elif send_mode == "buffered":
        print("UDP真实机械臂轨迹复现（PTP双实例缓存，按空槽发送）")
    else:
        print("UDP真实机械臂轨迹复现（PTP_ABS_ACP，固定延迟预发送）")
    print("=" * 60)

    paths, meta = load_trajectory_data(pkl_name)
    display_all_trajectories(paths, meta)

    cmd_sequence, skipped_duplicates, skipped_dense_points = build_command_sequence(
        paths,
        duplicate_tol_deg=duplicate_tol_deg,
        min_point_spacing_deg=min_point_spacing_deg,
    )
    print(f"\n待执行命令数: {len(cmd_sequence)}")
    print(f"压缩掉的重复关节点: {skipped_duplicates}")
    print(f"按最小点间距压掉的过密点: {skipped_dense_points}")
    print(f"发送模式: {send_mode}")
    print(f"最小点间距阈值: {float(min_point_spacing_deg):.3f}°")
    print(f"精确阶段末段设置: {PRECISION_STAGE_TAIL_STEPS}")
    if send_mode == "timed":
        print(f"相邻点预发送间隔: {point_send_interval_s:.3f}s")
    elif send_mode == "buffered":
        print("发送策略: 查询PLC双实例空槽后再发送下一点")

    print("\n初始化UDP客户端:")
    print(f"  PLC地址: {plc_ip}:{plc_port}")
    print(f"  本地地址: {local_ip}:{local_port}")
    plc = RawUdpPLCClient(plc_ip, plc_port, local_ip, local_port)
    relay_controller = None

    if relay_port:
        print("\n初始化夹爪继电器:")
        print(f"  串口设备: {relay_port}")
        print(f"  波特率: {relay_baudrate}")
        if not os.path.exists(relay_port):
            print(f"  [WARN] 串口不存在: {relay_port}")
            print("  [WARN] 自动退回手动夹爪模式")
        else:
            try:
                relay_controller = SerialRelayController(
                    port=relay_port,
                    baudrate=relay_baudrate,
                    timeout=relay_timeout,
                )
                close_action = "开启继电器" if close_finger_uses_relay_on else "关闭继电器"
                release_action = "关闭继电器" if close_finger_uses_relay_on else "开启继电器"
                print(f"  close_finger -> {close_action}")
                print(f"  release_finger -> {release_action}")
            except Exception as exc:
                print(f"  [WARN] 继电器初始化失败: {exc}")
                print("  [WARN] 自动退回手动夹爪模式")
                relay_controller = None

    try:
        ans = input("\n请确认机械臂与工件状态安全。继续执行? [Y|n]: ")
        if ans.strip().lower() == "n":
            print("操作已取消")
            return

        print("\n【步骤1】上电...")
        plc.power_on()
        time.sleep(1.0)

        print("\n【步骤2】查询初始状态...")
        plc.query_status()
        time.sleep(0.3)

        if send_mode == "wait":
            print("\n【步骤3】执行轨迹（按 path 内部顺序，每点等待完成）...")
        elif send_mode == "buffered":
            print("\n【步骤3】执行轨迹（按 path 内部顺序，双实例按空槽连续发送）...")
        else:
            print("\n【步骤3】执行轨迹（按 path 内部顺序，固定延迟预发送下一点）...")
        current_stage = None
        current_path = None
        motion_in_flight = False
        precision_segment_active = False

        for i, cmd in enumerate(cmd_sequence, start=1):
            stage = cmd["motion_key"]
            path_idx = cmd.get("path_idx", -1)

            if stage != current_stage or path_idx != current_path:
                if motion_in_flight:
                    print("\n[阶段切换] 等待上一阶段运动完成...")
                    ok = plc.wait_for_motion_complete(timeout=60.0)
                    if not ok:
                        print("✗ 上一阶段等待完成失败")
                        ans = input("是否继续? [Y|n]: ")
                        if ans.strip() != "Y":
                            break
                    motion_in_flight = False

                current_stage = stage
                current_path = path_idx
                precision_segment_active = False
                stage_cn = STAGE_NAME_MAP.get(stage, stage)
                ans = input(
                    f"\n=== Path {path_idx} | 即将开始阶段: {stage}（{stage_cn}）===\n"
                    f"按回车开始；输入 n 终止："
                )
                if ans.strip().lower() == "n":
                    print("已终止执行")
                    break

            if cmd["type"] == "gripper":
                if motion_in_flight:
                    print("\n[夹爪动作前] 等待当前运动完成...")
                    ok = plc.wait_for_motion_complete(timeout=60.0)
                    if not ok:
                        print("✗ 当前运动等待完成失败")
                        ans = input("是否继续? [Y|n]: ")
                        if ans.strip() != "Y":
                            break
                    motion_in_flight = False

                print(f"[{i}/{len(cmd_sequence)}] Path {path_idx} {stage} -> 夹爪{gripper_action_name(stage)}")
                ok = execute_gripper_action(
                    relay_controller=relay_controller,
                    motion_key=cmd["motion_key"],
                    close_finger_uses_relay_on=close_finger_uses_relay_on,
                    action_delay_s=relay_action_delay_s,
                )
                if not ok:
                    print("✗ 夹爪动作执行失败")
                    ans = input("是否继续? [Y|n]: ")
                    if ans.strip() != "Y":
                        break
                continue

            print(
                f"[{i}/{len(cmd_sequence)}] Path {path_idx} {stage} "
                f"{cmd['step'] + 1}/{cmd['total_steps']} -> "
                f"{[round(x, 2) for x in cmd['angles_deg']]}"
            )

            use_precision_send = send_mode == "buffered" and should_use_precision_send(cmd)
            if use_precision_send and not precision_segment_active:
                if motion_in_flight:
                    print("  [精确段] 先等待连续段运动完成...")
                    ok = plc.wait_for_motion_complete(timeout=60.0)
                    if not ok:
                        print("✗ 进入精确段前等待完成失败")
                        ans = input("是否继续? [Y|n]: ")
                        if ans.strip() != "Y":
                            break
                        continue
                    motion_in_flight = False
                precision_segment_active = True
                print("  [精确段] 切换为逐点等待模式")

            if send_mode == "buffered" and not use_precision_send:
                pre_status = plc.query_status(silent=True)
                pre_free_slot_count = None if pre_status is None else pre_status.get("free_slot_count")
                ok = plc.wait_until_buffer_ready(timeout=60.0, require_pipeline_active=motion_in_flight)
                if not ok:
                    print("✗ 当前点等待空槽失败")
                    ans = input("是否继续? [Y|n]: ")
                    if ans.strip() != "Y":
                        break
                    continue
                ok = plc.move_ptp_buffered_acp(*cmd["angles_deg"], wait_complete=False, timeout=60.0)
                if ok:
                    ok = plc.wait_until_buffer_consumed(previous_free_slot_count=pre_free_slot_count, timeout=8.0)
            elif use_precision_send:
                ok = plc.move_ptp_abs_acp(*cmd["angles_deg"], wait_complete=True, timeout=60.0)
            else:
                wait_complete = (send_mode == "wait")
                ok = plc.move_ptp_abs_acp(*cmd["angles_deg"], wait_complete=wait_complete, timeout=60.0)
            if not ok:
                print("✗ 当前点执行失败")
                ans = input("是否继续? [Y|n]: ")
                if ans.strip() != "Y":
                    break
            if send_mode == "timed":
                motion_in_flight = True
                time.sleep(max(0.0, float(point_send_interval_s)))
            elif send_mode == "buffered":
                if use_precision_send:
                    motion_in_flight = False
                else:
                    motion_in_flight = True
                    time.sleep(0.02)
            else:
                motion_in_flight = False

        if motion_in_flight:
            print("\n【步骤3.5】等待最后一个点完成...")
            ok = plc.wait_for_motion_complete(timeout=60.0)
            if not ok:
                print("✗ 最后一个点等待完成失败")

        print("\n【步骤4】查询最终状态...")
        plc.query_status()

        print(f"\n【步骤5】复位到安全位 {SAFE_HOME_JOINTS} ...")
        plc.move_ptp_abs_acp(*SAFE_HOME_JOINTS, wait_complete=True, timeout=60.0)

        print("\n测试完成")

    except KeyboardInterrupt:
        print("\n用户中断")
        plc.power_off()
    except Exception as exc:
        print(f"\n错误: {exc}")
        plc.power_off()
    finally:
        if relay_controller is not None:
            relay_controller.close()
        plc.close()


def summarize_paths(paths):
    print("\n" + "=" * 80)
    print("paths 内容摘要")
    print("=" * 80)

    for path_idx, total_path in enumerate(paths):
        if total_path is None:
            print(f"[path {path_idx}] None")
            continue
        if not isinstance(total_path, dict):
            print(f"[path {path_idx}] 非 dict，类型={type(total_path)}")
            continue

        print(f"\n[path {path_idx}]")
        for motion_key, path_data in total_path.items():
            if path_data is None:
                print(f"  - {motion_key}: None")
                continue

            if motion_key in GRIPPER_KEYS:
                print(f"  - {motion_key}: gripper")
                continue

            count = len(path_data) if hasattr(path_data, "__len__") else "unknown"
            print(f"  - {motion_key}: joint_points={count}")

    print("\n" + "=" * 80)


def print_path_stage(paths, path_idx=0, stage_key="change_pose"):
    if path_idx >= len(paths):
        print(f"[ERR] path_idx 越界: {path_idx}, 总数={len(paths)}")
        return

    path = paths[path_idx]
    if path is None:
        print(f"[ERR] path {path_idx} is None")
        return
    if not isinstance(path, dict):
        print(f"[ERR] path {path_idx} 不是 dict: {type(path)}")
        return
    if stage_key not in path:
        print(f"[ERR] path {path_idx} 不包含阶段: {stage_key}")
        return

    data = path[stage_key]
    if data is None:
        print(f"[ERR] path {path_idx}[{stage_key}] is None")
        return

    print(f"\n[path {path_idx}] {stage_key}，joint_points={len(data)}")
    for i, q in enumerate(data):
        print(f"  step {i + 1}/{len(data)}: {np.asarray(q).reshape(-1).tolist()}")


if __name__ == "__main__":
    pkl_file = os.path.join(os.path.dirname(__file__), "low_level_paths.pkl")
    replay_with_udp(
        pkl_file,
        relay_port="/dev/ttyUSB0",  # 改成实际串口名，例如 "/dev/ttyUSB0"
        relay_baudrate=9600,
        relay_timeout=1.0,
        relay_action_delay_s=0.5,
        close_finger_uses_relay_on=True,
    )
