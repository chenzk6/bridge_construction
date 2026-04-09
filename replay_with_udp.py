import pickle
import time
import numpy as np
from pathlib import Path
import sys

NUM_JOINTS = 6  # 六轴
# 导入PLC UDP客户端


def load_trajectory_data(pkl_name):
    """加载pickle文件中的轨迹数据"""
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


def display_all_trajectories(paths, meta):
    """
    显示所有轨迹中的关节角度
    """
    print("\n" + "=" * 80)
    print("轨迹详细信息")
    print("=" * 80)
    
    global_cmd_idx = 0
    valid_idx = 0
    
    for path_idx in range(len(paths)):
        total_path = paths[path_idx]
        
        if total_path is None:
            print(f"\n[路径 {path_idx}] 为空，跳过")
            continue
        
        print(f"\n{'='*80}")
        print(f"【路径 {path_idx}】(有效索引: {valid_idx})")
        print(f"{'='*80}")
        
        # 运动顺序
        motion_sequence = [
            "fetch_object",    # 抓取物体
            "close_finger",    # 闭合夹爪
            "change_pose",     # 改变姿态
            "release_finger",  # 释放夹爪
            "lift_up",         # 提起
            "move_back"        # 移回
        ]
        
        for motion_key in motion_sequence:
            if motion_key not in total_path:
                continue
            
            path_data = total_path[motion_key]
            
            if path_data is None:
                print(f"\n  [{motion_key}] 数据为空")
                continue
            
            print(f"\n  ┌─ [{motion_key}] 关节点数: {len(path_data)}")
            
            # 处理不同类型的动作
            if motion_key == "close_finger":
                print(f"  │  动作: 闭合夹爪 (位置: 450)")
                global_cmd_idx += 1
                
            elif motion_key == "release_finger":
                print(f"  │  动作: 释放夹爪 (位置: 850)")
                global_cmd_idx += 1
                
            elif motion_key in ["fetch_object", "change_pose", "lift_up", "move_back"]:
                # 显示关节运动轨迹
                for step_idx, q in enumerate(path_data):
                    q_arr = np.asarray(q).reshape(-1)
                    if q_arr.shape[0] < NUM_JOINTS:
                        print(f"  │  [WARN] 第 {step_idx+1} 步关节维度不足: {q_arr.shape[0]} < {NUM_JOINTS}，已跳过")
                        continue

                    angles = q_arr[:NUM_JOINTS].astype(float).tolist()  # 只取前6轴

                    # 补偿伺服角误差（若第4轴存在）
                    if NUM_JOINTS > 3:
                        angles[3] += 3 / 180 * np.pi

                    # 转换为度数显示
                    angles_deg = [a * 180 / np.pi for a in angles]

                    # 格式化输出
                    angles_str = " | ".join([f"J{i+1}: {a:8.2f}°" for i, a in enumerate(angles_deg)])

                    is_last = (step_idx == len(path_data) - 1)
                    prefix = "  └─" if is_last else "  ├─"
                    wait_mark = " [WAIT]" if is_last else ""

                    print(f"{prefix} [{global_cmd_idx:4d}] 步 {step_idx+1:3d}/{len(path_data):3d}: {angles_str}{wait_mark}")
                    global_cmd_idx += 1
        
        valid_idx += 1
    
    print(f"\n{'='*80}")
    print(f"总命令数: {global_cmd_idx}")
    print(f"{'='*80}\n")
    
    return global_cmd_idx


def preview_mode(pkl_name):
    """
    预览模式：仅加载和显示轨迹，不执行
    """
    print("=" * 80)
    print("轨迹预览模式 (仅显示，不执行)")
    print("=" * 80)
    
    # 加载数据
    paths, meta = load_trajectory_data(pkl_name)
    
    # 显示所有轨迹
    total_cmds = display_all_trajectories(paths, meta)
    
    print("\n预览完成！")


def replay_with_udp(pkl_name, plc_ip='192.168.1.101', plc_port=10011, 
                    local_ip='192.168.1.77', local_port=10002):
    """
    使用UDP在真实机械臂上复现轨迹
    """
    print("=" * 60)
    print("UDP真实机械臂轨迹复现")
    print("=" * 60)
    
    # 加载数据
    paths, meta = load_trajectory_data(pkl_name)
    
    # 显示轨迹信息
    display_all_trajectories(paths, meta)
    
    # 初始化UDP客户端
    print(f"\n初始化UDP客户端:")
    print(f"  PLC地址: {plc_ip}:{plc_port}")
    print(f"  本地地址: {local_ip}:{local_port}")
    
    plc = PLCUdpClient(plc_ip, plc_port, local_ip, local_port)
    
    try:
        # 确认开始
        ans = input("\n请确认场景配置。继续执行? [Y|n]: ")
        if ans != "Y":
            print("操作已取消")
            return
        
        # 上电
        print("\n【步骤1】上电...")
        plc.power_on()
        time.sleep(2)
        
        # 查询初始状态
        print("\n【步骤2】查询初始状态...")
        plc.query_status()
        time.sleep(1)
        
        # 下电
        print("\n【步骤3】下电...")
        plc.power_off()
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
        plc.power_off()
    except Exception as e:
        print(f"\n错误: {e}")
        plc.power_off()
    finally:
        plc.close()


if __name__ == "__main__":
    # 文件路径
    pkl_file = "low_level_paths.pkl"
    
    # 先运行预览模式，仅显示轨迹
    preview_mode(pkl_file)
    
    # 如需执行，取消下面注释：
    # replay_with_udp(
    #     pkl_file,
    #     plc_ip='192.168.1.101',
    #     plc_port=10011,
    #     local_ip='192.168.1.77',
    #     local_port=10002
    # )