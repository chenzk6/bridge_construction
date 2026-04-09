import pybullet as p
import pybullet_utils.bullet_client as bc
from env.robots import LHRobot
import pybullet_data
import numpy as np
import time
import threading
import sys

# 创建 GUI 客户端
pc = bc.BulletClient(connection_mode=p.GUI)
pc.setGravity(0, 0, -9.8)
pc.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=90,
    cameraPitch=-30,
    cameraTargetPosition=[0.85, 0.6, 0.0]
)

# 设置 PyBullet 数据路径
pc.setAdditionalSearchPath(pybullet_data.getDataPath())
pc.loadURDF("plane.urdf")

# 创建机器人
robot = LHRobot(pc)

print("\n" + "="*70)
print("🎨 交互式机械臂可视化工具（带滑块显示）")
print("="*70)

# 全局变量
current_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
gripper_value = 0.0

# ✅ 创建滑块（只读显示）
sliders = []
slider_ids = []
joint_names = ["基座旋转", "肩部", "肘部", "手腕1", "手腕2", "手腕3"]

for i in range(6):
    slider_id = pc.addUserDebugParameter(
        f"关节{i} ({joint_names[i]})",
        robot.joint_ll[i],
        robot.joint_ul[i],
        0.0
    )
    sliders.append(slider_id)
    slider_ids.append(slider_id)

# 夹爪滑块
gripper_slider_id = pc.addUserDebugParameter(
    "夹爪 (0=打开, 1=闭合)",
    0.0, 1.0, 0.0
)

# 预设配置
presets = {
    '1': ("零位", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    '2': ("起点", [0.382, 1.079, -0.222, 3.127, 0.695, 0.272]),
    '3': ("终点", [0.357, 0.657, -0.031, 1.841, 0.980, -0.448]),
    '4': ("抬起", [0.0, -np.pi/3, np.pi/4, 0.0, np.pi/6, 0.0]),
}

def print_help():
    """打印帮助信息"""
    print("\n" + "="*70)
    print("📖 使用说明")
    print("="*70)
    print("\n【输入关节角度】")
    print("  直接输入 6 个数字（用空格或逗号分隔）")
    print("  示例 1: 0.382 1.079 -0.222 3.127 0.695 0.272")
    print("  示例 2: 0.382, 1.079, -0.222, 3.127, 0.695, 0.272")
    print("  示例 3: [0.382, 1.079, -0.222, 3.127, 0.695, 0.272]")
    print("\n【快捷命令】")
    print("  1, 2, 3, 4  - 切换到预设配置")
    print("  p           - 打印当前关节角度和末端位置")
    print("  c           - 复制当前关节角度")
    print("  g [0-1]     - 设置夹爪开合度 (0=打开, 1=闭合)")
    print("  h           - 显示此帮助")
    print("  q           - 退出程序")
    print("\n【右侧滑块】")
    print("  ⚠️  滑块仅用于显示，不可直接拖动")
    print("  💡 请通过输入数字来设置关节角度")
    print("\n【预设配置】")
    for key, (name, joints) in presets.items():
        joints_str = ', '.join([f'{j:.3f}' for j in joints])
        print(f"  {key}. {name:8s} : [{joints_str}]")
    print("="*70 + "\n")

# 坐标轴绘制
def draw_axes(position, orientation, length=0.15):
    """绘制 XYZ 坐标轴"""
    rot_mat = p.getMatrixFromQuaternion(orientation)
    rot_mat = np.array(rot_mat).reshape(3, 3)
    
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB = XYZ
    for i in range(3):
        axis_end = position + rot_mat[:, i] * length
        pc.addUserDebugLine(
            position, axis_end,
            lineColorRGB=colors[i],
            lineWidth=3,
            lifeTime=0.1
        )

# ✅ 更新滑块显示
def update_sliders():
    """更新滑块位置（通过删除重建实现）"""
    global sliders, slider_ids, gripper_slider_id
    
    # PyBullet 不支持直接修改滑块值，需要删除重建
    # 删除旧滑块
    for slider_id in slider_ids:
        pc.removeUserDebugItem(slider_id)
    pc.removeUserDebugItem(gripper_slider_id)
    
    # 重建滑块
    slider_ids = []
    for i in range(6):
        slider_id = pc.addUserDebugParameter(
            f"关节{i} ({joint_names[i]})",
            robot.joint_ll[i],
            robot.joint_ul[i],
            current_joints[i]  # ← 使用当前值
        )
        slider_ids.append(slider_id)
    
    # 重建夹爪滑块
    gripper_slider_id = pc.addUserDebugParameter(
        "夹爪 (0=打开, 1=闭合)",
        0.0, 1.0, gripper_value
    )

# 输入处理线程
def input_thread():
    """处理用户输入"""
    global current_joints, gripper_value
    
    print_help()
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            # 退出
            if user_input.lower() == 'q':
                print("\n✅ 退出程序...")
                sys.exit(0)
            
            # 帮助
            elif user_input.lower() == 'h':
                print_help()
            
            # 打印状态
            elif user_input.lower() == 'p':
                print("\n" + "-"*70)
                print(f"当前关节角度 (弧度): {current_joints}")
                print(f"当前关节角度 (度):   {[f'{x:.2f}' for x in np.rad2deg(current_joints)]}")
                eef_pos = robot.get_end_effector_pos()
                eef_quat = robot.get_end_effector_orn(as_type="quat")
                eef_euler = robot.get_end_effector_orn(as_type="euler")
                print(f"末端位置: [{eef_pos[0]:.4f}, {eef_pos[1]:.4f}, {eef_pos[2]:.4f}]")
                print(f"末端姿态 (四元数): {[f'{x:.4f}' for x in eef_quat]}")
                print(f"末端姿态 (欧拉角): {[f'{x:.2f}°' for x in np.rad2deg(eef_euler)]}")
                print(f"夹爪状态: {gripper_value:.2f} (0=打开, 1=闭合)")
                print("-"*70)
            
            # 复制
            elif user_input.lower() == 'c':
                try:
                    import pyperclip
                    joints_str = ', '.join([f'{j:.6f}' for j in current_joints])
                    pyperclip.copy(joints_str)
                    print(f"✅ 已复制到剪贴板: [{joints_str}]")
                except ImportError:
                    joints_str = ', '.join([f'{j:.6f}' for j in current_joints])
                    print(f"📋 请手动复制: [{joints_str}]")
                    print("   (安装 pyperclip 以启用自动复制: pip install pyperclip)")
            
            # 夹爪控制
            elif user_input.lower().startswith('g '):
                try:
                    value = float(user_input.split()[1])
                    if 0 <= value <= 1:
                        gripper_value = value
                        update_sliders()  # ← 更新滑块
                        print(f"✅ 夹爪设置为: {gripper_value:.2f}")
                    else:
                        print("❌ 夹爪值必须在 0-1 之间")
                except:
                    print("❌ 格式错误，使用: g 0.5")
            
            # 预设配置
            elif user_input in presets:
                name, joints = presets[user_input]
                current_joints = joints.copy()
                update_sliders()  # ← 更新滑块
                print(f"✅ 切换到配置: {name}")
                print(f"   关节角度: {current_joints}")
                print(f"   角度(度): {[f'{x:.2f}°' for x in np.rad2deg(current_joints)]}")
            
            # 解析关节角度
            else:
                # 清理输入
                clean_input = user_input.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
                parts = clean_input.replace(',', ' ').split()
                
                if len(parts) == 6:
                    new_joints = [float(x) for x in parts]
                    
                    # 检查限位
                    valid = True
                    for i, j in enumerate(new_joints):
                        if not (robot.joint_ll[i] <= j <= robot.joint_ul[i]):
                            print(f"⚠️  关节{i} 超出限位: {j:.3f} rad "
                                  f"(范围: [{robot.joint_ll[i]:.3f}, {robot.joint_ul[i]:.3f}])")
                            valid = False
                    
                    if valid:
                        current_joints = new_joints
                        update_sliders()  # ← 更新滑块
                        print(f"✅ 设置关节角度: {current_joints}")
                        print(f"   角度(度): {[f'{x:.2f}°' for x in np.rad2deg(current_joints)]}")
                    else:
                        print("❌ 有关节超出限位，未应用")
                
                elif len(parts) == 1 and parts[0].replace('.', '').replace('-', '').isdigit():
                    print(f"❌ 只检测到 1 个数字，需要 6 个关节角度")
                    print(f"   提示: 请用空格或逗号分隔，例如: 0.1 0.2 0.3 0.4 0.5 0.6")
                
                else:
                    print(f"❌ 需要输入 6 个关节角度，检测到 {len(parts)} 个")
                    print(f"   您的输入: {parts}")
        
        except ValueError as e:
            print(f"❌ 输入格式错误: {e}")
            print("   提示: 确保输入的是数字，例如: 0.382 1.079 -0.222 3.127 0.695 0.272")
        except KeyboardInterrupt:
            print("\n✅ 退出程序...")
            sys.exit(0)
        except Exception as e:
            print(f"❌ 错误: {e}")

# 启动输入线程
input_thread_obj = threading.Thread(target=input_thread, daemon=True)
input_thread_obj.start()

# 主循环
print("\n🚀 可视化窗口已启动，请直接拖动右侧滑块控制机械臂\n")

try:
    while True:
        # 直接读取6个关节滑块
        current_joints = [pc.readUserDebugParameter(sid) for sid in slider_ids]

        # 读取夹爪滑块
        gripper_value = pc.readUserDebugParameter(gripper_slider_id)

        # 应用关节角度
        gripper_angles = [0.3 * gripper_value] * 6
        full_qpos = current_joints + gripper_angles + [0.0] * 3
        robot.reset_qpos(qpos=full_qpos)

        # 获取末端状态
        eef_pos = robot.get_end_effector_pos()
        eef_quat = robot.get_end_effector_orn(as_type="quat")
        eef_euler = np.rad2deg(robot.get_end_effector_orn(as_type="euler"))

        # 绘制坐标轴
        draw_axes(eef_pos, eef_quat, length=0.15)

        # 状态栏
        print(
            f"\r末端位置: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}] "
            f"姿态: [{eef_euler[0]:6.1f}°, {eef_euler[1]:6.1f}°, {eef_euler[2]:6.1f}°] "
            f"夹爪: {gripper_value:.2f}   ",
            end="",
            flush=True
        )

        pc.stepSimulation()
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n\n✅ 程序正常退出")