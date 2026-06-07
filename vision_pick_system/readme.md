vision_pick_system 

[xy_plane_calibration.py](https://github.com/chenzk6/bridge_construction/blob/my-saved-branch/vision_pick_system/xy_plane_calibration.py)平面标定，转换坐标到模拟环境；具体命令在源码里

[main_aruco.py](https://github.com/chenzk6/bridge_construction/blob/my-saved-branch/vision_pick_system/main_aruco.py) 获取木块的位置并转换到虚拟坐标，存储位置在data/initial_positions_xxxx-xxx.py





[initial.py](https://github.com/chenzk6/bridge_construction/blob/my-saved-branch/initial.py) 按照设置的位置初始化环境，并执行仿真，将机械臂动作存为[low_level_paths.pkl](https://github.com/chenzk6/bridge_construction/blob/my-saved-branch/low_level_paths.pkl)

[replay_with_udp.py](https://github.com/chenzk6/bridge_construction/blob/my-saved-branch/replay_with_udp.py)  发送运动命令到真实的机械臂

