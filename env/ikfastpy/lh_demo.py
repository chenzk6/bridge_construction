import sys  
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  
import env.ikfastpy.LH_ikFast as LH_ikFast  
from env.bullet_rotations import quat2mat, mat2quat, is_rotation_mat  
  
# Initialize LH kinematics  
lh_kin = LH_ikFast.PyKinematics()  
n_joints = lh_kin.getDOF()  
print(f"LH Robot IKFast - DOF: {n_joints}")  
  
success = 0  
failure_cases = []  
  
for i in range(1000):  
    # Test inverse kinematics with random poses  
    ee_pose = np.zeros((3, 4))  
      
    # Generate random rotation matrix  
    theta_noise = np.random.uniform(-np.pi, np.pi)  
    alpha = np.random.uniform(-np.pi, np.pi)  
    beta = np.random.uniform(-np.pi, np.pi)  
    ee_pose[:, :3] = quat2mat(np.array([np.sin(theta_noise / 2) * np.cos(alpha) * np.cos(beta),  
                                        np.sin(theta_noise / 2) * np.cos(alpha) * np.sin(beta),  
                                        np.sin(theta_noise / 2) * np.sin(alpha),  
                                        np.cos(theta_noise / 2)]))  
      
    assert is_rotation_mat(ee_pose[:, :3]), ee_pose[:, :3]  
    ee_pose[:, -1] = np.random.uniform(-1, 1, size=3)  
      
    # Call IKFast  
    joint_configs = lh_kin.inverse(ee_pose.reshape(-1).tolist())  
    n_solutions = int(len(joint_configs)/n_joints)  
      
    if n_solutions > 0:  
        success += 1  
        print(f"Test {i}: {n_solutions} solutions found")  
    else:  
        failure_cases.append(ee_pose)  
        print(f"Test {i}: No solutions found")  
  
print(f"\nSuccess rate: {success}/1000 ({success/10:.1f}%)")  
