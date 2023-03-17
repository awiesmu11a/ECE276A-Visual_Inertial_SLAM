import numpy as np
from pr3_utils import *
import scipy.linalg as la

def preprocess(t, features, imu_T_cam, K, b, feature_multiple):

    dt = np.sum(np.diff(t)) / (t.shape[1] - 1)
    feature_id = []
    for i in range(features.shape[1]):
        if i % feature_multiple == 0:
            feature_id.append(i)
    feature_id = np.array(feature_id)

    cam_T_imu = inversePose(imu_T_cam)

    imu_flip = np.array([[1, 0, 0, 0],
					    [0, -1, 0, 0],
					    [0, 0, -1, 0],
					    [0, 0, 0, 1]])
    
    K_s = np.array([[K[0,0], 0, K[0,2], 0],
		 			[0, K[1,1], K[1,2], 0],
		 			[K[0,0], 0, K[0,2], -K[0,0] * b],
		 			[0, K[1,1], K[1,2], 0]])

    return dt, feature_id, cam_T_imu, imu_flip, K_s

def dead_reckon(dt, initial_pose, linear_velocity, angular_velocity):

    T = initial_pose
    twist = np.concatenate((linear_velocity, angular_velocity), axis=0)
    twist_cap = axangle2twist(twist.T)
    twist_cap = dt * twist_cap
    world_T_imu = twist2pose(twist_cap)
    temp = []
    for i in range(world_T_imu.shape[0]):
        temp.append(T @ world_T_imu[i])
        T = T @ world_T_imu[i]
    temp = np.array(temp)
    pose = np.transpose(temp, (1,2,0))

    T_inverse = inversePose(temp)

    return pose, T_inverse, temp

def landmark_groundtruth(features, feature_id, imu_T_cam, K, b, temp):

    landmark_og = []

    for i in range(features.shape[1]):
        for j in range(features.shape[2]):
            if ((features[:, i, j] + 1).any()):
                z = (K[0,0] * b) / (features[0, i, j] - features[2, i, j])
                y = z * (features[1, i, j] - K[1,2]) / K[1,1]
                x = z * (features[0, i, j] - K[0,2]) / K[0,0]
                temp_pose = np.array([x, y, z, 1]).T
                temp_pose = (temp[j] @ imu_T_cam @ temp_pose).T
                landmark_og.append(temp_pose[:3])
                break

    landmark_og = np.array(landmark_og)

    return landmark_og