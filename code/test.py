import numpy as np
import time
from pr3_utils import *
from analysis import *
import scipy.linalg as la
import sys

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
        print("Dead Reckoning: ", i, "/", world_T_imu.shape[0])
        print("----------------------------------")
    temp = np.array(temp)
    pose = np.transpose(temp, (1,2,0))

    T_inverse = inversePose(temp)

    return pose, T_inverse, temp

def landmark_groundtruth(features, imu_T_cam, K, b, temp):

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
        
        print("Landmark Groundtruth: ", i, "/", features.shape[1])
        print("----------------------------------")

    landmark_og = np.array(landmark_og)

    return landmark_og

def feature_extraction(landmark_mean, K_s, cam_T_imu, pose_mean):

    pose_inverse = inversePose(pose_mean)
     
    feature_estimate = landmark_mean.reshape((-1, 3)).T
    feature_estimate = np.concatenate((feature_estimate, np.ones((1, feature_estimate.shape[1]))), axis=0)
    feature_estimate = K_s @ (projection((cam_T_imu @ pose_inverse @ feature_estimate).T).T)

    return feature_estimate


def find_H(T, cam_T_imu, estimate, K_s):

	H = np.zeros((4 * estimate.shape[1], 3 * estimate.shape[1]))
	estimate_jacobian = projectionJacobian(estimate.T)
	P = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
	for i in range(estimate.shape[1]):
		H[i*4:(i+1)*4, i*3:(i+1)*3] = (K_s @ estimate_jacobian[i] @ cam_T_imu @ T @ (P.T))
	
	return H

def mapping_update(landmark_og, landmark_mean_og, landmark_cov, features, feature_id, cam_T_imu, K_s, T_inverse):

    landmark_og = landmark_og[feature_id, :]
    RMSE = []
    landmark_mean = landmark_mean_og

    for i in range(features.shape[2]):

        start = time.time()
        features_obs = features[:, feature_id, i]
        
        observed = []
        for j in range(features_obs.shape[1]):
            if not ((features_obs[0, j] == -1) and (features_obs[1, j] == -1) and (features_obs[2, j] == -1) and (features_obs[3, j] == -1)):
                observed.append(j)
        observed = np.array(observed)

        if observed.shape[0] == 0:
                print("No features observed")
                print(i)
                print("Time taken: ", time.time() - start)
                print("------------------------------------------------------------------")
                continue
        
        features_obs = features_obs[:, observed]
        
        noise_obs_cov = np.eye(4 * observed.shape[0]) * 1
        noise_obs = np.random.multivariate_normal(np.zeros(4 * observed.shape[0]), noise_obs_cov)

        z = features_obs.flatten('F') + noise_obs

        landmark_mean_obs = np.zeros((3 * observed.shape[0]))
        landmark_cov_obs = np.zeros((3 * observed.shape[0], 3 * observed.shape[0]))

        for j in range(observed.shape[0]):
            landmark_mean_obs[j*3:(j+1)*3] = landmark_mean[observed[j]*3:(observed[j]+1)*3]
            landmark_cov_obs[j*3:(j+1)*3, j*3:(j+1)*3] = landmark_cov[observed[j]*3:(observed[j]+1)*3, observed[j]*3:(observed[j]+1)*3]
        
        feature_estimate = landmark_mean_obs.reshape((-1, 3)).T
        feature_estimate = np.concatenate((feature_estimate, np.ones((1, feature_estimate.shape[1]))), axis=0)
        features_estimate = projection((cam_T_imu @ T_inverse[i] @ feature_estimate).T).T

        H = find_H(T_inverse[i], cam_T_imu, features_estimate, K_s)

        features_estimate = K_s @ features_estimate
        z_cap = features_estimate.flatten('F')
        innovation = z - z_cap

        Kalman_gain = landmark_cov_obs @ H.T @ np.linalg.pinv(H @ landmark_cov_obs @ H.T + noise_obs_cov)

        if is_Singular(H @ landmark_cov_obs @ H.T + noise_obs_cov):
            print("Singular Matrix in Mapping Update")
            print(i)
            sys.exit()

        landmark_mean_obs = landmark_mean_obs + (Kalman_gain @ innovation)
        landmark_cov_obs = (np.eye(3 * observed.shape[0]) - (Kalman_gain @ H)) @ landmark_cov_obs

        for j in range(observed.shape[0]):
            landmark_mean[observed[j]*3:(observed[j]+1)*3] = landmark_mean_obs[j*3:(j+1)*3]
            landmark_cov[observed[j]*3:(observed[j]+1)*3, observed[j]*3:(observed[j]+1)*3] = landmark_cov_obs[j*3:(j+1)*3, j*3:(j+1)*3]
        
        landmark_og_obs_x = landmark_og[observed, 0]
        landmark_og_obs_y = landmark_og[observed, 1]

        landmark_mean_obs_x = landmark_mean_obs.reshape((-1, 3))[:, 0]
        landmark_mean_obs_y = landmark_mean_obs.reshape((-1, 3))[:, 1]

        RMSE.append(np.sqrt(np.mean((landmark_og_obs_x - landmark_mean_obs_x)**2 + (landmark_og_obs_y - landmark_mean_obs_y)**2)))

        print("Mapping Update: ", i, "/", features.shape[2])
        print("Time taken: ", time.time() - start)
        print("Innovation: ", np.mean(innovation))
        print("Number of features observed: ", observed.shape[0])
        print("RMSE: ", np.sqrt(np.mean((landmark_og_obs_x - landmark_mean_obs_x)**2 + (landmark_og_obs_y - landmark_mean_obs_y)**2)))
        print("------------------------------------------------------------------")
    
    RMSE = np.array(RMSE)

    return landmark_mean.reshape((-1, 3)), RMSE
