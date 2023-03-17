import numpy as np
import time
import sys
from pr3_utils import *
from test import *
from EKF import *

def find_H(T, cam_T_imu, estimate, K_s):

	H = np.zeros((4 * estimate.shape[1], 3 * estimate.shape[1]))
	estimate_jacobian = projectionJacobian(estimate.T)
	P = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
	for i in range(estimate.shape[1]):
		H[i*4:(i+1)*4, i*3:(i+1)*3] = (K_s @ estimate_jacobian[i] @ cam_T_imu @ T @ (P.T))
	
	return H

if __name__ == '__main__':

	# Load the measurements
	filename = "../data/10.npz"
	t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename)

	# Number of features to scaled down
	# Feature id with below mentioned multiple will be selected
	feature_multiple = 2

	initial_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

	dt, feature_id, cam_T_imu, imu_flip, K_s = preprocess(t, features, imu_T_cam, K, b, feature_multiple)
	features_eff = feature_id.shape[0]

	dead_reckon_plot, T_inverse, pose_dead_reckon = dead_reckon(dt, initial_pose, linear_velocity, angular_velocity)

	landmark_og = landmark_groundtruth(features, feature_id, imu_T_cam, K, b, pose_dead_reckon)

	m_x = landmark_og[:, 0]
	m_y = landmark_og[:, 1]
	m_z = landmark_og[:, 2]

	landmark_og = landmark_og[feature_id, :]

	pose_mean = initial_pose
	landmark_mean = landmark_og.flatten('C')
	
	covariance = np.eye(3 * features_eff + 6)
	init_cov_pose = 1
	init_cov_landmark = 1
	covariance[:6, :6] = np.eye(6) * init_cov_pose
	covariance[6:, 6:] = np.eye(3 * features_eff) * init_cov_landmark

	for i in range(features.shape[2]):

		start = time.time()
		features_obs = features[:, feature_id, i]

		observed = []
		for j in range(landmarks.shape[1]):
			if not  ((landmarks[0, j] == -1) and (landmarks[1, j] == -1) and (landmarks[2, j] == -1) and (landmarks[3, j] == -1)):
				observed.append(j)
		observed = np.array(observed)

		if (observed.shape[0] == 0):
			print("No features observed")
			print(i)
			print("Time taken: ", time.time() - start)
			print("------------------------------------------------------------------")
			continue

		features_obs = features_obs[:, observed]

		landmark_mean_obs = np.zeros((3 * observed.shape[0]))
		covariance_obs = np.zeros((3 * observed.shape[0] + 6, 3 * observed.shape[0] + 6))
		covariance_obs[:6, :6] = covariance[:6, :6]

		for j in range(observed.shape[0]):
			landmark_mean_obs[j*3:(j+1)*3] = landmark_mean[observed[j]*3:(observed[j]+1)*3]
			covariance_obs[6+j*3:6+(j+1)*3, :6] = covariance[6+observed[j]*3:6+(observed[j]+1)*3, :6]
			covariance_obs[:6, 6+j*3:6+(j+1)*3] = covariance[:6, 6+observed[j]*3:6+(observed[j]+1)*3]
		
		for j in range(observed.shape[0]):
			for k in range(observed.shape[0]):
				covariance_obs[6+j*3:6+(j+1)*3, 6+k*3:6+(k+1)*3] = covariance[6+observed[j]*3:6+(observed[j]+1)*3, 6+observed[k]*3:6+(observed[k]+1)*3]
				covariance_obs[6+k*3:6+(k+1)*3, 6+j*3:6+(j+1)*3] = covariance[6+observed[k]*3:6+(observed[k]+1)*3, 6+observed[j]*3:6+(observed[j]+1)*3]
		
		u = np.concatenate((linear_velocity[:, i], angular_velocity[:, i]))
		noise_motion_cov = np.eye(6) * 1e-3

		pose_mean, covarinace_obs = prediction(pose_mean, covariance_obs, noise_motion_cov, u, dt)

		noise_obs_cov = np.eye(4 * observed.shape[0]) * 1e-1
		noise_obs = np.random.multivariate_normal(np.zeros(4 * observed.shape[0]), noise_obs_cov)

		z = features_obs.flatten('F') + noise_obs

		pose_mean, landmark_mean_obs, covariance_obs = update(pose_mean, landmark_mean_obs, covariance_obs, z, noise_obs_cov, observed, K_s, cam_T_imu)
		
		for j in range(observed.shape[0]):
			landmark_mean[observed[j]*3:(observed[j]+1)*3] = landmark_mean_obs[j*3:(j+1)*3]
			covariance[6+observed[j]*3:6+(observed[j]+1)*3, :6] = covariance_obs[6+j*3:6+(j+1)*3, :6]
			covariance[:6, 6+observed[j]*3:6+(observed[j]+1)*3] = covariance_obs[:6, 6+j*3:6+(j+1)*3]
		
		for j in range(observed.shape[0]):
			for k in range(observed.shape[0]):
				covariance[6+observed[j]*3:6+(observed[j]+1)*3, 6+observed[k]*3:6+(observed[k]+1)*3] = covariance_obs[6+j*3:6+(j+1)*3, 6+k*3:6+(k+1)*3]
				covariance[6+observed[k]*3:6+(observed[k]+1)*3, 6+observed[j]*3:6+(observed[j]+1)*3] = covariance_obs[6+k*3:6+(k+1)*3, 6+j*3:6+(j+1)*3]
		
		print("Iteration: ", i)
		print("Time taken: ", time.time() - start)
		print("Time remaining: ", (time.time() - start) * (features.shape[2] - i))
		print("------------------------------------------------------------------")


	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update
	m_mean = landmark_og.flatten('C')
	m_cov = np.eye(3 * features_eff) * 1
	

	Error = []

	for i in range(features.shape[2]):

		start = time.time()
		landmarks = features[:, feature_id, i]
		observed = []

		for j in range(landmarks.shape[1]):
			if ((landmarks[0, j] != -1) and (landmarks[1, j] != -1) and (landmarks[2, j] != -1) and (landmarks[3, j] != -1)):
				observed.append(j)
		observed = np.array(observed)

		if (observed.shape[0] == 0):
			print("No features observed")
			print(i)
			print("Time taken: ", time.time() - start)
			print("------------------------------------------------------------------")
			continue

		noise = np.random.multivariate_normal(np.zeros(4 * observed.shape[0]), np.eye(4 * observed.shape[0]) * 1e-1)

		landmarks = landmarks[:, observed]
		og = (landmark_og[observed, :]).T
		Landmarks = landmarks.flatten('F') + noise

		m_mean_temp = np.zeros((3 * observed.shape[0]))
		m_cov_temp = np.zeros((3 * observed.shape[0], 3 * observed.shape[0]))

		for j in range(observed.shape[0]):
			m_mean_temp[j*3:(j+1)*3] = m_mean[observed[j]*3:(observed[j]+1)*3]
			m_cov_temp[j*3:(j+1)*3, j*3:(j+1)*3] = m_cov[observed[j]*3:(observed[j]+1)*3, observed[j]*3:(observed[j]+1)*3]

		landmark_estimate = m_mean.reshape((-1, 3))
		landmark_estimate = landmark_estimate[observed, :].T
		landmark_estimate = np.concatenate((landmark_estimate, np.ones((1, landmark_estimate.shape[1]))), axis=0)
		features_estimate = projection((cam_T_imu @ T_inverse[i] @ landmark_estimate).T).T
		
		H = find_H(T_inverse[i], cam_T_imu, features_estimate, K_s)

		features_estimate = K_s @ features_estimate

		Kalman_gain = m_cov_temp @ H.T @ np.linalg.pinv(H @ m_cov_temp @ H.T + (np.eye(4 * observed.shape[0]) * 1e-1))

		z_cap = features_estimate.flatten('F')
		z = Landmarks
		m_mean_temp = m_mean_temp + Kalman_gain @ (z - z_cap)
		m_cov_temp = (np.eye(3 * observed.shape[0]) - Kalman_gain @ H) @ m_cov_temp

		for j in range(observed.shape[0]):
			m_mean[observed[j]*3:(observed[j]+1)*3] = m_mean_temp[j*3:(j+1)*3]
			m_cov[observed[j]*3:(observed[j]+1)*3, observed[j]*3:(observed[j]+1)*3] = m_cov_temp[j*3:(j+1)*3, j*3:(j+1)*3]


		error = (abs(og.flatten('F') - m_mean_temp)) / (og.flatten('F'))
		error = np.sum(error) / error.shape[0]
		error * 100
		print(i)
		print("Time: ", time.time() - start)
		print("Time left: ", (time.time() - start) * (features.shape[2] - i) / 60, " minutes")
		print("Error: ", error)
		if error < 1000: Error.append(error)
		print("=========================")

		#if i == 2000:
		#	break	
	#print("Error: ", error)


	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	Error = np.array(Error)
	m_mean = m_mean.reshape((-1, 3))
	m2_x = m_mean[:, 0]
	m2_y = m_mean[:, 1]	
	#plt.scatter(m_x, m_y, c = 'r', s=0.1)
	plt.plot(Error)
	plt.show()
	visualize_trajectory_2d(dead_reckon_plot, m2_x, m2_y, m_x, m_y, show_ori = True)