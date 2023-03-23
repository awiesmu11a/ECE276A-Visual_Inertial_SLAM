import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from pr3_utils import *
from test import *
from EKF import *
from analysis import *

if __name__ == '__main__':

	dataset = sys.argv[1]
	# Load the measurements
	filename = "../data/" + dataset + ".npz"
	t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename)

	# Number of features to scaled down
	# Feature id with below mentioned multiple will be selected
	feature_multiple = 3

	initial_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

	preprocess_time = time.time()

	dt, feature_id, cam_T_imu, imu_flip, K_s = preprocess(t, features, imu_T_cam, K, b, feature_multiple)
	features_eff = feature_id.shape[0]

	dead_reckon_plot, T_inverse, pose_dead_reckon = dead_reckon(dt, initial_pose, linear_velocity, angular_velocity)

	landmark_og = landmark_groundtruth(features, imu_T_cam, K, b, pose_dead_reckon)

	print("Time taken for Dead reckon and Landmark Groundtruth: ", time.time() - preprocess_time)

	#plot(dead_reckon_plot, landmark_og, "Dead Reckon", "Dead_Reckon with Ground Truth Landmarks")

	pose_mean = initial_pose
	landmark_mean = landmark_og[feature_id, :].flatten('C')
	
	covariance = np.eye(3 * features_eff + 6)
	init_cov_pose = 5
	init_cov_landmark = 5
	covariance[:6, :6] = np.eye(6) * init_cov_pose
	covariance[6:, 6:] = np.eye(3 * features_eff) * init_cov_landmark

	
	landmark_cov = covariance[6:, 6:]
	"""
	mapping_time = time.time()

	landmark_mapping_update, RMSE_landmarks = mapping_update(landmark_og, landmark_mean, landmark_cov, features, feature_id, imu_flip @ cam_T_imu, K_s, T_inverse)

	print("Time taken for mapping update: ", time.time() - mapping_time)

	plt.plot(RMSE_landmarks, label = "RMSE of Landmarks estimates")
	plt.title("RMSE of Landmarks estimates")
	plt.show(block=True)
	plt.close

	plot(dead_reckon_plot, landmark_mapping_update, "Dead Reckon", "Dead_Reckon with Mapping Update Landmarks")
	"""
	pose_pred_step = []
	pose_update_step = []

	Error_obs = []
	Error_pred = []

	ekf_time = time.time()

	for i in range(features.shape[2]):

		start = time.time()
		features_obs = features[:, feature_id, i]

		observed = []
		for j in range(features_obs.shape[1]):
			if not ((features_obs[0, j] == -1) and (features_obs[1, j] == -1) and (features_obs[2, j] == -1) and (features_obs[3, j] == -1)):
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
			landmark_mean_obs[j * 3: (j + 1) * 3] = landmark_mean[observed[j] * 3: (observed[j] + 1) * 3]
			covariance_obs[6 + j * 3: 6 + (j + 1) * 3, :6] = covariance[6 + observed[j] * 3: 6 + (observed[j] + 1) * 3, :6]
			covariance_obs[:6, 6 + j * 3: 6 + (j + 1) * 3] = covariance[:6, 6 + observed[j] * 3: 6 + (observed[j] + 1) * 3]
		
		for j in range(observed.shape[0]):
			for k in range(observed.shape[0]):
				covariance_obs[6 + j * 3: 6 + (j + 1) * 3, 6 + k * 3: 6 + (k + 1) * 3] = covariance[6 + observed[j] * 3: 6 + (observed[j] + 1) * 3, 6 + observed[k] * 3: 6 + (observed[k] + 1) * 3]
				covariance_obs[6 + k * 3: 6 + (k + 1) * 3, 6 + j * 3: 6 + (j + 1) * 3] = covariance[6 + observed[k] * 3: 6 + (observed[k] + 1) * 3, 6 + observed[j] * 3: 6 + (observed[j] + 1) * 3]
		
		u = np.concatenate((linear_velocity[:, i], angular_velocity[:, i]))
		noise_motion_cov = np.eye(6) * 1

		pose_mean, covarinace_obs = prediction(pose_mean, covariance_obs, noise_motion_cov, u, dt)

		noise_obs_cov = np.eye(4 * observed.shape[0]) * 10
		noise_obs = np.random.multivariate_normal(np.zeros(4 * observed.shape[0]), noise_obs_cov)

		z = features_obs.flatten('F') + noise_obs

		z_prefilter = feature_extraction(landmark_mean_obs, K_s, imu_flip @ cam_T_imu, pose_mean)
		z_prefilter = z_prefilter.flatten('F')
		error_pred = z - z_prefilter
		
		pose_pred_step.append(pose_mean)
		"""
		if not verify_covariance(covariance_obs):
			print("Covariance error in prediction step")
			print(i)
			print(covariance_obs)
			sys.exit()
		"""

		pose_mean, landmark_mean_obs, covariance_obs = update(pose_mean, landmark_mean_obs, covariance_obs, z, noise_obs_cov, K_s, imu_flip @ cam_T_imu)

		z_postfilter = feature_extraction(landmark_mean_obs, K_s, imu_flip @ cam_T_imu, pose_mean)
		z_postfilter = z_postfilter.flatten('F')
		error_update = z - z_postfilter

		pose_update_step.append(pose_mean)
		"""
		if not verify_covariance(covariance_obs):
			print("Covariance error in update step")
			print(i)
			print(covariance_obs)
			sys.exit()
		"""
		
		for j in range(observed.shape[0]):
			landmark_mean[observed[j]*3:(observed[j]+1)*3] = landmark_mean_obs[j*3:(j+1)*3]
			covariance[6+observed[j]*3:6+(observed[j]+1)*3, :6] = covariance_obs[6+j*3:6+(j+1)*3, :6]
			covariance[:6, 6+observed[j]*3:6+(observed[j]+1)*3] = covariance_obs[:6, 6+j*3:6+(j+1)*3]
		
		for j in range(observed.shape[0]):
			for k in range(observed.shape[0]):
				covariance[6+observed[j]*3:6+(observed[j]+1)*3, 6+observed[k]*3:6+(observed[k]+1)*3] = covariance_obs[6+j*3:6+(j+1)*3, 6+k*3:6+(k+1)*3]
				covariance[6+observed[k]*3:6+(observed[k]+1)*3, 6+observed[j]*3:6+(observed[j]+1)*3] = covariance_obs[6+k*3:6+(k+1)*3, 6+j*3:6+(j+1)*3]
		
		avg_relative_error_pred = np.mean(error_pred / z) * 100
		Error_pred.append(avg_relative_error_pred)
		
		avg_relative_error = np.mean(error_update / z) * 100
		Error_obs.append(avg_relative_error)

		print("EKF SLAM: Iteration ", i, " of ", features.shape[2])
		print("Time taken: ", time.time() - start)
		print("Time remaining: ", (time.time() - start) * (features.shape[2] - i))
		print("Obs. Error after prediction step: ", np.mean(error_pred))
		print("Obs. Error after update step: ", np.mean(error_update))
		print("------------------------------------------------------------------")
	
	print("Time taken for EKF SLAM: ", time.time() - ekf_time)

	Error_obs = np.array(Error_obs)
	Error_pred = np.array(Error_pred)

	pose_pred_step = np.array(pose_pred_step)
	pose_pred_step = np.transpose(pose_pred_step, (1, 2, 0))

	pose_update_step = np.array(pose_update_step)
	pose_update_step = np.transpose(pose_update_step, (1, 2, 0))

	landmarks = landmark_mean.reshape((-1, 3))

	plt.plot(Error_obs, label="Average relative error in innovation")
	plt.plot(Error_pred, label="Average realtive error after prediction step")
	plt.title("Average relative error")
	plt.show()
	plt.close()

	plot(pose_pred_step, landmark_og, "Trajectory after prediction step", "Trajectory after prediction step with landmark mapping update")

	plot(pose_update_step, landmarks, "Final EKF SLAM trajectory", "Final SLAM result")

	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update
	
	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time