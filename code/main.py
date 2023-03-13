import numpy as np
from pr3_utils import *
import time

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

	# (a) IMU Localization via EKF Prediction
	
	dt = np.sum(np.diff(t)) / (t.shape[1] - 1)

	T = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
	twist = np.concatenate((linear_velocity, angular_velocity), axis=0)
	twist_cap = axangle2twist(twist.T)
	twist_cap = dt * twist_cap
	world_T_imu = twist2pose(twist_cap)
	temp = []
	for i in range(world_T_imu.shape[0]):
		temp.append(T@world_T_imu[i])
		T = T@world_T_imu[i]
	temp = np.array(temp)
	pose = np.transpose(temp, (1,2,0))

	# (b) Landmark Mapping via EKF Update

	feature_id = []
	for i in range(features.shape[1]):
		if i % 2 == 0:
			feature_id.append(i)
	feature_id = np.array(feature_id)
	features_consider = feature_id.shape[0]

	landmark_og = []
	T_inverse = inversePose(temp)
	cam_T_imu = inversePose(imu_T_cam)

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
	m_x = landmark_og[:, 0]
	m_y = landmark_og[:, 1]
	m_z = landmark_og[:, 2]
	landmark_og = landmark_og[feature_id, :]
	
	m_mean = landmark_og.flatten('F')
	m_cov = np.eye(3 * features_consider)
	K_s = np.array([[K[0,0], 0, K[0,2], 0],
		 			[0, K[1,1], K[1,2], 0],
		 			[K[0,0], 0, K[0,2], -K[0,0] * b],
		 			[0, K[1,1], K[1,2], 0]])

	Error = []


	for i in range(features.shape[2]):

		start = time.time()
		landmarks = features[:, feature_id, i]
		observed = []

		for j in range(landmarks.shape[1]):
			if ((landmarks[:, j] + 1).any()):
				observed.append(j)
		observed = np.array(observed)

		if (observed.shape[0] == 0):
			print("No features observed")
			print(i)
			print("Time taken: ", time.time() - start)
			print("------------------------------------------------------------------")
			continue

		noise = np.random.multivariate_normal(np.zeros(4 * observed.shape[0]), np.eye(4 * observed.shape[0]))

		landmarks = landmarks[:, observed]
		og = (landmark_og[observed, :]).T
		Landmarks = landmarks.flatten('F') + noise

		m_mean_temp = np.zeros((3 * observed.shape[0]))
		m_cov_temp = np.zeros((3 * observed.shape[0], 3 * observed.shape[0]))

		for j in range(observed.shape[0]):
			m_mean_temp[j*3:(j+1)*3] = m_mean[observed[j]*3:(observed[j]+1)*3]
			m_cov_temp[j*3:(j+1)*3, j*3:(j+1)*3] = m_cov[observed[j]*3:(observed[j]+1)*3, observed[j]*3:(observed[j]+1)*3]

		landmark_estimate = m_mean.reshape((-1, 3))
		og_read = og
		landmark_estimate = landmark_estimate[observed, :].T
		og_read = np.concatenate((og_read, np.ones((1, og_read.shape[1]))), axis=0)
		landmark_estimate = np.concatenate((landmark_estimate, np.ones((1, landmark_estimate.shape[1]))), axis=0)
		og_read = projection((cam_T_imu @ T_inverse[i] @ og_read).T).T
		features_estimate = projection((cam_T_imu @ T_inverse[i] @ landmark_estimate).T).T
		
		H = find_H(T_inverse[i], cam_T_imu, features_estimate, K_s)

		og_read = K_s @ og_read
		features_estimate = K_s @ features_estimate

		Kalman_gain = m_cov_temp @ H.T @ np.linalg.inv(H @ m_cov_temp @ H.T + (np.eye(4 * observed.shape[0])))

		#Kalman_gain = m_cov @ H.T @ np.linalg.inv(H @ m_cov @ H.T + np.eye(4 * features_consider))
		z_cap = features_estimate.flatten('F')
		z = Landmarks
		m_mean_temp = m_mean_temp + Kalman_gain @ (z - z_cap)
		#m_mean = m_mean + Kalman_gain @ (z - z_cap)
		#m_cov = (np.eye(3 * features_consider) - Kalman_gain @ H) @ m_cov
		m_cov_temp = (np.eye(3 * observed.shape[0]) - Kalman_gain @ H) @ m_cov_temp

		for j in range(observed.shape[0]):
			m_mean[observed[j]*3:(observed[j]+1)*3] = m_mean_temp[j*3:(j+1)*3]
			m_cov[observed[j]*3:(observed[j]+1)*3, observed[j]*3:(observed[j]+1)*3] = m_cov_temp[j*3:(j+1)*3, j*3:(j+1)*3]

		og_read = (og_read.T).flatten('F')
		error = abs(og.flatten('F') - m_mean)
		#error = abs(og_read - z)
		error = np.sum(error) / error.shape[0]
		print(i)
		print("Time: ", time.time() - start)
		print("Error: ", error)
		if error < 1000: Error.append(error)
		print("=========================")
	
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
	visualize_trajectory_2d(pose, m2_x, m2_y, m_x, m_y, show_ori = True)