import numpy as np
from pr3_utils import *
import time

def find_H(T, imu_T_cam, estimate):

	H = np.zeros((4 * estimate.shape[1], 3 * estimate.shape[1]))
	estimate_jacobian = projectionJacobian(estimate.T)
	P = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
	for i in range(estimate.shape[1]):
		H[i*4:(i+1)*4, i*3:(i+1)*3] = (estimate_jacobian[i] @ imu_T_cam @ T @ (P.T))
	
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

	features_consider = 1000
	feature_id = np.random.choice(features.shape[1], features_consider, replace=False)
	m_mean = np.zeros((3 * features_consider))
	m_cov = np.eye(3 * features_consider)
	K_s = np.array([[K[0,0], 0, K[0,2], 0],[0, K[1,1], K[1,2], 0],[K[0,0], 0, K[0,2], -K[0,0]*b],[0, K[1,1], K[1,2], 0]])

	T_inverse = inversePose(temp)

	noise = np.random.multivariate_normal(np.zeros(4 * features_consider), np.eye(4 * features_consider))

	for i in range(features.shape[2]):	

		start = time.time()
		landmarks = features[:, feature_id, i]
		Landmarks = landmarks.flatten('F') + noise

		for j in range(landmarks.shape[1]):
			if not ((landmarks[:, j] + 1).all()):
				Landmarks[j*4:(j+1)*4] = np.zeros(4)

		landmark_estimate = m_mean.reshape((-1, 3)).T
		landmark_estimate = np.concatenate((landmark_estimate, np.ones((1, landmark_estimate.shape[1]))), axis=0)
		features_estimate = projection((imu_T_cam @ T_inverse[i] @ landmark_estimate).T).T
		features_estimate = K_s @ features_estimate

		H = find_H(T_inverse[i], imu_T_cam, features_estimate)

		Kalman_gain = m_cov @ H.T @ np.linalg.inv(H @ m_cov @ H.T + np.eye(4 * features_consider))
		z_cap = features_estimate.flatten('F')
		z = Landmarks
		m_mean = m_mean + Kalman_gain @ (z - z_cap)
		m_cov = (np.eye(3 * features_consider) - Kalman_gain @ H) @ m_cov

		error = abs(z - z_cap)
		error = np.sum(error) / error.shape[0]
		print(i)
		print("Time: ", time.time() - start)
		print("Error: ", error)
		print("=========================")
		if i == 100:
			break
	
	print("Error: ", error)


	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	visualize_trajectory_2d(pose, show_ori = True)