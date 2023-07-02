import numpy as np
from pr3_utils import *
from analysis import *
import scipy.linalg as la
import sys

def calc_H(pose_inverse, feature_estimate, landmark_estimate, K_s, cam_T_imu):

    P = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    H = np.zeros((4 * feature_estimate.shape[1], 6 + 3 * feature_estimate.shape[1]))
    jacobian_projection = projectionJacobian(feature_estimate.T)

    temp_dot = np.zeros((4, 6))

    for i in range(feature_estimate.shape[1]):

        temp = pose_inverse @ landmark_estimate[:, i]
        temp_dot[:3, :3] = np.eye(3)
        temp_dot[:3, 3:] = -1 * axangle2skew(temp[:3])


        H[i*4:(i+1)*4, 6+i*3:6+(i+1)*3] = (K_s @ jacobian_projection[i] @ cam_T_imu @ pose_inverse @ (P.T))
        H[i*4:(i+1)*4, :6] = -1 * (K_s @ jacobian_projection[i] @ cam_T_imu @ temp_dot)
    
    return H

def prediction(pose_mean, covariance, noise_motion_cov, u, dt):

    u_hat = axangle2twist(u.T)
    u_hat = dt * u_hat

    u_curly_hat = axangle2adtwist(u.T)
    u_curly_hat = -dt * u_curly_hat
    F = la.expm(u_curly_hat)

    pose_mean = pose_mean @ twist2pose(u_hat)
    covariance[:6, :6] = F @ covariance[:6, :6] @ F.T + noise_motion_cov

    covariance[6:, :6] = covariance[6:, :6] @ F.T
    covariance[:6, 6:] = F @ covariance[:6, 6:]

    return pose_mean, covariance

def update(pose_mean, landmark_mean, covariance, z, noise_obs_cov, K_s, cam_T_imu):

    pose_inverse = inversePose(pose_mean)

    feature_estimate = landmark_mean.reshape((-1, 3)).T
    feature_estimate = np.concatenate((feature_estimate, np.ones((1, feature_estimate.shape[1]))), axis=0)

    landmark_estimate = feature_estimate

    feature_estimate = projection((cam_T_imu @ pose_inverse @ feature_estimate).T).T

    H = calc_H(pose_inverse, feature_estimate, landmark_estimate, K_s, cam_T_imu)

    feature_estimate = K_s @ feature_estimate
    z_cap = feature_estimate.flatten('F')
    innovation = z - z_cap

    Kalman_gain = covariance @ H.T @ np.linalg.pinv(H @ covariance @ H.T + noise_obs_cov)

    pose_mean = pose_mean @ twist2pose(axangle2twist(Kalman_gain[:6, :] @ innovation))
    landmark_mean = landmark_mean + (Kalman_gain[6:, :] @ innovation)
    covariance = (np.eye(6 + 3 * landmark_estimate.shape[1]) - (Kalman_gain @ H)) @ covariance

    return pose_mean, landmark_mean, covariance