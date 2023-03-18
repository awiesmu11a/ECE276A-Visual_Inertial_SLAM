import numpy as np
from pr3_utils import *
import scipy.linalg as la

def verify_covariance(covariance):
    # Verify that the covariance is positive semi definite
    # Input:
    #   covariance: 3x3 covariance matrix
    # Output:
    #   is_positive_semi_definite: True if the covariance is positive semi definite, False otherwise
    is_positive_semi_definite = True
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        print('Covariance is not positive semi definite')
        is_positive_semi_definite = False
    
    # verify that the covariance is symmetric
    if not np.allclose(covariance, covariance.T):
        print('Covariance is not symmetric')
        is_positive_semi_definite = False
    
    return is_positive_semi_definite

def is_Singular(A):
    
    # Verify that the matrix is singular

    is_singular = False
    if np.linalg.det(A) == 0:
        is_singular = True
    return is_singular

def plot(path, landmarks, path_name, Title):
    landmarks_x = landmarks[: , 0]
    landmarks_y = landmarks[: , 1]
    visualize_trajectory_2d(path, landmarks_x, landmarks_y, Title, path_name, show_ori = True)