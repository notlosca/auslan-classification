from typing import Tuple
import numpy as np

def float_to_radiant(roll:float, pitch:float, yaw:float) -> Tuple[float,float,float]:
    """
    Function that returns the degree in radiants.
    To retrieve them, we multiply the float value by pi as described on the dataset information page.

    Args:
        roll (float): roll value belonging to [0,1]
        pitch (float): pitch value belonging to [0,1]
        yaw (float): yaw value belonging to [0,1]

    Returns:
        Tuple[float,float,float]: degrees in radiants.
    """
    roll_angle = np.pi*roll
    pitch_angle = np.pi*(pitch)
    yaw_angle = np.pi*(yaw)
    return roll_angle, pitch_angle, yaw_angle

def x_rotation(theta:float=np.pi/2) -> np.ndarray:
    """
    Returns the rotation matrix used to perform a counter clockwise rotation around the x axis

    Args:
        theta (float, optional): degree in radiants. Defaults to np.pi/2.

    Returns:
        np.ndarray: rotation matrix
    """
    # Usually roll. Since ENU framework was used, this will be the pitch.
    rot_mat = np.array([
        [1,             0,              0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), +np.cos(theta)]
        ], dtype=float)
    return rot_mat

def y_rotation(theta:float=np.pi/2) -> np.ndarray:
    """    
    Returns the rotation matrix used to perform a counter clockwise rotation around the y axis

    Args:
        theta (float, optional): degree in radiants. Defaults to np.pi/2.

    Returns:
        np.ndarray: rotation matrix
    """
    # Usually pitch. Since ENU framework was used, this will be the roll.
    rot_mat = np.array([
        [+np.cos(theta), 0,  np.sin(theta)],
        [0,              1,              0],
        [-np.sin(theta), 0, +np.cos(theta)]
        ], dtype=float)
    return rot_mat

def z_rotation(theta:float=np.pi/2) -> np.ndarray:
    """    
    Returns the rotation matrix used to perform a counter clockwise rotation around the z axis

    Args:
        theta (float, optional): degree in radiants. Defaults to np.pi/2.

    Returns:
        np.ndarray: rotation matrix
    """
    # Yaw 
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), +np.cos(theta), 0],
        [            0,              0, 1]
        ], dtype=float)
    return rot_mat

def rotation_matrix(roll:float, pitch:float, yaw:float) -> np.ndarray:
    """
    Returns the matrix used to perform rotation along each axis. It's computed by composing rotation around the x, ,y and z axis.

    Args:
        roll (float): roll value belonging to [0,1]
        pitch (float): pitch value belonging to [0,1]
        yaw (float): yaw value belonging to [0,1]

    Returns:
        np.ndarray: Rotation matrix
    """
    roll_angle, pitch_angle, yaw_angle = float_to_radiant(roll, pitch, yaw)
    roll_mat = y_rotation(roll_angle)
    pitch_mat = x_rotation(pitch_angle)
    yaw_mat = z_rotation(yaw_angle)
    R = yaw_mat@pitch_mat@roll_mat
    return R.round(decimals=2)