import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utils.rotation as urotation

def plot_frames(data:np.ndarray, lim:float, sign:str, folder:str, dx_plot_dict:dict, sx_plot_dict:dict, plot_axis:bool=True,):
    """
    Plot motion frames of the input data.
    Input data contains x, y, z, roll, pitch, and yaw of left and right hands.

    Args:
        data (np.ndarray): input data in array format. Shape [time_series_length, num_predictors]
        lim (float): set limit bounds to x, y, and z
        sign (str): label of the passed sign
        folder (str): folder in which save frames
        dx_plot_dict (dict): **kwargs concerning right hand (x,y,z) plot
        sx_plot_dict (dict): **kwargs concerning left hand (x,y,z) plot
        plot_axis (bool, optional): whether to show or not axis. Defaults to True.
    """
    n_features_per_hand = 11
    sx_stack = data[:,3:6]
    dx_stack = data[:,3+n_features_per_hand:6+n_features_per_hand]

    v_dirs = np.array([[1, 0, 0], # versors' direction
                [0, 1, 0],
                [0, 0, 1]])

    origin = np.array([0,0,0]) # origin point, same origin point for x,y and, z
    
    for i in range(data.shape[0]): # data.shape = [ts_length, num_predictors]
    
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(projection='3d')
        
        # pick x, y, z of the left hand and plot the movements
        sx_xs = data[:,0][:i + 1]
        sx_ys = data[:,1][:i + 1]
        sx_zs = data[:,2][:i + 1]
        sx_origin = np.array([sx_xs[-1], sx_ys[-1], sx_zs[-1]])
        ax.plot(sx_xs, sx_ys, sx_zs, **sx_plot_dict)#marker='o', label='sx', color='tab:blue', alpha=1, fillstyle=None)
        
        # pick x, y, z of the right hand and plot the movements
        dx_xs = data[:,11][:i + 1]
        dx_ys = data[:,12][:i + 1]
        dx_zs = data[:,13][:i + 1]
        dx_origin = np.array([dx_xs[-1], dx_ys[-1], dx_zs[-1]])
        ax.plot(dx_xs, dx_ys, dx_zs, **dx_plot_dict)#marker='o', label='dx', color='tab:orange', alpha=1, fillstyle=None)
        
        # pick pitch, roll, yaw of the left hand and plot the orientation
        sx_roll_angle = sx_stack[i,0]
        sx_pitch_angle = sx_stack[i,1] - 0.5 # offset due to normalization
        sx_yaw_angle = sx_stack[i,2] - 0.5 # offset due to normalization
        # compute the overall rotation matrix for the left hand
        sx_R = urotation.rotation_matrix(sx_roll_angle, sx_pitch_angle, sx_yaw_angle)
        
        # pick pitch, roll, yaw of the left hand and plot the orientation
        dx_roll_angle = dx_stack[i,0]
        dx_pitch_angle = dx_stack[i,1] - 0.5 # offset due to normalization
        dx_yaw_angle = dx_stack[i,2] - 0.5 # offset due to normalization
        # compute the overall rotation matrix for the right hand
        dx_R = urotation.rotation_matrix(dx_roll_angle, dx_pitch_angle, dx_yaw_angle)
        
        U, V, W = zip(*v_dirs)
        # perform the rotation of the versors for both left and right hand
        sx_UVW = sx_R@np.array([U, V, W])
        dx_UVW = dx_R@np.array([U, V, W])

        # sx_x
        ax.quiver(*sx_origin, *sx_UVW[:,0], color=['r'], length=.1, normalize=True, label='pitch axis', arrow_length_ratio=0.1, linewidths=1.5)
        # sx_y
        ax.quiver(*sx_origin, *sx_UVW[:,1], color=['g'], length=.1, normalize=True, label='roll axis', arrow_length_ratio=0.1, linewidths=1.5)
        # sx_z
        ax.quiver(*sx_origin, *sx_UVW[:,2], color=['b'], length=.1, normalize=True, label='yaw axis', arrow_length_ratio=0.1, linewidths=1.5)

        # dx_x
        ax.quiver(*dx_origin, *dx_UVW[:,0], color=['r'], length=.1, normalize=True, label='_dx_x: roll axis', arrow_length_ratio=0.1, linewidths=1.5)
        # dx_y
        ax.quiver(*dx_origin, *dx_UVW[:,1], color=['g'], length=.1, normalize=True, label='_dx_y: pitch axis', arrow_length_ratio=0.1, linewidths=1.5)
        # dx_z
        ax.quiver(*dx_origin, *dx_UVW[:,2], color=['b'], length=.1, normalize=True, label='_dx_z: yaw axis', arrow_length_ratio=0.1, linewidths=1.5)
        
        # set the limit of the axis
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.title(f"Sign: '{sign}'")
        
        plt.legend()
        if not plot_axis:
            plt.axis('off')
        
        if i < 10:
            plt.savefig(f"{folder}/0{i}_{sign.lower()}.png", format='png', dpi=300)
        else:
            plt.savefig(f"{folder}/{i}_{sign.lower()}.png", format='png', dpi=300)
        plt.show()