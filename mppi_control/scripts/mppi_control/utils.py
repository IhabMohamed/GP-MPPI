#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Ihab S. Mohamed, Vehicle Autonomy and Intelligence Lab - Indiana University, Bloomington, USA
"""
import yaml
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv

''' @brief: Plotting the current state of the robot (x, y, theta)'''
def statePlotting(state_history, results_rootpath):
    state_length = len(state_history)

    x, y, Yaw = [s[0]
                 for s in state_history], [s[1] for s in state_history
                                           ], [s[2] for s in state_history]
    # States
    plt.figure()
    plt.subplots_adjust(hspace=0.4)
    plt.title('Jackal Full States')
    plt.subplot(211)
    plt.grid(True)
    plt.plot(range(state_length), x, 'r--', range(state_length), y, 'b--')
    plt.legend(('$x$ [m]', '$y$ [m]'))
    plt.subplot(212)
    plt.grid(True)
    plt.plot(range(state_length), Yaw, 'g-')
    plt.legend(('$\theta$ [rad.]'))
    plt.savefig(results_rootpath + '/states.png')
    plt.close()
    #plt.show()


''' @brief: Plotting the control input (linear and angular velocities)'''
def controlPlotting(control_history, average_control_sequence_history,
                    recovery_threshold_l, recovery_threshold_h,
                    results_rootpath):
    control_length = len(control_history)

    U0, U1 = [U[0] for U in control_history], [U[1] for U in control_history]
    average_U0 = np.array(average_control_sequence_history)

    plt.figure()
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(311)
    plt.grid(True)
    plt.plot(range(control_length), U0, '-', label='$v_x$')
    plt.legend(loc='upper right')
    #plt.title('Control Signal')
    plt.subplot(312)
    plt.grid(True)
    plt.plot(range(control_length), average_U0, '-', label='mean|$_{v_x}$')
    plt.axhline(y=recovery_threshold_l, color='r', linestyle='--')
    plt.axhline(y=recovery_threshold_h, color='r', linestyle='--')
    plt.legend(loc='upper right')
    plt.subplot(313)
    plt.grid(True)
    plt.plot(range(control_length), U1, 'r-', label='$w_z$')
    plt.legend(loc='upper right')
    plt.savefig(results_rootpath + '/control.png')
    #tikzplotlib.save(results_rootpath + '/control.tex')
    plt.close()


''' @brief: Plotting the running cost (state and control costs), the minimum sampled cost, 
           and the average execution time of MPPI per iteration '''
def costPlotting(state_cost_history, control_cost_history, min_cost_history,
                 iter_time_history, results_rootpath):
    cost_length = len(state_cost_history)
    iter_length = len(iter_time_history)

    plt.figure()
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 2, 1)
    plt.grid(True)
    plt.plot(range(cost_length), state_cost_history, '-')
    plt.legend(('State Cost'))
    #plt.xlabel('Iterations')
    plt.ylabel('Average Running Cost')
    #plt.title('Average Cost of the Optimal Control $u^*_0$')

    plt.subplot(2, 2, 2)
    plt.grid(True)
    plt.plot(range(cost_length), min_cost_history, '-')
    plt.legend(('Min Cost'))
    #plt.xlabel('Iterations')
    plt.ylabel('Min Cost')

    plt.subplot(2, 2, 3)
    plt.grid(True)
    plt.plot(range(cost_length), control_cost_history, '-')
    plt.legend(('Control Cost'))
    #plt.xlabel('Iterations')
    plt.ylabel('Cont. Cost')

    # computation time
    plt.subplot(2, 2, 4)
    plt.grid(True)
    plt.plot(range(iter_length), iter_time_history, '-')
    plt.xlabel('Iter')
    plt.ylabel('Time [s]')
    #plt.title('GPU Computation Time per Iteration')
    plt.savefig(results_rootpath + '/costs.png')
    

''' @brief: Plotting the robot trajectory generated by the controller within a given map'''
def trajectoryPath(state_history, obstacle_grid, map_size, results_rootpath):
    x, y = [s[0] for s in state_history], [s[1] for s in state_history]
    # Get the xy indices of the obstacles map
    obstacle_idx = grid2index(obstacle_grid, map_size)

    plt.figure()
    plt.plot(x, y, color='k', linewidth=1)
    plt.scatter([o[1] for o in obstacle_idx], [o[0] for o in obstacle_idx],
                s=3,
                c='r')
    plt.savefig(results_rootpath + '/obs_map.png')
    #plt.show()
    plt.close()


''' @brief: Retrieving the controllers' parameters, the costmap information, and summary of the performance'''
def testSummaryMPPI(init_pose, pose_desired, max_linear_velocity,
                    max_angular_velocity, av_linear_vel, map_info,
                    time_horizon, hz, weights, num_trajectories, R, Sigma_du,
                    exploration_variance, gamma, SG_window, SG_PolyOrder,
                    Distibution_Type, local_minima, violate_ctrl_const,
                    pathLength, task_completion, recovery_mode,
                    recovery_threshold_l, recovery_mode_percentage, av_t_mppi,
                    real_time_mppi, results_rootpath):
    # For MPPI
    summary = [{
        'Initial Pose [m & Rad]':
        '[' + str(init_pose[0]) + ',' + str(init_pose[1]) + ',' +
        str(init_pose[2]) + ']'
    }, {
        'Desired Pose [m & Rad]':
        '[' + str(pose_desired[0]) + ',' + str(pose_desired[1]) + ',' +
        str(pose_desired[2]) + ']'
    }, {
        'Max linear Velocity': str(max_linear_velocity)
    }, {
        'Average linear Velocity': str(av_linear_vel)
    }, {
        'Max Angular Velocity': str(max_angular_velocity)
    }, {
        'Map':
        'Costmap',
        'Map X/Y-Size [cell]':
        str(map_info["map_size"]),
        'Map Resolution [m/cell]':
        str(map_info["costmap_resolution"]),
        'Map Width [m]':
        str(map_info["map_size"] * map_info["costmap_resolution"]),
        'Map Height [m]':
        str(map_info["map_size"] * map_info["costmap_resolution"])
    }, {
        'Parameters': 'SG Filter',
        'Window Length': str(SG_window),
        'Poly. Order': str(SG_PolyOrder)
    }, {
        'Distibution Type (0: Normal, 1: Log-Normal)': Distibution_Type
    }, {
        'Control Scheme': 'MPPI',
        'Time Horizon [s]': str(time_horizon),
        'Sampling Rate [Hz]': str(hz),
        'Num of Trajectories': num_trajectories,
        'Weights': str(weights),
        'R': str(R),
        'Sigma_du': str(Sigma_du),
        'Exploration Variance': str(exploration_variance),
        'Gamma or Lambda': str(gamma)
    }, {
        'Traveled Distance [m]': str(pathLength)
    }, {
        'Task Completion [%]': str(task_completion)
    }, {
        'Recovery Mode ': recovery_mode
    }, {
        'Recovery Mode Th [m/s]': recovery_threshold_l
    }, {
        'Recovery Mode Percentage[%]': recovery_mode_percentage
    }, {
        'Average MPPI Excution Time [ms]': str(av_t_mppi * 1000)
    }, {
        'Real Time MPPI': real_time_mppi
    }, {
        'Reach Local Minima': local_minima
    }, {
        'Violate Ctrl Constraints': violate_ctrl_const
    }]

    with open(results_rootpath + '/test_summary.yaml', 'w') as f:
        yaml.dump(summary, f, sort_keys=True)


''' @brief: Saving the results of the control mission in txt file'''
def save_results(state_history, counter_history, desired_state_history,
                 state_cost_history, min_cost_history, control_history,
                 average_control_sequence_history, iter_time_history,
                 results_rootpath):
    # Get Control inputs
    v, w = [U[0] for U in control_history], [U[1] for U in control_history]
    x, y, theta = [s[0]
                   for s in state_history], [s[1] for s in state_history
                                             ], [s[2] for s in state_history]
    x_d, y_d, theta_d = [s[0] for s in desired_state_history
                         ], [s[1] for s in desired_state_history
                             ], [s[2] for s in desired_state_history]
    # counter = [s[0] for s in counter_history]

    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df['theta'] = np.asarray(theta) * 180 / np.pi
    df['counter'] = np.asarray(counter_history)
    df['x_d'] = x_d
    df['y_d'] = y_d
    df['theta_d'] = np.asarray(theta_d) * 180 / np.pi
    df['v'] = v
    df['w'] = w
    df['mean_v'] = average_control_sequence_history
    df['state_cost'] = state_cost_history
    df['min_Traj_cost'] = min_cost_history
    df['t_mppi'] = iter_time_history
    df.to_csv(results_rootpath + '/results.csv', index=False)


''' @brief: Retrieving the travelled distance by the robot'''
def getTraveledDistances(folder_path):
    TraveledDistances = readDataFromFile(folder_path +
                                         '/average_travelled_distance.csv')
    return TraveledDistances


''' @brief: Retrieving the summary over the intensive simulations'''
def intensiveSimulationSummary(folder_path):
    counter_unreachableGoals, counter_reachableGoals, counter_successful_tests, counter_semi_successful_tests, counter_real_time_MPPI, counter_unreal_time_MPPI, violate_Ctrl_constraints = 0, 0, 0, 0, 0, 0, 0
    folder_unreachableGoals, folder_reachableGoals, folder_success_tests, folder_semi_success_tests, folder_violate_Ctrl_constraints, folder_real_time_MPPI, folder_unreal_time_MPPI = [], [], [], [], [], [], []
    save_average_real_time_MPPI, save_average_unreal_time_MPPI, save_average_travelled_distance = [], [], []
    # List all directories in a certain folder
    for root, dirs, files in os.walk(folder_path, topdown=True):
        dirs.sort()  # sort files in order
        for name in dirs:
            roots = os.path.join(root, name)
            data = getDataFromYamlFile(roots + '/test_summary.yaml')

            # Extract the relevent information
            Traveled_Distance = data[8]
            Traveled_Distance = Traveled_Distance['Traveled Distance [m]']
            save_average_travelled_distance.append(Traveled_Distance)

            Average_MPPI_Excution_Time = data[9]
            Average_MPPI_Excution_Time = Average_MPPI_Excution_Time[
                'Average MPPI Excution Time [ms]']

            Real_Time_MPPI = data[10]
            Real_Time_MPPI = Real_Time_MPPI['Real Time MPPI']

            Reach_Local_Minima = data[11]
            Reach_Local_Minima = Reach_Local_Minima['Reach Local Minima']

            Violate_Ctrl_Constraints = data[12]
            Violate_Ctrl_Constraints = Violate_Ctrl_Constraints[
                'Violate Ctrl Constraints']

            # Count the unreachable goal tests and save the corresponding folder
            if Reach_Local_Minima == True:
                counter_unreachableGoals += 1
                folder_unreachableGoals.append(name)
            else:
                counter_reachableGoals += 1
                folder_reachableGoals.append(name)

            if Violate_Ctrl_Constraints == True:
                violate_Ctrl_constraints += 1
                folder_violate_Ctrl_constraints.append(name)

            if Real_Time_MPPI == True:
                counter_real_time_MPPI += 1
                folder_real_time_MPPI.append(name)
                save_average_real_time_MPPI.append(Average_MPPI_Excution_Time)
            else:
                counter_unreal_time_MPPI += 1
                folder_unreal_time_MPPI.append(name)
                save_average_unreal_time_MPPI.append(
                    Average_MPPI_Excution_Time)

            if Violate_Ctrl_Constraints == False and Reach_Local_Minima == False:
                counter_successful_tests += 1
                folder_success_tests.append(name)

            if Violate_Ctrl_Constraints == True and Reach_Local_Minima == False:
                counter_semi_successful_tests += 1
                folder_semi_success_tests.append(name)

    # Compute the average traveled distance over all tasks, average MPPI excution time
    average_traveled_distance = np.mean(
        np.array(save_average_travelled_distance).astype(np.float))
    total_traveled_distance = np.sum(
        np.array(save_average_travelled_distance).astype(np.float))
    average_real_time_MPPI = np.mean(
        np.array(save_average_real_time_MPPI).astype(np.float))
    average_unreal_time_MPPI = np.mean(
        np.array(save_average_unreal_time_MPPI).astype(np.float))
    # Save all results in a Yaml file
    summary = [{
        '# of Tasks': counter_unreachableGoals + counter_reachableGoals
    }, {
        'Reachable Goal Tasks': counter_reachableGoals
    }, {
        'Non-Reachable Goal Tasks': counter_unreachableGoals
    }, {
        'Violated Ctrl Constraints Tasks': violate_Ctrl_constraints
    }, {
        'Successful Tasks (Unviolated Ctrl Const. + Reachable)':
        counter_successful_tests
    }, {
        'Semi-Successful Tasks (Violated Ctrl Const. + Reachable)':
        counter_semi_successful_tests
    }, {
        'Average Traveled Distance [m]': str(average_traveled_distance)
    }, {
        'Total Traveled Distance [m]': str(total_traveled_distance)
    }, {
        'Real-Time Tasks': counter_real_time_MPPI
    }, {
        'Average Real-Time MPPI Excution Time [ms]':
        str(average_real_time_MPPI)
    }, {
        'Average Un-Real-Time MPPI Excution Time [ms]':
        str(average_unreal_time_MPPI)
    }]

    with open(folder_path + '/intensive_simulation_summary.yaml', 'w') as f:
        yaml.dump(summary, f, sort_keys=True)
    # Save convergence index and their folder name
    np.savetxt(folder_path + '/unreachableGoals_tests_folders.csv',
               folder_unreachableGoals,
               fmt='%s')
    np.savetxt(folder_path + '/reachableGoals_tests_folders.csv',
               folder_reachableGoals,
               fmt='%s')
    np.savetxt(folder_path + '/real_time_mppi.csv',
               folder_real_time_MPPI,
               fmt='%s')
    np.savetxt(folder_path + '/unreal_time_mppi.csv',
               folder_unreal_time_MPPI,
               fmt='%s')
    np.savetxt(folder_path + '/violate_ctrl_constraints.csv',
               folder_violate_Ctrl_constraints,
               fmt='%s')
    np.savetxt(folder_path + '/success_tests_folders.csv',
               folder_success_tests,
               fmt='%s')
    np.savetxt(folder_path + '/semi_success_tests_folders.csv',
               folder_semi_success_tests,
               fmt='%s')
    np.savetxt(folder_path + '/average_travelled_distance.csv',
               save_average_travelled_distance,
               fmt='%s')

def readDataFromFile(data_path):
    filename = data_path
    with open(filename, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=' ')
        data = list(readCSV)
        data = np.array(data).astype(float)
        #data = np.array(data)
    return data

def readStringDataFromFile(data_path):
    filename = data_path
    with open(filename, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=' ')
        data = list(readCSV)
    return data

def getDataFromYamlFile(fileName):
    data = yaml.load(open(fileName), Loader=yaml.FullLoader)
    return data


''' @brief: Returning the mean and standard deviation of the lognormal distribution, 
           given mean and variance of Normal distribution'''
def Normal2LogN(m, v):
    """ m: mean, v: variance
    Return: mu: mean, sigma: standard deviation of LN dist"""
    mu = np.exp(m + 0.5 * v)
    var = np.exp(2 * m + v) * (np.exp(v) - 1)
    sigma = np.sqrt(var)
    return mu, sigma


''' @brief: Returning the mean and variance of the Normal distribution, 
           given mean and standard deviation of lognormal distribution'''
def LogN2Normal(m, v):
    """ m: mean, v: standard deviation of LN dist.
    Return: mu: mean, var: variance"""
    mu = 2 * np.log(m) - 0.5 * np.log(np.square(m) + np.square(v))
    var = -2 * np.log(m) + np.log(np.square(m) + np.square(v))
    return mu, var


''' @brief: Returning the mean and variance of the product of lognormal and Normal distributions, 
           given the mean and variance of both Normal and Log-Normal distributions.
'''
def NLN(mu_N, var_N, mu_LN, var_LN):
    mu = mu_N * np.exp(mu_LN + 0.5 * var_LN)
    var = (np.square(mu_N) + var_N) * np.exp(
        2 * mu_LN + 2 * var_LN) - np.square(mu_N) * np.exp(2 * mu_LN + var_LN)
    return mu, var


''' @brief: Mapping from real-world poses (in meters) into positive grid '''
def map2grid(obstacles_gazebo_map, params):
    Xmin, Xmax, Ymin, Ymax = params[0], params[1], params[2], params[3]
    XmapSize, YmapSize = params[4], params[5]

    # Mapping from Gazebo obstacles pose into positive grid
    Xmap = XmapSize * (obstacles_gazebo_map[:, 0] - Xmin) / (Xmax - Xmin)
    Ymap = YmapSize * (obstacles_gazebo_map[:, 1] - Ymin) / (Ymax - Ymin)
    Xmap = np.floor(Xmap)
    Ymap = np.floor(Ymap)

    obstacle_grid = np.array([Xmap, Ymap])
    return obstacle_grid.T


''' @brief: Getting the 2D coordinates (x,y) of the obstacles in the map '''
def grid2index(obstacle_grid, map_size):
    obstacle_idx = []
    for i in range(0, map_size):
        for j in range(0, map_size):
            if obstacle_grid[i, j] > 0:
                obstacle_idx.append(np.array([i, j]))
    return obstacle_idx


# @brief:  Converting from map coordinates to world coordinates
def mapToWorld(mx, my, params):
    origin_x, origin_y, resolution = params[0], params[1], params[2]
    wx = origin_x + (mx + 0.5) * resolution
    wy = origin_y + (my + 0.5) * resolution
    return wx, wy


''' @brief:  Converting from world coordinates to map coordinates'''
def worldToMap(wx, wy, params):
    origin_x, origin_y, resolution = params[0], params[1], params[2]
    mx = int((wx - origin_x) / resolution)
    my = int((wy - origin_y) / resolution)
    return mx, my


''' @brief: Returning the pose of the obstacles, given the costmap in cells '''
def mapInMeters(costmap, params):
    map_size = params[3]
    obstacles_xy = []
    for i in range(0, int(map_size)):
        for j in range(0, int(map_size)):
            if costmap[i, j] > 0:
                wx, wy = mapToWorld(i, j, params)
                obstacles_xy.append(np.array([wx, wy]))
    return obstacles_xy


''' @brief:  Given an index... compute the associated costmap coordinates '''
def index2cell(index, size_x):
    my = np.floor(index / size_x)
    mx = index - (my * size_x)
    return mx, my


''' @brief: Calculating the travalled distance between set of points '''
def pathLength(x, y):
    n = len(x)
    lv = [
        np.sqrt((x[i + 1] - x[i])**2 + (y[i + 1] - y[i])**2)
        for i in range(n - 1)
    ]
    L = sum(lv)
    return L


''' @brief: Updating the robot states '''
def update_kinematics(s, u, dt):
    # Set the control input: linear and angular velocities
    v, w = u
    # Update x, y, yaw of the vehicle
    s[0] += np.cos(s[2]) * v * dt
    s[1] += np.sin(s[2]) * v * dt
    s[2] += w * dt
    return s