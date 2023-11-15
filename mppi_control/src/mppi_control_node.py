#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ihab S. Mohamed & Mahmoud Ali, Vehicle Autonomy and Intelligence Lab - Indiana University, Bloomington, USA
"""
"""
The main ROS node for performing autonomous navigation of a differential wheeled robot (e.g., ClearPath Jackal robot)
based on MPPI, log-MPPI, OR GP-MPPI, assuming that the map is genereted online using the onboard sensor (i.e., 2d_costmap).
"""

import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile

import time
import rospy
import sys, os, datetime

# Import MPPI, Jackal Math. models, and utils
import mppi_control.utils as uls
from mppi_control.jackal import Jackal
from mppi_control.mppi import MPPI_Controller

# ROS messages
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, Point, Quaternion, Pose
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion


class mppiControllerNode():

    def __init__(self):
        rospy.init_node('mppi_control_node', anonymous=True)
        
        # Get the control frequency
        self.hz = np.array(rospy.get_param("~samplingRate"))
        self.rate = rospy.Rate(self.hz)

        # To print out the current state and the optimal control generated by the controller
        self.print_out = rospy.get_param("~print_out")

        # Retrieve the parameters of the costmap
        self.costmap_size = rospy.get_param("~costmap_size")
        self.local_costmap_resolution = rospy.get_param("~costmap_resolution")
        self.local_costmap_Origin_x = rospy.get_param("~costmap_origin_x")
        self.local_costmap_Origin_y = rospy.get_param("~costmap_origin_y")
        self.collision_cost = rospy.get_param("~collision_cost")
        self.footprint = rospy.get_param("~footprint")

        # \param "without_heading" to navigate without taking into account the heading of the robot, since we have 360 [deg.] LiDaR
        self.without_heading = rospy.get_param("~without_heading")
        if self.without_heading:
            rospy.loginfo("Headless navigation task is performed, ENJOY ;-)")
        else:
            rospy.loginfo(
                "The navigation task is being performed taken into account the robot heading angle, ENJOY ;-)"
            )
        
        """ Publishers and subscribers """
        self.Odometry_sub = rospy.Subscriber("odom",
                                             Odometry,
                                             self.OdometryCallback,
                                             queue_size=10)

        # To subscribe to the sub-goal published by the GP Subgoal Recommender node
        self.desired_pose_sub = rospy.Subscriber("gp_subgoal",
                                                 PoseStamped,
                                                 self.desiredPoseCallback,
                                                 queue_size=10)

        # To subscribe to the local costmap published by move_base (namely, costmap_2d)
        self.local_costmap_sub = rospy.Subscriber("local_costmap",
                                                  OccupancyGrid,
                                                  self.localCostmapCallback,
                                                  queue_size=1)

        self.cmd_vel_pub = rospy.Publisher("mppi/cmd_vel", Twist, queue_size=1)
        self.predicted_path_pub = rospy.Publisher("visualization_marker",
                                                  Marker,
                                                  queue_size=1)

        # Setup regular publisher of Twist message
        self.cmd_vel_msg = Twist()
    
        # Get the states and the control input dimensions
        self.state_dim = np.array(rospy.get_param("~state_dim"))
        self.control_dim = np.array(rospy.get_param("~control_dim"))

        # Initialize the states and desired states
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.targets = np.zeros(self.state_dim, dtype=np.float32)

        self.minimumDistance = np.array(rospy.get_param("~minimumDistance"))

        # Load MPPI controller parameters
        rospy.loginfo("MPPI parameters are loaded..... Enjoy :-)")
        self.time_horizon = np.array(rospy.get_param("~time_horizon"))
        self.exploration_variance = np.array(
            rospy.get_param("~exploration_variance"))
        self.num_trajectories = rospy.get_param("~num_trajectories")
        self.weights = np.array(rospy.get_param("~weights"))
        self.weights = self.weights * np.ones(self.state_dim, dtype=np.float32)
        self.std_n = np.array(rospy.get_param("~std_n"))
        self.lambda_ = rospy.get_param("~lambda")
        self.R = 1 * self.lambda_ / self.std_n
        
        """ Get the type of the distribution that will be used for updating the control inputs
        self.dist_type = 0, for Normal
        self.dist_type = 1, for logNormal
        self.dist_type = 2, ------
        """
        self.dist_type = rospy.get_param("~dist_type")
        # Set the mean and standard deviation of Log-Normal dist based on the corresponding Normal distribution.
        self.mu_LogN, self.std_LogN = uls.Normal2LogN(0, np.mean(self.std_n))
        self.LogN_info = [self.dist_type, self.mu_LogN, self.std_LogN]

        # Get the injected control noise variance
        # For MPPI
        if self.dist_type == 0:
            self.Sigma_du = np.square(self.std_n)
        # For log-MPPI
        else:
            self.Sigma_du = uls.NLN(0, np.square(self.std_n), self.mu_LogN,
                                    np.square(self.std_LogN))[1]

        # Load the parameters of the Savitsky Galoy filter
        self.SG_window = np.array(rospy.get_param("~SG_window"))
        self.SG_PolyOrder = np.array(rospy.get_param("~SG_PolyOrder"))

        # Get the maximum allowable velocities of the robot
        self.max_linear_velocity = np.array(
            rospy.get_param("~max_linear_velocity"))
        self.max_angular_velocity = np.array(
            rospy.get_param("~max_angular_velocity"))

        """ \param "gp_mppi" activates the GP-MPPI control strategy (with a "true" value). 
        Otherwise, the baselines, MPPI or log-MPPI, will be launched based on the value given to "normal_dist" 
        """
        self.gp_mppi = rospy.get_param("~gp_mppi")
        
        # Do not use the following lines with the real robot, will not move
        """
        if not self.gp_mppi:
        rospy.wait_for_service('/gazebo/reset_world')
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_world()
        """

        # Load the parameters of the recovery mode (RM) of the GP-MPPI control strategy
        self.final_goal = np.array(rospy.get_param("~final_goal"))
        self.recovery_mode = rospy.get_param("~recovery_mode")
        self.recovery_threshold_l = rospy.get_param("~recovery_threshold_l")
        self.recovery_threshold_h = rospy.get_param("~recovery_threshold_h")

        # Retrieve the map information in dictionary form
        self.map_info = {
            "map_size": self.costmap_size,
            "costmap_origin_x": self.local_costmap_Origin_x,
            "costmap_origin_y": self.local_costmap_Origin_y,
            "costmap_resolution": self.local_costmap_resolution,
            "collision_cost": self.collision_cost,
            "footprint": self.footprint
        }

        ''' Create a new instance of "Jackal" class and assign this object to the local variable "self.robot" '''
        self.robot = Jackal(self.state_dim, 1 / float(self.hz),
                            self.max_linear_velocity,
                            self.max_angular_velocity, self.map_info)
        
        ''' Create a new instance of "MPPI_Controller" class and assign this object to the local 
        variable "self.jackal_controller" '''
        self.jackal_controller = MPPI_Controller(self.state_dim,
                                                 self.control_dim,
                                                 self.num_trajectories,
                                                 self.time_horizon,
                                                 self.hz,
                                                 self.exploration_variance,
                                                 self.robot.cuda_kinematics(),
                                                 self.robot.cuda_state_cost(),
                                                 self.SG_window,
                                                 self.SG_PolyOrder,
                                                 self.LogN_info,
                                                 lambda_=self.lambda_)

        # Costs initialization
        self.state_cost, self.control_cost, self.fail = 0, 0, False
        self.jackal_controller.reset_controls()

        # Repositories for saving the results
        self.state_history, self.desired_state_history, self.counter_history = [], [], []
        self.average_control_sequence_history, self.control_history = [], []
        self.state_cost_history, self.control_cost_history, self.min_cost_history = [], [], []
        self.mppi_time_history = []

        # Create folder for saving data of the running mission
        results_folder = rospy.get_param("~results_folder")
        if results_folder:
            self.results_rootpath = os.path.join(
                results_folder,
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            os.makedirs(self.results_rootpath)
        else:
            self.results_rootpath = None

    """ @brief: Callback function for the Odometry message (i.e., robot pose) """
    def OdometryCallback(self, odometry_msg):
        # Get the robot's Pose
        Jackal_quaternion = odometry_msg.pose.pose.orientation
        Jackal_position = odometry_msg.pose.pose.position

        qx, qy, qz, qw = Jackal_quaternion.x, Jackal_quaternion.y, Jackal_quaternion.z, Jackal_quaternion.w
        x, y = Jackal_position.x, Jackal_position.y
        """ from quaternion to Euler angles (Note that: Quaternions w+ix+jy+kz are represented as [w, x, y, z]) """
        yaw = euler_from_quaternion([qx, qy, qz, qw])[2]  
        self.Jackal_states = ([x, y, yaw])

    """ @brief: Callback function triggered when the GP recommender policy suggests a new goal pose. """
    def desiredPoseCallback(self, desired_pose_msg):
        desired_position = desired_pose_msg.pose.position
        desired_quaternion = desired_pose_msg.pose.orientation
        qx, qy, qz, qw = desired_quaternion.x, desired_quaternion.y, desired_quaternion.z, desired_quaternion.w
        x, y = desired_position.x, desired_position.y
        self.yaw_desired = euler_from_quaternion([qx, qy, qz, qw])[2]
        self.desired_pose = np.array([x, y, self.yaw_desired],
                                     dtype=np.float32)
    
    """ @brief: Callback function triggered upon updating/publishing the 2D costmap """
    def localCostmapCallback(self, local_costmap_msg):
        # Retrieve the costmap's size in [pixels], data [cost/cell], and its Origin
        self.local_costmap_width = local_costmap_msg.info.width
        self.local_costmap_height = local_costmap_msg.info.height
        
        # Built a 2D costmap from the given raw data by costmap_2d package
        local_costmap = np.array(local_costmap_msg.data, dtype=np.float32)
        
        # Get the obstacle grid (i.e., 2D costmap) by directly reshape the given raw data
        local_costmap = local_costmap.reshape(
            (self.local_costmap_width, self.local_costmap_height))
        self.local_costmap = local_costmap

        # Retrieve the updated Origin
        self.local_costmap_updated_origin_x = local_costmap_msg.info.origin.position.x
        self.local_costmap_updated_origin_y = local_costmap_msg.info.origin.position.y
        self.costmap_updated_origin = np.array([
            self.local_costmap_updated_origin_x,
            self.local_costmap_updated_origin_y
        ])
        self.local_costmap_resolution = round(
            local_costmap_msg.info.resolution, 3)
        
        # Retrieve the Origin when the robot at (0,0)
        self.local_costmap_Origin_x = -self.local_costmap_width * self.local_costmap_resolution / 2
        self.local_costmap_Origin_y = -self.local_costmap_height * self.local_costmap_resolution / 2

    """ @brief: Publisher function for publishing the optimal predicted path obtained by the controller
                for visualization propose """
    def publish_predicted_path(self):
        # Publish the optimal predicted path obtained by MPPI
        self.line_strip = Marker()
        self.line_strip.id = 0
        self.line_strip.header.frame_id = "map"
        self.line_strip.header.stamp = rospy.Time.now()
        self.line_strip.type = self.line_strip.LINE_STRIP
        self.line_strip.action = self.line_strip.ADD
        self.line_strip.scale.x = 0.1
        self.line_strip.color.a = 1.0
        self.line_strip.color.b = 1.0
        self.line_strip.pose.orientation.w = 1.0
        self.line_strip.pose.position.x = 0
        self.line_strip.pose.position.y = 0
        self.line_strip.pose.position.z = 0
        self.line_strip.points = []
        for i in range(len(self.jackal_controller.U)):
            self.p = Point()
            u = self.jackal_controller.U[i, :]
            if i == 0:
                state = self.robot.update_kinematics(self.state, u)
            else:
                state = self.robot.update_kinematics(state, u)
            #print(state)
            self.p.x = state[0]
            self.p.y = state[1]
            self.p.z = 0
            self.line_strip.points.append(self.p)
        self.predicted_path_pub.publish(self.line_strip)

    """ @brief: Handling the robot's heading angle, based on each quarter, for performing a heading navigation """
    def headingAngle(self, current_state, desired_state):
        delta_x = desired_state[0] - current_state[0]
        delta_y = desired_state[1] - current_state[1]
        desired_heading = np.arctan2(delta_y, delta_x)

        if ((desired_heading - current_state[2]) <
            (-np.pi / 2.0)) and ((desired_heading - current_state[2]) >
                                 (-3.0 * np.pi / 2.0)):
            desired_heading = desired_heading - np.pi
            case = 1
        elif (((desired_heading - current_state[2]) > (np.pi / 2))
              and ((desired_heading - current_state[2]) <
                   (3.0 * np.pi / 2.0))):
            desired_heading = desired_heading + np.pi
            case = 2
        elif ((desired_heading - current_state[2]) < (-3.0 * np.pi / 2.0)):
            desired_heading = desired_heading + 2 * np.pi
            case = 3
        elif ((desired_heading - current_state[2]) > (3.0 * np.pi / 2.0)):
            desired_heading = desired_heading - 2 * np.pi
            case = 4
        else:
            desired_heading = desired_heading
            case = 5
        return desired_heading, case

    """ @brief: The primary function for running the MPPI-based algorithm """
    def run_mppi(self):
        try:
            # Sleep for 2 seconds to be sure that all sensors are active
            rospy.sleep(2.0)
            rospy.loginfo("The MPPI controller got first odometry message.")
            self.init_pose = self.Jackal_states
            self.average_control_sequence = 0
            self.counter = 0
            self.recovery_status = False

            # Set the desired pose as the final pose only if we are running MPPI or log-MPPI
            if not self.gp_mppi:
                self.desired_pose = np.copy(self.final_goal)
                self.yaw_desired = self.desired_pose[2]

            while not rospy.is_shutdown():
                """ Update the obstacle map based on the costmap acquired by the sensor """
                self.robot.update_obstacle_grid(self.local_costmap,
                                                self.costmap_updated_origin)

                # Read the current and desired states
                self.state = np.copy(self.Jackal_states)
                if self.recovery_mode or not self.gp_mppi:
                    self.targets = np.copy(self.final_goal)
                else:
                    self.targets = np.copy(self.desired_pose)

                self.distanceToGoal = np.sqrt(
                    np.square(self.targets[1] - self.state[1]) +
                    np.square(self.targets[0] - self.state[0]))

                # Condition for shutting down the node
                orientation_tolerance = 0.15
                if (self.distanceToGoal <= self.minimumDistance - 0.1) and (
                        abs(self.targets[2] - self.state[2]) <
                        orientation_tolerance):
                    rospy.signal_shutdown("Node shutting down")

                """ \param "self.counter_history" tracks the instances when the MPPI/log-MPPI
                 relies on the subgoal recommended by the GP"""    
                if not self.recovery_mode:
                    self.counter_history.append(0)
                # If recovery mode is active    
                if self.recovery_mode:
                    if self.final_goal[0] == self.desired_pose[
                            0] and self.final_goal[1] == self.desired_pose[1]:
                        self.targets = np.copy(self.final_goal)
                        self.counter_history.append(0)
                    elif self.distanceToGoal > 1 and self.average_control_sequence <= self.recovery_threshold_l:
                        sys.stdout.write("Recovery Counter: %d \n" %
                                         (self.counter))
                        self.targets = np.copy(self.desired_pose)
                        print(self.targets)
                        self.counter += 1
                        self.counter_history.append(1)
                    else:
                        self.targets = np.copy(self.final_goal)
                        self.counter_history.append(0)

                if self.without_heading:
                    if self.distanceToGoal > self.minimumDistance:
                        self.targets[2] = self.state[2]
                    else:
                        self.targets[2] = self.yaw_desired
                else:
                    # Calculate the desired yaw angle (keep the robot heading towards the goal)
                    desired_heading, case = self.headingAngle(
                        self.state, self.targets)
                    if self.distanceToGoal > self.minimumDistance:
                        self.targets[2] = desired_heading
                    else:
                        self.targets[2] = self.yaw_desired
                    # Increase the weight of the 3rd state (yaw) so that the robot can head towards the goal as soon as possible
                    self.weights[2] = 30.0

                # For computing the excution time of MPPI
                start = time.time()
                
                # Compute the optimal control
                u, normalizer, min_cost = self.jackal_controller.compute_control(
                    self.state,
                    [self.std_n, self.R, self.weights, self.targets])
                
                # \param "t_mppi" returns the excution time of MPPI
                t_mppi = time.time() - start
                
                # Record the costs
                costs = self.robot.cost(self.state, u,
                                        [self.weights, self.targets, self.R])
                self.state_cost = costs[0]
                self.control_cost = costs[1]
                
                # Record and update the state, and the mppi excution time
                self.control_history.append(u)
                self.state_history.append(np.copy(self.state))
                self.desired_state_history.append(np.copy(self.targets))
                self.state_cost_history.append(self.state_cost)
                self.min_cost_history.append(min_cost)
                self.control_cost_history.append(self.control_cost)
                self.mppi_time_history.append(t_mppi)

                # print("--------------------------------------")
                U = np.array(self.jackal_controller.U)
                
                """ Calculate the mean linear velocity over the entire control horizon to 
                anticipate the occurrence of local minima.""" 
                average_control_sequence = np.mean(np.abs(U[:, 0]))
                self.average_control_sequence = average_control_sequence
                self.average_control_sequence_history.append(
                    average_control_sequence)

                if self.print_out:
                    sys.stdout.write(
                        "Current States: (%f, %f, %f), t_mppi: %f \n" %
                        (self.state[0], self.state[1],
                         self.state[2] * 180 / np.pi, t_mppi))
                    sys.stdout.write("U: (%f, %f) \n" % (u[0], u[1]))
                    print(
                        "------------------------------------------------------"
                    )
                    sys.stdout.flush()
                if np.isnan(np.sum(self.state)):
                    print("Breaking MPPI due to numerical errors")
                    self.fail = True
                    break
                
                # Publish the linear and angular velocities to the robot
                self.cmd_vel_msg.linear.x = u[0]
                self.cmd_vel_msg.linear.y = 0
                self.cmd_vel_msg.linear.z = 0
                self.cmd_vel_msg.angular.x = 0
                self.cmd_vel_msg.angular.y = 0
                self.cmd_vel_msg.angular.z = u[1]
                self.cmd_vel_pub.publish(self.cmd_vel_msg)

                # Publish the optimal predicted path obtained by MPPI
                self.publish_predicted_path()
                self.rate.sleep()
        except rospy.ROSInterruptException:
            print("ROS Terminated")
            pass

    """@brief: Plotting the states, corresponding optimal control action, and instantaneous running cost """        
    def dataPlotting(self):
        # Plot all states and their corresponding optimal control sequence
        uls.statePlotting(self.state_history, self.results_rootpath)
        uls.controlPlotting(self.control_history,
                            self.average_control_sequence_history,
                            self.recovery_threshold_l,
                            self.recovery_threshold_h, self.results_rootpath)
        uls.costPlotting(self.state_cost_history, self.control_cost_history,
                         self.min_cost_history, self.mppi_time_history,
                         self.results_rootpath)

    ''' @brief: Retrieving the controllers' parameters, the costmap information, and summary of the performance'''    
    def test_summary(self):
        if self.distanceToGoal > 1.0:
            self.local_minima = True
        else:
            self.local_minima = False
        control_history = np.array(self.control_history)
        if max(control_history[:, 0]) > self.max_linear_velocity or max(
                control_history[:, 1]) > self.max_angular_velocity:
            self.violate_ctrl_const = True
        else:
            self.violate_ctrl_const = False
        x, y = [s[0] for s in self.state_history
                ], [s[1] for s in self.state_history]

        self.pathLength = uls.pathLength(x, y)

        # Compute the Task completion percentage
        if self.distanceToGoal <= self.minimumDistance:
            self.task_completion = 100
            # rospy.signal_shutdown("Node shutting down")
        else:
            self.task_completion = (
                self.pathLength /
                (self.distanceToGoal + self.pathLength)) * 100
            if self.task_completion >= 98.5:
                self.task_completion = 100

        # Calculate the percentage of GP assistant contribution to MPPI/log-MPPI when recovery mode is activated.
        num_zeros = self.counter_history.count(0)
        num_ones = self.counter_history.count(1)
        self.recovery_mode_percentage = 100 * (num_ones /
                                               (num_ones + num_zeros))

        # Calculate the average linear velocity of the robot throughout the entire assigned control mission
        v_history = [U[0] for U in self.control_history]
        v_history = np.abs(v_history)
        v_history = [v for v in v_history if v >= 0.08]
        self.av_linear_vel = np.mean(np.array(v_history))

        # Compute the average execution time of the controller across all iterations.
        self.av_t_mppi = np.mean(np.array(self.mppi_time_history)[20:])
        if self.av_t_mppi < 1 / float(self.hz):
            self.real_time_mppi = True
        else:
            self.real_time_mppi = False
            self.real_time_mppi = False
        
        uls.testSummaryMPPI(
            self.init_pose, self.targets, self.max_linear_velocity,
            self.max_angular_velocity, self.av_linear_vel, self.map_info,
            self.time_horizon, self.hz, self.weights, self.num_trajectories,
            self.R, self.Sigma_du, self.exploration_variance, self.lambda_,
            self.SG_window, self.SG_PolyOrder, self.dist_type,
            self.local_minima, self.violate_ctrl_const, self.pathLength,
            self.task_completion, self.recovery_mode,
            self.recovery_threshold_l, self.recovery_mode_percentage,
            self.av_t_mppi, self.real_time_mppi, self.results_rootpath)
        
        uls.save_results(self.state_history, self.counter_history,
                         self.desired_state_history, self.state_cost_history,
                         self.min_cost_history, self.control_history,
                         self.average_control_sequence_history,
                         self.mppi_time_history, self.results_rootpath)


if __name__ == "__main__":
    MPPI_Controller_Node = mppiControllerNode()
    MPPI_Controller_Node.run_mppi()
    MPPI_Controller_Node.test_summary()
    MPPI_Controller_Node.dataPlotting()