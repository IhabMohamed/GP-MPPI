from typing import Tuple, Optional
import tempfile
import pathlib
import warnings

import io
import os
import math
import numpy as np
from time import time
from scipy import stats
from datetime import datetime

import pcl
import rospy
import ros_numpy
from tf.transformations import quaternion_from_euler

### import ros msgs
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from gp_subgoal.msg import PosePcl2
from geometry_msgs.msg import PoseStamped, PointStamped
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

### import opencv
import cv2
from cv_bridge import CvBridge, CvBridgeError  #after cv2


### to disable GPU for GP training and prediction: 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
### to select GPU
# tf.device("gpu:0")


### import tensrflow and gpflow and related lib
import tensorflow as tf
import gpflow
from gpflow.config import default_float
from gpflow.ci_utils import ci_niter
from gpflow.utilities import to_default_float
from gpflow import set_trainable
from gpflow.utilities import print_summary


#### configurations
warnings.filterwarnings("ignore")
gpflow.config.set_default_float(np.float32)
np.random.seed(0)
tf.random.set_seed(0)



""" @brief: class to construct a 2D Sparse GP using gpflow """
class SGP2D:

    def __init__(self):
        self.model = None
        self.data = None
        self.kernel1 = None
        self.kernel2 = None
        self.kernel = None
        self.indpts = None
        self.meanf = gpflow.mean_functions.Constant(0)

    """ @brief: to initialize the parameters to the RQ kernel """
    def set_kernel_param(self, ls1, ls2, var, alpha, noise, noise_var):
        self.kernel1 = gpflow.kernels.RationalQuadratic(
            lengthscales=[ls1, ls2])
        self.kernel1.variance.assign(var)
        self.kernel1.alpha.assign(alpha)
        self.kernel2 = gpflow.kernels.White(noise)
        self.kernel2.variance.assign(noise_var)
        self.kernel = self.kernel1 + self.kernel2


    """ @brief: to intialize training data to empty data set """
    def set_empty_data(self):
        in_init, out_init = np.zeros((0, 2)), np.zeros((0, 1))  # input dim:2
        mdl_in = tf.Variable(in_init, shape=(None, 2), dtype=tf.float32)
        mdl_out = tf.Variable(out_init, shape=(None, 1), dtype=tf.float32)
        self.data = (mdl_in, mdl_out)

    """ @brief: to intialize training data as tf tensors """
    def set_training_data(self, d_in, d_out):
        mdl_in = tf.Variable(d_in, dtype=tf.float32)
        mdl_out = tf.Variable(d_out, dtype=tf.float32)
        self.data = (mdl_in, mdl_out)


    """ @brief: to intialize indusing points as empty set"""
    def set_empty_indpts(self):
        indpts_init = np.zeros((0, 2))
        self.indpts = tf.Variable(indpts_init,
                                  shape=(None, 2),
                                  dtype=tf.float32)

    """ @brief: to intialize indusing points from the training data"""
    def set_indpts_from_training_data(self, indpts_size, in_data):
        data_size = np.shape(in_data)[0]
        pts_idx = range(0, data_size, int(data_size / indpts_size))
        self.indpts = in_data[[idx for idx in pts_idx], :]

    """ @brief: to intialize kernel mean"""
    def set_init_mean(self, init_mean):
        self.meanf = gpflow.mean_functions.Constant(init_mean)

    """ @brief: to intiate a SGP model using gpflow lib"""
    def set_sgp_model(self):
        self.model = gpflow.models.SGPR(self.data,
                                        self.kernel,
                                        self.indpts,
                                        mean_function=self.meanf)

    """ @brief: to select which params set to trainable and which are fixed"""
    def select_trainable_param(self):
        set_trainable(self.kernel1.variance, False)
        set_trainable(self.kernel1.lengthscales, False)
        set_trainable(self.kernel2.variance, False)
        set_trainable(self.model.likelihood.variance, False)

    """ @brief: minimize the loss during training"""
    def minimize_loss(self):
        self.model.training_loss_closure(
        ) 

    """ @brief: to select optimizer: here we are using adam""" 
    def adam_optimize_param(self):
        # tm= time()
        optimizer = tf.optimizers.Adam()
        optimizer.minimize(self.model.training_loss,
                           self.model.trainable_variables)
        # print("SGP2D:: adam_optimize_param time: ", time() - tm)


""" @brief: class to train a 2D Sparse GP using the ocupancy surface, 
    to predict the variance surface, and to learn subgoals around the robot """
class VSGPNavGlb:

    def __init__(self):
        ### Node initialization
        rospy.init_node("gp_subgoal")
        print("##############################################")
        print("              Initialize gp_subgoal           ")
        print("##############################################")

        ### subscriber to pose and occupancy surface
        self.pose_pcl_sub = rospy.Subscriber("pose_pcl",
                                             PosePcl2,
                                             self.pose_pcl_cb,
                                             queue_size=1)

        ### publishers
        ## variance surface
        self.gp_var_pub = rospy.Publisher("gp_nav_var",
                                              PointCloud2,
                                              queue_size=1)
        ## occupancy surface
        self.gp_oc_pub = rospy.Publisher("gp_nav_oc",
                                             PointCloud2,
                                             queue_size=1)
        ## navigation (points) subgoals in the robot frame (velodyne frame)
        self.gp_nav_pts_pub = rospy.Publisher("gp_nav_pts",
                                              PointCloud2,
                                              queue_size=1)

        ## navigation (points) subgoals in the world frame (world frame)
        self.gp_actul_xy_subgls_pub = rospy.Publisher("gp_nav_actul_xy_gls",
                                                       PointCloud2,
                                                       queue_size=1)

        ## final goal 
        self.gl_pub = rospy.Publisher("gl_wrt_wrld", PointCloud2, queue_size=1)

        ## recommended subgoal to MPPI controller
        self.gp_rcmndd_subgl_pub = rospy.Publisher("gp_subgoal",
                                               PoseStamped,
                                               queue_size=1)

        ## variables to store data for GP training and prediction 
        self.oc_srfc_rds = rospy.get_param('/oc_srfc_rds')
        self.pcl_skp = rospy.get_param('/pcl_skp')
        self.pose = None
        self.org_unq_thetas = None
        self.pcl_unq_thetas = None
        self.pcl_thetas = None
        self.pcl_alphas = None
        self.pcl_rds = None
        self.pcl_oc = None
        self.pcl_sz = None

        ## grid to reconstruct the variance surface 
        self.gp_grd = None
        self.gp_grd_w = None
        self.gp_grd_h = None
        self.gp_grd_ths = None
        self.gp_grd_als = None
        self.gp_grd_oc = None
        self.gp_grd_rds = None
        self.gp_grd_var = None
        self.sample_gp_nav_grid()

        ## variables to store GPFrontiers and their cost 
        self.gp_nav_pt = None
        self.gp_nav_pts = None
        self.gap_utlty_fun = None
        self.gp_nav_frntr_cntrs = None
        self.gp_nav_frntr_areas = None
        
        ## GP param
        self.gp_nav_indpts_sz = rospy.get_param('/gp_nav_indpts_sz')
        self.gp_nav_var_thrshld = rospy.get_param('/gp_nav_var_thrshld')
        
        ## cost funstion param
        self.gap_k_dir = rospy.get_param('/gap_k_dir')
        self.gap_k_dst = rospy.get_param('/gap_k_dst')
        self.gp_nav_goal_dst = rospy.get_param('/gp_nav_goal_dst')

        ## visualization param
        self.gp_nav_var_img_viz = rospy.get_param('/gp_nav_var_img_viz')
        self.gp_nav_var_viz = rospy.get_param('/gp_nav_var_viz')

        ## final goal
        self.gl_x = rospy.get_param('/gl_x')
        self.gl_y = rospy.get_param('/gl_y')
        self.gl_yaw = rospy.get_param('/gl_yaw')
        self.gl_wrt_wrld = np.array([self.gl_x, self.gl_y, 1], dtype="float32")
        self.gp_nav_frame_id = "velodyne"
        self.gp_nav_var_pblsh = True

        self.gp_nav_xypts = None
        self.goal_published = False

        ## limit the environemnent to maximum dimension
        self.map_2d_h = 500
        self.map_2d_w = 500

        ## for generated pointcloud 
        self.header = Header()
        self.header.seq = 0
        self.header.stamp = None
        self.header.frame_id = "world" 
        self.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)
        ]

        ## print all params
        self.print_ros_param()
        rospy.spin()




    """ @brief: to print all parameters"""
    def print_ros_param(self):
        print("######## OC SRFC ######")
        print("oc_srfc_rds                : ", self.oc_srfc_rds)
        print("######## Nav ########")
        print("gp_nav_indpts_sz    : ", self.gp_nav_indpts_sz)
        print("gap_k_dir        : ", self.gap_k_dir)
        print("gap_k_dst       : ", self.gap_k_dst)
        print("gp_nav_var_thrshld  : ", self.gp_nav_var_thrshld)
        print("gp_nav_goal_dst     : ", self.gp_nav_goal_dst)
        print("gp_nav_var_img_viz  : ", self.gp_nav_var_img_viz)
        print("gl_x         : ", self.gl_x)
        print("gl_y         : ", self.gl_y)
        print("##########################################")

    """ @brief: to initiate and train 2D SGP using the SGP2D class"""
    def gp_nav_fit(self, ls1, ls2, var, alpha, noise, noise_var):
        self.gp_nav = SGP2D()
        self.gp_nav.set_kernel_param(ls1, ls2, var, alpha, noise, noise_var)
        self.gp_nav.set_training_data(self.gp_nav_din, self.gp_nav_dout)
        self.gp_nav.set_indpts_from_training_data(
            self.gp_nav_indpts_sz, self.gp_nav_din)  #(indpts_size, data_size)
        self.gp_nav.set_sgp_model()
        self.gp_nav.select_trainable_param()
        self.gp_nav.minimize_loss()
        # self.gp_nav.adam_optimize_param()

    """ @brief: callback funtion to process the occupancy surface"""
    def pose_pcl_cb(self, pose_pcl_msg):
        #################### GP Nav ####################
        print("\n\n\n##### OBSV: seq=",pose_pcl_msg.header.seq, ", RosTime=", pose_pcl_msg.header.stamp.to_sec() , " #####")
        msg_rcvd_time = time()
        self.header.stamp = pose_pcl_msg.header.stamp
        self.pose = pose_pcl_msg.pose

        ## transformation matrices between robot and world
        self.tf_rbt_2_wrld()  
        self.tf_wrld_2_rbt()
        self.rbt2gl_error()

        self.publish_goal() # publish final goal as a point for rviz visualization

        ## retrieve th, al, rds points from pointcloud
        pcl_arr = ros_numpy.point_cloud2.pointcloud2_to_array(
            pose_pcl_msg.pcl2, squeeze=True)
        pcl_arr = np.round(np.array(pcl_arr.tolist(), dtype='float'), 4)  

        print("org_pcl_size: ", np.shape(pcl_arr))
        ## Downsample and assign thetas, alphas, occs, rds variables
        self.downsample_pcl(pcl_arr)  

        ## define input and output data for training  
        self.gp_nav_din = np.column_stack((self.pcl_thetas, self.pcl_alphas))
        self.gp_nav_dout = np.array(self.pcl_oc, dtype='float').reshape(-1, 1)
        self.gp_nav_fit(0.09, 0.11, 0.7, 10, 10,
                        0.005)  #(ls1, ls2, var, alpha, noise, noise_var)
        nav_grd_oc, nav_grd_var = self.gp_nav.model.predict_f(self.gp_grd)
        self.gp_grd_var = nav_grd_var.numpy()
        self.gp_grd_oc = nav_grd_oc.numpy()
        self.gp_grd_rds = self.oc_srfc_rds - self.gp_grd_oc

        ## process the variance surface to define GPFrontiers and assign subgoals
        self.gp_nav_mask_thrshld()
        self.gp_grd_var_img()
        self.gp_nav_pkup_nav_pt()
        self.gp_nav_xypts_actul_pcl()

        #### occupancy and variance surfaces visualization
        self.gp_grd_oc_pcl()
        self.gp_grd_var_pcl()

        # goal in polar coordinate wrt to velodyne
        self.gl_wrt_rbt = self.tf_2d_inv @ self.gl_wrt_wrld

        # predict occupancy in direction of final goal using GP occupancy model
        self.gl_th = np.arctan2(self.gl_wrt_rbt[1], self.gl_wrt_rbt[0])
        self.gl_al = np.pi / 2
        gl_oc, gl_var = self.gp_nav.model.predict_f(
            np.array([self.gl_th, self.gl_al], dtype="float32").reshape(1, 2))
        gl_rds = self.oc_srfc_rds - gl_oc.numpy()
        # print("gl_var, var_thrshld: ", gl_var.numpy(), self.gp_nav_var_thrshld)
        # print("gl_dst_err, gl_rds: ", self.gl_dst_err, gl_rds)

        ### mode: no obstacle betwen goal and robot, robots go directly to final goal
        if (self.gl_dst_err < gl_rds):
            if not self.goal_published:
                self.glbl_gl_pbl()  ## not sure if this correct situation
                self.goal_published = False
                print("Navigation Mode: FinalGoal ")

            else:
                quit()

        ### mode: there is obstacle betwen goal and robot, robots follow recommended subgoal        
        else:
            print("Navigation Mode: SubGoal ")
            self.gp_nav_glbl_gl_pbl()

        print("Total Processing Time (with visualization): ", time() - msg_rcvd_time)




    """ @brief: error bet. robot and goal"""
    def rbt2gl_error(self):
        self.gl_dst_err = np.sqrt((self.gl_x - self.pose.x)**2 +
                                  (self.gl_y - self.pose.y)**2)
        self.gl_dir = np.arctan2(self.gl_y - self.pose.y,
                                 self.gl_x - self.pose.x)
        self.gl_dir_err = self.gl_dir - self.pose.theta
        # print("gl_dst_err, gl_dir_err: ", self.gl_dst_err, self.gl_dir_err )



    """ @brief: varinace threshoild based on the variance distribution (mean and variance)"""
    def gp_nav_mask_thrshld(self):
        gp_nav_var_stats = stats.describe(self.gp_grd_var)
        gp_nav_var_mean = gp_nav_var_stats.mean[0]
        gp_nav_var_var = gp_nav_var_stats.variance[0]
        self.gp_nav_var_thrshld = 0.4 * (gp_nav_var_mean - 3 * gp_nav_var_var)
        print("variance distribution mean and var: ", gp_nav_var_mean, gp_nav_var_var)
        print("variance threshold: ", self.gp_nav_var_thrshld)


    """ @brief: convert variance to cv image and detect high variance regions usign the variance threshold"""
    def gp_grd_var_img(self):
        img = np.zeros((self.gp_grd_h, 3 * self.gp_grd_w), np.uint8)
        ## normlize variance
        var_xtnd = np.array([]).reshape(-1, 1)
        var_xtnd = np.append(var_xtnd, self.gp_grd_var)
        var_xtnd = np.append(var_xtnd, self.gp_grd_var)
        var_xtnd = np.append(var_xtnd, self.gp_grd_var)
        # print("var_xtnd: ", np.shape(var_xtnd))
        var_norm = np.linalg.norm(var_xtnd)
        normalized_var = var_xtnd / var_norm
        bw_var = (255 / normalized_var[normalized_var.argmax(axis=0)]
                  ) * normalized_var

        ## img[0:self.gp_grd_h, 0:self.gp_grd_w] = 400*self.gp_oc_srfc_grd_var.reshape(self.gp_oc_srfc_grd_w, self.gp_oc_srfc_grd_h).T
        img[0:self.gp_grd_h,
            0:3 * self.gp_grd_w] = bw_var.reshape(3 * self.gp_grd_w, self.gp_grd_h).T

        ## grey to binay
        self.gp_nav_img_thrshld = 0.5 * int(
            np.mean(bw_var) + self.gp_nav_var_thrshld * np.var(bw_var))
        _, bw_img = cv2.threshold(img, self.gp_nav_img_thrshld, 255,
                                  cv2.THRESH_BINARY)
        ### resize image
        scale_factor = 10
        width = int(bw_img.shape[1] * scale_factor)
        height = int(bw_img.shape[0] * scale_factor)
        dsize = (width, height)
        scaled_img = cv2.resize(bw_img, dsize)
        contours, hierarchy = cv2.findContours(scaled_img.copy(),
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        l_frntr_cntrs = []
        l_frntr_ars = []
        # print(">> l_frntr_areas: ", [cv2.contourArea(i) for i in contours])
        for i in contours:
            M = cv2.moments(i)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.drawContours(scaled_img, [i], -1, (0, 255, 0), 2)
                cv2.circle(scaled_img, (cx, cy), 7, (0, 0, 255), -1)

            cx = int(cx / scale_factor)
            cy = int(cy / scale_factor)
            # print(f"x: {cx} y: {cy}")
            if cx >= 127 and cx < 254 and cv2.contourArea(
                    i) > 250 * scale_factor and cy < 5:  # and cy > 5:
                l_frntr_cntrs = l_frntr_cntrs + [
                    self.gp_grd_ths[cx - 127], self.gp_grd_als[cy]
                ]
                l_frntr_ars.append(cv2.contourArea(i))

        # print(">> l_frntr_cntrs: ", l_frntr_cntrs)
        # print(">> l_frntr_areas: ", l_frntr_ars)
        if (len(l_frntr_ars) == 1
                and l_frntr_ars[0] > 50000) or len(l_frntr_ars) == 0:
            l_frntr_cntrs = [[0, np.pi / 2], [np.pi / 2, np.pi / 2],
                             [np.pi, np.pi / 2], [-np.pi / 2, np.pi / 2]]
            l_frntr_ars = [12500, 12500, 12500, 12500]
        self.gp_nav_frntr_cntrs = np.array(l_frntr_cntrs).reshape(-1, 2)
        self.gp_nav_frntr_areas = np.array(l_frntr_ars)  #.reshape(-1)
        
        ## draw contours
        if self.gp_nav_var_img_viz:
            cv2.drawContours(scaled_img, contours, -1, 255, 3, cv2.LINE_AA,
                             hierarchy, abs(-1))
            cv2.imshow('contours', scaled_img)
            cv2.waitKey()

    """ @brief: robot to wolrd 2D TF matrix"""
    def tf_rbt_2_wrld(self):
        self.tf_2d = np.array(
            [[np.cos(self.pose.theta), -np.sin(self.pose.theta), self.pose.x],
             [np.sin(self.pose.theta),
              np.cos(self.pose.theta), self.pose.y], [0, 0, 1]])

    """ @brief:  wolrd to robot 2D TF matrix"""
    def tf_wrld_2_rbt(self):
        cos_th = np.cos(self.pose.theta)
        sin_th = np.sin(self.pose.theta)
        self.tf_2d_inv = np.array(
            [[cos_th, sin_th, -self.pose.x * cos_th - self.pose.y * sin_th],
             [-sin_th, cos_th, self.pose.x * sin_th - self.pose.y * cos_th],
             [0, 0, 1]])

    """ @brief:  wolrd to robot 2D TF matrix"""
    def gls_in_wrld_frame(self, gls):
        xy_gls = np.empty([0, 3])
        for gl in gls:
            xy_gls = np.vstack((xy_gls, self.tf_2d @ gl))
        # limit to map  border,  here consider  w=h (this part is not important)
        xy_gls = np.clip(xy_gls, -self.map_2d_h / 2 + 1, self.map_2d_h / 2 - 1) 
        return xy_gls

    """ @brief:  calculate xy location in world frame for subgoals (navigation points)"""
    def xy_actul_nav_gls(self): 
        actul_gls_rds = self.gp_nav_goal_dst * np.ones(
            self.gp_nav_gls_sz, dtype='float32').reshape(-1, 1)
        x, y, z = self.convert_spherical_2_cartesian(
            self.gp_nav_pts.T[0].reshape(-1, 1),
            self.gp_nav_pts.T[1].reshape(-1, 1), actul_gls_rds)
        z = np.ones(self.gp_nav_gls_sz).reshape(-1, 1)
        actul_gls = np.hstack((x, y, z))
        self.gp_nav_actul_xy_gls = self.gls_in_wrld_frame(actul_gls)

    """ @brief:  calculate the recommended subgoal (navigation point) based on cost function"""
    def gp_nav_pkup_nav_pt(self):
        # print("gp_nav_pkup_nav_pt:: ")
        self.gp_nav_pts = self.gp_nav_frntr_cntrs
        self.gp_nav_gls_sz = np.shape(self.gp_nav_pts)[0]
        # print("# goals: ", self.gp_nav_gls_sz)
        self.xy_actul_nav_gls()

        gap_to_gl_dst = np.sqrt((self.gl_y -
                                 self.gp_nav_actul_xy_gls.T[1])**2 +
                                (self.gl_x - self.gp_nav_actul_xy_gls.T[0])**2)

        self.gap_utlty_fun = self.gap_k_dst * (
            self.gp_nav_goal_dst + gap_to_gl_dst) + self.gap_k_dir * abs(
                self.gp_nav_frntr_cntrs.T[0])**2
        # print("gap_dir: ", abs(self.gp_nav_frntr_cntrs.T[0]))
        # print("gap_to_gl_dst : ", gap_to_gl_dst)
        # print("self.gap_utlty_fun: ", self.gap_utlty_fun)

        self.chsn_gl_idx = self.gap_utlty_fun.argmin(axis=0)  
        self.gp_nav_pt = self.gp_nav_pts[self.chsn_gl_idx]
        print("recommended subgoal id and direction: ", self.chsn_gl_idx, self.gp_nav_pt[0])


    """ @brief:  publish recommended goal to MPPI controller"""
    def gp_nav_glbl_gl_pbl(self):
        nav_xy_gl = self.gp_nav_actul_xy_gls[self.chsn_gl_idx].reshape(3, -1)
        yaw = np.arctan2(nav_xy_gl[1], nav_xy_gl[0])
        qtrn = quaternion_from_euler(0, 0, yaw)
        gl_msg = PoseStamped()
        gl_msg.pose.position.x = nav_xy_gl[0]
        gl_msg.pose.position.y = nav_xy_gl[1]
        gl_msg.pose.position.z = 0
        gl_msg.pose.orientation.x = qtrn[0]
        gl_msg.pose.orientation.y = qtrn[1]
        gl_msg.pose.orientation.z = qtrn[2]
        gl_msg.pose.orientation.w = qtrn[3]

        self.header.frame_id = "world"
        gl_msg.header = self.header
        self.gp_rcmndd_subgl_pub.publish(gl_msg)
        print("recomended subgoal in world frame: ", nav_xy_gl[0], nav_xy_gl[1])

    """ @brief:  publish final goal"""
    def glbl_gl_pbl(self):
        # yaw = np.arctan2(self.gl_x, self.gl_y)
        # qtrn = quaternion_from_euler(0, 0, yaw)
        qtrn = quaternion_from_euler(0, 0, self.gl_yaw)
        gl_msg = PoseStamped()
        gl_msg.pose.position.x = self.gl_x
        gl_msg.pose.position.y = self.gl_y
        gl_msg.pose.position.z = 0
        gl_msg.pose.orientation.x = qtrn[0]
        gl_msg.pose.orientation.y = qtrn[1]
        gl_msg.pose.orientation.z = qtrn[2]
        gl_msg.pose.orientation.w = qtrn[3]
        self.header.frame_id = "world"
        gl_msg.header = self.header
        self.gp_rcmndd_subgl_pub.publish(gl_msg)
        # print("glbl_nav goal: ", nav_xy_gl[0], nav_xy_gl[1])

    """ @brief:  stop command when robot reach final goal"""
    def stop_cmd(self):
        gl_msg = PoseStamped()
        gl_msg.pose.position.x = self.pose.x
        gl_msg.pose.position.y = self.pose.y
        gl_msg.pose.position.z = 0
        gl_msg.pose.orientation.x = 0
        gl_msg.pose.orientation.y = 0
        gl_msg.pose.orientation.z = 0
        gl_msg.pose.orientation.w = 1
        self.header.frame_id = "world"
        gl_msg.header = self.header
        self.gp_rcmndd_subgl_pub.publish(gl_msg)

   
    """ @brief:  publish variance surface for visualization"""
    def gp_grd_var_pcl(self):
        rds = self.gp_nav_var_viz * np.ones(np.shape(self.gp_grd_var)[0],
                                            dtype='float32').reshape(-1, 1)
        x, y, z = self.convert_spherical_2_cartesian(
            self.gp_grd.T[:][0].reshape(-1, 1),
            self.gp_grd.T[:][1].reshape(-1, 1), rds)
        intensity = np.array(self.gp_grd_var,
                             dtype='float32').reshape(-1, 1)
        gp_var_pcl = np.column_stack((x, y, z, intensity))
        self.header.frame_id = self.gp_nav_frame_id
        pc2 = point_cloud2.create_cloud(self.header, self.fields, gp_var_pcl)
        self.gp_var_pub.publish(pc2)

    """ @brief:  publish occupancy surface for visualization"""
    def gp_grd_oc_pcl(self):
        rds = self.gp_nav_var_viz * np.ones(np.shape(self.gp_grd_rds)[0],
                                            dtype='float32').reshape(-1, 1)
        x, y, z = self.convert_spherical_2_cartesian(
            self.gp_grd.T[:][0].reshape(-1, 1),
            self.gp_grd.T[:][1].reshape(-1, 1), rds)
        intensity = np.array(self.gp_grd_rds,
                             dtype='float32').reshape(-1, 1)
        gp_oc_pcl = np.column_stack((x, y, z, intensity))
        self.header.frame_id = self.gp_nav_frame_id
        pc2 = point_cloud2.create_cloud(self.header, self.fields, gp_oc_pcl)
        self.gp_oc_pub.publish(pc2)

    """ @brief:  publish occupancy surface for visualization"""
    def gp_nav_pts_pcl(self):
        rds = self.gp_nav_goal_dst * np.ones(self.gp_nav_gls_sz,
                                             dtype='float32').reshape(-1, 1)
        x, y, z = self.convert_spherical_2_cartesian(
            self.gp_nav_pts.T[0].reshape(-1, 1),
            self.gp_nav_pts.T[1].reshape(-1, 1), rds)
        intensity = np.array(self.gap_utlty_fun,
                             dtype='float32').reshape(-1, 1)
        nav_pts_pcl = np.column_stack((x, y, z, intensity))
        self.header.frame_id = self.gp_nav_frame_id
        pc2 = point_cloud2.create_cloud(self.header, self.fields, nav_pts_pcl)
        self.gp_nav_pts_pub.publish(pc2)

    """ @brief:  publish navigation points in world frame for visualization"""
    def gp_nav_xypts_actul_pcl(self):
        # print(">> gp_nav_xypts_actul_pcl:: ")
        intensity = np.array(self.gap_utlty_fun,
                             dtype='float32').reshape(-1, 1)
        # print("intensity: ", intensity)
        # print("self.gp_nav_actul_xy_gls: ", self.gp_nav_actul_xy_gls)
        nav_pts_pcl = np.column_stack((self.gp_nav_actul_xy_gls, intensity))
        self.header.frame_id = "world"
        pc2 = point_cloud2.create_cloud(self.header, self.fields, nav_pts_pcl)
        self.gp_actul_xy_subgls_pub.publish(pc2)

    """ @brief:  publish final goal for visualization"""
    def publish_goal(self):
        # print(">> publish_goal:: ")
        nav_pts_pcl = np.column_stack((self.gl_x, self.gl_y, 0, 1))
        self.header.frame_id = "world"
        pc2 = point_cloud2.create_cloud(self.header, self.fields, nav_pts_pcl)
        self.gl_pub.publish(pc2)

    """ @brief:  downsample pointcloud"""
    def downsample_pcl(self, pcl_arr):
        pcl_arr = pcl_arr[np.argsort(pcl_arr[:, 0])]  ## sort based on thetas
        thetas = pcl_arr.transpose()[:][0].reshape(-1, 1)
        self.org_unq_thetas = np.array(sorted(set(
            thetas.flatten())))  #.reshape(-1,1)
        #### percentage to keep  or fraction to delete
        keep_th_ids = [
            t for t in range(0, np.shape(self.org_unq_thetas)[0], self.pcl_skp)]
        ids = []
        for t in keep_th_ids:
            ids = ids + list(np.where(thetas == self.org_unq_thetas[t])[0])
        pcl_arr = pcl_arr[ids]
        pcl_arr = pcl_arr.transpose()
        self.pcl_thetas = np.round(pcl_arr[:][0].reshape(-1, 1), 4)
        self.pcl_alphas = np.round(pcl_arr[:][1].reshape(-1, 1), 4)
        self.pcl_rds = np.round(pcl_arr[:][2].reshape(-1, 1), 4)
        self.pcl_oc = np.round(pcl_arr[:][3].reshape(-1, 1), 4)
        self.pcl_sz = np.shape(self.pcl_thetas)[0]
        self.pcl_unq_thetas = np.array(sorted(set(
            self.pcl_thetas.flatten())))  #.reshape(-1,1)
        self.unq_smpld_th_size = np.shape(self.pcl_unq_thetas)[0]

    """ @brief:  grid to reconstruct variance surface"""
    def sample_gp_nav_grid(self):
        th_rsltion = 0.05  #0.00174  # 0.02 # #from -pi to pi rad -> 0 35999
        al_rsltion = 0.05  #0.0349  #vpl16 resoltuion is 2 deg (from -15 to 15 deg)
        self.gp_grd_ths = np.arange(-np.pi + 0.0, np.pi + 0.02,
                                        th_rsltion, dtype='float32')
        self.gp_grd_als = np.arange(np.pi / 2 - 0.261799, np.pi / 2 + 0.261799,
                                        al_rsltion, dtype='float32')
        self.gp_grd = np.array( np.meshgrid(self.gp_grd_ths,
                        self.gp_grd_als)).T.reshape(-1, 2)
        self.gp_grd_w = np.shape(self.gp_grd_ths)[0]
        self.gp_grd_h = np.shape(self.gp_grd_als)[0]
        # print("\ngrid: ", np.shape(self.gp_grd))
        # print("grid w, h: ", self.gp_grd_w, self.gp_grd_h)

    """ @brief:  convert spherical to cartesian coordinates"""
    def convert_spherical_2_cartesian(self, theta, alpha, dist):
        x = np.array(dist * np.sin(alpha) * np.cos(theta),
                     dtype='float32').reshape(-1, 1)
        y = np.array(dist * np.sin(alpha) * np.sin(theta),
                     dtype='float32').reshape(-1, 1)
        z = np.array(dist * np.cos(alpha), dtype='float32').reshape(-1, 1)
        return x, y, z

    """ @brief:  convert  cartesian  to spherical coordinates"""
    def convert_cartesian_2_spherical(self, x, y, z):
        dist = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        alpha = np.arccos(z / dist)
        return theta, alpha, dist


if __name__ == "__main__":
    try:
        VSGPNavGlb()
    except rospy.ROSInterruptException:
        pass
