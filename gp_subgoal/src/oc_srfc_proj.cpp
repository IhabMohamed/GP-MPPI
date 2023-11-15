
#include <iostream>
#include <ros/ros.h>
#include <vector>
#include <math.h>


#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/PointCloud2.h> //for sensor_msgs::PointCloud2
#include <gp_subgoal/PosePcl2.h>
#include <geometry_msgs/TransformStamped.h>

#include <pcl/point_types.h> // for pcl::PointXYZ
#include <pcl_conversions/pcl_conversions.h> // for pcl::fromROSMsg()

#include <pcl_ros/point_cloud.h>
#include<tf/tf.h>



#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>


// #include <Eigen/Dense>


ros::Publisher lfrq_sph_pcl_pub;     // to send (azumith, elevation, radius) values to gp_subgoal node
ros::Publisher lfrq_org_oc_srfc_pub; //for visualizing original oc_srfc in rviz


// float px_old , py_old, yaw_old;
std::string velodyne_frame="velodyne";
float oc_srfc_rds= 0, org_oc_srfc_rds_viz=0, proj_lfrq=10.0;
float t =0, t_prvs = 0;



int sync_callback(const  sensor_msgs::PointCloud2::ConstPtr &pcl_in, const nav_msgs::Odometry::ConstPtr &pose_in)
{

  if (pcl_in->header.stamp.toSec() - t_prvs > (1/proj_lfrq)) {
    t = pcl_in->header.stamp.toSec(); 
    ROS_INFO_STREAM("t1: " << t - t_prvs);
    t_prvs = t;
  }
  else
    return 0;

  // calculate robot orientation interms of (roll, pitch, yaw)
  tf::Quaternion q( pose_in->pose.pose.orientation.x, pose_in->pose.pose.orientation.y, 
                    pose_in->pose.pose.orientation.z, pose_in->pose.pose.orientation.w );
  tf::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);


  //convert the input into a pcl::PointCloud< pcl::PointXYZ> object
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*pcl_in, *cloudPtr);


  //*********** define pcl variables ***********
  pcl::PointCloud<pcl::PointXYZI> org_pcl, sph_pcl, occ_srfc_pcl;
  std::copy(cloudPtr->points.begin(), cloudPtr->points.end(),std::back_inserter(org_pcl));



 
  //*********** project carteesian pcl into different form ***********
  std::vector<pcl::PointXYZI> sph_pcl_vec, occ_srfc_pcl_vec;
  for(auto pt: org_pcl){
    // ROS_INFO_STREAM("pt: " << pt.x);
    float dst = sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z );
    float th = atan2(pt.y , pt.x );
    float al = acos(pt.z / dst );

    // spherical form of occupancy surface (th, al, dst)
    pcl::PointXYZI shp_pt;
    shp_pt.x = th;
    shp_pt.y = al;
    shp_pt.z = dst;
    shp_pt.intensity = oc_srfc_rds - dst;
    sph_pcl_vec.push_back(shp_pt);

   
    // occupancy surface in cartesian for visualization in rviz
    pcl::PointXYZI oc_srfc_pt;
    oc_srfc_pt.x = org_oc_srfc_rds_viz * sin(al) * cos(th);
    oc_srfc_pt.y = org_oc_srfc_rds_viz * sin(al) * sin(th);
    oc_srfc_pt.z = org_oc_srfc_rds_viz * cos(al); 
    oc_srfc_pt.intensity = dst;
    occ_srfc_pcl_vec.push_back(oc_srfc_pt);
  }
  // ROS_INFO_STREAM("sph_pcl_vec: "<< sph_pcl_vec.size() );
  // ROS_INFO_STREAM("occ_srfc_pcl_vec: "<< occ_srfc_pcl_vec.size() );




  //*********** std vector to pcl ***********
  std::copy(sph_pcl_vec.begin(), sph_pcl_vec.end(),std::back_inserter(sph_pcl));
  sph_pcl.header.frame_id = velodyne_frame; 
  std::copy(occ_srfc_pcl_vec.begin(), occ_srfc_pcl_vec.end(),std::back_inserter(occ_srfc_pcl));
  occ_srfc_pcl.header.frame_id = velodyne_frame; 



  //*********** Publish msgs ***********
  // compose the pose_pcl msg (pose + pointcloud) 
  gp_subgoal::PosePcl2 pose_pcl_msg;
  pose_pcl_msg.header.stamp = pcl_in->header.stamp; //.toSec(); 
  pose_pcl_msg.pose.x = pose_in->pose.pose.position.x;
  pose_pcl_msg.pose.y = pose_in->pose.pose.position.y;
  pose_pcl_msg.pose.theta = yaw;
  pcl::toROSMsg(sph_pcl, pose_pcl_msg.pcl2);
  lfrq_sph_pcl_pub.publish(pose_pcl_msg);

  // compose the PointCloud2 msg to visualize the occupancy surface
  sensor_msgs::PointCloud2 occ_srfc_pcl_msg;
  pcl::toROSMsg(occ_srfc_pcl, occ_srfc_pcl_msg);
  occ_srfc_pcl_msg.header.stamp = pcl_in->header.stamp;//.toSec(); 
  lfrq_org_oc_srfc_pub.publish(occ_srfc_pcl_msg);


  return 0;
}


void print_param(){
  ROS_INFO("######### oc_srfc_proj: param ########");
  ROS_INFO_STREAM("velodyne_frame      : " << velodyne_frame);
  ROS_INFO_STREAM("oc_srfc_rds         : " << oc_srfc_rds);
  ROS_INFO_STREAM("org_oc_srfc_rds_viz : " << org_oc_srfc_rds_viz);
  ROS_INFO_STREAM("proj_lfrq : " << proj_lfrq);
  ROS_INFO("#####################################");
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "oc_srfc_proj");
  ROS_INFO("oc_srfc_proj node ... ");

  ros::NodeHandle nh, priv_nh("~");
  priv_nh.getParam("velodyne_frame", velodyne_frame);
  nh.getParam("oc_srfc_rds", oc_srfc_rds);
  nh.getParam("org_oc_srfc_rds_viz", org_oc_srfc_rds_viz);
  nh.getParam("proj_lfrq", proj_lfrq);
  print_param();

  //create publisher object
  lfrq_sph_pcl_pub = nh.advertise<gp_subgoal::PosePcl2>("pose_pcl", 1);
  lfrq_org_oc_srfc_pub = nh.advertise<sensor_msgs::PointCloud2>("lfrq_org_oc_srfc", 1);

  // message filter time synchronization
  message_filters::Subscriber<nav_msgs::Odometry> pose_sub(nh, "ground_truth/state", 10);
  message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub(nh, "mid/points", 10);
  message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, nav_msgs::Odometry> sync(pcl_sub, pose_sub, 10);
 
  sync.registerCallback(boost::bind(&sync_callback, _1, _2)); // try std instead boost not working as well
  ros::spin();

  return 0;
}


