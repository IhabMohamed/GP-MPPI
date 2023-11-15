# gp_subgoal
This package identifies navigation subgoals in the free space around the robot and recommends one of them to the controller. 


## Usage
```
roslaunch gp_subgoal gp_subgoal_sim.launch
```

## Nodes
### oc_srfc_proj
#### Subscribe to:
- Localization topic:  [nav_msgs/msg/Odometry](https://docs.ros2.org/foxy/api/nav_msgs/msg/Odometry.html)
- Poitcloud topic:  [sensor_msgs/PointCloud2](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud2.html)
#### Publish to:
- PosePCL topic: gp_subgoal/msg/PosePCL2
  
### gp_subgoals
#### Subscribe to:
- PosePCL topic: gp_subgoal/msg/PosePCL2

#### Publish to:
- recommended_subgoal topic: [geometry_msgs/PoseStamped](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html)


