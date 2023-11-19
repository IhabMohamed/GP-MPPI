# GP-MPPI

This repository contains the ROS implementation of the GP-MPPI control strategy, an online learning-based control strategy that integrates the state-of-the-art sampling-based Model Predictive Path Integral [(MPPI)](https://arc.aiaa.org/doi/pdf/10.2514/1.G001921) controller with a local perception model based on Sparse Gaussian Process (SGP).

The key idea is to leverage the learning capability of SGP to construct a variance (uncertainty) surface, enabling the robot to learn about the navigable space surrounding it, identify a set of suggested subgoals, and ultimately recommend the optimal subgoal that minimizes a predefined cost function for the local MPPI planner. MPPI then computes the optimal control sequence that satisfies the robot and collision avoidance constraints. This approach eliminates the need for a global map of the environment or an offline training process.

![GP-MPPI-Architecture](media/GP-MPPI-Architecture.png) 
![](media/MU1_SM_T2_2x.gif "Maze 1, Sample Mode (SM)")

<img src="media/GP-MPPI-Architecture.png" alt="GP-MPPI-Architecture" width="400">
<img src="media/MU1_SM_T2_2x.gif" alt="Maze 1, Sample Mode (SM)" width="400">

## Paper:

If you find this code useful in your research, please cite:

Ihab S. Mohamed, Mahmoud Ali, and Lantao Liu. "GP-guided MPPI for Efficient Navigation in Complex Unknown Cluttered Environments." IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023.

Paper: [https://arxiv.org/abs/2306.12369](https://arxiv.org/abs/2307.04019)

Bibtex:
```
@article{mohamed2023gp,
  title={GP-guided MPPI for Efficient Navigation in Complex Unknown Cluttered Environments},
  author={Mohamed, Ihab S and Ali, Mahmoud and Liu, Lantao},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```
## Media:
**Video**: https://youtu.be/et9t8X1wHKI

## ROS Packages:

The entire project will be uploaded soon; however, we are currently referring to our related repositories for the baselines [(namely, MPPI and log-MPPI)](https://github.com/IhabMohamed/log-MPPI_ros) and [GP perception model](https://github.com/mahmoud-a-ali/vsgp_pcl).

# To run GP-MPPI
1. roslaunch jackal_gazebo world_stage.launch env_name:=forest1 (0.2 tres/m2) OR maze1
2. roslaunch  gp_subgoal gp_subgoal_sim.launch
4. roslaunch mppi_control control_stage.launch gp_mppi:=true; gp_mppi:=false means that MPPI or log-mppi will be run 


### Primary code maintainer:
Ihab S. Mohamed and Mahmoud Ali (e-mail: {mohamedi, alimaa}@iu.edu)\
Vehicle Autonomy and Intelligence Lab ([VAIL](https://vail.sice.indiana.edu/))\
Indiana University - Bloomington, USA


