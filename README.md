# LVIMOT

This repository was originally created for the LVIMOT paper: "LVIMOT: Accurate and Robust LiDAR-Visual-Inertial Localization and Multi-Object Tracking in Dynamic Environments via Tightly Coupled Integration." At this stage, we are open-sourcing the FGO-MOT module first — a core 3D MOT component within LVIMOT. It performs continuous object tracking and trajectory estimation for robust online tracking and smooth object trajectories.

## Highlights
- Factor-graph optimization: models motion priors, observation constraints, and marginalization as factors and solves in a unified optimization framework.

### Common dependencies
- ROS 1
- PCL, Eigen
- Ceres-Solver

## Quick Start
1) Prepare a catkin workspace and place this package in `src/`:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/shqfeng/LVIMOT.git
cd ..
catkin_make
source devel/setup.bash
```

2) Launch tracking (example):

```bash
roslaunch fgo-mot run.launch
```

## Data and Configuration
- Datasets:
  - KITTI tracking pose data and object detections from [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT).

  Organize KITTI tracking data as:
  ```
  # KITTI Tracking Dataset
  └── kitti_tracking
         ├── testing 
         |      ├──calib
         |      |    ├──0000.txt
         |      |    ├──....txt
         |      |    └──0028.txt
         |      ├──image_02
         |      |    ├──0000
         |      |    ├──....
         |      |    └──0028
         |      ├──pose
         |      |    ├──0000
         |      |    |    └──pose.txt
         |      |    ├──....
         |      |    └──0028
         |      |         └──pose.txt
         |      ├──label_02
         |      |    ├──0000.txt
         |      |    ├──....txt
         |      |    └──0028.txt
         |      └──velodyne
         |           ├──0000
         |           ├──....
         |           └──0028      
         └── training  # same structure as testing
                ├──calib
                ├──image_02
                ├──pose
                ├──label_02
                └──velodyne 
  ```

  Example detections layout:
  ```
  └── point-rcnn
         ├── training
         |      ├──0000
         |      |    ├──000001.txt
         |      |    ├──....txt
         |      |    └──000153.txt
         |      ├──...
         |      └──0020
         └── testing
  ```

- Parameters:
  - tracking/optimization parameters: [config/config.yaml](./config/config.yaml).
  - If you maintain a custom tracker config (paths, thresholds), place it under [config](./config).

- Evaluation:
  - Results are saved to `~/catkin_ws/output/tracking_kitti/`.
  - Evaluate using 3DMOT metrics [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT).

## Citation
Please cite the following related works:

```bibtex
@article{feng2023fgomot,
       title   = {Accurate and Real-Time 3D-LiDAR Multi-Object Tracking Using Factor Graph Optimization},
       author  = {Feng, S. and Li, X. and Yan, Z. and others},
       journal = {IEEE Sensors Journal},
       year    = {2023}
}

@article{feng2023vimot,
       title   = {VIMOT: A Tightly-Coupled Estimator for Stereo Visual-Inertial Navigation and Multi-Object Tracking},
       author  = {Feng, S. and Li, X. and Xia, C. and others},
       journal = {IEEE Transactions on Instrumentation and Measurement},
       year    = {2023}
}
```
