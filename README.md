# FGO-MOT: Factor Graph Optimization-based 3D Multi-Object Tracking

This repository was originally created for the LVIMOT paper: "LVIMOT: Accurate and Robust LiDAR-Visual-Inertial Localization and Multi-Object Tracking in Dynamic Environments via Tightly Coupled Integration." At this stage, we are open-sourcing the FGO-MOT module first — a core 3D MOT component within LVIMOT. It performs continuous detection fusion, data association, and trajectory estimation by combining geometric priors and observation factors for robust online tracking and smooth trajectories.

## Highlights
- Factor-graph optimization: models motion priors, observation constraints, and marginalization as factors and solves in a unified optimization framework.
- Data association: optimal matching via Hungarian algorithm (see [fgo-mot/include/HungarianAlg.hpp](fgo-mot/include/HungarianAlg.hpp)).
- Geometry fitting and clustering: L-Shape fitting and CVC clustering (see [fgo-mot/include/LShapeFit.hpp](fgo-mot/include/LShapeFit.hpp), [fgo-mot/include/CVC_cluster.hpp](fgo-mot/include/CVC_cluster.hpp)).
- Ground segmentation: Patchwork integration (see [fgo-mot/include/patchwork.hpp](fgo-mot/include/patchwork.hpp); parameters in [fgo-mot/config/patchwork_params.yaml](fgo-mot/config/patchwork_params.yaml)).
- Odometry integration: works with LiDAR odometry (see [fgo-mot/src/lidarOdometry.cpp](fgo-mot/src/lidarOdometry.cpp), [fgo-mot/src/imageProjection.cpp](fgo-mot/src/imageProjection.cpp)).
- ROS-native: custom messages and RViz visualization (see [fgo-mot/msg](fgo-mot/msg), [fgo-mot/launch/test.rviz](fgo-mot/launch/test.rviz)).

## Repository Layout
- Core modules:
       - Tracking logic: [fgo-mot/src/objectTracker.cpp](fgo-mot/src/objectTracker.cpp)
       - LiDAR odometry: [fgo-mot/src/lidarOdometry.cpp](fgo-mot/src/lidarOdometry.cpp)
       - Projection / feature extraction: [fgo-mot/src/imageProjection.cpp](fgo-mot/src/imageProjection.cpp), [fgo-mot/src/featureExtraction.cpp](fgo-mot/src/featureExtraction.cpp)
- Factors and optimization:
       - Factor definitions: [fgo-mot/include/factors/objectFactor.h](fgo-mot/include/factors/objectFactor.h), [fgo-mot/include/factors/marginalizationFactor.h](fgo-mot/include/factors/marginalizationFactor.h)
       - Pose local parameterization: [fgo-mot/include/factors/pose_local_parameterization.h](fgo-mot/include/factors/pose_local_parameterization.h)
- Managers and frames:
       - Feature/object frames: [fgo-mot/include/featureManager/laserFrame.h](fgo-mot/include/featureManager/laserFrame.h), [fgo-mot/include/featureManager/objectFrame.h](fgo-mot/include/featureManager/objectFrame.h)
- Configs and launch:
       - LIO config: [fgo-mot/config/lio_config.yaml](fgo-mot/config/lio_config.yaml)
       - Ground segmentation params: [fgo-mot/config/patchwork_params.yaml](fgo-mot/config/patchwork_params.yaml)
       - Launch files: [fgo-mot/launch/run.launch](fgo-mot/launch/run.launch)
- Message types:
       - Detection and annotations: [fgo-mot/msg/detect_object.msg](fgo-mot/msg/detect_object.msg), [fgo-mot/msg/boundingBox2D.msg](fgo-mot/msg/boundingBox2D.msg), [fgo-mot/msg/boundingBox2DArray.msg](fgo-mot/msg/boundingBox2DArray.msg)
       - Cloud info and indices: [fgo-mot/msg/cloud_info.msg](fgo-mot/msg/cloud_info.msg), [fgo-mot/msg/index_vector.msg](fgo-mot/msg/index_vector.msg)

## Dependencies
We recommend building and running on ROS 1 (Melodic/Noetic) with PCL/Eigen; optimization typically depends on Ceres or similar (refer to [fgo-mot/CMakeLists.txt](fgo-mot/CMakeLists.txt) and [fgo-mot/package.xml](fgo-mot/package.xml)). Native ROS on Windows is limited—Ubuntu or WSL2 is recommended for development.

### Common dependencies (example; check actual list)
- ROS 1 (catkin workflow)
- C++14/17 compiler (gcc/clang, or MSVC inside WSL2)
- PCL, Eigen
- Ceres-Solver (for nonlinear least-squares in FGO)

## Quick Start
1) Prepare a catkin workspace and place this package in `src/`:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone <this-repo-url>
cd ..
rosdep install --from-paths src --ignore-src -r -y
catkin build # or catkin_make
source devel/setup.bash
```

2) Launch tracking (example):

```bash
roslaunch fgo-mot run.launch
```

3) Visualization:

```bash
rviz -d $(rospack find fgo-mot)/launch/test.rviz
```

> Note: topic names and parameters follow [fgo-mot/launch/run.launch](fgo-mot/launch/run.launch) and individual module sources.

## Data and Configuration
- Datasets:
       - KITTI helpers provided in [fgo-mot/src/kittiHelper.cpp](fgo-mot/src/kittiHelper.cpp). For offline playback, prepare rosbag files or converted point cloud topics.
- Parameters:
       - LIO and optimization parameters: [fgo-mot/config/lio_config.yaml](fgo-mot/config/lio_config.yaml).
       - Ground segmentation parameters: [fgo-mot/config/patchwork_params.yaml](fgo-mot/config/patchwork_params.yaml).

## Interfaces and Topics (Overview)
- Inputs:
       - LiDAR point cloud topics (e.g., `/points_raw`), subject to your setup.
- Outputs:
       - Tracked objects and bounding boxes (see [fgo-mot/msg/detect_object.msg](fgo-mot/msg/detect_object.msg), [fgo-mot/msg/boundingBox2DArray.msg](fgo-mot/msg/boundingBox2DArray.msg)).
       - Cloud processing info (see [fgo-mot/msg/cloud_info.msg](fgo-mot/msg/cloud_info.msg)).

## Algorithm Overview
- Each time-step’s object state is a variable node. We add observation factors (e.g., geometric fits, detections), motion factors (odometry/dynamics), and marginalization factors to control problem size.
- Data association via the Hungarian algorithm establishes correspondences across objects, followed by global trajectory optimization over the factor graph.
- Ground segmentation and shape fitting improve detection quality; optimization yields smooth and robust object-level trajectories.

## Demo and Results
- RViz configuration example: [fgo-mot/launch/test.rviz](fgo-mot/launch/test.rviz).
- If you have demo videos or screenshots, feel free to link them here.

## Citation
Please cite the following related works:

```bibtex
@article{feng2025lvimot,
       title   = {LVIMOT: Accurate and robust LiDAR-visual-inertial localization and multi-object tracking in dynamic environments via tightly coupled integration},
       author  = {Feng, S. and Li, X. and Yan, Z. and others},
       journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
       year    = {2025},
       volume  = {230},
       pages   = {675--692}
}

@article{li2024liolot,
       title   = {LIO-LOT: Tightly-Coupled Multi-object Tracking and LiDAR-Inertial Odometry},
       author  = {Li, X. and Yan, Z. and Feng, S. and others},
       journal = {IEEE Transactions on Intelligent Transportation Systems},
       year    = {2024}
}

@article{feng2023vimot,
       title   = {VIMOT: A Tightly-Coupled Estimator for Stereo Visual-Inertial Navigation and Multi-Object Tracking},
       author  = {Feng, S. and Li, X. and Xia, C. and others},
       journal = {IEEE Transactions on Instrumentation and Measurement},
       year    = {2023}
}

@article{feng2023fgo3dlidar,
       title   = {Accurate and Real-Time 3D-LiDAR Multi-Object Tracking Using Factor Graph Optimization},
       author  = {Feng, S. and Li, X. and Yan, Z. and others},
       journal = {IEEE Sensors Journal},
       year    = {2023}
}

@article{feng2024tightlycoupled,
       title   = {Tightly Coupled Integration of LiDAR and Vision for 3D Multiobject Tracking},
       author  = {Feng, S. and Li, X. and Yan, Z. and others},
       journal = {IEEE Transactions on Intelligent Vehicles},
       year    = {2024}
}
```

## License
This repository is open-source. Please add and update a `LICENSE` file in the root to declare your chosen license (e.g., MIT/Apache-2.0/BSD-3-Clause).

## Acknowledgments
- Thanks to the open-source community for LiDAR processing, graph optimization, clustering, and data association implementations and insights.
- Patchwork ground segmentation, Ceres Solver, and the ROS ecosystem provide key building blocks for this project.

# LVIMOT
## 
## Prepare data 
You can download the Kitti tracking pose data from [here](https://drive.google.com/drive/folders/1Vw_Mlfy_fJY6u0JiCD-RMb6_m37QAXPQ?usp=sharing), and
you can download the point-rcnn, second-iou and pv-rcnn detections from [here](https://drive.google.com/file/d/1zVWFGwRqF_CBP4DFJJa4nBcu-z6kpF1R/view?usp=sharing).
You can download the CasA detections(including training & testing set) from [here](https://drive.google.com/file/d/1LaousWNTldOV1IhdcGDRM_UGi5BFWDoN/view?usp=sharing).

To run this code, you should organize Kitti tracking dataset as below:
```
# Kitti Tracking Dataset       
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
       └── training # the structure is same as testing set
              ├──calib
              ├──image_02
              ├──pose
              ├──label_02
              └──velodyne 
```
Detections
```
└── point-rcnn
       ├── training
       |      ├──0000
       |      |    ├──000001.txt
       |      |    ├──....txt
       |      |    └──000153.txt
       |      ├──...
       |      └──0020
       └──testing 
```

## Requirements
```
python3
numpy
opencv
yaml
```

## Quick start
* Please modify the dataset path and detections path in the [yaml file](./config/online/pvrcnn_mot.yaml) 
to your own path.
* Then run ``` python3 kitti_3DMOT.py config/online/pvrcnn_mot.yaml``` 
* The results are automatically saved to ```evaluation/results/sha_key/data```, and 
evaluated by HOTA metrics.

