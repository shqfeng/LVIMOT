#ifndef COMMON_H
#define COMMON_H

#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <eigen_conversions/eigen_msg.h>
#include <message_filters/subscriber.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <sensor_msgs/NavSatFix.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include "tf2_msgs/TFMessage.h"
// PCL
#define PCL_NO_PRECOMPILE
#include <pcl/common/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/impl/point_types.hpp>

#include <pcl/impl/pcl_base.hpp>
#include <pcl/impl/point_types.hpp>

#include <pcl/filters/filter.h>
#include <pcl/filters/impl/filter.hpp>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/impl/passthrough.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/registration/icp.h>
#include <pcl/registration/impl/icp.hpp>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include "fgo_mot/cloud_info.h"
#include "fgo_mot/index_vector.h"
#include "fgo_mot/detect_object.h"
#include "fgo_mot/boundingBox2D.h"
#include "fgo_mot/boundingBox2DArray.h"

#include <boost/thread/thread.hpp>
// Eigen
#include <Eigen/Dense>
#include <queue>

#include <cmath>
#include <ctime>
#include <array>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <queue>
#include <assert.h>
#include <map>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <memory>

using namespace std;

#define WINDOW_SIZE 10
#define NUM_OF_F 1000

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12,
};

// Eigen::Vector3d G = Eigen::Vector3d(0,0,9.8);

struct PointPoseInfo
{
    double x;
    double y;
    double z;
    double qw;
    double qx;
    double qy;
    double qz;
    int idx;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointPoseInfo,
                                  (double, x, x)(double, y, y)(double, z, z)(double, qw, qw)(double, qx, qx)(double, qy, qy)(double, qz, qz)(int, idx, idx)(double, time, time))

// PCL point types

struct PointXYZILISS
{
    PCL_ADD_POINT4D; // quad-word XYZ
    float intensity; ///< laser intensity reading
    // std::uint16_t ring;
    std::int16_t label; ///< point label
    std::uint16_t id;   //   > 1 obj,0 others
    float alpha;
    double score; //   detection score
    double smooth;
    double cov00;
    double cov01;
    double cov02;
    double cov10;
    double cov11;
    double cov12;
    double cov20;
    double cov21;
    double cov22;
    int frame_id;
    int seq_id;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // ensure proper alignment
} EIGEN_ALIGN16;

// Register custom point struct according to PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZILISS,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::int16_t, label, label)(std::uint16_t, id, id)(float, alpha, alpha)(double, score, score)(double, smooth, smooth)(double, cov00, cov00)(double, cov01, cov01)(double, cov02, cov02)(double, cov10, cov10)(double, cov11, cov11)(double, cov12, cov12)(double, cov20, cov20)(double, cov21, cov21)(double, cov22, cov22)(int, frame_id, frame_id)(int, seq_id, seq_id))

struct PointXYZIKITTI
{
    PCL_ADD_POINT4D
    float intensity;
    // std::uint16_t ring;
    std::uint16_t id;
    float score;
    float alpha;
    int frame_id;
    int seq_id;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIKITTI,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint16_t, id, id)(float, score, score)(float, alpha, alpha)(int, frame_id, frame_id)(int, seq_id, seq_id))

using pcl::PointXYZI;
// using pcl::PointXYZINormal;
typedef PointXYZILISS PointType;

/*
 * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
 */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY; // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))

typedef PointXYZIRPYT PointTypePose;

enum FeatureType
{
    UNKNOWN = 0,
    CORNER = 1,
    CORNERLESS = 2,
    SURFACE = 3,
    SURFACELESS = 4
};

// enum FeatureStatus {
//     INTIAL = 0,
//     GROUND = 1,
//     BACKGROUND = 2,
//     SURFACE = 3,
//     SURFACELESS = 4
// };

// Get parameters from yaml file
template <class class_name>
bool getParameter(const std::string &paramName, class_name &param)
{
    std::string nodeName = ros::this_node::getName();
    std::string paramKey;
    if (!ros::param::search(paramName, paramKey))
    {
        ROS_ERROR("%s: Failed to search for parameter '%s'.", nodeName.c_str(), paramName.c_str());
        return false;
    }

    if (!ros::param::has(paramKey))
    {
        ROS_ERROR("%s: Missing required parameter '%s'.", nodeName.c_str(), paramName.c_str());
        return false;
    }

    if (!ros::param::get(paramKey, param))
    {
        ROS_ERROR("%s: Failed to get parameter '%s'.", nodeName.c_str(), paramName.c_str());
        return false;
    }

    return true;
}

template <typename T>
sensor_msgs::PointCloud2 publishPointCloud(const ros::Publisher &thisPub, const T &thisCloud, std_msgs::Header inputHeader)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header = inputHeader;
    // tempCloud.header.frame_id = thisFrame;
    if (thisPub.getNumSubscribers() != 0)
        thisPub.publish(tempCloud);
    return tempCloud;
};

class Timer
{
public:
    Timer(const char *nameIn)
    {
        name = nameIn;
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> dt = end - start;
        return dt.count() * 1000;
    }

    void tic_toc()
    {
        printf("The process %s takes %f ms.\n", name, toc());
    }

private:
    const char *name;
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

struct ObjTraj
{
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> velos;
    std::vector<Eigen::Quaterniond> orientations;
    int id;
};


template <typename T>
pair<T, T> calVarStdev(vector<T> vecNums) // 均值、方差和标准差计算
{
    pair<T, T> res;
    T sumNum = accumulate(vecNums.begin(), vecNums.end(), 0.0);
    T mean = sumNum / vecNums.size(); // 均值
    T accum = 0.0;
    for_each(vecNums.begin(), vecNums.end(), [&](const T d)
    { 
        accum += (d - mean) * (d - mean); 
    });
    T variance = accum / vecNums.size(); // 方差
    T stdev = sqrt(variance);            // 标准差

    res.first = mean;
    res.second = stdev;
    return res;
}

#endif // COMMON_H
