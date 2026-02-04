#pragma once
#include "utils/common.h"
#include "factors/integrationBase.h"

struct LaserFeature
{
    std_msgs::Header header;
    pcl::PointCloud<PointType> laser_cloud_full;
    pcl::PointCloud<PointType> laser_cloud_corner;
    pcl::PointCloud<PointType> laser_cloud_surf;
	pcl::PointCloud<PointType> laser_cloud_full_ds;
	pcl::PointCloud<PointType> laser_cloud_corner_ds;
    pcl::PointCloud<PointType> laser_cloud_surf_ds;
    std::vector<pcl::PointCloud<PointType>> laser_cloud_obj;
};

typedef std::shared_ptr<LaserFeature> LaserFeaturePtr;
typedef std::shared_ptr<LaserFeature const> LaserFeatureConstPtr;

/**
*@brief LaserFrame Class to store laser frame info for lidar odometry
*
*/
class LaserFrame
{
public:
	LaserFrame() {};
	/**
	* @brief constructing function for assigning values to parameters
	* @param[in] laser_feature			laser feature points info(shared_ptr)
	* @param[in] t						time
	*/
	LaserFrame(LaserFeaturePtr laser_feature, double _t) :t{ _t }, is_key_frame{ false }
	{
		laser_cloud_full = laser_feature->laser_cloud_full.makeShared();							
		laser_cloud_corner = laser_feature->laser_cloud_corner.makeShared();				
		laser_cloud_surf = laser_feature->laser_cloud_surf.makeShared();
		laser_cloud_full_ds = laser_feature->laser_cloud_full_ds.makeShared();	
		laser_cloud_corner_ds = laser_feature->laser_cloud_corner_ds.makeShared();				
		laser_cloud_surf_ds = laser_feature->laser_cloud_surf_ds.makeShared();
		laser_cloud_obj = laser_feature->laser_cloud_obj;
	};
	/**
	* @brief constructing function for assigning values to parameters
	* @param[in] laser_feature			laser feature points info(sensor_msg_class)
	* @param[in] t						time
	*/
	LaserFrame(LaserFeature laser_feature, double _t) :t{ _t }, is_key_frame{ false }
	{
		laser_cloud_full = laser_feature.laser_cloud_full.makeShared();							
		laser_cloud_corner = laser_feature.laser_cloud_corner.makeShared();				
		laser_cloud_surf = laser_feature.laser_cloud_surf.makeShared();
		laser_cloud_full_ds = laser_feature.laser_cloud_full_ds.makeShared();	
		laser_cloud_corner_ds = laser_feature.laser_cloud_corner_ds.makeShared();				
		laser_cloud_surf_ds = laser_feature.laser_cloud_surf_ds.makeShared();
		laser_cloud_obj = laser_feature.laser_cloud_obj;
	};


	pcl::PointCloud<PointType>::Ptr laser_cloud_full;					/// laser cloud info
	pcl::PointCloud<PointType>::Ptr laser_cloud_corner;				/// received corner points 
	pcl::PointCloud<PointType>::Ptr laser_cloud_surf;			/// 
	pcl::PointCloud<PointType>::Ptr laser_cloud_full_ds;			/// 
	pcl::PointCloud<PointType>::Ptr laser_cloud_corner_ds;				/// received corner points 
	pcl::PointCloud<PointType>::Ptr laser_cloud_surf_ds;			/// 
	std::vector<pcl::PointCloud<PointType>> laser_cloud_obj;					/// recceived surface points


	double t;														/// time
	Eigen::Matrix3d R;												/// rotation matrix
	Eigen::Vector3d T;												/// translation matrix
	IntegrationBase *pre_integration;								/// pointer to pre_integration items
	bool is_key_frame;												/// sign of the frame is key frame or not
};

typedef std::shared_ptr<LaserFrame> LaserFramePtr;
typedef std::shared_ptr<LaserFrame const> LaserFrameConstPtr;

struct LaserFrameMsg
{
	/**
	* @brief update lidar info
	* @param[in] _laser_frame		current laser frame
	* @param[in] _frame_index		index of the frame
	*/
	LaserFrameMsg(LaserFramePtr _laser_frame, int _frame_index) :
		laser_frame{ _laser_frame }, frame_index{ _frame_index }
	{
		// lidar_info = new LidarInfo();
	};
	LaserFramePtr laser_frame;
	
	int frame_index;				/// index of the frame
	bool is_correct = false;		/// unused
	bool is_fix = false;			/// whether residual is fixed or not

	// //for corner
	std::vector<Eigen::Vector3d> laserCloudCornerFeatureVec;		/// current corner_feature  point of the laser scan
	std::vector<Eigen::Vector3d> laserCloudCornerPointAVec;		/// the closest point to the feature point 
	std::vector<Eigen::Vector3d> laserCloudCornerPointBVec;		/// another point of the correspond edge line

	// //for surf
	std::vector<Eigen::Vector3d> laserCloudSurfFeatureVec;		/// current surface point of laser cloud
	std::vector<Eigen::Vector3d> surfNorm;				/// norm of the correspond surface
	std::vector<double> surfNegativeOADotNorm;			/// distance from the plane to the origin

	//�洢����ͼ�Ż��Ĺ۲���Ϣ	
	// LidarInfo* lidar_info;							/// lidar observation for graph optimization
};
