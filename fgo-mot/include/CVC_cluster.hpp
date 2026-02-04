#ifndef CVC_CLUSTER_H
#define CVC_CLUSTER_H

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/features/integral_image_normal.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/impl/passthrough.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>

// C++
#include <iostream>
#include <stdlib.h>
#include <unordered_map>
#include <algorithm>
#include <limits>

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/PolygonArray.h>

#include <utils/common.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
template <typename T>
std::string toString(const T &t)
{
	std::ostringstream oss;
	oss << t;
	return oss.str();
}

struct PointAPR
{
	float azimuth;
	float polar_angle;
	float range;
};

struct Voxel
{
	bool haspoint = false;
	int cluster = -1;
	std::vector<int> index;
};

bool compare_cluster(std::pair<int, int> a, std::pair<int, int> b)
{
	return a.second > b.second;
}; // upper sort

float Polar_angle_cal(float x, float y)
{
	float temp_tangle = 0;
	if (x == 0 && y == 0)
	{
		temp_tangle = 0;
	}
	else if (y >= 0)
	{
		temp_tangle = (float)atan2(y, x);
	}
	else if (y < 0)
	{
		temp_tangle = (float)atan2(y, x) + 2 * M_PI;
	}
	return temp_tangle;
};

class CVC
{
public:
	CVC();
	CVC(std::vector<float> &param)
	{
		if (param.size() != 5)
		{
			printf("Param number is not correct!");
			std::abort();
		}
		for (int i = 0; i < param.size(); ++i)
		{
			deltaA_ = param[0];
			deltaR_ = param[1];
			deltaP_ = param[2];
			min_range_ = param[3];
			max_range_ = param[4];
		}
	}

	~CVC() {}
	template <typename PointT>
	void calculateAPR(const pcl::PointCloud<PointT> &cloud_IN, std::vector<PointAPR> &vapr);

	void build_hash_table(const std::vector<PointAPR> &vapr, std::unordered_map<int, Voxel> &map_out);
	void find_neighbors(int polar, int range, int azimuth, std::vector<int> &neighborindex);
	bool most_frequent_value(std::vector<int> values, std::vector<int> &cluster_index);
	void mergeClusters(std::vector<int> &cluster_indices, int idx1, int idx2);
	std::vector<int> cluster(std::unordered_map<int, Voxel> &map_in, const std::vector<PointAPR> &vapr);
	void process();

private:
	float deltaA_ = 2;
	float deltaR_ = 0.35;
	float deltaP_ = 1.2;
	float min_range_ = std::numeric_limits<float>::max();
	float max_range_ = std::numeric_limits<float>::min();
	float min_azimuth_ = -24.8 * M_PI / 180;
	float max_azimuth_ = 2.0 * M_PI / 180;
	int length_ = 0;
	int width_ = 0;
	int height_ = 0;
};

template <typename PointT>
void CVC::calculateAPR(const pcl::PointCloud<PointT> &cloud_IN, std::vector<PointAPR> &vapr)
{
	for (int i = 0; i < cloud_IN.points.size(); ++i)
	{
		PointAPR par;
		par.polar_angle = Polar_angle_cal(cloud_IN.points[i].x, cloud_IN.points[i].y);
		par.range = sqrt(cloud_IN.points[i].x * cloud_IN.points[i].x + cloud_IN.points[i].y * cloud_IN.points[i].y);
		par.azimuth = (float)atan2(cloud_IN.points[i].z, par.range);
		if (par.range < min_range_)
		{
			min_range_ = par.range;
		}
		if (par.range > max_range_)
		{
			max_range_ = par.range;
		}
		vapr.push_back(par);
	}
	length_ = int((max_range_ - min_range_) / deltaR_) + 1;
	width_ = round(360 / deltaP_);
	height_ = int(((max_azimuth_ - min_azimuth_) * 180 / M_PI) / deltaA_) + 1;
}

void CVC::build_hash_table(const std::vector<PointAPR> &vapr, std::unordered_map<int, Voxel> &map_out)
{
	std::vector<int> ri;
	std::vector<int> pi;
	std::vector<int> ai;
	for (int i = 0; i < vapr.size(); ++i)
	{
		int azimuth_index = int(((vapr[i].azimuth - min_azimuth_) * 180 / M_PI) / deltaA_);
		int polar_index = int(vapr[i].polar_angle * 180 / M_PI / deltaP_);
		int range_index = int((vapr[i].range - min_range_) / deltaR_);

		int voxel_index = (polar_index * (length_) + range_index) + azimuth_index * (length_) * (width_);
		ri.push_back(range_index);
		pi.push_back(polar_index);
		ai.push_back(azimuth_index);
		std::unordered_map<int, Voxel>::iterator it_find;
		it_find = map_out.find(voxel_index);
		if (it_find != map_out.end())
		{
			it_find->second.index.push_back(i);
		}
		else
		{
			Voxel vox;
			vox.haspoint = true;
			vox.index.push_back(i);
			vox.index.swap(vox.index);
			map_out.insert(std::make_pair(voxel_index, vox));
		}
	}
	auto maxPosition = max_element(ai.begin(), ai.end());
	auto maxPosition1 = max_element(ri.begin(), ri.end());
	auto maxPosition2 = max_element(pi.begin(), pi.end());
}

void CVC::find_neighbors(int polar, int range, int azimuth, std::vector<int> &neighborindex)
{
	for (int z = azimuth - 1; z <= azimuth + 1; z++)
	{
		if (z < 0 || z > (height_ - 1))
		{
			continue;
		}

		for (int y = range - 1; y <= range + 1; y++)
		{
			if (y < 0 || y > (length_ - 1))
			{
				continue;
			}

			for (int x = polar - 1; x <= polar + 1; x++)
			{
				int px = x;
				if (x < 0)
				{
					px = width_ - 1;
				}
				if (x > (width_ - 1))
				{
					px = 0;
				}
				neighborindex.push_back((px * (length_) + y) + z * (length_) * (width_));
			}
		}
	}
}

bool CVC::most_frequent_value(std::vector<int> values, std::vector<int> &cluster_index)
{
	std::unordered_map<int, int> histcounts;
	for (int i = 0; i < values.size(); i++)
	{
		if (histcounts.find(values[i]) == histcounts.end())
		{
			histcounts[values[i]] = 1;
		}
		else
		{
			histcounts[values[i]] += 1;
		}
	}
	int max = 0, maxi;
	std::vector<std::pair<int, int>> tr(histcounts.begin(), histcounts.end());
	sort(tr.begin(), tr.end(), compare_cluster);
	for (int i = 0; i < tr.size(); ++i)
	{
		if (tr[i].second > 10)
		{
			cluster_index.push_back(tr[i].first);
		}
	}

	return true;
}

std::vector<int> CVC::cluster(std::unordered_map<int, Voxel> &map_in, const std::vector<PointAPR> &vapr)
{
	int current_cluster = 0;
	std::vector<int> cluster_indices = std::vector<int>(vapr.size(), -1);

	for (int i = 0; i < vapr.size(); ++i)
	{

		if (cluster_indices[i] != -1)
			continue;
		int azimuth_index = int((vapr[i].azimuth - min_azimuth_) * 180 / M_PI / deltaA_);
		int polar_index = int(vapr[i].polar_angle * 180 / M_PI / deltaP_);
		int range_index = int((vapr[i].range - min_range_) / deltaR_);
		int voxel_index = (polar_index * (length_) + range_index) + azimuth_index * (length_) * (width_);

		std::unordered_map<int, Voxel>::iterator it_find;
		std::unordered_map<int, Voxel>::iterator it_find2;

		it_find = map_in.find(voxel_index);
		std::vector<int> neightbors;

		if (it_find != map_in.end())
		{

			std::vector<int> neighborid;
			find_neighbors(polar_index, range_index, azimuth_index, neighborid);
			for (int k = 0; k < neighborid.size(); ++k)
			{

				it_find2 = map_in.find(neighborid[k]);

				if (it_find2 != map_in.end())
				{

					for (int j = 0; j < it_find2->second.index.size(); ++j)
					{
						neightbors.push_back(it_find2->second.index[j]);
					}
				}
			}
		}

		neightbors.swap(neightbors);

		if (neightbors.size() > 0)
		{
			for (int j = 0; j < neightbors.size(); ++j)
			{
				int oc = cluster_indices[i];
				int nc = cluster_indices[neightbors[j]];
				if (oc != -1 && nc != -1)
				{
					if (oc != nc)
						mergeClusters(cluster_indices, oc, nc);
				}
				else
				{
					if (nc != -1)
					{
						cluster_indices[i] = nc;
					}
					else
					{
						if (oc != -1)
						{
							cluster_indices[neightbors[j]] = oc;
						}
					}
				}
			}
		}

		if (cluster_indices[i] == -1)
		{
			current_cluster++;
			cluster_indices[i] = current_cluster;
			for (int s = 0; s < neightbors.size(); ++s)
			{
				cluster_indices[neightbors[s]] = current_cluster;
			}
		}
	}
	return cluster_indices;
}

void CVC::mergeClusters(std::vector<int> &cluster_indices, int idx1, int idx2)
{
	for (int i = 0; i < cluster_indices.size(); i++)
	{
		if (cluster_indices[i] == idx1)
		{
			cluster_indices[i] = idx2;
		}
	}
}

class ObjectCluster
{

public:
	ObjectCluster() = default;

	~ObjectCluster() = default;

public:
	std::vector<PointType> cluster_points_;

	std::vector<PointType> convex_hull_points_;
	double length_;
	double width_;
	double height_;
	double yaw_; //方块的朝向
	double min_y_;
	double max_y_;
	double min_x_;
	double max_x_;
	double min_z_;
	double max_z_;
	double average_x_;
	double average_y_;
	double average_z_;
	double alpha;
	double score;
	Eigen::Vector3d center_ = Eigen::Vector3d::Zero();
	Eigen::Vector3d direction_ = Eigen::Vector3d(1, 0, 0);
	std::vector<PointType> bbox_points_;
	PointType center_point_;
	jsk_recognition_msgs::BoundingBox bounding_box_;
	geometry_msgs::PolygonStamped polygon_;

	fgo_mot::boundingBox2D bounding_box2d_;

	double average_intensity_ = -1.0;
	int box2d_min_x;
	int box2d_min_y;
	int box2d_max_x;
	int box2d_max_y;
	int box2d_width;
	int box2d_height;

	int id;

	cv::RotatedRect box_;
	std::vector<cv::Point2f> convex_hull_;
	std::vector<cv::Point2f> origin_point2f_;
	std::vector<Eigen::Vector3d> origin_point3f_;

	bool box_initial;

	/***
	* @description: label = -1 未知
					label = 0  不可用
					label = 1  可用
	* @return {*}
	*/
	bool usable = true;

	bool object_classify()
	{
		if (cluster_points_.size() < 3)
			return false;
		float dx = bounding_box_.dimensions.x;
		float dy = bounding_box_.dimensions.y;
		float dz = bounding_box_.dimensions.z;
		double ratio1 = dz / (dx < dy ? dx : dy);
		double ratio2 = (dx / dy > dy / dx ? dx / dy : dy / dx);
		double min_width = dx < dy ? dx : dy;
		double volume = dx * dy * dz;
		if (dx > 6 || dy > 6 || dz > 5)
			usable = false;
		else if (min_width > 3)
			usable = false;
		else if (volume > 20)
			usable = false;
		// else if (ratio1 > 1.2 || ratio1 < 0.5 || (dx < dy ? dx : dy) < 0.1)
		// 	usable = false;
		// else if (ratio2 > 5.5)
		// 	usable = false;
		else if (max_z_ < -0.8 || max_z_ > 0.2)
			usable = false;
		else
			usable = true;

		return usable;
	}
};

typedef std::vector<ObjectCluster> ClusterVector;
typedef std::shared_ptr<ClusterVector> ClusterVectorPtr;

#endif
