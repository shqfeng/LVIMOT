#pragma once
#include "utils/common.h"
#include "factors/marginalizationFactor.h"

class objectOptimization
{
public:
	int findPoseInd(pair<int, int> _index);
	int findFeatureInd(int _index);

public:
	int obj_id;
	bool isStatic[1500] = {false};

	map<pair<int, int>, int> obj_double_map;
	map<int, int> obj_feature_double_map;

	// 待优化参数，w系位姿(x,y,z,qw,qx,qy,qz)、w系速度(vx,vy,vz)、w系加速度(ax,ay,az)
	double para_objPose[1500][SIZE_POSE];
	double para_objLinearVel[1500][3];
	double para_objAngularVel[1500][3];
	double para_objAcc[1500][3];

	double para_objDimensions[1500][3];
	double para_objFeature[1500][3];

	double ceres_cov[1500][15 * 15];

	MarginalizationInfo *last_marginalization_info_tracking = nullptr;
	vector<double *> last_marginalization_parameter_blocks_tracking;
};

int objectOptimization::findPoseInd(pair<int, int> _index)
{
	map<pair<int, int>, int>::iterator it = obj_double_map.find(_index);
	if (it != obj_double_map.end())
	{
		// return obj_double_map[_index];
		return it->second;
	}
	else
	{
		return -1;
	}
	// std::cout << "Before map find operation..." << std::endl;
    
    // 关键操作 - 这里可能崩溃
//     map<pair<int, int>, int>::iterator it = obj_double_map.find(_index);
//     // std::cout << "After map find operation..." << std::endl;
    
//     if (it != obj_double_map.end())
//     {
//         // std::cout << "Found index: " << it->second << std::endl;
//         // std::cout << "=== findPoseInd DEBUG END ===" << std::endl;
//         return it->second;
//     }
//     else
//     {
//         std::cout << "Index not found in map" << std::endl;
//         std::cout << "=== findPoseInd DEBUG END ===" << std::endl;
//         return -1;
//     }
}

// int objectOptimization::findFeatureInd(int  _index)
// {
// 	auto& it_feature = obj_feature_double_map.find(_index);
// 	if (it_feature != obj_feature_double_map.end())
// 	{
// 		return obj_feature_double_map[_index];
// 	}
// 	else
// 	{
// 		return -1;
// 	}
// }
