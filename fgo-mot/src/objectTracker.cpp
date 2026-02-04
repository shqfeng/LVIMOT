#define PCL_NO_PRECOMPILE
#include "utils/common.h"
#include "utils/math_tools.h"
#include "featureManager/objectFrame.h"
#include "utils/msg.h"
#include "HungarianAlg.hpp"
#include "utils/utility.h"
#include "factors/objectFactor.h"
#include "factors/pose_local_parameterization.h"
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <fstream>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>

class ObjectTracker
{
private:
    ros::NodeHandle nh;

    ros::Subscriber subCloudInfo;

    ros::Publisher pubObjectPointCloudFull;
    ros::Publisher pubObjectCornerPoints;
    ros::Publisher pubObjectSurfacePoints;
    ros::Publisher pubObjectBboxArray;
    ros::Publisher pubBoxCorners;
    ros::Publisher pubObjectBboxArrayPrev;
    ros::Publisher pubTrackBboxArray;
    ros::Publisher pubMarkers;
    ros::Publisher pubVeloArrowArray;
    ros::Publisher pubVeloTextArray;
    ros::Publisher pubObjectPointCloudLast;
    ros::Publisher pubGlobalTrajs;
    ros::Publisher pubObjPosesGlobal;
    ros::Publisher pubTrackBboxArrayAfterOpt;
    ros::Publisher pubTrackBboxArrayPredict;
    ros::Publisher pubTrackPosArrayPredictCov;
    ros::Publisher pubObjectCloudCorner;
    ros::Publisher pubObjectCloudSurf;
    ros::Publisher pubObjectPointCloudFullInWorld;
    ros::Publisher pubPath;
    ros::Publisher pubCar;
    ros::Publisher pubCloudinCamera;

    nav_msgs::Path path;

    ros::Publisher pubSuccessBbox;
    ros::Publisher pubFailedBbox;

    std_msgs::Header cloudHeader;
    fgo_mot::cloud_info cloudInfo;

    pcl::PointCloud<PointType>::Ptr laserCloudFeatureOrigin;
    std::vector<fgo_mot::detect_object> detectObjectMsg;
    std::vector<pcl::PointCloud<PointType>> objFullCloudVec;
    std::vector<pcl::PointCloud<PointType>> objCornerCloudVec;
    std::vector<pcl::PointCloud<PointType>> objSurfaceCloudVec;
    jsk_recognition_msgs::BoundingBoxArray objBboxVec;
    jsk_recognition_msgs::BoundingBoxArray objBboxVecPrev;
    jsk_recognition_msgs::BoundingBoxArray objBboxVecPred;
    visualization_msgs::MarkerArray objPosCovVecPred;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr objectCornerCloudVis;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr objectSurfaceCloudVis;

    int last_frame_id = -1;

    std::vector<sensor_msgs::ObjectTrack> allObjectsData;

    double timeNewCloudInfo = 0;
    bool isFirstFrame;
    int obj_num_count_curr;
    int obj_id;
    int global_frame_count;
    int seq_ind;
    string frame_id = "fgo_mot";

    Eigen::Vector3d pos_body;
    Eigen::Quaterniond quater_body;

    std::vector<objectOptimization *> objOptbuffer;

    std::string output_path;
    fstream obj_tracking_opt_kitti;
    fstream obj_tracking_opt_kitti_test;

    double ZUPT_Linear_vel, ZUPT_Angular_vel;
    Eigen::Vector3d tempdimension;
    int maxBoxLossNum;

    bool sliding_window_opt;// 0--->null, 1--->const vel, 2--->const acc
    int motion_model;
    bool use_ZUPT;
    bool use_box_model;
    bool use_marginalization;

    double max_costmatrixvalue;

    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterCorner;

    double voxel_leaf_size;

    std::vector<int> sequences;
    std::string sequence_num;

    int birthmin;
    std::vector<double> dataAssociation_timer;
    std::vector<double> optimization_timer;
    double motion_weight;

    bool output_predict;

    std::map<int, std::map<int, std::vector<double>>> frame_last_dicts;

    std::map<int, int> cloudinfo_bbox_map;

    std::map<int, Eigen::Matrix4d> Ego_pose;

    Eigen::Matrix4d P2, R_rect, Tr_velo_cam;

    std::string base_path;

public:
    ObjectTracker() : nh("~")
    {
        initializeParameters();
        allocateMemory();

        subCloudInfo = nh.subscribe<fgo_mot::cloud_info>("/odometry/cloud_info", 10000, &ObjectTracker::laserCloudInfoHandler, this);

        pubObjectPointCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/tracker/object_feature_full", 1);
        pubObjectCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("/tracker/cloud_object_corner", 1);
        pubObjectSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("/tracker/cloud_object_surface", 1);
        pubObjectPointCloudLast = nh.advertise<sensor_msgs::PointCloud2>("/tracker/object_cloud_local_map", 1);

        pubObjectPointCloudFullInWorld = nh.advertise<sensor_msgs::PointCloud2>("/tracker/object_feature_full_world", 1);

        pubObjectCloudCorner = nh.advertise<sensor_msgs::PointCloud2>("/tracker/object_corner_last", 1);

        pubObjectCloudSurf = nh.advertise<sensor_msgs::PointCloud2>("/tracker/object_surf_last", 1);

        pubBoxCorners = nh.advertise<sensor_msgs::PointCloud2>("/tracker/box_corners", 1);

        pubObjectBboxArray = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/tracker/object_bbox_array", 1);

        pubObjectBboxArrayPrev = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/tracker/object_bbox_array_prev", 1);

        pubTrackBboxArray = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/tracker/track_object_bbox_array", 1);

        pubTrackBboxArrayAfterOpt = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/tracker/track_object_bbox_array_aft_opt", 1);

        pubTrackBboxArrayPredict = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/tracker/track_object_bbox_array_predict", 1);

        pubTrackPosArrayPredictCov = nh.advertise<visualization_msgs::MarkerArray>("/tracker/track_object_position_predict_covariance", 1);

        pubMarkers = nh.advertise<visualization_msgs::MarkerArray>("/tracker/object_info", 1);

        pubVeloArrowArray = nh.advertise<visualization_msgs::MarkerArray>("/tracker/object_velos_arrow", 1);

        pubVeloTextArray = nh.advertise<visualization_msgs::MarkerArray>("/tracker/object_velos_text", 1);

        pubGlobalTrajs = nh.advertise<visualization_msgs::MarkerArray>("/tracker/obj_trajs", 1);

        pubObjPosesGlobal = nh.advertise<geometry_msgs::PoseArray>("/tracker/obj_poses", 1);

        pubSuccessBbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/tracker/Success_bbox", 1);

        pubFailedBbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/tracker/Failed_Bbox", 1);

        pubCar = nh.advertise<visualization_msgs::Marker>("/odometry/ego_car", 1);
        pubPath = nh.advertise<nav_msgs::Path>("/path", 100);
        pubCloudinCamera = nh.advertise<sensor_msgs::PointCloud2>("/tracker/Camera_cloud", 1);
    }

    virtual ~ObjectTracker()
    {
        save_tracking_result();
        std::pair<double, double> mean_std_timer_dataAssociation, mean_std_timer_optimization;
        mean_std_timer_dataAssociation = calVarStdev(dataAssociation_timer);
        mean_std_timer_optimization = calVarStdev(optimization_timer);
        printf("\033[1;32mdataAssociation   Time[ms] : %0.2f ± %0.2f, %0.0f FPS. \033[0m \n", mean_std_timer_dataAssociation.first, mean_std_timer_dataAssociation.second, floor(1000.0 / mean_std_timer_dataAssociation.first));
        printf("\033[1;32moptimization      Time[ms] : %0.2f ± %0.2f, %0.0f FPS. \033[0m \n", mean_std_timer_optimization.first, mean_std_timer_optimization.second, floor(1000.0 / mean_std_timer_optimization.first));
    }

    void initializeParameters()
    {
        if (!getParameter("/common/frame_id", frame_id))
        {
            ROS_WARN("frame_id not set, use default value: fgo_mot");
            frame_id = "fgo_mot";
        }
        if (!getParameter("/object_tracker/output_path", output_path))
        {
            ROS_WARN("output_path not set, use default value: output_path");
            output_path = "output_path";
        }
        if (!getParameter("/object_tracker/maxBoxLossNum", maxBoxLossNum))
        {
            ROS_WARN("maxBoxLossNum not set, use default value: 3");
            maxBoxLossNum = 3;
        }
        if (!getParameter("/object_tracker/sliding_window_opt", sliding_window_opt))
        {
            ROS_WARN("sliding_window_opt not set, use default value: true");
            sliding_window_opt = true;
        }
        if (!getParameter("/object_tracker/motion_model", motion_model))
        {
            ROS_WARN("motion_model not set, use default value: const_acc_w");
            motion_model = 2;
        }
        if (!getParameter("/object_tracker/use_ZUPT", use_ZUPT))
        {
            ROS_WARN("use_ZUPT not set, use default value: true");
            use_ZUPT = true;
        }
        if (!getParameter("/object_tracker/ZUPT_Linear_vel", ZUPT_Linear_vel))
        {
            ROS_WARN("ZUPT_Linear_vel not set, use default value: 0.5");
            ZUPT_Linear_vel = 0.5;
        }
        if (!getParameter("/object_tracker/ZUPT_Angular_vel", ZUPT_Angular_vel))
        {
            ROS_WARN("ZUPT_Angular_vel not set, use default value: 0.1");
            ZUPT_Angular_vel = 0.1;
        }
        if (!getParameter("/object_tracker/use_box_model", use_box_model))
        {
            ROS_WARN("use_box_model not set, use default value: true");
            use_box_model = true;
        }
        if (!getParameter("/object_tracker/use_marginalization", use_marginalization))
        {
            ROS_WARN("use_marginalization not set, use default value: true");
            use_marginalization = true;
        }
        if (!getParameter("/object_tracker/max_costmatrixvalue", max_costmatrixvalue))
        {
            ROS_WARN("max_costmatrixvalue not set, use default value: 0.9");
            max_costmatrixvalue = 0.9;
        }
        if (!getParameter("/object_tracker/voxel_leaf_size", voxel_leaf_size))
        {
            ROS_WARN("voxel_leaf_size not set, use default value: 0.2");
            voxel_leaf_size = 0.2;
        }
        if (!getParameter("/kitti_helper/sequences", sequences))
        {
            ROS_WARN("sequence not set, use default value: [1,6,8,10,12,13,14,15,16,18,19]");
            sequences = {1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19};
        }
        if (!getParameter("/object_tracker/birthmin", birthmin))
        {
            ROS_WARN("sequence not set, use default value: 5");
            birthmin = 5;
        }
        if (!getParameter("/object_tracker/motion_weight", motion_weight))
        {
            ROS_WARN("motion_weight not set, use default value: 1.0");
            motion_weight = 1.0;
        }
        if (!getParameter("/object_tracker/output_predict", output_predict))
        {
            ROS_WARN("output_predict not set, use default value: false");
            output_predict = false;
        }
        if (!getParameter("/kitti_helper/base_path", base_path))
        {
            ROS_WARN("base_path not set, use default value: /home/hickeytom/Workspace/F-MOT-V2_WS/tracking/training/");
            base_path = "/home/hickeytom/Workspace/F-MOT-V2_WS/tracking/training/";
        }

        std::stringstream ss;
        ss << setw(4) << setfill('0') << sequences[0];
        ss >> sequence_num;
        obj_tracking_opt_kitti.open(output_path + "/" +"tracking_kitti/" + "/"+ sequence_num + ".txt", ios::out | ios::trunc);  // keep
        obj_tracking_opt_kitti_test.open(output_path + "/" +"tracking_kitti_test/"+ sequence_num + ".txt", ios::out | ios::trunc);  // keep

        std::string calib_file = base_path + "/calib/" + sequence_num + ".txt";
        read_calib_data(calib_file);

        isFirstFrame = true;

        obj_id = 0;

        global_frame_count = 0;

        pos_body = Eigen::Vector3d(0, 0, 0);
        quater_body = Eigen::Quaterniond(1, 0, 0, 0);

        downSizeFilterSurf.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
        downSizeFilterCorner.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);

    }

    bool read_calib_data(std::string calib_file_path)
    {
        P2.setIdentity();
        R_rect.setIdentity();
        Tr_velo_cam.setIdentity();
        std::ifstream calib_data_file(calib_file_path, std::ifstream::in);

        if (!calib_data_file)
            cerr << "calib_data_file does not exist " << std::endl;

        std::string line;
        int count = 0;
        while (getline(calib_data_file, line))
        {
            if (count == 2)
            {
                std::stringstream calib_stream(line);
                std::string s;
                std::getline(calib_stream, s, ' ');
                if (s != "P2:")
                    return false;
                for (std::size_t i = 0; i < 3; ++i)
                {
                    for (std::size_t j = 0; j < 4; ++j)
                    {
                        std::getline(calib_stream, s, ' ');
                        P2(i, j) = stof(s);
                    }
                }
                P2(3, 0) = P2(3, 1) = P2(3, 2) = 0.0;
                P2(3, 3) = 1.0;
            }
            else if (count == 4)
            {
                std::stringstream calib_stream(line);
                std::string s;
                std::getline(calib_stream, s, ' ');
                if (s != "R_rect")
                    return false;
                for (std::size_t i = 0; i < 3; ++i)
                {
                    for (std::size_t j = 0; j < 3; ++j)
                    {
                        std::getline(calib_stream, s, ' ');
                        R_rect(i, j) = stof(s);
                    }
                }
                R_rect(3, 0) = R_rect(3, 1) = R_rect(3, 2) = R_rect(0, 3) = R_rect(1, 3) = R_rect(2, 3) = 0.0;
                R_rect(3, 3) = 1.0;
            }
            else if (count == 5)
            {
                std::stringstream calib_stream(line);
                std::string s;
                std::getline(calib_stream, s, ' ');
                if (s != "Tr_velo_cam")
                    return false;
                for (std::size_t i = 0; i < 3; ++i)
                {
                    for (std::size_t j = 0; j < 4; ++j)
                    {
                        std::getline(calib_stream, s, ' ');
                        Tr_velo_cam(i, j) = stof(s);
                    }
                }
                Tr_velo_cam(3, 0) = Tr_velo_cam(3, 1) = Tr_velo_cam(3, 2) = 0.0;
                Tr_velo_cam(3, 3) = 1.0;
            }
            else{}
        
            count++;
        }
        Tr_velo_cam = R_rect * Tr_velo_cam;
        calib_data_file.close();
        return true;
    }

    void allocateMemory()
    {
        laserCloudFeatureOrigin.reset(new pcl::PointCloud<PointType>());

        objectCornerCloudVis.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
        objectSurfaceCloudVis.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    }

    jsk_recognition_msgs::BoundingBox transformBox2World(const jsk_recognition_msgs::BoundingBox &bbox)
    {
        jsk_recognition_msgs::BoundingBox bbox_trans = bbox;
        Eigen::Vector3d pos_obj(bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z);
        Eigen::Quaterniond quater_obj;
        quater_obj.w() = bbox.pose.orientation.w;
        quater_obj.x() = bbox.pose.orientation.x;
        quater_obj.y() = bbox.pose.orientation.y;
        quater_obj.z() = bbox.pose.orientation.z;
        pos_obj = quater_body * pos_obj + pos_body;
        quater_obj = quater_body * quater_obj;
        bbox_trans.header.frame_id = "map";
        bbox_trans.pose.position.x = pos_obj.x();
        bbox_trans.pose.position.y = pos_obj.y();
        bbox_trans.pose.position.z = pos_obj.z();
        bbox_trans.pose.orientation.w = quater_obj.w();
        bbox_trans.pose.orientation.x = quater_obj.x();
        bbox_trans.pose.orientation.y = quater_obj.y();
        bbox_trans.pose.orientation.z = quater_obj.z();
        return bbox_trans;
    }

    PointType transformPoint2World(const PointType &point)
    {
        PointType point_trans = point;
        Eigen::Vector3d p(point.x, point.y, point.z);
        p = quater_body * p + pos_body;
        point_trans.x = p.x();
        point_trans.y = p.y();
        point_trans.z = p.z();

        return point_trans;
    }

    Eigen::Vector3d transformPoint2World(const Eigen::Vector3d &point)
    {
        Eigen::Vector3d point_trans = quater_body * point + pos_body;
        return point_trans;
    }

    Eigen::Vector3d transformPoint(const Eigen::Vector3d &point, const Eigen::Vector3d &position, const Eigen::Quaterniond &quater)
    {
        Eigen::Vector3d point_trans = quater * point + position;
        return point_trans;
    }

    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int numPts = cloudIn->points.size();
        cloudOut->resize(numPts);

        for (int i = 0; i < numPts; ++i)
        {
            Eigen::Vector3d ptIn(cloudIn->points[i].x, cloudIn->points[i].y, cloudIn->points[i].z);
            Eigen::Vector3d ptOut = quater_body * ptIn + pos_body;

            PointType pt = cloudIn->points[i];
            pt.x = ptOut.x();
            pt.y = ptOut.y();
            pt.z = ptOut.z();

            cloudOut->points[i] = pt;
        }

        return cloudOut;
    }

    void objCloudInfoHandle()
    {
        obj_num_count_curr = 0;
        detectObjectMsg.clear();
        objFullCloudVec.clear();
        objCornerCloudVec.clear();
        objSurfaceCloudVec.clear();
        objBboxVec.boxes.clear();
        objBboxVec.header = cloudHeader;

        cloudinfo_bbox_map.clear();

        for (int i = 0; i < cloudInfo.objArray.size(); ++i)
        {
            if (cloudInfo.objArray[i].objPointIndex.size() < 1)
                continue;

            if (cloudInfo.objArray[i].bounding_box.pose.position.x > 60)
                continue;

            detectObjectMsg.push_back(cloudInfo.objArray[i]);
            jsk_recognition_msgs::BoundingBox bbox;
            bbox = cloudInfo.objArray[i].bounding_box;

            bbox.header = cloudHeader;
            bbox.label = cloudInfo.objArray[i].id;
            objBboxVec.boxes.push_back(transformBox2World(bbox));

            pcl::PointCloud<PointType> obj_cloud;
            pcl::PointCloud<PointType> obj_cloud_corner;
            pcl::PointCloud<PointType> obj_cloud_surface;
            for (int j = 0; j < cloudInfo.objArray[i].objPointIndex.size(); ++j)
            {
                obj_cloud.push_back(transformPoint2World(laserCloudFeatureOrigin->points[cloudInfo.objArray[i].objPointIndex[j]]));
                if (laserCloudFeatureOrigin->points[cloudInfo.objArray[i].objPointIndex[j]].label == SURFACE)
                {
                    obj_cloud_corner.push_back(transformPoint2World(laserCloudFeatureOrigin->points[cloudInfo.objArray[i].objPointIndex[j]]));
                }
                else if (laserCloudFeatureOrigin->points[cloudInfo.objArray[i].objPointIndex[j]].label == SURFACE)
                {
                    obj_cloud_surface.push_back(transformPoint2World(laserCloudFeatureOrigin->points[cloudInfo.objArray[i].objPointIndex[j]]));
                }
            }
            objFullCloudVec.push_back(obj_cloud);
            objCornerCloudVec.push_back(obj_cloud_corner);
            objSurfaceCloudVec.push_back(obj_cloud_surface);

            cloudinfo_bbox_map[obj_num_count_curr] = i;

            obj_num_count_curr++;
        }

        objectCornerCloudVis->clear();
        objectSurfaceCloudVis->clear();
        pcl::PointCloud<pcl::PointXYZRGB> box_corner_pc;
        cv::RNG rng(12345);
        for (int i = 0; i < objFullCloudVec.size(); ++i)
        {
            int r = rng.uniform(20, 255);
            int g = rng.uniform(20, 255);
            int b = rng.uniform(20, 255);

            for (int j = 0; j < objFullCloudVec[i].points.size(); ++j)
            {
                pcl::PointXYZRGB point;
                point.x = objFullCloudVec[i].points[j].x;
                point.y = objFullCloudVec[i].points[j].y;
                point.z = objFullCloudVec[i].points[j].z;
                point.r = r;
                point.g = g;
                point.b = b;
                if (objFullCloudVec[i].points[j].label == CORNER)
                    objectCornerCloudVis->points.push_back(point);
                else if (objFullCloudVec[i].points[j].label == SURFACE)
                    objectSurfaceCloudVis->points.push_back(point);
            }
            std::vector<Eigen::Vector3d> corners = convertBox2Corners(objBboxVec.boxes[i]);
            for (int k = 0; k < corners.size(); ++k)
            {
                pcl::PointXYZRGB point;
                point.x = corners[k].x();
                point.y = corners[k].y();
                point.z = corners[k].z();
                point.r = r;
                point.g = g;
                point.b = b;
                box_corner_pc.push_back(point);
            }
        }
        
        if (pubObjectPointCloudFull.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 laserCloudTemp;
            pcl::toROSMsg(*transformCloud(laserCloudFeatureOrigin), laserCloudTemp);
            laserCloudTemp.header = cloudHeader;
            laserCloudTemp.header.frame_id = "map";
            pubObjectPointCloudFull.publish(laserCloudTemp);
        }
        
        if (pubObjectCornerPoints.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 laserCloudTemp;
            pcl::toROSMsg(*objectCornerCloudVis, laserCloudTemp);
            laserCloudTemp.header = cloudHeader;
            laserCloudTemp.header.frame_id = "map";
            pubObjectCornerPoints.publish(laserCloudTemp);
        }
        
        if (pubObjectSurfacePoints.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 laserCloudTemp;
            pcl::toROSMsg(*objectSurfaceCloudVis, laserCloudTemp);
            laserCloudTemp.header = cloudHeader;
            laserCloudTemp.header.frame_id = "map";
            pubObjectSurfacePoints.publish(laserCloudTemp);
        }
        
        if (pubBoxCorners.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 laserCloudTemp;
            pcl::toROSMsg(box_corner_pc, laserCloudTemp);
            laserCloudTemp.header = cloudHeader;
            laserCloudTemp.header.frame_id = "map";
            pubBoxCorners.publish(laserCloudTemp);
        }
        
        if (pubObjectBboxArray.getNumSubscribers() != 0)
        {
            pubObjectBboxArray.publish(objBboxVec);
        }
    }

    std::vector<Eigen::Vector3d> convertBox2Corners(jsk_recognition_msgs::BoundingBox box)
    {
        double width = box.dimensions.x;
        double length = box.dimensions.y;
        double height = box.dimensions.z;
        Eigen::Quaterniond q;
        q.w() = box.pose.orientation.w;
        q.x() = box.pose.orientation.x;
        q.y() = box.pose.orientation.y;
        q.z() = box.pose.orientation.z;

        Eigen::Vector3d center = Eigen::Vector3d(box.pose.position.x, box.pose.position.y, box.pose.position.z);
        Eigen::VectorXd x_corners(9), y_corners(9), z_corners(9);
        x_corners << -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, 0;
        y_corners << length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2, 0;
        z_corners << -height / 2, -height / 2, -height / 2, -height / 2, height / 2, height / 2, height / 2, height / 2, 0;
        std::vector<Eigen::Vector3d> corners_xyz;

        for (int i = 0; i < 9; ++i)
        {
            Eigen::Vector3d point(x_corners[i], y_corners[i], z_corners[i]);
            point = q.toRotationMatrix() * point;
            corners_xyz.push_back(point + center);
        }
        return corners_xyz;
    }

    bool generateNewObj(int obj_index)
    {
        sensor_msgs::ObjectTrack obj_tmp;
        obj_tmp.id = obj_id;

        sensor_msgs::ObjectPerFrameObs per_obj_detect;
        per_obj_detect.header = cloudHeader;
        per_obj_detect.frame_id = global_frame_count;
        per_obj_detect.seq_id = seq_ind;
        per_obj_detect.time = cloudHeader.stamp.toSec();
        per_obj_detect.alpha = cloudInfo.objArray[cloudinfo_bbox_map[obj_index]].alpha;
        per_obj_detect.score = cloudInfo.objArray[cloudinfo_bbox_map[obj_index]].score;

        per_obj_detect.feature_box.bbox3d_in_local = objBboxVec.boxes[obj_index];
        per_obj_detect.feature_box.bounding_box_2d = detectObjectMsg[obj_index].bounding_box2d;

        per_obj_detect.feature_box.corners_in_local = convertBox2Corners(objBboxVec.boxes[obj_index]);
        
        per_obj_detect.Pi = (Eigen::Vector3d(objBboxVec.boxes[obj_index].pose.position.x, objBboxVec.boxes[obj_index].pose.position.y,
                                             objBboxVec.boxes[obj_index].pose.position.z));
        per_obj_detect.Qi = Eigen::Quaterniond(objBboxVec.boxes[obj_index].pose.orientation.w, objBboxVec.boxes[obj_index].pose.orientation.x,
                                               objBboxVec.boxes[obj_index].pose.orientation.y, objBboxVec.boxes[obj_index].pose.orientation.z);
        
        per_obj_detect.Pi_obs = (Eigen::Vector3d(objBboxVec.boxes[obj_index].pose.position.x, objBboxVec.boxes[obj_index].pose.position.y,
                                                 objBboxVec.boxes[obj_index].pose.position.z));
        per_obj_detect.Qi_obs = Eigen::Quaterniond(objBboxVec.boxes[obj_index].pose.orientation.w, objBboxVec.boxes[obj_index].pose.orientation.x,
                                                   objBboxVec.boxes[obj_index].pose.orientation.y, objBboxVec.boxes[obj_index].pose.orientation.z);

        per_obj_detect.centroid = transformPoint2World(Eigen::Vector3d(detectObjectMsg[obj_index].average_x, detectObjectMsg[obj_index].average_y,
                                                                       objBboxVec.boxes[obj_index].pose.position.z));
        per_obj_detect.Dimensions = Eigen::Vector3d(objBboxVec.boxes[obj_index].dimensions.x, objBboxVec.boxes[obj_index].dimensions.y,
                                                    objBboxVec.boxes[obj_index].dimensions.z);

        per_obj_detect.Vi.setZero();
        per_obj_detect.Wi.setZero();
        per_obj_detect.Ai.setZero();
        per_obj_detect.covariance.setZero();

        per_obj_detect.pointcloudFull = objFullCloudVec[obj_index];
        per_obj_detect.pointcloudCorner = objCornerCloudVec[obj_index];
        per_obj_detect.pointcloudSurf = objSurfaceCloudVec[obj_index];

        obj_tmp.object_perframes.push_back(per_obj_detect);
        allObjectsData.push_back(obj_tmp);

        allObjectsData.back().track_box_num = 1;

        obj_id++;
        return true;
    }

    bool detectZeroVelocity(int obj_global_index)
    {
        if (!use_ZUPT)
            return false;
        if (obj_global_index < 0 || obj_global_index >= allObjectsData.size()) {
            return false;
        }
        
        const auto& track = allObjectsData[obj_global_index];
        
        int count = 0;
        std::vector<double> linear_vel_x, linear_vel_y, linear_vel_z;
        std::vector<double> angular_vel_x, angular_vel_y, angular_vel_z;
        
        for (int j = track.object_perframes.size() - 2; j >= 0; --j) 
        {
                const auto& frame = track.object_perframes[j];
                linear_vel_x.push_back(frame.Vi.x());
                linear_vel_y.push_back(frame.Vi.y());
                linear_vel_z.push_back(frame.Vi.z());
                angular_vel_x.push_back(frame.Wi.x());
                angular_vel_y.push_back(frame.Wi.y());
                angular_vel_z.push_back(frame.Wi.z());
                count++;
        }
        
        if (count < 3)
            return false;
        
        auto calculateStd = [](const std::vector<double>& data) {
            if (data.empty()) return 0.0;
            
            double sum = 0.0;
            for (double val : data) sum += val;
            double mean = sum / data.size();
            
            double variance = 0.0;
            for (double val : data) variance += (val - mean) * (val - mean);
            return std::sqrt(variance / data.size());
        };

        auto calculateMean = [](const std::vector<double>& data) {
            if (data.empty()) return 0.0;
            double sum = 0.0;
            for (double val : data) sum += val;
            return sum / data.size();
        };
        
        double std_lin_x = calculateStd(linear_vel_x);
        double std_lin_y = calculateStd(linear_vel_y);
        double std_lin_z = calculateStd(linear_vel_z);
        double std_ang_x = calculateStd(angular_vel_x);
        double std_ang_y = calculateStd(angular_vel_y);
        double std_ang_z = calculateStd(angular_vel_z);

        double mean_lin_x = calculateMean(linear_vel_x);
        double mean_lin_y = calculateMean(linear_vel_y);
        double mean_lin_z = calculateMean(linear_vel_z);
        double mean_ang_x = calculateStd(angular_vel_x);
        double mean_ang_y = calculateStd(angular_vel_y);
        double mean_ang_z = calculateStd(angular_vel_z);
        
        bool is_zero_velocity = ((mean_lin_x < ZUPT_Linear_vel) && (mean_lin_y < ZUPT_Linear_vel)) || (mean_ang_z < ZUPT_Angular_vel && mean_lin_y > 2* mean_lin_x);

        if (mean_lin_x > ZUPT_Linear_vel && mean_ang_z > ZUPT_Angular_vel)
            is_zero_velocity = false;
        return is_zero_velocity;
    }

    double normalizeYaw(double yaw) {
        while (yaw > 180.0) yaw = yaw - 360.0;
        while (yaw < -180.0) yaw = yaw + 360.0;
        return yaw;
    }

    bool pushTrackObj(int obj_global_index, int obj_index)
    {
        sensor_msgs::ObjectPerFrameObs per_obj_detect;
        per_obj_detect.header = cloudHeader;
        per_obj_detect.time = cloudHeader.stamp.toSec();
        per_obj_detect.frame_id = global_frame_count;
        per_obj_detect.seq_id = seq_ind;

        per_obj_detect.Pi = (Eigen::Vector3d(objBboxVec.boxes[obj_index].pose.position.x, objBboxVec.boxes[obj_index].pose.position.y,
                                            objBboxVec.boxes[obj_index].pose.position.z));
        per_obj_detect.Qi = Eigen::Quaterniond(objBboxVec.boxes[obj_index].pose.orientation.w, objBboxVec.boxes[obj_index].pose.orientation.x,
                                            objBboxVec.boxes[obj_index].pose.orientation.y, objBboxVec.boxes[obj_index].pose.orientation.z);
        
        per_obj_detect.Pi_obs = (Eigen::Vector3d(objBboxVec.boxes[obj_index].pose.position.x, objBboxVec.boxes[obj_index].pose.position.y,
                                                 objBboxVec.boxes[obj_index].pose.position.z));
        per_obj_detect.Qi_obs = Eigen::Quaterniond(objBboxVec.boxes[obj_index].pose.orientation.w, objBboxVec.boxes[obj_index].pose.orientation.x,
                                                   objBboxVec.boxes[obj_index].pose.orientation.y, objBboxVec.boxes[obj_index].pose.orientation.z);
        
        per_obj_detect.centroid = transformPoint2World(Eigen::Vector3d(detectObjectMsg[obj_index].average_x, detectObjectMsg[obj_index].average_y,
                                                                       objBboxVec.boxes[obj_index].pose.position.z));

        per_obj_detect.Dimensions = Eigen::Vector3d(objBboxVec.boxes[obj_index].dimensions.x, objBboxVec.boxes[obj_index].dimensions.y,
                                                    objBboxVec.boxes[obj_index].dimensions.z);

        per_obj_detect.pointcloudFull = objFullCloudVec[obj_index];
        per_obj_detect.pointcloudCorner = objCornerCloudVec[obj_index];
        per_obj_detect.pointcloudSurf = objSurfaceCloudVec[obj_index];
        per_obj_detect.feature_box.bbox3d_in_local = objBboxVec.boxes[obj_index];
        per_obj_detect.feature_box.corners_in_local = convertBox2Corners(objBboxVec.boxes[obj_index]);
        per_obj_detect.alpha = cloudInfo.objArray[cloudinfo_bbox_map[obj_index]].alpha;
        per_obj_detect.score = cloudInfo.objArray[cloudinfo_bbox_map[obj_index]].score;

        per_obj_detect.feature_box.bounding_box_2d = detectObjectMsg[obj_index].bounding_box2d;

        per_obj_detect.covariance.setZero();

        if (motion_model == 2)
        {
            for (int j = allObjectsData[obj_global_index].object_perframes.size() - 1; j >= 0; j--)
            {
                if (allObjectsData[obj_global_index].object_perframes[j].feature_box.status != -1)
                {
                    sensor_msgs::ObjectPerFrameObs obj_per_frame = allObjectsData[obj_global_index].object_perframes[j];
                    double dt = per_obj_detect.header.stamp.toSec() - obj_per_frame.header.stamp.toSec();
                    Eigen::Vector3d delta_p = per_obj_detect.Pi - obj_per_frame.Pi;

                    per_obj_detect.Ai.setZero();
                    per_obj_detect.Wi.setZero();
                    double s = 0.1 / dt;
                    Eigen::Quaterniond q_identity(1, 0, 0, 0);
                    Eigen::Vector3d p_predict = s * delta_p;

                    for (int k = 0; k < per_obj_detect.feature_box.corners_in_local.size(); ++k)
                    {
                        per_obj_detect.feature_box.corners_in_local_predict.push_back(transformPoint(per_obj_detect.feature_box.corners_in_local[k], p_predict, q_identity));
                    }

                    Eigen::Vector3d ypr = Utility::R2ypr(obj_per_frame.Qi.toRotationMatrix());
                    Eigen::Vector3d ypr_detection = Utility::R2ypr(per_obj_detect.Qi.toRotationMatrix());

                    per_obj_detect.Qi = Eigen::Quaterniond(Utility::ypr2R(Eigen::Vector3d(ypr_detection.x(), ypr_detection.y(), ypr_detection.z())));

                    per_obj_detect.Vi = delta_p / dt;
                    per_obj_detect.Vi_obs = delta_p / dt;

                    per_obj_detect.Qi_obs = per_obj_detect.Qi;

                    break;
                }
            }
        }

        if (motion_model == 1)
        {
            for (int j = allObjectsData[obj_global_index].object_perframes.size() - 1; j >= 0; j--)
            {
                if (allObjectsData[obj_global_index].object_perframes[j].feature_box.status != -1)
                {
                    sensor_msgs::ObjectPerFrameObs obj_per_frame = allObjectsData[obj_global_index].object_perframes[j];
                    double dt = per_obj_detect.header.stamp.toSec() - obj_per_frame.header.stamp.toSec();
                    Eigen::Vector3d delta_p = per_obj_detect.Pi - obj_per_frame.Pi;

                    per_obj_detect.Ai.setZero();
                    per_obj_detect.Wi.setZero();
                    double s = 0.1 / dt;
                    Eigen::Quaterniond q_identity(1, 0, 0, 0);
                    Eigen::Vector3d p_predict = s * delta_p;

                    for (int k = 0; k < per_obj_detect.feature_box.corners_in_local.size(); ++k)
                    {
                        per_obj_detect.feature_box.corners_in_local_predict.push_back(transformPoint(per_obj_detect.feature_box.corners_in_local[k], p_predict, q_identity));
                    }

                    Eigen::Vector3d ypr = Utility::R2ypr(obj_per_frame.Qi.toRotationMatrix());
                    Eigen::Vector3d ypr_detection = Utility::R2ypr(per_obj_detect.Qi.toRotationMatrix());

                    per_obj_detect.Qi = Eigen::Quaterniond(Utility::ypr2R(Eigen::Vector3d(ypr_detection.x(), ypr_detection.y(), ypr_detection.z())));

                    per_obj_detect.Vi = per_obj_detect.Qi.conjugate() * (delta_p / dt);
                    per_obj_detect.Vi_obs = per_obj_detect.Qi.conjugate() * (delta_p / dt);
                    per_obj_detect.Qi_obs = per_obj_detect.Qi;

                    break;
                }
            }
        }
        
        
        allObjectsData[obj_global_index].object_perframes.back() = per_obj_detect;

        correctYawWithMotionDirection(allObjectsData[obj_global_index]);

        if (allObjectsData[obj_global_index].object_perframes.size() < 3)
            return false;

        allObjectsData[obj_global_index].object_perframes.back().isStatic = detectZeroVelocity(obj_global_index);

        int count = 0;
        for (int j = allObjectsData[obj_global_index].object_perframes.size() - 1; j >= 0; j--)
        {
            if (allObjectsData[obj_global_index].object_perframes[j].feature_box.status != -1)
            {
                count++;
            }
        }
        if (count >= birthmin)
        {
            for (int j = allObjectsData[obj_global_index].object_perframes.size() - 1; j >= 0; j--)
            {
                if (allObjectsData[obj_global_index].object_perframes[j].feature_box.status != -1)
                {
                    sensor_msgs::ObjectPerFrameObs obj_per_frame = allObjectsData[obj_global_index].object_perframes[j];
                    Eigen::Vector3d pry = Utility::R2ypr(obj_per_frame.Qi.toRotationMatrix());
                }
            }
        }

        if (count >= birthmin)
        {
            for (int j = 0; j < allObjectsData[obj_global_index].object_perframes.size(); j++)
            {
                if (allObjectsData[obj_global_index].object_perframes[j].feature_box.status != -1)
                {
                   
                    sensor_msgs::ObjectPerFrameObs obj_per_frame = allObjectsData[obj_global_index].object_perframes[j];
                    Eigen::Vector3d pry                          = Utility::R2ypr(obj_per_frame.Qi.toRotationMatrix());
                    std::vector<double> line_values{obj_per_frame.feature_box.bbox3d_in_local.dimensions.x, obj_per_frame.feature_box.bbox3d_in_local.dimensions.y,
                                                    obj_per_frame.feature_box.bbox3d_in_local.dimensions.z, obj_per_frame.Pi.x(), obj_per_frame.Pi.y(), obj_per_frame.Pi.z(),
                                                    pry.x(), pry.y(), pry.z(), obj_per_frame.feature_box.bbox3d_in_local.value, obj_per_frame.alpha};

                    if (fabs(obj_per_frame.feature_box.bounding_box_2d.score - obj_per_frame.feature_box.bbox3d_in_local.value) > 0.1)
                    {}
                    else
                    {
                        line_values.push_back(obj_per_frame.feature_box.bounding_box_2d.xmin);
                        line_values.push_back(obj_per_frame.feature_box.bounding_box_2d.ymin);
                        line_values.push_back(obj_per_frame.feature_box.bounding_box_2d.xmax);
                        line_values.push_back(obj_per_frame.feature_box.bounding_box_2d.ymax);
                        line_values.push_back(obj_per_frame.feature_box.bounding_box_2d.score);
                    }

                    if (frame_last_dicts.find(obj_per_frame.frame_id) != frame_last_dicts.end())
                    {
                        frame_last_dicts[obj_per_frame.frame_id][allObjectsData[obj_global_index].id] = line_values;
                    }
                    else
                    {
                        frame_last_dicts[obj_per_frame.frame_id] = {{allObjectsData[obj_global_index].id, line_values}};
                    }

                }
                else if (output_predict)
                {
                    sensor_msgs::ObjectPerFrameObs obj_per_frame = allObjectsData[obj_global_index].object_perframes[j];
                    Eigen::Vector3d pry                          = Utility::R2ypr(obj_per_frame.Qi.toRotationMatrix());
                    std::vector<double> line_values{obj_per_frame.feature_box.bbox3d_in_local.dimensions.x, obj_per_frame.feature_box.bbox3d_in_local.dimensions.y,
                                                    obj_per_frame.feature_box.bbox3d_in_local.dimensions.z, obj_per_frame.Pi.x(), obj_per_frame.Pi.y(), obj_per_frame.Pi.z(),
                                                    pry.x(), pry.y(), pry.z(), obj_per_frame.feature_box.bbox3d_in_local.value, obj_per_frame.alpha};

                    if (frame_last_dicts.find(obj_per_frame.frame_id) != frame_last_dicts.end())
                    {
                        frame_last_dicts[obj_per_frame.frame_id][allObjectsData[obj_global_index].id] = line_values;
                    }
                    else
                    {
                        frame_last_dicts[obj_per_frame.frame_id] = {{allObjectsData[obj_global_index].id, line_values}};
                    }
                }
                else
                {
                }
            }
        }

        return true;
    }

    void getPredictionAndCov(sensor_msgs::ObjectPerFrameObs& obj_per_frame)
    {
        static int marker_id = 0;
        Eigen::Vector3d Pj = obj_per_frame.Pi;
        Eigen::Vector3d Pj_cov = obj_per_frame.Pi_cov;
        Eigen::Matrix3d covariance;

        covariance << Pj_cov.x()*Pj_cov.x(), 0.0, 0.0, 
                    0.0, Pj_cov.y()*Pj_cov.y(), 0.0,    
                    0.0, 0.0, Pj_cov.z()*Pj_cov.z();    

        visualization_msgs::Marker pos_cov_mark;

        pos_cov_mark.header.frame_id = "map";
        pos_cov_mark.header.stamp = cloudHeader.stamp;
        pos_cov_mark.ns = "covariance_markers";
        pos_cov_mark.id = marker_id++;
        pos_cov_mark.lifetime = ros::Duration(1);
        pos_cov_mark.frame_locked = false;
        pos_cov_mark.type = visualization_msgs::Marker::SPHERE;
        pos_cov_mark.action = visualization_msgs::Marker::ADD;
        
        pos_cov_mark.pose.position.x = Pj.x();
        pos_cov_mark.pose.position.y = Pj.y();
        pos_cov_mark.pose.position.z = Pj.z();
    
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance);
        Eigen::Vector3d eigenvalues = eigen_solver.eigenvalues();
        Eigen::Matrix3d eigenvectors = eigen_solver.eigenvectors();
        
        pos_cov_mark.pose.orientation.x = obj_per_frame.Qi.x();
        pos_cov_mark.pose.orientation.y = obj_per_frame.Qi.y();
        pos_cov_mark.pose.orientation.z = obj_per_frame.Qi.z();
        pos_cov_mark.pose.orientation.w = obj_per_frame.Qi.w();
        
        double confidence_scale = 2.0;
        pos_cov_mark.scale.x = 2.0 * confidence_scale * Pj_cov.x();
        pos_cov_mark.scale.y = 2.0 * confidence_scale * Pj_cov.y(); 
        pos_cov_mark.scale.z = 2.0 * confidence_scale * Pj_cov.z();
        
        pos_cov_mark.color.r = 1.0f;
        pos_cov_mark.color.g = 0.0f;
        pos_cov_mark.color.b = 0.0f;
        pos_cov_mark.color.a = 0.3f;

        objPosCovVecPred.markers.push_back(pos_cov_mark);
    }

    map<int, int> performDataAssociationWithPrediction()
    {
        objBboxVecPrev.boxes.clear();
        objBboxVecPrev.header = cloudHeader;
        objBboxVecPrev.header.frame_id = "map";

        std::vector<sensor_msgs::obj_box_msg> pre_track_boxs;
        std::vector<sensor_msgs::obj_box_msg> track_boxs;
        std::vector<double> predict_conf_vec;

        map<int, int> matchedPairs;
        map<int, int> track_ind_map;
        int all_obj_ind = 0;

        objBboxVecPred.header = cloudHeader;
        objBboxVecPred.header.frame_id = "map";
        objBboxVecPred.boxes.clear();
        objPosCovVecPred.markers.clear();

        for (size_t i = 0; i < allObjectsData.size(); i++)
        {
            if (allObjectsData[i].isFailTracking)
            {
                continue;
            }
            sensor_msgs::ObjectPerFrameObs obj_per_frame = allObjectsData[i].object_perframes.back();
            sensor_msgs::obj_box_msg box_tmp;
            box_tmp.bbox3d_in_local = obj_per_frame.feature_box.bbox3d_in_local;
            Eigen::Vector3d Pj = obj_per_frame.Pi;

            Eigen::Quaterniond Qj = obj_per_frame.Qi;
            jsk_recognition_msgs::BoundingBox box_tmp2;
            box_tmp2.header = cloudHeader;
            box_tmp2.header.frame_id = "map";
            box_tmp2.dimensions = obj_per_frame.feature_box.bbox3d_in_local.dimensions;
            box_tmp2.pose.position.x = Pj.x();
            box_tmp2.pose.position.y = Pj.y();
            box_tmp2.pose.position.z = Pj.z();
            box_tmp2.pose.orientation.x = Qj.x();
            box_tmp2.pose.orientation.y = Qj.y();
            box_tmp2.pose.orientation.z = Qj.z();
            box_tmp2.pose.orientation.w = Qj.w();
            box_tmp.corners_in_local = convertBox2Corners(box_tmp2);

            objBboxVecPred.boxes.push_back(box_tmp2);
            
            getPredictionAndCov(obj_per_frame);
            
            pre_track_boxs.push_back(box_tmp);

            box_tmp.bbox3d_in_local.label = 0;
            box_tmp2.label = 0;
            objBboxVecPrev.boxes.push_back(box_tmp2);
            
            double dt = 0.1;
            Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(0, 6) = -obj_per_frame.Qi.toRotationMatrix() * Utility::skewSymmetric(obj_per_frame.Ai) * dt;
            F.block<3, 3>(0, 9) = obj_per_frame.Qi.toRotationMatrix() * dt;
            F.block<3, 3>(3, 6) = -obj_per_frame.Qi.toRotationMatrix() * Utility::skewSymmetric(obj_per_frame.Ai);
            F.block<3, 3>(3, 9) = obj_per_frame.Qi.toRotationMatrix();

            F.block<3, 3>(6, 6) = -Utility::skewSymmetric(obj_per_frame.Wi);
            F.block<3, 3>(6, 12) = Eigen::Matrix3d::Identity();

            Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15, 6);
            V.block<3, 3>(9, 0) = Eigen::Matrix3d::Identity();
            V.block<3, 3>(12, 3) = Eigen::Matrix3d::Identity();

            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(15, 15);
            Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(6, 6);
            noise.block<3, 3>(0, 0) = (0.1 * 0.1) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(3, 3) = (0.01 * 0.01) * Eigen::Matrix3d::Identity();

            Eigen::Matrix<double, 15, 15> cov_pred = (I + F * dt) * obj_per_frame.covariance * (I + F * dt).transpose() + (V * dt) * noise * (V * dt).transpose();

            double conf = sqrt(cov_pred(0, 0) + cov_pred(1, 1) + cov_pred(2, 2));
            conf = 1.0;
            predict_conf_vec.push_back(conf);

            track_ind_map[all_obj_ind] = i;
            all_obj_ind++;
        }

        for (int i = 0; i < objBboxVec.boxes.size(); ++i)
        {
            sensor_msgs::obj_box_msg box_tmp;
            box_tmp.bbox3d_in_local = objBboxVec.boxes[i];
            box_tmp.corners_in_local = convertBox2Corners(box_tmp.bbox3d_in_local);

            track_boxs.push_back(box_tmp);
        }

        const size_t trkNum = pre_track_boxs.size();
        const size_t detNum = track_boxs.size();
        std::vector<int> assignment(trkNum, -1);

        AssignmentProblemSolver hungAlgo;
        std::vector<double> costMatrix(trkNum * detNum);
        hungAlgo.createDistaceMatrixByGIoU(pre_track_boxs, track_boxs, costMatrix);
        hungAlgo.solve(costMatrix, trkNum, detNum, assignment);

        set<int> unmatchedDetections;
        set<int> unmatchedTrajectories;
        set<int> allItems;
        set<int> matchedItems;
        unmatchedTrajectories.clear();
        unmatchedDetections.clear();
        allItems.clear();
        matchedItems.clear();

        if (detNum > trkNum)
        {
            for (unsigned int n = 0; n < detNum; n++)
                allItems.insert(n);
            for (unsigned int i = 0; i < trkNum; ++i)
                matchedItems.insert(assignment[i]);
            set_difference(allItems.begin(), allItems.end(),
                           matchedItems.begin(), matchedItems.end(),
                           insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        }
        else
        {
            for (unsigned int i = 0; i < trkNum; ++i)
                if (assignment[i] == -1)                {
                    unmatchedTrajectories.insert(i);
                }
        }

        for (unsigned int i = 0; i < trkNum; ++i)
        {
            if (assignment[i] == -1)
            {
                unmatchedDetections.insert(assignment[i]);
                continue;
            }
            if (costMatrix[i + assignment[i] * trkNum] > max_costmatrixvalue)
            {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            }
            else
                matchedPairs[track_ind_map[i]] = assignment[i];
        }

        return matchedPairs;
    }

    void predictAllTracks()
    {
        for (size_t i = 0; i < allObjectsData.size(); ++i)
        {
            if (allObjectsData[i].isFailTracking)
            {
                continue;
            }
            sensor_msgs::ObjectPerFrameObs per_object_tmp;
            per_object_tmp.header = cloudHeader;
            per_object_tmp.time = cloudHeader.stamp.toSec();
            per_object_tmp.frame_id = global_frame_count;
            per_object_tmp.seq_id = seq_ind;
            per_object_tmp.feature_box.status = -1;

            Eigen::Vector3d average_velo = Eigen::Vector3d(0, 0, 0);
            Eigen::Vector3d average_acc = Eigen::Vector3d(0, 0, 0);
            int valid_frame_cnt = 0;
            sensor_msgs::ObjectPerFrameObs obj_per_frame;
            for (int j = 0; j < allObjectsData[i].object_perframes.size(); ++j)
            {
                sensor_msgs::ObjectPerFrameObs obj_per_frame_tmp = allObjectsData[i].object_perframes[j];
                if (obj_per_frame_tmp.feature_box.status != -1)
                {
                    average_velo += allObjectsData[i].object_perframes[j].Vi;
                    average_acc += allObjectsData[i].object_perframes[j].Ai;
                    valid_frame_cnt++;
                    obj_per_frame = obj_per_frame_tmp;
                }
            }
            obj_per_frame.Ai = average_acc / (valid_frame_cnt * 1.0);
            obj_per_frame.Vi = average_velo / (valid_frame_cnt * 1.0);
            Eigen::Matrix3d R_pre = obj_per_frame.Qi.toRotationMatrix();
            double dt = cloudHeader.stamp.toSec() - obj_per_frame.header.stamp.toSec();
            
            if (dt <= 0) 
            {
                dt = 0.1;
            }
            
            if (motion_model == 2)
            {
                per_object_tmp.Pi = obj_per_frame.Pi + obj_per_frame.Vi * dt + R_pre * (obj_per_frame.Ai * (0.5 * dt * dt));
                per_object_tmp.Vi = obj_per_frame.Vi + R_pre * obj_per_frame.Ai * dt;
            }
            else
            {
                per_object_tmp.Pi = obj_per_frame.Pi +  R_pre * (obj_per_frame.Vi * dt);
                per_object_tmp.Vi = obj_per_frame.Vi;
            }
            
            if ((obj_per_frame.Wi * dt).norm() < 0.001) 
                per_object_tmp.Qi = obj_per_frame.Qi * Utility::deltaQ(obj_per_frame.Wi * dt);
            else
                per_object_tmp.Qi = obj_per_frame.Qi;
            

            per_object_tmp.Dimensions = obj_per_frame.Dimensions;
            per_object_tmp.feature_box.bbox3d_in_local.dimensions = obj_per_frame.feature_box.bbox3d_in_local.dimensions;
            
            allObjectsData[i].object_perframes.push_back(per_object_tmp);
        }
    }

    void objectTracking()
    {
        if (isFirstFrame)
        {
            allObjectsData.clear();
            
            for (int obj_index = 0; obj_index < obj_num_count_curr; ++obj_index) {
                generateNewObj(obj_index);
            }
            isFirstFrame = false;
            return;
        }

        predictAllTracks();

        std::vector<bool> isObjectExit(obj_num_count_curr, false);
        map<int, int> matchedPairs = performDataAssociationWithPrediction();

        int hit_num = 0;
        for (map<int, int>::iterator per_match = matchedPairs.begin(); per_match != matchedPairs.end(); per_match++)
        {
            hit_num++;
            isObjectExit[per_match->second] = true;
        }

        for (size_t i = 0; i < allObjectsData.size(); ++i)
        {
            if (allObjectsData[i].isFailTracking)
            {
                continue;
            }
            map<int, int>::iterator iter = matchedPairs.find(i);
            if (iter == matchedPairs.end())
            {
                allObjectsData[i].loss_box_num++;
                if (allObjectsData[i].loss_box_num > maxBoxLossNum)
                {
                    allObjectsData[i].isFailTracking = true;
                }
                else
                {
                    continue;
                }
                
            }
            
            allObjectsData[i].track_box_num++;
            pushTrackObj(i, matchedPairs[i]);

            allObjectsData[i].loss_box_num = 0;
        }

        for (size_t i = 0; i < obj_num_count_curr; i++)
        {
            if (isObjectExit[i])
                continue;
            generateNewObj(i);
        }
    }

    double get_registration_angle(const Eigen::MatrixXd &mat)
    {
        double cos_theta = mat(0, 0);
        double sin_theta = mat(1, 0);
        cos_theta        = std::max(-1.0, std::min(1.0, cos_theta));
        double theta_cos = std::acos(cos_theta);
        if (sin_theta >= 0)
        return theta_cos;
        else
        return 2 * M_PI - theta_cos;
    }

    void save_tracking_result()
    {
        for (auto i = frame_last_dicts.begin(); i != frame_last_dicts.end(); i++)
        {
            int frame_index                                 = i->first;
            const std::map<int, std::vector<double>> &track = i->second;
            for (auto j = track.begin(); j != track.end(); j++)
            {
                int object_id                     = j->first;
                std::vector<double> track_data = j->second;
                Eigen::Matrix4d ego_pose       = Ego_pose[frame_index];
                Eigen::Vector4d twj;
                twj << track_data[3], track_data[4], track_data[5], 1.0;
                Eigen::Vector3d dimj(track_data[0], track_data[1], track_data[2]);
                Eigen::Vector3d ypr(track_data[6] * M_PI / 180.0, track_data[7] * M_PI / 180.0, track_data[8] * M_PI / 180.0);
                double ang          = get_registration_angle(ego_pose.inverse());
                Eigen::Vector4d tlj = ego_pose.inverse() * twj;
                double r_y          = -ypr(0) - ang - M_PI * 0.5;
                tlj(2) -= 0.5 * dimj(2);
                Eigen::Vector4d tcj = Tr_velo_cam * tlj;
                Eigen::VectorXd x_corners(8), y_corners(8), z_corners(8);
                double l = dimj(0);
                double w = dimj(1);
                double h = dimj(2);
                x_corners << l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2;
                y_corners << 0, 0, 0, 0, -h, -h, -h, -h;
                z_corners << w / 2, -w / 2, w / 2, -w / 2, w / 2, -w / 2, w / 2, -w / 2;
                Eigen::Matrix4d transpose;
                transpose << cos(M_PI - r_y), 0, -sin(M_PI - r_y), tcj(0),
                    0., 1., 0., tcj(1),
                    sin(M_PI - r_y), 0., cos(M_PI - r_y), tcj(2),
                    0., 0., 0., 1.;
                double x1 = std::numeric_limits<double>::max();
                double y1 = std::numeric_limits<double>::max();
                double x2 = std::numeric_limits<double>::min();
                double y2 = std::numeric_limits<double>::min();
                for (int i = 0; i < 8; ++i)
                {
                    Eigen::Vector4d point(x_corners[i], y_corners[i], z_corners[i], 1.0);
                    point                   = transpose * point;
                    Eigen::Vector4d img_pts = P2 * point;
                    img_pts /= img_pts(2);
                    x1 = std::min(x1, img_pts(0));
                    y1 = std::min(y1, img_pts(1));
                    x2 = std::max(x2, img_pts(0));
                    y2 = std::max(y2, img_pts(1));

                    x1 = x1 > 0 ? x1 : 0;
                    x2 = x2 < 1242 ? x2 : 1242;
                    y1 = y2 > 0 ? y1 : 0;
                    y2 = y2 < 375 ? y2 : 375;
                }

                double score = track_data[9];
                obj_tracking_opt_kitti_test << std::fixed << std::setprecision(0) << frame_index << " " << object_id << " "
                                    << "Car 0 0 ";
                obj_tracking_opt_kitti_test << std::setprecision(4) << track_data[10] << " " << x1 << " " << y1 << " " << x2 << " " << y2 << " "
                                    << h << " " << w << " " << l << " " << tcj(0) << " " << tcj(1) << " "
                                    << tcj(2) << " " << r_y << " " << score << " " 
                                    << track_data[11] << " " << track_data[12] << " " << track_data[13] << " " 
                                    << track_data[14] << " " << track_data[15] << " "
                                    << track_data[11] - x1 << " " <<  track_data[12] - y1 << " " 
                                    << track_data[13] - x2 << " " << track_data[14] - y2 << " " <<  track_data[15] - score << "\n";
                                    
                if (track_data.size() > 12)
                {
                    x1 = track_data[11];
                    y1 = track_data[12];
                    x2 = track_data[13];
                    y2 = track_data[14];
                }
                else
                {}

                if (x2- x1 > 300 && y2- y1 > 1000)
                {
                    continue;
                }
                else
                {
                    double score = track_data[9];
                    obj_tracking_opt_kitti << std::fixed << std::setprecision(0) << frame_index << " " << object_id << " "
                                        << "Car 0 0 ";
                    obj_tracking_opt_kitti << std::setprecision(4) << track_data[10] << " " << x1 << " " << y1 << " " << x2 << " " << y2 << " "
                                        << h << " " << w << " " << l << " " << tcj(0) << " " << tcj(1) << " "
                                        << tcj(2) << " " << r_y << " " << score << "\n";
                }
                
            }
        }
        obj_tracking_opt_kitti.close();
        obj_tracking_opt_kitti_test.close();
    }

    void laserCloudInfoHandler(const fgo_mot::cloud_infoConstPtr &msgIn)
    {
        cloudInfo = *msgIn;
        cloudHeader = cloudInfo.header;
        timeNewCloudInfo = cloudInfo.header.stamp.toSec();
        seq_ind = cloudInfo.seq_id;

        if (cloudInfo.frame_id < last_frame_id)
        {
            save_tracking_result();
            isFirstFrame = true;
            global_frame_count = 0;
            Ego_pose.clear();
            frame_last_dicts.clear();
            obj_id = 0;

            objOptbuffer.clear();
            allObjectsData.clear();

            std::stringstream ss;
            ss << setw(4) << setfill('0') << seq_ind;
            ss >> sequence_num;
            
            obj_tracking_opt_kitti.open(output_path + "/" +"tracking_kitti/"+ sequence_num + ".txt", ios::out | ios::trunc);
            obj_tracking_opt_kitti_test.open(output_path + "/" +"tracking_kitti_test/"+ sequence_num + ".txt", ios::out | ios::trunc);
            
            std::string calib_file = base_path + "/calib/" + sequence_num + ".txt";
            read_calib_data(calib_file);
        }
        last_frame_id = cloudInfo.frame_id;

        std::cout << "======================seq id: " << cloudInfo.seq_id << " frame_id: " << global_frame_count << "  ==================" << std::endl;
        
        pos_body.x() = cloudInfo.pose_lidar_odom.pose.pose.position.x;
        pos_body.y() = cloudInfo.pose_lidar_odom.pose.pose.position.y;
        pos_body.z() = cloudInfo.pose_lidar_odom.pose.pose.position.z;
        quater_body.w() = cloudInfo.pose_lidar_odom.pose.pose.orientation.w;
        quater_body.x() = cloudInfo.pose_lidar_odom.pose.pose.orientation.x;
        quater_body.y() = cloudInfo.pose_lidar_odom.pose.pose.orientation.y;
        quater_body.z() = cloudInfo.pose_lidar_odom.pose.pose.orientation.z;
        quater_body = quater_body.normalized();

        Eigen::Matrix<double, 4, 4> lidar_pose;
        lidar_pose.setIdentity();
        lidar_pose.block<3, 3>(0, 0) = quater_body.normalized().toRotationMatrix();
        lidar_pose.block<3, 1>(0, 3) = pos_body;
        Ego_pose[cloudInfo.frame_id] = lidar_pose;

        pcl::fromROSMsg(cloudInfo.cloud_feature_origin, *laserCloudFeatureOrigin);

        geometry_msgs::PoseStamped poseStamped;
        poseStamped.header = cloudInfo.pose_lidar_odom.header;
        poseStamped.pose = cloudInfo.pose_lidar_odom.pose.pose;
        poseStamped.header.stamp = cloudInfo.pose_lidar_odom.header.stamp;
        path.header.stamp = cloudInfo.pose_lidar_odom.header.stamp;
        if (isFirstFrame)
            path.poses.clear();
        path.poses.push_back(poseStamped);
        path.header.frame_id = "map";
        pubPath.publish(path);

        static tf::TransformBroadcaster tfMap2Odom;
        tf::Quaternion quat;
        tf::quaternionMsgToTF(cloudInfo.pose_lidar_odom.pose.pose.orientation, quat);
        tf::Transform map_to_odom = tf::Transform(quat, tf::Vector3(cloudInfo.pose_lidar_odom.pose.pose.position.x, cloudInfo.pose_lidar_odom.pose.pose.position.y, cloudInfo.pose_lidar_odom.pose.pose.position.z));
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, cloudInfo.pose_lidar_odom.header.stamp, "map", "fgo_mot"));

        pubCloudinCamera.publish(cloudInfo.Cloud_in_Camera);

        if (pubCar.getNumSubscribers() != 0)
        {
            visualization_msgs::Marker visiual_ego_car;
            visiual_ego_car.header = cloudInfo.pose_lidar_odom.header;
            visiual_ego_car.header.frame_id = "map";
            visiual_ego_car.id = -1;
            visiual_ego_car.type = visualization_msgs::Marker::MESH_RESOURCE;
            visiual_ego_car.action = visualization_msgs::Marker::ADD;
            visiual_ego_car.lifetime = ros::Duration();
            visiual_ego_car.mesh_resource = "package://fgo_mot/config/ego-Car.dae";

            visiual_ego_car.pose.position.x = cloudInfo.pose_lidar_odom.pose.pose.position.x;
            visiual_ego_car.pose.position.y = cloudInfo.pose_lidar_odom.pose.pose.position.y;
            visiual_ego_car.pose.position.z = cloudInfo.pose_lidar_odom.pose.pose.position.z - 1.73;

            tf::Quaternion quat;
            tf::quaternionMsgToTF(cloudInfo.pose_lidar_odom.pose.pose.orientation, quat);
            double roll, pitch, yaw;
            tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
            quat.setRPY(roll, pitch, yaw + 0.5 * M_PI);

            visiual_ego_car.pose.orientation.w = quat.w();
            visiual_ego_car.pose.orientation.x = quat.x();
            visiual_ego_car.pose.orientation.y = quat.y();
            visiual_ego_car.pose.orientation.z = quat.z();
            visiual_ego_car.color.r = 0;
            visiual_ego_car.color.g = 0;
            visiual_ego_car.color.b = 1;
            visiual_ego_car.color.a = 1.0;

            visiual_ego_car.scale.x = 1.0;
            visiual_ego_car.scale.y = 1.0;
            visiual_ego_car.scale.z = 1.0;

            pubCar.publish(visiual_ego_car);
        }
        objCloudInfoHandle();

        if (obj_num_count_curr == 0  || obj_num_count_curr == 0)
        {
            global_frame_count++;
            objBboxVecPrev = objBboxVec;
            return;
        }

        Timer dataAssociation_t("dataAssociation");
        dataAssociation_t.tic();
        objectTracking();
        dataAssociation_timer.push_back(dataAssociation_t.toc());

        Timer optimization_t("dataAssociation");
        optimization_t.tic();
        updateObjectInfo();
        publishTrackingResult();

        objBboxVecPrev = objBboxVec;
        global_frame_count++;
        optimization_timer.push_back(optimization_t.toc());
    }

    int findObjBufInd(int _index)
    {
        int isfind = -1;
        for (int i = 0; i < objOptbuffer.size(); i++)
        {
            if (objOptbuffer[i]->obj_id == _index)
            {
                isfind = i;
                return isfind;
            }
        }
        return isfind;
    }

    double directionToYaw(const Eigen::Vector3d& direction) {
        return std::atan2(direction.y(), direction.x()) * 180.0 / M_PI;
    }

    Eigen::Vector3d calculateMotionDirectionRobust(const std::vector<sensor_msgs::ObjectPerFrameObs>& poses, int frame_idx) {
        int total_frames = poses.size();
        
        if (total_frames == 1) {
            return Eigen::Vector3d::Zero();
        }
        else if (frame_idx == 0) {
            Eigen::Vector3d motion = poses[1].Pi - poses[0].Pi;
            return (motion.norm() > 0.05) ? motion : Eigen::Vector3d::Zero();
        }
        else if (frame_idx == total_frames - 1) {
            Eigen::Vector3d motion = poses[frame_idx].Pi - poses[frame_idx - 1].Pi;
            return (motion.norm() > 0.05) ? motion : Eigen::Vector3d::Zero();
        }
        else {
            Eigen::Vector3d forward_motion = poses[frame_idx + 1].Pi - poses[frame_idx].Pi;
            Eigen::Vector3d backward_motion = poses[frame_idx].Pi - poses[frame_idx - 1].Pi;
            
            double forward_weight = (forward_motion.norm() > 0.05) ? 0.4 : 0.0;
            double backward_weight = (backward_motion.norm() > 0.05) ? 0.6 : 0.0;
            
            if (forward_weight + backward_weight < 0.1) {
                return Eigen::Vector3d::Zero();
            }
            
            return (backward_motion * backward_weight + forward_motion * forward_weight) / (forward_weight + backward_weight);
        }
    }

    bool applyYawCorrection(std::vector<double>& yaw_angles, int frame_idx, const Eigen::Vector3d& motion_direction) {
        double optimal_yaw = directionToYaw(motion_direction);
        double current_yaw = yaw_angles[frame_idx];
        
        double diff_normal = normalizeYaw(current_yaw - optimal_yaw);
        double diff_reversed = normalizeYaw((current_yaw + 180.0) - optimal_yaw);
        
        if (std::abs(diff_reversed) < std::abs(diff_normal) && std::abs(diff_reversed) < 80.0) {
            yaw_angles[frame_idx] = normalizeYaw(current_yaw + 180.0);
            return true;
        }
        return false;
    }

    void handleLowMotionFrames(const std::vector<sensor_msgs::ObjectPerFrameObs>& poses, 
                            std::vector<double>& yaw_angles, 
                            int frame_idx) {
        
        double first_yaw = yaw_angles[0];
        double flip_ratio = 0.0;
        
        int flip_count = 0;
        int total_count = 0;
        
        for (int i = 0; i < yaw_angles.size(); i++) {
            double diff = normalizeYaw(yaw_angles[i] - first_yaw);
            if (std::abs(diff - 180.0) < 90.0) {
                flip_count++;
            }
            total_count++;
        }
        
        if (total_count > 0) {
            flip_ratio = (double)flip_count / total_count;
        }
        else
        {
            return;
        }
        
        double current_yaw = yaw_angles[frame_idx];
        double diff = normalizeYaw(current_yaw - first_yaw);
        
        if (flip_ratio > 0.5 && frame_idx == 0)
        {
            yaw_angles[frame_idx] = normalizeYaw(current_yaw + 180.0);
        }
        else if (flip_ratio > 0.5 && std::abs(diff - 180.0) < 90.0) {
            yaw_angles[frame_idx] = normalizeYaw(current_yaw + 180.0);
        } 
        else{}
    }

    void correctYawWithMotionDirection(sensor_msgs::ObjectTrack &_obj) {
        if (_obj.object_perframes.size() < 5) {
            return;
        }
        
        std::vector<double> yaw_angles;
        std::vector<int> valid_indices;
        
        for (int i = 0; i < _obj.object_perframes.size(); i++) {
            Eigen::Vector3d ypr = Utility::R2ypr(_obj.object_perframes[i].Qi.normalized().toRotationMatrix());
            yaw_angles.push_back(ypr.x());
            valid_indices.push_back(i);
        }
        
        std::vector<double> corrected_yaw = yaw_angles;
        int correction_count = 0;
        
        for (int i = 0; i < _obj.object_perframes.size(); i++) {
            Eigen::Vector3d motion_direction = calculateMotionDirectionRobust(_obj.object_perframes, i);
            
            if (motion_direction.norm() > 1.0)
            {
                if (applyYawCorrection(corrected_yaw, i, motion_direction)) 
                {
                    correction_count++;
                }
            } 
            else 
            {
                handleLowMotionFrames(_obj.object_perframes, corrected_yaw, i);
            }
        }
        updateYawAndAngularVelocity(_obj, valid_indices, corrected_yaw);
    }

    void updateYawAndAngularVelocity(sensor_msgs::ObjectTrack &_obj, 
                                    const std::vector<int> &valid_indices,
                                    const std::vector<double> &corrected_yaw) {
        for (int i = 0; i < valid_indices.size(); i++) {
            int frame_idx = valid_indices[i];
            double yaw_deg = corrected_yaw[i];

            double normalized_yaw = normalizeYaw(yaw_deg);
            tf2::Quaternion orientation;
            orientation.setRPY(0.0, 0.0, normalized_yaw * M_PI / 180.0);

            if (!_obj.isOpt)
            {

                _obj.object_perframes[frame_idx].Qi_obs.w() = orientation.w();
                _obj.object_perframes[frame_idx].Qi_obs.x() = orientation.x();
                _obj.object_perframes[frame_idx].Qi_obs.y() = orientation.y();
                _obj.object_perframes[frame_idx].Qi_obs.z() = orientation.z();
            }
            else
            {
                if (frame_idx == _obj.object_perframes.size()-1)
                {

                    _obj.object_perframes[frame_idx].Qi_obs.w() = orientation.w();
                    _obj.object_perframes[frame_idx].Qi_obs.x() = orientation.x();
                    _obj.object_perframes[frame_idx].Qi_obs.y() = orientation.y();
                    _obj.object_perframes[frame_idx].Qi_obs.z() = orientation.z();
                }
            }

            if (i > 0) {
                int prev_frame_idx = valid_indices[i-1];
                double dt = 0.1 * (frame_idx - prev_frame_idx);
                double delta_yaw = normalizeYaw(corrected_yaw[i] - corrected_yaw[i-1]);
                
                _obj.object_perframes[frame_idx].Wi.z() = delta_yaw * M_PI / 180.0 / dt;
            }
        }
    }

    void obj_vector2double(objectOptimization *_objOpt, sensor_msgs::ObjectTrack &_obj)
    {
        int global_pose_num = _objOpt->obj_double_map.size();
        for (int i = 0; i < _obj.object_perframes.size(); i++)
        {
            pair<int, int> obj_ind = make_pair(_obj.id, _obj.object_perframes[i].frame_id);
            int pos_ind = _objOpt->findPoseInd(obj_ind);

            Eigen::Vector3d ypr = Utility::R2ypr(_obj.object_perframes[i].Qi.toRotationMatrix());
            tempdimension[0] = _obj.object_perframes[i].Dimensions[0];
            tempdimension[1] = _obj.object_perframes[i].Dimensions[1];
            tempdimension[2] = _obj.object_perframes[i].Dimensions[2];

            if (pos_ind == -1)
            {
                _objOpt->obj_double_map[obj_ind] = global_pose_num;
                _objOpt->isStatic[global_pose_num] = _obj.object_perframes[i].isStatic;

                _objOpt->para_objPose[global_pose_num][0] = _obj.object_perframes[i].Pi.x();
                _objOpt->para_objPose[global_pose_num][1] = _obj.object_perframes[i].Pi.y();
                _objOpt->para_objPose[global_pose_num][2] = _obj.object_perframes[i].Pi.z();

                _objOpt->para_objLinearVel[global_pose_num][0] = _obj.object_perframes[i].Vi.x();
                _objOpt->para_objLinearVel[global_pose_num][1] = _obj.object_perframes[i].Vi.y();
                _objOpt->para_objLinearVel[global_pose_num][2] = _obj.object_perframes[i].Vi.z();

                _objOpt->para_objAngularVel[global_pose_num][0] = _obj.object_perframes[i].Wi.x();
                _objOpt->para_objAngularVel[global_pose_num][1] = _obj.object_perframes[i].Wi.y();
                _objOpt->para_objAngularVel[global_pose_num][2] = _obj.object_perframes[i].Wi.z();

                _objOpt->para_objAcc[global_pose_num][0] = _obj.object_perframes[i].Ai.x();
                _objOpt->para_objAcc[global_pose_num][1] = _obj.object_perframes[i].Ai.y();
                _objOpt->para_objAcc[global_pose_num][2] = _obj.object_perframes[i].Ai.z();

                Eigen::Quaterniond q{_obj.object_perframes[i].Qi.normalized()};
                _objOpt->para_objPose[global_pose_num][3] = q.x();
                _objOpt->para_objPose[global_pose_num][4] = q.y();
                _objOpt->para_objPose[global_pose_num][5] = q.z();
                _objOpt->para_objPose[global_pose_num][6] = q.w();

                _objOpt->para_objDimensions[global_pose_num][0] = _obj.object_perframes[i].Dimensions.x();
                _objOpt->para_objDimensions[global_pose_num][1] = _obj.object_perframes[i].Dimensions.y();
                _objOpt->para_objDimensions[global_pose_num][2] = _obj.object_perframes[i].Dimensions.z();

                global_pose_num++;
            }
            else
            {
                _objOpt->isStatic[pos_ind] = _obj.object_perframes[i].isStatic;
                _objOpt->para_objPose[pos_ind][0] = _obj.object_perframes[i].Pi.x();
                _objOpt->para_objPose[pos_ind][1] = _obj.object_perframes[i].Pi.y();
                _objOpt->para_objPose[pos_ind][2] = _obj.object_perframes[i].Pi.z();

                _objOpt->para_objLinearVel[pos_ind][0] = _obj.object_perframes[i].Vi.x();
                _objOpt->para_objLinearVel[pos_ind][1] = _obj.object_perframes[i].Vi.y();
                _objOpt->para_objLinearVel[pos_ind][2] = _obj.object_perframes[i].Vi.z();

                _objOpt->para_objAngularVel[pos_ind][0] = _obj.object_perframes[i].Wi.x();
                _objOpt->para_objAngularVel[pos_ind][1] = _obj.object_perframes[i].Wi.y();
                _objOpt->para_objAngularVel[pos_ind][2] = _obj.object_perframes[i].Wi.z();

                _objOpt->para_objAcc[pos_ind][0] = _obj.object_perframes[i].Ai.x();
                _objOpt->para_objAcc[pos_ind][1] = _obj.object_perframes[i].Ai.y();
                _objOpt->para_objAcc[pos_ind][2] = _obj.object_perframes[i].Ai.z();

                Eigen::Quaterniond q{_obj.object_perframes[i].Qi.normalized()};
                _objOpt->para_objPose[pos_ind][3] = q.x();
                _objOpt->para_objPose[pos_ind][4] = q.y();
                _objOpt->para_objPose[pos_ind][5] = q.z();
                _objOpt->para_objPose[pos_ind][6] = q.w();

                _objOpt->para_objDimensions[pos_ind][0] = _obj.object_perframes[i].Dimensions.x();
                _objOpt->para_objDimensions[pos_ind][1] = _obj.object_perframes[i].Dimensions.y();
                _objOpt->para_objDimensions[pos_ind][2] = _obj.object_perframes[i].Dimensions.z();
            }
            char buffer[200];
            sprintf(buffer, "%5d%8d%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f \n", _obj.object_perframes[i].frame_id, _obj.id, tempdimension[0], tempdimension[1], tempdimension[2],
                    _obj.object_perframes[i].Pi.x(), _obj.object_perframes[i].Pi.y(), _obj.object_perframes[i].Pi.z(),
                    ypr.x(), ypr.y(), ypr.z());
        }
    }

    void obj_double2vector(objectOptimization *_objOpt, sensor_msgs::ObjectTrack &_obj)
    {
        for (int i = 0; i < _obj.object_perframes.size(); i++)
        {

            pair<int, int> obj_ind = make_pair(_obj.id, _obj.object_perframes[i].frame_id);
            int pos_ind = _objOpt->findPoseInd(obj_ind);

            if (pos_ind != -1)
            {
                _obj.object_perframes[i].Pi[0] = _objOpt->para_objPose[pos_ind][0];
                _obj.object_perframes[i].Pi[1] = _objOpt->para_objPose[pos_ind][1];
                _obj.object_perframes[i].Pi[2] = _objOpt->para_objPose[pos_ind][2];

                Eigen::Quaterniond q(_objOpt->para_objPose[pos_ind][6], _objOpt->para_objPose[pos_ind][3],
                                     _objOpt->para_objPose[pos_ind][4], _objOpt->para_objPose[pos_ind][5]);
                _obj.object_perframes[i].Qi = q.normalized();

                _obj.object_perframes[i].Vi[0] = _objOpt->para_objLinearVel[pos_ind][0];
                _obj.object_perframes[i].Vi[1] = _objOpt->para_objLinearVel[pos_ind][1];
                _obj.object_perframes[i].Vi[2] = _objOpt->para_objLinearVel[pos_ind][2];

                _obj.object_perframes[i].Wi[0] = _objOpt->para_objAngularVel[pos_ind][0];
                _obj.object_perframes[i].Wi[1] = _objOpt->para_objAngularVel[pos_ind][1];
                _obj.object_perframes[i].Wi[2] = _objOpt->para_objAngularVel[pos_ind][2];

                _obj.object_perframes[i].Ai[0] = _objOpt->para_objAcc[pos_ind][0];
                _obj.object_perframes[i].Ai[1] = _objOpt->para_objAcc[pos_ind][1];
                _obj.object_perframes[i].Ai[2] = _objOpt->para_objAcc[pos_ind][2];

                _obj.object_perframes[i].Dimensions[0] = _objOpt->para_objDimensions[pos_ind][0];
                _obj.object_perframes[i].Dimensions[1] = _objOpt->para_objDimensions[pos_ind][1];
                _obj.object_perframes[i].Dimensions[2] = _objOpt->para_objDimensions[pos_ind][2];

                Eigen::Vector3d ypr = Utility::R2ypr(_obj.object_perframes[i].Qi.toRotationMatrix());
            }
        }
    }

    void obj_vector2double_marg(objectOptimization *_objOpt, sensor_msgs::ObjectTrack &_obj)
    {
        int global_pose_num = _objOpt->obj_double_map.size();
        for (int i = 0; i < _obj.object_perframes.size(); i++)
        {
            pair<int, int> obj_ind = make_pair(_obj.id, _obj.object_perframes[i].frame_id);
            int pos_ind = _objOpt->findPoseInd(obj_ind);

            if (pos_ind == -1)
            {
                _objOpt->obj_double_map[obj_ind] = global_pose_num;

                _objOpt->para_objPose[global_pose_num][0] = _obj.object_perframes[i].Pi.x();
                _objOpt->para_objPose[global_pose_num][1] = _obj.object_perframes[i].Pi.y();
                _objOpt->para_objPose[global_pose_num][2] = _obj.object_perframes[i].Pi.z();
                Eigen::Quaterniond q{_obj.object_perframes[i].Qi.normalized()};
                _objOpt->para_objPose[global_pose_num][3] = q.x();
                _objOpt->para_objPose[global_pose_num][4] = q.y();
                _objOpt->para_objPose[global_pose_num][5] = q.z();
                _objOpt->para_objPose[global_pose_num][6] = q.w();

                _objOpt->para_objLinearVel[global_pose_num][0] = _obj.object_perframes[i].Vi.x();
                _objOpt->para_objLinearVel[global_pose_num][1] = _obj.object_perframes[i].Vi.y();
                _objOpt->para_objLinearVel[global_pose_num][2] = 0;

                _objOpt->para_objAngularVel[global_pose_num][0] = _obj.object_perframes[i].Wi.x();
                _objOpt->para_objAngularVel[global_pose_num][1] = _obj.object_perframes[i].Wi.y();
                _objOpt->para_objAngularVel[global_pose_num][2] = _obj.object_perframes[i].Wi.z();

                _objOpt->para_objAcc[global_pose_num][0] = _obj.object_perframes[i].Ai.x();
                _objOpt->para_objAcc[global_pose_num][1] = _obj.object_perframes[i].Ai.y();
                _objOpt->para_objAcc[global_pose_num][2] = 0;

                _objOpt->para_objDimensions[global_pose_num][0] = _obj.object_perframes[i].Dimensions.x();
                _objOpt->para_objDimensions[global_pose_num][1] = _obj.object_perframes[i].Dimensions.y();
                _objOpt->para_objDimensions[global_pose_num][2] = _obj.object_perframes[i].Dimensions.z();

                global_pose_num++;
            }
            else
            {
                _objOpt->para_objPose[pos_ind][0] = _obj.object_perframes[i].Pi.x();
                _objOpt->para_objPose[pos_ind][1] = _obj.object_perframes[i].Pi.y();
                _objOpt->para_objPose[pos_ind][2] = _obj.object_perframes[i].Pi.z();
                Eigen::Quaterniond q{_obj.object_perframes[i].Qi.normalized()};
                _objOpt->para_objPose[pos_ind][3] = q.x();
                _objOpt->para_objPose[pos_ind][4] = q.y();
                _objOpt->para_objPose[pos_ind][5] = q.z();
                _objOpt->para_objPose[pos_ind][6] = q.w();

                _objOpt->para_objLinearVel[pos_ind][0] = _obj.object_perframes[i].Vi.x();
                _objOpt->para_objLinearVel[pos_ind][1] = _obj.object_perframes[i].Vi.y();
                _objOpt->para_objLinearVel[pos_ind][2] = 0;

                _objOpt->para_objAngularVel[pos_ind][0] = _obj.object_perframes[i].Wi.x();
                _objOpt->para_objAngularVel[pos_ind][1] = _obj.object_perframes[i].Wi.y();
                _objOpt->para_objAngularVel[pos_ind][2] = _obj.object_perframes[i].Wi.z();

                _objOpt->para_objAcc[pos_ind][0] = _obj.object_perframes[i].Ai.x();
                _objOpt->para_objAcc[pos_ind][1] = _obj.object_perframes[i].Ai.y();
                _objOpt->para_objAcc[pos_ind][2] = 0;

                _objOpt->para_objDimensions[pos_ind][0] = _obj.object_perframes[i].Dimensions.x();
                _objOpt->para_objDimensions[pos_ind][1] = _obj.object_perframes[i].Dimensions.y();
                _objOpt->para_objDimensions[pos_ind][2] = _obj.object_perframes[i].Dimensions.z();
            }
        }
    }

    void optimization_muti_objTracking_BA(objectOptimization *_objOpt, sensor_msgs::ObjectTrack &_obj)
    {
        if (sliding_window_opt)
        {
            obj_vector2double(_objOpt, _obj);

            ceres::Problem problem;
            ceres::LossFunction *tracking_small_loss;
            tracking_small_loss = new ceres::HuberLoss(0.1);

            ceres::LossFunction *tracking_large_loss;
            tracking_large_loss = new ceres::HuberLoss(0.5);
            for (int i = 0; i < _obj.object_perframes.size(); i++)
            {
                pair<int, int> obj_ind_i = make_pair(_obj.id, _obj.object_perframes[i].frame_id);

                int pos_i = _objOpt->findPoseInd(obj_ind_i);
                if (pos_i != -1)
                {
                    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
                    problem.AddParameterBlock(_objOpt->para_objPose[pos_i], SIZE_POSE, local_parameterization);
                    problem.AddParameterBlock(_objOpt->para_objLinearVel[pos_i], 3);
                    problem.AddParameterBlock(_objOpt->para_objAngularVel[pos_i], 3);
                    problem.AddParameterBlock(_objOpt->para_objAcc[pos_i], 3);
                    problem.AddParameterBlock(_objOpt->para_objDimensions[pos_i], 3);
                }
            }

            int window_first_opt_index = _obj.object_perframes[0].frame_id;

            double best_detection_score = std::numeric_limits<double>::min();  // 初始化为最大值

            for (int i = 0; i < _obj.object_perframes.size(); i++)
            {
                if (_obj.object_perframes[i].feature_box.status == -1)
                    continue;

                window_first_opt_index = _obj.object_perframes[i].frame_id;
                break;
            }

            pair<int, int> first_obj_ind = make_pair(_obj.id, window_first_opt_index);
            int first_pose_ind = _objOpt->findPoseInd(first_obj_ind);

            if (first_pose_ind < 0)
                return;

            problem.SetParameterBlockConstant(_objOpt->para_objPose[first_pose_ind]);

            if (_objOpt->last_marginalization_info_tracking != nullptr && _objOpt->last_marginalization_info_tracking->valid)
            {
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(_objOpt->last_marginalization_info_tracking);
                problem.AddResidualBlock(marginalization_factor, NULL,
                                         _objOpt->last_marginalization_parameter_blocks_tracking);
            }

            bool motion_model_marg = false;
            
            if (motion_model)
            {
                for (int i = 0; i < _obj.object_perframes.size() - 1; i++)
                {

                    for (int j = i + 1; j < _obj.object_perframes.size(); ++j)
                    {
                        pair<int, int> obj_ind_i = make_pair(_obj.id, _obj.object_perframes[i].frame_id);
                        int pos_i = _objOpt->findPoseInd(obj_ind_i);

                        pair<int, int> obj_ind_j = make_pair(_obj.id, _obj.object_perframes[j].frame_id);
                        int pos_j = _objOpt->findPoseInd(obj_ind_j);

                        double dt = _obj.object_perframes[j].time - _obj.object_perframes[i].time;

                        if ((pos_i != -1) && (pos_j != -1) && dt != 0)
                        {
                            if (motion_model == 2)
                            {
                                Eigen::Map<Eigen::Matrix<double, 15, 15, Eigen::RowMajor>> Cov(_objOpt->ceres_cov[pos_i]);
                                ceres::CostFunction *const_acc_functor = ConstAccFactorAuto::Create(dt, Cov, motion_weight);

                                std::vector<double> const_acc_residuals = checkConstAccFactorAuto(_objOpt->para_objPose[pos_i], _objOpt->para_objLinearVel[pos_i], 
                                                    _objOpt->para_objAngularVel[pos_i], _objOpt->para_objAcc[pos_i],
                                                    _objOpt->para_objPose[pos_j], _objOpt->para_objLinearVel[pos_j], 
                                                    _objOpt->para_objAngularVel[pos_j], _objOpt->para_objAcc[pos_j],
                                                    dt, 1.0, Cov);
                                    
                                std::pair<double, Eigen::Vector3d> residuals = calculateResidualError(const_acc_residuals);
                                if (residuals.first > 10.0 || residuals.second.norm() > 40.0) continue;

                                if (i==0) motion_model_marg = true;

                                problem.AddResidualBlock(const_acc_functor, nullptr, _objOpt->para_objPose[pos_i], _objOpt->para_objLinearVel[pos_i], _objOpt->para_objAngularVel[pos_i], _objOpt->para_objAcc[pos_i],
                                                     _objOpt->para_objPose[pos_j], _objOpt->para_objLinearVel[pos_j], _objOpt->para_objAngularVel[pos_j], _objOpt->para_objAcc[pos_j]);

                            }
                            else
                            {
                                std::vector<double> cvrt_residuals = checkCVRTFactorAuto(_objOpt->para_objPose[pos_i], _objOpt->para_objLinearVel[pos_i], 
                                                _objOpt->para_objAngularVel[pos_i],
                                                _objOpt->para_objPose[pos_j], _objOpt->para_objLinearVel[pos_j], 
                                                _objOpt->para_objAngularVel[pos_j],
                                                dt, 1.0);
                                std::pair<double, Eigen::Vector3d> residuals = calculateResidualError(cvrt_residuals);
                                if (residuals.first > 10.0 || residuals.second.norm() > 40.0) continue;

                                if (i==0) motion_model_marg = true;

                                ceres::CostFunction *cvrt_functor = CVRTFactorAuto::Create(dt,motion_weight);
                                problem.AddResidualBlock(cvrt_functor, nullptr, _objOpt->para_objPose[pos_i], _objOpt->para_objLinearVel[pos_i], _objOpt->para_objAngularVel[pos_i],
                                                     _objOpt->para_objPose[pos_j], _objOpt->para_objLinearVel[pos_j], _objOpt->para_objAngularVel[pos_j]);

                            }
                            break;
                        }
                        
                    }
                }
            }
            else
            {}
            
            if (use_box_model)
            {
                for (int i = 0; i < _obj.object_perframes.size(); i++)
                {
                    if (_obj.object_perframes[i].feature_box.status == -1)
                        continue;
                    bool is_static = false;
                    if (_obj.object_perframes[i].isStatic)
                        is_static = true;
                    double score = _obj.object_perframes[i].score;

                    int obj_frame_id = _obj.object_perframes[i].frame_id;
                    pair<int, int> obj_ind = make_pair(_obj.id, obj_frame_id);
                    int pose_ind = _objOpt->findPoseInd(obj_ind);

                    _objOpt->para_objPose[pose_ind][0] = _obj.object_perframes[i].Pi_obs.x();
                    _objOpt->para_objPose[pose_ind][1] = _obj.object_perframes[i].Pi_obs.y();
                    _objOpt->para_objPose[pose_ind][2] = _obj.object_perframes[i].Pi_obs.z();

                    _objOpt->para_objPose[pose_ind][3] = _obj.object_perframes[i].Qi_obs.x();
                    _objOpt->para_objPose[pose_ind][4] = _obj.object_perframes[i].Qi_obs.y();
                    _objOpt->para_objPose[pose_ind][5] = _obj.object_perframes[i].Qi_obs.z();
                    _objOpt->para_objPose[pose_ind][6] = _obj.object_perframes[i].Qi_obs.w();

                    std::vector<double> measure_residuals = checkMeasureFactor(_objOpt->para_objPose[pose_ind], 
                                    _obj.object_perframes[i].Pi_obs,
                                    _obj.object_perframes[i].Qi_obs.normalized(), 
                                    1.0);
                    std::pair<double, Eigen::Vector3d> residuals = calculateResidualError(measure_residuals);
                    

                    if (residuals.first > 10.0 || residuals.second.norm() > 40.0) continue;
                    
                    ceres::CostFunction *obj_measurement_functor = MeasureFactor::Create(_obj.object_perframes[i].Pi_obs, 
                                                                    _obj.object_perframes[i].Qi_obs.normalized(), 10);
                    problem.AddResidualBlock(obj_measurement_functor, tracking_large_loss, _objOpt->para_objPose[pose_ind]);
                }
            }
            
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.trust_region_strategy_type = ceres::DOGLEG;
            options.minimizer_progress_to_stdout = true;
            options.max_num_iterations = 8;
            options.function_tolerance = 1e-4;

            options.minimizer_progress_to_stdout = false;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            
            obj_double2vector(_objOpt, _obj);
            if (use_marginalization)
            {
                bool marginlization_status = false;
                MarginalizationInfo *marginalization_info = new MarginalizationInfo();
                obj_vector2double_marg(_objOpt, _obj);

                vector<double *> drop_para;
                getRemoveParaVec(drop_para, _objOpt, _obj);

                if (_objOpt->last_marginalization_info_tracking != nullptr && _objOpt->last_marginalization_info_tracking->valid)
                {
                    vector<int> drop_set;

                    for (int i = 0; i < static_cast<int>(_objOpt->last_marginalization_parameter_blocks_tracking.size()); i++)
                    {
                        for (int j = 0; j < drop_para.size(); j++)
                        {
                            if (_objOpt->last_marginalization_parameter_blocks_tracking[i] == drop_para[j])
                            {
                                drop_set.push_back(i);
                            }
                        }
                    }

                    MarginalizationFactor *marginalization_factor = new MarginalizationFactor(_objOpt->last_marginalization_info_tracking);
                    
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                                   _objOpt->last_marginalization_parameter_blocks_tracking, drop_set);
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }

                
                if(motion_model)
                {
                    for (int i = 0; i < _obj.object_perframes.size() - 1; i++)
                    {
                        if (!motion_model_marg) break;

                        for (int j = i + 1; j < _obj.object_perframes.size(); ++j)
                        {
                            pair<int, int> obj_ind_i = make_pair(_obj.id, _obj.object_perframes[i].frame_id);
                            int pos_i = _objOpt->findPoseInd(obj_ind_i);

                            pair<int, int> obj_ind_j = make_pair(_obj.id, _obj.object_perframes[j].frame_id);
                            int pos_j = _objOpt->findPoseInd(obj_ind_j);

                            double dt = _obj.object_perframes[j].time - _obj.object_perframes[i].time;

                            if ((pos_i != -1) && (pos_j != -1) && dt != 0)
                            {
                                if (motion_model == 2)
                                {
                                    Eigen::Map<Eigen::Matrix<double, 15, 15, Eigen::RowMajor>> Cov(_objOpt->ceres_cov[pos_i]);
                                    ceres::CostFunction *const_acc_functor = ConstAccFactorAuto::Create(dt, Cov, motion_weight);
                                    
                                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(const_acc_functor, tracking_small_loss, 
                                                        vector<double *>{_objOpt->para_objPose[pos_i], _objOpt->para_objLinearVel[pos_i], _objOpt->para_objAngularVel[pos_i], _objOpt->para_objAcc[pos_i],
                                                        _objOpt->para_objPose[pos_j], _objOpt->para_objLinearVel[pos_j], _objOpt->para_objAngularVel[pos_j], _objOpt->para_objAcc[pos_j]},
                                                        vector<int>{0, 1, 2, 3});
                                
                                    marginalization_info->addResidualBlockInfo(residual_block_info);
                                    marginlization_status = true;
                                }
                                else
                                {
                                   
                                    ceres::CostFunction *cvrt_functor = CVRTFactorAuto::Create(dt,motion_weight);
                                    
                                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(cvrt_functor, tracking_large_loss, 
                                                        vector<double *>{_objOpt->para_objPose[pos_i], _objOpt->para_objLinearVel[pos_i], _objOpt->para_objAngularVel[pos_i],
                                                        _objOpt->para_objPose[pos_j], _objOpt->para_objLinearVel[pos_j], _objOpt->para_objAngularVel[pos_j]},
                                                        vector<int>{0, 1, 2});
                                    marginalization_info->addResidualBlockInfo(residual_block_info);

                                    marginlization_status = true;
                                }
                                break;
                            }
                            
                        }
                        break;
                    }
                }     

                if (!marginlization_status)
                {
                    _objOpt->last_marginalization_info_tracking = nullptr;
                    return;
                }
                marginalization_info->preMarginalize();
                marginalization_info->marginalize();
                std::unordered_map<long, double *> addr_shift;
                getObjTrackingParameterBlocks(addr_shift, _objOpt, _obj);
                vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

                _objOpt->last_marginalization_info_tracking = nullptr;
                if (_objOpt->last_marginalization_info_tracking != nullptr)
                    delete _objOpt->last_marginalization_info_tracking;

                _objOpt->last_marginalization_info_tracking = marginalization_info;
                _objOpt->last_marginalization_parameter_blocks_tracking = parameter_blocks;
            }
        }
    }

    std::pair<double, Eigen::Vector3d> calculateResidualError(const std::vector<double>& residuals) {
        
        double position_norm = 0.0;
        Eigen::Vector3d angle_deg = Eigen::Vector3d(0,0,0);
        
        if (residuals.size() >= 6) {
            position_norm = sqrt(residuals[0]*residuals[0] + residuals[1]*residuals[1] + residuals[2]*residuals[2]);
            
            Eigen::Vector3d angle_rad = 2* Eigen::Vector3d(residuals[3], residuals[4], residuals[5]);
            angle_deg = angle_rad * 180.0 / M_PI;
        }
        
        return std::make_pair(position_norm, angle_deg);
    }

    std::vector<double> checkCVRTFactorAuto(double* pose_i, double* linearVel_i, double* angularVel_i,
                                        double* pose_j, double* linearVel_j, double* angularVel_j,
                                        double dt, double motion_weight) {
        
        double residuals[12];
        
        CVRTFactorAuto factor(dt, motion_weight);
        factor(pose_i, linearVel_i, angularVel_i, pose_j, linearVel_j, angularVel_j, residuals);
        
        std::vector<double> residual_vec(residuals, residuals + 12);
        
        double total_residual = 0;
        for (int i = 0; i < 12; ++i) {
            total_residual += fabs(residuals[i]);
        }
        
        return residual_vec;
    }

    std::vector<double> checkConstAccFactorAuto(double* pose_i, double* linearVel_i, double* angularVel_i, double* acc_i,
                                            double* pose_j, double* linearVel_j, double* angularVel_j, double* acc_j,
                                            double dt, double motion_weight, Eigen::MatrixXd Cov_prior) {
        
        double residuals[15];
        
        ConstAccFactorAuto factor(dt, Cov_prior, motion_weight);
        factor(pose_i, linearVel_i, angularVel_i, acc_i, pose_j, linearVel_j, angularVel_j, acc_j, residuals);
        
        std::vector<double> residual_vec(residuals, residuals + 15);
        
        double total_residual = 0;
        for (int i = 0; i < 15; ++i) {
            total_residual += fabs(residuals[i]);
        }
        
        return residual_vec;
    }

    std::vector<double> checkMeasureFactor(double* pose, Eigen::Vector3d Pi_obs, Eigen::Quaterniond Qi_obs, double score) {
        
        double residuals[6];
        
        MeasureFactor factor(Pi_obs, Qi_obs, score);
        factor(pose, residuals);
        
        std::vector<double> residual_vec(residuals, residuals + 6);
        
        double position_residual = sqrt(residuals[0]*residuals[0] + residuals[1]*residuals[1] + residuals[2]*residuals[2]);
        double rotation_residual = sqrt(residuals[3]*residuals[3] + residuals[4]*residuals[4] + residuals[5]*residuals[5]);
        double total_residual = position_residual + rotation_residual;
        
        return residual_vec;
    }


    void getObjTrackingParameterBlocks(std::unordered_map<long, double *> &addr_shift, objectOptimization *_objOpt, sensor_msgs::ObjectTrack &_obj)
    {
        int global_pose_num = 0;
        map<pair<int, int>, int> obj_map_tmp;

        int start_frame_id = _obj.object_perframes[0].frame_id;

        for (int i = 0; i < _obj.object_perframes.size(); i++)
        {
            if (_obj.object_perframes[i].frame_id == start_frame_id)
                continue;
            
            pair<int, int> obj_ind = make_pair(_obj.id, _obj.object_perframes[i].frame_id);
            obj_map_tmp[obj_ind] = global_pose_num;

            _objOpt->para_objPose[global_pose_num][0] = _obj.object_perframes[i].Pi.x();
            _objOpt->para_objPose[global_pose_num][1] = _obj.object_perframes[i].Pi.y();
            _objOpt->para_objPose[global_pose_num][2] = _obj.object_perframes[i].Pi.z();
            Eigen::Quaterniond q{_obj.object_perframes[i].Qi};
            _objOpt->para_objPose[global_pose_num][3] = q.x();
            _objOpt->para_objPose[global_pose_num][4] = q.y();
            _objOpt->para_objPose[global_pose_num][5] = q.z();
            _objOpt->para_objPose[global_pose_num][6] = q.w();

            _objOpt->para_objLinearVel[global_pose_num][0] = _obj.object_perframes[i].Vi.x();
            _objOpt->para_objLinearVel[global_pose_num][1] = _obj.object_perframes[i].Vi.y();
            _objOpt->para_objLinearVel[global_pose_num][2] = _obj.object_perframes[i].Vi.z();

            _objOpt->para_objAngularVel[global_pose_num][0] = _obj.object_perframes[i].Wi.x();
            _objOpt->para_objAngularVel[global_pose_num][1] = _obj.object_perframes[i].Wi.y();
            _objOpt->para_objAngularVel[global_pose_num][2] = _obj.object_perframes[i].Wi.z();

            _objOpt->para_objAcc[global_pose_num][0] = _obj.object_perframes[i].Ai.x();
            _objOpt->para_objAcc[global_pose_num][1] = _obj.object_perframes[i].Ai.y();
            _objOpt->para_objAcc[global_pose_num][2] = _obj.object_perframes[i].Ai.z();

            _objOpt->para_objDimensions[global_pose_num][0] = _obj.object_perframes[i].Dimensions.x();
            _objOpt->para_objDimensions[global_pose_num][1] = _obj.object_perframes[i].Dimensions.y();
            _objOpt->para_objDimensions[global_pose_num][2] = _obj.object_perframes[i].Dimensions.z();

            global_pose_num++;
        }

        map<int, int> obj_index_map;
        for (auto obj_iter = obj_map_tmp.begin(); obj_iter != obj_map_tmp.end(); obj_iter++)
        {
            int obj_old_index = _objOpt->findPoseInd(obj_iter->first);
            if (obj_old_index != -1)
                obj_index_map[obj_old_index] = obj_iter->second;
            else{}
        }

        for (auto obj_iter = obj_index_map.begin(); obj_iter != obj_index_map.end(); obj_iter++)
        {
            addr_shift[reinterpret_cast<long>(_objOpt->para_objPose[obj_iter->first])] = _objOpt->para_objPose[obj_iter->second];
        }

        for (auto obj_iter = obj_index_map.begin(); obj_iter != obj_index_map.end(); obj_iter++)
        {
            addr_shift[reinterpret_cast<long>(_objOpt->para_objLinearVel[obj_iter->first])] = _objOpt->para_objLinearVel[obj_iter->second];
        }

        for (auto obj_iter = obj_index_map.begin(); obj_iter != obj_index_map.end(); obj_iter++)
        {
            addr_shift[reinterpret_cast<long>(_objOpt->para_objAngularVel[obj_iter->first])] = _objOpt->para_objAngularVel[obj_iter->second];
        }

        for (auto obj_iter = obj_index_map.begin(); obj_iter != obj_index_map.end(); obj_iter++)
        {
            addr_shift[reinterpret_cast<long>(_objOpt->para_objAcc[obj_iter->first])] = _objOpt->para_objAcc[obj_iter->second];
        }

        for (auto obj_iter = obj_index_map.begin(); obj_iter != obj_index_map.end(); obj_iter++)
        {
            addr_shift[reinterpret_cast<long>(_objOpt->para_objDimensions[obj_iter->first])] = _objOpt->para_objDimensions[obj_iter->second];
        }

        _objOpt->obj_double_map = obj_map_tmp;
    }

    void getRemoveParaVec(vector<double *> &drop_para, objectOptimization *_objOpt, sensor_msgs::ObjectTrack &_obj)
    {
        int start_frame_id = _obj.object_perframes[0].frame_id;
        pair<int, int> obj_ind = make_pair(_obj.id, start_frame_id);
        int pose_double_ind = _objOpt->findPoseInd(obj_ind);

        if (pose_double_ind != -1)
        {
            drop_para.push_back(_objOpt->para_objPose[pose_double_ind]);
            drop_para.push_back(_objOpt->para_objLinearVel[pose_double_ind]);
            drop_para.push_back(_objOpt->para_objAngularVel[pose_double_ind]);
            drop_para.push_back(_objOpt->para_objAcc[pose_double_ind]);
            drop_para.push_back(_objOpt->para_objDimensions[pose_double_ind]);
        }
    }

    void updateObjectInfo()
    {
        for (vector<sensor_msgs::ObjectTrack>::iterator iter_obj = allObjectsData.begin(); iter_obj != allObjectsData.end();)
        {
            sensor_msgs::ObjectTrack &per_obj = *iter_obj;

            int object_id = per_obj.id;
            int buf_id = findObjBufInd(object_id);
            if (buf_id == -1)
            {

                if (per_obj.isFailTracking)
                {
                    iter_obj = allObjectsData.erase(iter_obj);
                    continue;
                }

                if (per_obj.object_perframes.size() < (WINDOW_SIZE + 1))
                {
                    iter_obj++;
                    continue;
                }

                objectOptimization *perObjOpt = new objectOptimization;
                perObjOpt->obj_id = object_id;
                objOptbuffer.push_back(perObjOpt);

                optimization_muti_objTracking_BA(perObjOpt, per_obj);

                per_obj.isOpt = true;

                slidingWindowObj(per_obj);
                iter_obj++;
            }
            else
            {

                if (per_obj.isFailTracking)
                {
                    iter_obj = allObjectsData.erase(iter_obj);
                    objOptbuffer.erase(objOptbuffer.begin() + buf_id);
                    continue;
                }

                if (per_obj.object_perframes.size() < (WINDOW_SIZE + 1))
                {
                    iter_obj++;
                    continue;
                }

                if (per_obj.object_perframes.back().feature_box.status == -1)
                {
                    slidingWindowObj(per_obj);
                    objOptbuffer[buf_id]->last_marginalization_info_tracking = nullptr;
                    iter_obj++;
                    continue;
                }

                optimization_muti_objTracking_BA(objOptbuffer[buf_id], per_obj);

                slidingWindowObj(per_obj);
                per_obj.isOpt = true;
                iter_obj++;
            }
        }
    }

    void slidingWindowObj(sensor_msgs::ObjectTrack &_obj)
    {
        int start_frame_id = _obj.object_perframes[0].frame_id;
        if (_obj.object_perframes[0].feature_box.status != -1)
        {
            _obj.traj_history[start_frame_id] = std::make_pair(_obj.object_perframes[0].Pi, _obj.object_perframes[0].Qi);
            _obj.vehicle_corner_cloud_total[start_frame_id] = _obj.object_perframes[0].pointcloudCorner;
            _obj.vehicle_surf_cloud_total[start_frame_id] = _obj.object_perframes[0].pointcloudSurf;
            _obj.vehicle_full_cloud_total[start_frame_id] = _obj.object_perframes[0].pointcloudFull;
        }
        _obj.object_perframes.erase(_obj.object_perframes.begin());
    }

    void publishTrackingResult()
    {
        jsk_recognition_msgs::BoundingBoxArray objBboxTrack;
        jsk_recognition_msgs::BoundingBoxArray objBboxTrackOpt;

        static std::map<int, jsk_recognition_msgs::BoundingBoxArray> tracks_bboxes;

        objBboxTrack.header = cloudHeader;
        objBboxTrackOpt.header = cloudHeader;
        objBboxTrackOpt.header.frame_id = "map";

        visualization_msgs::MarkerArray label_markers;
        
        pcl::PointCloud<PointType> objs_pc_full;
        pcl::PointCloud<PointType> objs_pc_full_in_world;
        pcl::PointCloud<PointType> objs_corner_full;
        pcl::PointCloud<PointType> objs_surf_full;
        int valid_obj_num = 0;
        std::vector<ObjTraj> objTrajs;
        objTrajs.clear();
        
        geometry_msgs::PoseArray obj_pose_array;
        obj_pose_array.header = cloudHeader;
        obj_pose_array.header.frame_id = "map";
        for (size_t i = 0; i < allObjectsData.size(); ++i)
        {
            if (allObjectsData[i].isFailTracking)
            {
                continue;
            }

            if (allObjectsData[i].object_perframes.size() < birthmin)
            {
                continue;
            }

            int n = allObjectsData[i].object_perframes.size() - 1;
            
            sensor_msgs::ObjectPerFrameObs obj_per_frame = allObjectsData[i].object_perframes[n];

            jsk_recognition_msgs::BoundingBox box = obj_per_frame.feature_box.bbox3d_in_local;
            box.header.frame_id = "map";
            box.pose.position.x = allObjectsData[i].object_perframes[n].Pi.x();
            box.pose.position.y = allObjectsData[i].object_perframes[n].Pi.y();
            box.pose.position.z = allObjectsData[i].object_perframes[n].Pi.z();
            box.pose.orientation.w = allObjectsData[i].object_perframes[n].Qi.w();
            box.pose.orientation.x = allObjectsData[i].object_perframes[n].Qi.x();
            box.pose.orientation.y = allObjectsData[i].object_perframes[n].Qi.y();
            box.pose.orientation.z = allObjectsData[i].object_perframes[n].Qi.z();
            box.dimensions.x = allObjectsData[i].object_perframes[n].Dimensions.x();
            box.dimensions.y = allObjectsData[i].object_perframes[n].Dimensions.y();
            box.dimensions.z = allObjectsData[i].object_perframes[n].Dimensions.z();

            if (allObjectsData[i].object_perframes[n].feature_box.status != -1)
            {
                box.label = 1;
                tracks_bboxes[allObjectsData[i].id].boxes.push_back(box);
            }
            else
            {
                box.label = 0;
                tracks_bboxes[allObjectsData[i].id].boxes.push_back(box);
            }

            visualization_msgs::Marker label_marker;
            label_marker.lifetime = ros::Duration(3);
            label_marker.header = cloudHeader;
            label_marker.header.frame_id = "map";
            label_marker.ns = "/label_markers";
            label_marker.action = visualization_msgs::Marker::ADD;
            label_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            label_marker.scale.x = 1.5;
            label_marker.scale.y = 1.5;
            label_marker.scale.z = 1.5;
            std_msgs::ColorRGBA label_color_;
            label_color_.r = 1;
            label_color_.g = 0;
            label_color_.b = 0;
            label_color_.a = 1.0;
            label_marker.id = allObjectsData[i].id;

            for (int j = allObjectsData[i].object_perframes.size() - 1; j >= 0; j--)
            {
                if (1)
                {
                    sensor_msgs::ObjectPerFrameObs obj_per_frame = allObjectsData[i].object_perframes[j];
                    objBboxTrack.boxes.push_back(obj_per_frame.feature_box.bbox3d_in_local);

                    jsk_recognition_msgs::BoundingBox box = obj_per_frame.feature_box.bbox3d_in_local;
                    box.header.frame_id = "map";
                    box.pose.position.x = allObjectsData[i].object_perframes[j].Pi.x();
                    box.pose.position.y = allObjectsData[i].object_perframes[j].Pi.y();
                    box.pose.position.z = allObjectsData[i].object_perframes[j].Pi.z();
                    box.pose.orientation.w = allObjectsData[i].object_perframes[j].Qi.w();
                    box.pose.orientation.x = allObjectsData[i].object_perframes[j].Qi.x();
                    box.pose.orientation.y = allObjectsData[i].object_perframes[j].Qi.y();
                    box.pose.orientation.z = allObjectsData[i].object_perframes[j].Qi.z();
                    box.dimensions.x = allObjectsData[i].object_perframes[j].Dimensions.x();
                    box.dimensions.y = allObjectsData[i].object_perframes[j].Dimensions.y();
                    box.dimensions.z = allObjectsData[i].object_perframes[j].Dimensions.z();
                    objBboxTrackOpt.boxes.push_back(box);
                   
                    std::stringstream id_stream;
                    id_stream << (allObjectsData[i].id);
                    std::string id_str = "ID: " + id_stream.str();
                    
                    label_marker.color = label_color_;
                    label_marker.text = id_str;
                    label_marker.pose.position.x = obj_per_frame.Pi.x();
                    label_marker.pose.position.y = obj_per_frame.Pi.y();
                    label_marker.pose.position.z = obj_per_frame.Pi.z() +
                                                   obj_per_frame.feature_box.bbox3d_in_local.dimensions.z / 2.0 + 1.5;
                    label_marker.scale.z = 1.0;
                    if (!label_marker.text.empty())
                        label_markers.markers.push_back(label_marker);
                    break;
                }
            }

            if (allObjectsData[i].object_perframes.back().feature_box.status != -1)
            {
                valid_obj_num++;
                objs_pc_full += allObjectsData[i].vehiclePointCloud;
                objs_pc_full += allObjectsData[i].vehicleFullCloudMap;
                objs_corner_full += allObjectsData[i].vehicleCornerCloudMap;
                objs_surf_full += allObjectsData[i].vehicleSurfCloudMap;
                objs_pc_full_in_world += allObjectsData[i].vehiclePointCloudInWorld;
            }

            ObjTraj objTraj;
            for (int j = 0; j < allObjectsData[i].object_perframes.size(); j++)
            {
                if (allObjectsData[i].object_perframes[j].feature_box.status != -1)
                {
                    sensor_msgs::ObjectPerFrameObs obj = allObjectsData[i].object_perframes[j];
                    objTraj.points.push_back(obj.Pi);
                    objTraj.orientations.push_back(obj.Qi);
                    objTraj.velos.push_back(obj.Vi);

                    geometry_msgs::Pose obj_pose;
                    obj_pose.position.x = obj.Pi.x();
                    obj_pose.position.y = obj.Pi.y();
                    obj_pose.position.z = obj.Pi.z();
                    Eigen::Quaterniond qs_last(obj.Qi);
                    qs_last = qs_last.normalized();
                    obj_pose.orientation.w = qs_last.w();
                    obj_pose.orientation.x = qs_last.x();
                    obj_pose.orientation.y = qs_last.y();
                    obj_pose.orientation.z = qs_last.z();

                    obj_pose_array.poses.push_back(obj_pose);
                }
            }
            pubObjPosesGlobal.publish(obj_pose_array);
            objTraj.id = allObjectsData[i].id;
            objTrajs.push_back(objTraj);
        }

        jsk_recognition_msgs::BoundingBoxArray objBboxTracked;

        objBboxTracked.header = cloudHeader;
        objBboxTracked.header.frame_id = "map";

        jsk_recognition_msgs::BoundingBoxArray objBboxPredicted;

        objBboxPredicted.header = cloudHeader;
        objBboxPredicted.header.frame_id = "map";

        for (size_t i = 0; i < allObjectsData.size(); ++i)
        {
            if (allObjectsData[i].isFailTracking)
            {
                continue;
            }

            if (allObjectsData[i].object_perframes.size() < 3)
            {
                continue;
            }

            bool is_static = false;
            for (int k = allObjectsData[i].object_perframes.size() - 1; k > 0; k--)
            {
                if (allObjectsData[i].object_perframes[k].isStatic)
                {
                    if (allObjectsData[i].object_perframes[k].feature_box.status != -1 && k != 0)
                    {
                        sensor_msgs::ObjectPerFrameObs obj_per_frame = allObjectsData[i].object_perframes[k];
                        jsk_recognition_msgs::BoundingBox box = obj_per_frame.feature_box.bbox3d_in_local;
                        box.label = 1;
                        box.header.frame_id = "map";
                        box.pose.position.x = allObjectsData[i].object_perframes[k].Pi.x();
                        box.pose.position.y = allObjectsData[i].object_perframes[k].Pi.y();
                        box.pose.position.z = allObjectsData[i].object_perframes[k].Pi.z();
                        box.pose.orientation.w = allObjectsData[i].object_perframes[k].Qi.w();
                        box.pose.orientation.x = allObjectsData[i].object_perframes[k].Qi.x();
                        box.pose.orientation.y = allObjectsData[i].object_perframes[k].Qi.y();
                        box.pose.orientation.z = allObjectsData[i].object_perframes[k].Qi.z();
                        box.dimensions.x = allObjectsData[i].object_perframes[k].Dimensions.x();
                        box.dimensions.y = allObjectsData[i].object_perframes[k].Dimensions.y();
                        box.dimensions.z = allObjectsData[i].object_perframes[k].Dimensions.z();
                        objBboxTracked.boxes.push_back(box);
                        is_static = true;
                        break;
                    }
                }
            }

            jsk_recognition_msgs::BoundingBoxArray bboxes = tracks_bboxes[allObjectsData[i].id];
            for (int j = 0; j < bboxes.boxes.size(); ++j)
            {
                if (bboxes.boxes[j].label == 1)
                    objBboxTracked.boxes.push_back(bboxes.boxes[j]);
                else
                    objBboxPredicted.boxes.push_back(bboxes.boxes[j]);
            }
        }

        if (pubGlobalTrajs.getNumSubscribers() != 0 && objTrajs.size() > 0)
        {
            visualization_msgs::MarkerArray mk;
            trajs2Markers(objTrajs, mk, cloudHeader);
            pubGlobalTrajs.publish(mk);
        }

        if (pubVeloArrowArray.getNumSubscribers() != 0 && objTrajs.size() > 0)
        {
            visualization_msgs::MarkerArray velo_arrow_mks, velo_text_mks;
            velos2Markers(objTrajs, velo_arrow_mks, velo_text_mks, cloudHeader);
            pubVeloArrowArray.publish(velo_arrow_mks);
            pubVeloTextArray.publish(velo_text_mks);
        }

        if (pubObjectPointCloudLast.getNumSubscribers() != 0)
        {

            sensor_msgs::PointCloud2 laserCloudTemp;
            pcl::toROSMsg(objs_pc_full, laserCloudTemp);
            laserCloudTemp.header = cloudHeader;
            laserCloudTemp.header.frame_id = "map";
            pubObjectPointCloudLast.publish(laserCloudTemp);
        }

        if (pubObjectCloudCorner.getNumSubscribers() != 0)
        {

            sensor_msgs::PointCloud2 laserCloudTemp;
            pcl::toROSMsg(objs_corner_full, laserCloudTemp);
            laserCloudTemp.header = cloudHeader;
            laserCloudTemp.header.frame_id = "map";
            pubObjectCloudCorner.publish(laserCloudTemp);
        }

        if (pubObjectCloudSurf.getNumSubscribers() != 0)
        {

            sensor_msgs::PointCloud2 laserCloudTemp;
            pcl::toROSMsg(objs_surf_full, laserCloudTemp);
            laserCloudTemp.header = cloudHeader;
            laserCloudTemp.header.frame_id = "map";
            pubObjectCloudSurf.publish(laserCloudTemp);
        }

        if (pubMarkers.getNumSubscribers() != 0)
        {
            pubMarkers.publish(label_markers);
        }

        if (pubTrackBboxArray.getNumSubscribers() != 0)
        {
            pubTrackBboxArray.publish(objBboxTrack);
        }

        if (pubObjectBboxArrayPrev.getNumSubscribers() != 0)
        {
            pubObjectBboxArrayPrev.publish(objBboxVecPrev);
        }

        if (pubTrackBboxArrayAfterOpt.getNumSubscribers() != 0)
        {
            pubTrackBboxArrayAfterOpt.publish(objBboxTrackOpt);
        }

        if (pubTrackBboxArrayPredict.getNumSubscribers() != 0)
        {
            pubTrackBboxArrayPredict.publish(objBboxVecPred);
            objBboxVecPred.boxes.clear();
        }

        if (pubTrackPosArrayPredictCov.getNumSubscribers() != 0)
        {
            pubTrackPosArrayPredictCov.publish(objPosCovVecPred);
        }

        if (pubObjectPointCloudFullInWorld.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 laserCloudTemp;
            pcl::toROSMsg(objs_pc_full_in_world, laserCloudTemp);
            laserCloudTemp.header = cloudHeader;
            laserCloudTemp.header.frame_id = "map";
            pubObjectPointCloudFullInWorld.publish(laserCloudTemp);
        }

        if (pubSuccessBbox.getNumSubscribers() != 0)
        {
            pubSuccessBbox.publish(objBboxTracked);
        }
        if (pubFailedBbox.getNumSubscribers() != 0)
        {
            pubFailedBbox.publish(objBboxPredicted);
        }
    }

    
    void trajs2Markers(const std::vector<ObjTraj> &trajs, visualization_msgs::MarkerArray &mks, const std_msgs::Header &header)
    {
        mks.markers.clear();
        visualization_msgs::Marker mk;
        mk.header = header;
        mk.header.frame_id = "map";
        mk.lifetime = ros::Duration(1);
        mk.type = visualization_msgs::Marker::LINE_STRIP;
        mk.color.r = 1;
        mk.color.g = 0;
        mk.color.b = 0;
        mk.color.a = 0.7;
        mk.scale.x = 0.05;
        mk.scale.y = 0.05;
        mk.frame_locked = false;
        mk.action = visualization_msgs::Marker::ADD;
        for (auto traj : trajs)
        {
            mk.id = traj.id;
            mk.points.clear();
            if (traj.points.size() <= 2)
            {
                continue;
            }
            for (auto point : traj.points)
            {
                geometry_msgs::Point m_point;
                m_point.x = point.x();
                m_point.y = point.y();
                m_point.z = point.z();
                mk.points.emplace_back(m_point);
            }
            mks.markers.emplace_back(mk);
        }
        return;
    }
    void velos2Markers(const std::vector<ObjTraj> &trajs, visualization_msgs::MarkerArray &velo_arrow_array,
            visualization_msgs::MarkerArray &velo_text_array, const std_msgs::Header &header)
    {
        for (auto traj : trajs)
        {
            if (traj.velos.empty() || traj.points.empty() || traj.orientations.empty()) continue;
            
            const auto& latest_vel = traj.velos.back();
            const auto& latest_pos = traj.points.back();
            const auto& latest_orient = traj.orientations.back();
            
            visualization_msgs::Marker arrow;
            arrow.header = header;
            arrow.header.frame_id = "map";
            arrow.ns = "velocity_arrows";
            arrow.id = traj.id;
            arrow.type = visualization_msgs::Marker::ARROW;
            arrow.action = visualization_msgs::Marker::ADD;
            arrow.lifetime = ros::Duration(1.0);
            
            geometry_msgs::Point start, end;
            start.x = latest_pos.x();
            start.y = latest_pos.y();
            start.z = latest_pos.z();
            
            Eigen::Vector3d world_vel = latest_orient * latest_vel;
            
            double scale = 0.5;
            end.x = start.x + world_vel.x() * scale;
            end.y = start.y + world_vel.y() * scale;
            end.z = start.z + world_vel.z() * scale;
            
            arrow.points.push_back(start);
            arrow.points.push_back(end);
            
            double speed = latest_vel.norm();
            arrow.color.r = 0.0;
            arrow.color.g = 0.0;
            arrow.color.b = 1.0;
            arrow.color.a = 1.0;
            
            arrow.scale.x = 0.4;
            arrow.scale.y = 0.8;
            arrow.scale.z = 1.0; 

            velo_arrow_array.markers.push_back(arrow);
            
            visualization_msgs::Marker text;
            text.header = header;
            text.header.frame_id = "map";
            text.ns = "velocity_text";
            text.id = traj.id + 1000;
            text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text.action = visualization_msgs::Marker::ADD;
            text.lifetime = ros::Duration(1.0);
            
            text.pose.position.x = end.x;
            text.pose.position.y = end.y;
            text.pose.position.z = end.z;
            
            std::stringstream ss;
            ss << "velo:" << speed << "m/s";
            
            text.text = ss.str();
            
            text.color.r = 0.0;
            text.color.g = 0.0;
            text.color.b = 1.0;
            text.color.a = 1.0;
            
            text.scale.z = 1.0;
            
            velo_text_array.markers.push_back(text);
        }

    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fgo_mot");

    ObjectTracker OT;

    ROS_INFO("\033[1;32m---->\033[0m object tacking Started.");

    ros::spin();
    return 0;
}