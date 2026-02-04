// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.
//   Dong-Uk Seo, Hyungtae Lim and Seungjae Lee, PaGO-LOAM Robust Ground-Optimized LiDAR Odometry

#include "utils/common.h"
#include "utils/math_tools.h"
#include "patchwork.hpp"
#include "CVC_cluster.hpp"
#include "LShapeFit.hpp"
#include "LShapeDetect.hpp"

class ImageProjection
{
private:
    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    ros::Subscriber subImu;
    ros::Subscriber subBbArray;
    ros::Subscriber subBb2dArray;

    ros::Publisher pubFullCloud;
    // ros::Publisher pubFullInfoCloud;

    ros::Publisher pubGroundCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubobjectCloud;
    ros::Publisher pubCloudinCamera;
    // ros::Publisher pubSegmentedCloudInfo;
    // ros::Publisher pubOutlierCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Publisher pubBboxArray;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;

    pcl::PointCloud<PointType>::Ptr fullCloud; // projected velodyne raw cloud, but saved in the form of 1-D matrix
    // pcl::PointCloud<PointType>::Ptr fullInfoCloud; // same as fullCloud, but with intensity - range

    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr nongroundCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;

    pcl::PointCloud<PointType>::Ptr CloudinCamera;

    // pcl::PointCloud<PointType>::Ptr outlierCloud;
    // pcl::PointCloud<PointType>::Ptr cloudCluster;//初始化

    PointType nanPoint; // fill in fullCloud at each iteration

    cv::Mat rangeMat;  // range matrix for range image
    cv::Mat labelMat;  // label matrix for segmentaiton marking
    cv::Mat groundMat; // ground matrix for ground cloud marking  1：地面点 2：非地面点 0：初始值（此处无点云）
    int labelCount;
    std::mutex mBuf;

    float startOrientation;
    float endOrientation;

    fgo_mot::cloud_info cloudInfo;

    std_msgs::Header cloudHeader;

    sensor_msgs::PointCloud2 currentCloudMsg;
    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    std::vector<sensor_msgs::ImuConstPtr> imuBuf;
    int idxImu;
    double currentTimeImu;

    Eigen::Vector3d gyr_0;
    Eigen::Quaterniond qIMU;
    Eigen::Vector3d rIMU;
    bool firstImu = false;
    string imu_topic;

    double timeScanNext;
    double runtime = 0;

    int N_SCANS;
    int Horizon_SCANS;
    double qlb0, qlb1, qlb2, qlb3;
    Eigen::Quaterniond q_lb;

    string frame_id;
    string lidarInputTopic;

    string patchworkConfigFile;

    boost::shared_ptr<PatchWork<PointType>> PatchworkGroundSeg;

    float ang_res_x;
    float ang_res_y;
    float ang_bottom;
    float sensorMinimumRange;
    int groundScanInd;
    int segmentation_strategy;

    std::map<long, int> imgIndex;
    std::map<int, vector<int>> objIndexMap;

    ClusterVectorPtr cluster_vector_ptr_curr;
    map<int, ObjectCluster> cluster_vector_ptr_map;
    nav_msgs::Odometry odom;
    nav_msgs::Path path;
    std::deque<jsk_recognition_msgs::BoundingBoxArray> objBboxVecBuf;
    std::deque<fgo_mot::boundingBox2DArray> objBbox2DVecBuf;
    jsk_recognition_msgs::BoundingBoxArray currentBboxArragMsg;
    jsk_recognition_msgs::BoundingBoxArray stableBboxArragMsg;
    fgo_mot::boundingBox2DArray bbox2dArragMsg;
    int minBboxpointnum;
    double minBboxscore;
    std::string output_path;
    fstream temp_file;

    int sequence;
    int frame_ind;
    int seq_ind;

    std::vector<double> imageProjection_timer;

    bool bbox_2d_is_avaliable = false;

public:
    ImageProjection() : nh("~")
    {
        initializeParameters();
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(lidarInputTopic, 10000, &ImageProjection::cloudHandler, this);
        subImu = nh.subscribe<sensor_msgs::Imu>(imu_topic, 500000, &ImageProjection::imuHandler, this);
        subBbArray = nh.subscribe<jsk_recognition_msgs::BoundingBoxArray>("/detection", 10000, &ImageProjection::BbArrayHandler, this);
        subBb2dArray = nh.subscribe<fgo_mot::boundingBox2DArray>("/detection2d_array", 10000, &ImageProjection::Bbox2dArrayHandler, this);

        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2>("/full_cloud_projected", 1);
        // pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_info", 1);
        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2>("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2>("/nonground_cloud", 1);
        pubCloudinCamera = nh.advertise<sensor_msgs::PointCloud2>("/Camera_cloud", 1);
        // pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info> ("/segmented_cloud_info", 1);
        // pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2> ("/outlier_cloud", 1);
        // pubCluster = nh.advertise<sensor_msgs::PointCloud2> ("/cluster", 1);
        pubLaserCloudInfo = nh.advertise<fgo_mot::cloud_info>("/image_project/cloud_info", 10000);
        pubBboxArray = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/image_project/object_bbox_array", 1);
        pubobjectCloud = nh.advertise<sensor_msgs::PointCloud2>("/objectcloud", 1);

        allocateMemory();
        resetParameters();

        // temp_file.open(output_path + "/temp_file_img.txt", ios::out | ios::trunc);
    }
    virtual ~ImageProjection()
    {
        std::pair<double, double> mean_std_timer;
        mean_std_timer = calVarStdev(imageProjection_timer);
        printf("\033[1;32mimageProjection   Time[ms] : %0.2f ± %0.2f, %0.0f FPS. \033[0m \n", mean_std_timer.first, mean_std_timer.second, floor(1000.0 / mean_std_timer.first));
    }

    void initializeParameters()
    {
        if (!getParameter("/common/frame_id", frame_id))
        {
            ROS_WARN("frame_id not set, use default value: fgo_mot");
            frame_id = "fgo_mot";
        }

        if (!getParameter("/common/imu_topic", imu_topic))
        {
            ROS_WARN("imu_topic not set, use default value: /imu/data");
            imu_topic = "/imu/data";
        }

        if (!getParameter("/common/line_num", N_SCANS))
        {
            ROS_WARN("line_num not set, use default value: 64");
            N_SCANS = 64;
        }

        if (!getParameter("/common/horizon_scans", Horizon_SCANS))
        {
            ROS_WARN("horizon_scans not set, use default value: 2000");
            Horizon_SCANS = 2000;
        }

        // extrinsic parameters
        if (!getParameter("/common/ql2b_w", qlb0))
        {
            ROS_WARN("ql2b_w not set, use default value: 1");
            qlb0 = 1;
        }

        if (!getParameter("/common/ql2b_x", qlb1))
        {
            ROS_WARN("ql2b_x not set, use default value: 0");
            qlb1 = 0;
        }

        if (!getParameter("/common/ql2b_y", qlb2))
        {
            ROS_WARN("ql2b_y not set, use default value: 0");
            qlb2 = 0;
        }

        if (!getParameter("/common/ql2b_z", qlb3))
        {
            ROS_WARN("ql2b_z not set, use default value: 0");
            qlb3 = 0;
        }

        if (!getParameter("/image_projection/patchwork_config_file", patchworkConfigFile))
        {
            ROS_WARN("patchworkConfigFile not set, use default value: config.yaml");
            patchworkConfigFile = "src/LIOMOT/config/patchwork_params.yaml";
        }

        if (!getParameter("/common/lidar_input_topic", lidarInputTopic))
        {
            ROS_WARN("lidarInputTopic not set, use default value: /velodyne_points");
            lidarInputTopic = "/velodyne_points";
        }

        if (!getParameter("/image_projection/ang_res_x", ang_res_x))
        {
            ROS_WARN("ang_res_x not set, use default value: 0.2");
            ang_res_x = 0.2;
        }

        if (!getParameter("/image_projection/ang_res_y", ang_res_y))
        {
            ROS_WARN("ang_res_y not set, use default value: 0.427");
            ang_res_y = 0.427;
        }

        if (!getParameter("/image_projection/ang_bottom", ang_bottom))
        {
            ROS_WARN("ang_bottom not set, use default value: 24.9");
            ang_bottom = 24.9;
        }
        if (!getParameter("/image_projection/sensorMinimumRange", sensorMinimumRange))
        {
            ROS_WARN("sensorMinimumRange not set, use default value: 0.3");
            sensorMinimumRange = 0.3;
        }
        if (!getParameter("/image_projection/groundScanInd", groundScanInd))
        {
            ROS_WARN("groundScanInd not set, use default value: 25");
            groundScanInd = 25;
        }
        if (!getParameter("/image_projection/minBboxpointnum", minBboxpointnum))
        {
            ROS_WARN("minBboxpointnum not set, use default value: 20");
            minBboxpointnum = 20;
        }
        if (!getParameter("/image_projection/minBboxscore", minBboxscore))
        {
            ROS_WARN("minBboxsocre not set, use default value: 0.7");
            minBboxscore = 0.7;
        }
        if (!getParameter("/object_tracker/output_path", output_path))
        {
            ROS_WARN("output_path not set, use default value: output_path");
            output_path = "output_path";
        }
        if (!getParameter("/image_projection/segmentation_strategy", segmentation_strategy))
        {
            ROS_WARN("segmentation_strategy not set, use default value: 0");
            segmentation_strategy = 0;
        }
        if (!getParameter("/common/sequence", sequence))
        {
            ROS_WARN("sequence not set, use default value: 0");
            sequence = 0;
        }

        q_lb = Eigen::Quaterniond(qlb0, qlb1, qlb2, qlb3);
        PatchworkGroundSeg.reset(new PatchWork<PointType>(patchworkConfigFile));

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;
    }

    void allocateMemory()
    {
        cluster_vector_ptr_curr = std::make_shared<ClusterVector>();

        laserCloudIn.reset(new pcl::PointCloud<PointType>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        // fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        nongroundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        CloudinCamera.reset(new pcl::PointCloud<PointType>());
        // outlierCloud.reset(new pcl::PointCloud<PointType>());

        // cloudCluster.reset(new pcl::PointCloud<pcl::PointXYZRGB>());

        fullCloud->points.resize(N_SCANS * Horizon_SCANS);
        // fullInfoCloud->points.resize(N_SCANS*Horizon_SCANS);

        cloudInfo.startRingIndex.assign(N_SCANS, 0);
        cloudInfo.endRingIndex.assign(N_SCANS, 0);

        cloudInfo.groundFlag.assign(N_SCANS * Horizon_SCANS, 0);
        cloudInfo.pointColInd.assign(N_SCANS * Horizon_SCANS, 0);
        cloudInfo.pointRange.assign(N_SCANS * Horizon_SCANS, 0);

        qIMU = Eigen::Quaterniond::Identity();
        rIMU = Eigen::Vector3d::Zero();

        idxImu = 0;
        currentTimeImu = -1;
    }

    void resetParameters()
    {
        cluster_vector_ptr_curr->clear();
        cluster_vector_ptr_map.clear();
        laserCloudIn->clear();
        // fullCloud->clear();
        groundCloud->clear();
        nongroundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        CloudinCamera->clear();
        imgIndex.clear();
        objIndexMap.clear();
        // outlierCloud->clear();
        // cloudCluster->clear();

        rangeMat = cv::Mat(N_SCANS, Horizon_SCANS, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCANS, Horizon_SCANS, CV_8S, cv::Scalar::all(0));

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        // std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);

        cloudInfo.startRingIndex.assign(N_SCANS, 0);
        cloudInfo.endRingIndex.assign(N_SCANS, 0);

        cloudInfo.groundFlag.assign(N_SCANS * Horizon_SCANS, 0);
        cloudInfo.pointColInd.assign(N_SCANS * Horizon_SCANS, 0);
        cloudInfo.pointRange.assign(N_SCANS * Horizon_SCANS, 0);

        cloudInfo.objIndexVec.clear();
        cloudInfo.objArray.clear();

        qIMU = Eigen::Quaterniond::Identity();
        rIMU = Eigen::Vector3d::Zero();

        currentBboxArragMsg.boxes.clear();
        stableBboxArragMsg.boxes.clear();
        currentBboxArragMsg.boxes.resize(0);
        stableBboxArragMsg.boxes.resize(0);

        bbox2dArragMsg.boxes.clear();
        bbox2dArragMsg.boxes.resize(0);
    }

    template <typename PointT1, typename PointT2>
    void removeClosedPointCloud(const pcl::PointCloud<PointT1> &cloud_in, pcl::PointCloud<PointT2> &cloud_out, float thres)
    {

        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());

        size_t j = 0;
        frame_ind = cloud_in.points.front().frame_id;
        seq_ind = cloud_in.points.front().seq_id;

        // std::cout<< "======================img_frame_id : " << frame_ind << "  ==================" << std::endl;
        for (size_t i = 0; i < cloud_in.points.size(); ++i)
        {
            if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
                continue;

            // if(cloud_in.points[i].x < 0) continue;

            if (isnan(cloud_in.points[i].x) || isnan(cloud_in.points[i].y) || isnan(cloud_in.points[i].z))
                continue;
            if (isinf(cloud_in.points[i].x) || isinf(cloud_in.points[i].y) || isinf(cloud_in.points[i].z))
                continue;
            cloud_out.points[j].x = cloud_in.points[i].x;
            cloud_out.points[j].y = cloud_in.points[i].y;
            cloud_out.points[j].z = cloud_in.points[i].z;
            cloud_out.points[j].intensity = cloud_in.points[i].intensity;
            cloud_out.points[j].id = cloud_in.points[i].id;
            cloud_out.points[j].score = cloud_in.points[i].score;
            cloud_out.points[j].alpha = cloud_in.points[i].alpha;
            cloud_out.points[j].label = UNKNOWN;
            cloud_out.points[j].frame_id = cloud_in.points[i].frame_id;
            cloud_out.points[j].seq_id = cloud_in.points[i].seq_id;
            j++;
        }
        if (j != cloud_in.points.size())
        {
            cloud_out.points.resize(j);
        }

        cloud_out.height = 1;
        cloud_out.width = static_cast<uint32_t>(j);
        cloud_out.is_dense = true;
    }

    void BbArrayHandler(const jsk_recognition_msgs::BoundingBoxArrayConstPtr &msg_Bbarray)
    {
        if (segmentation_strategy != 0)
        {
            jsk_recognition_msgs::BoundingBoxArray objBboxVec;
            objBboxVec.header = msg_Bbarray->header;
            objBboxVec.header.frame_id = "fgo_mot";
            for (int i = 0; i < msg_Bbarray->boxes.size(); ++i)
            {
                jsk_recognition_msgs::BoundingBox bbox = msg_Bbarray->boxes[i];
                bbox.header.frame_id = "fgo_mot";
                objBboxVec.boxes.push_back(bbox);
            }
            objBboxVecBuf.push_back(objBboxVec);
        }
    }

    void Bbox2dArrayHandler(const fgo_mot::boundingBox2DArrayConstPtr &msg_Bbarray)
    {
        if (segmentation_strategy != 0)
        {
            fgo_mot::boundingBox2DArray objBboxVec;
            objBboxVec.header = msg_Bbarray->header;
            objBboxVec.header.frame_id = "fgo_mot";
            for (int i = 0; i < msg_Bbarray->boxes.size(); ++i)
            {
                fgo_mot::boundingBox2D bbox = msg_Bbarray->boxes[i];
                bbox.header.frame_id = "fgo_mot";
                objBboxVec.boxes.push_back(bbox);
            }
            objBbox2DVecBuf.push_back(objBboxVec);
        }
    }

    bool copyPointCloud()
    {
        int tmpIdx = 0;
        if (idxImu > 0)
            tmpIdx = idxImu - 1;
        if (0)
        {
            if (imuBuf.empty() || imuBuf[tmpIdx]->header.stamp.toSec() > timeScanNext)
            {
                ROS_WARN("Waiting for IMU data ...");
                return false;
            }
        }

        pcl::PointCloud<PointXYZIKITTI> kitti_cloud_input;

        pcl::fromROSMsg(currentCloudMsg, kitti_cloud_input);
        std::vector<int> indices;
        removeClosedPointCloud(kitti_cloud_input, *laserCloudIn, sensorMinimumRange);

        return true;
    }

    void imuCompensation()
    {
        if (firstImu)
            processIMU(timeScanNext);
        if (isnan(qIMU.w()) || isnan(qIMU.x()) || isnan(qIMU.y()) || isnan(qIMU.z()))
        {
            qIMU = Eigen::Quaterniond::Identity();
        }
    }

    void findStartEndAngle()
    {
        // start and end orientation of this cloud
        cloudInfo.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
        cloudInfo.endOrientation = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                                          laserCloudIn->points[laserCloudIn->points.size() - 1].x) +
                                   2 * M_PI;
        if (cloudInfo.endOrientation - cloudInfo.startOrientation > 3 * M_PI)
        {
            cloudInfo.endOrientation -= 2 * M_PI;
        }
        else if (cloudInfo.endOrientation - cloudInfo.startOrientation < M_PI)
            cloudInfo.endOrientation += 2 * M_PI;
        cloudInfo.orientationDiff = cloudInfo.endOrientation - cloudInfo.startOrientation;
    }

    double groundSegAndCluster()
    {
        double time_taken;
        PatchworkGroundSeg->estimate_ground(*laserCloudIn, *groundCloud, *nongroundCloud, time_taken);

        std::map<int, std::vector<int>> id_nongroundCloudindex;
        cluster_vector_ptr_curr->clear();
        if (segmentation_strategy == 0)
        {
            vector<float> param(5, 0);
            param[0] = 2;
            param[1] = 0.5;
            param[2] = 1;
            param[3] = 1.0;
            param[4] = 50;

            CVC Cluster(param);
            std::vector<PointAPR> capr;
            Cluster.calculateAPR(*nongroundCloud, capr);
            std::unordered_map<int, Voxel> hash_table;
            Cluster.build_hash_table(capr, hash_table);
            vector<int> cluster_indices;
            cluster_indices = Cluster.cluster(hash_table, capr);
            vector<int> cluster_id;
            Cluster.most_frequent_value(cluster_indices, cluster_id);
            ClusterVectorPtr cluster_vector_ptr = std::make_shared<ClusterVector>();
            cluster_vector_ptr->resize(cluster_id.size() + 1);
            cluster_vector_ptr_curr->resize(cluster_id.size() + 1);
            int id = 1;
            for (int j = 0; j < cluster_id.size(); ++j)
            {
                ObjectCluster clusterj;
                for (int i = 0; i < cluster_indices.size(); ++i)
                {
                    if (cluster_indices[i] == cluster_id[j]) // nongroundCloud index map cluster id
                    {
                        nongroundCloud->points[i].id = id;
                        PointType pt;
                        pt.x = nongroundCloud->points[i].x;
                        pt.y = nongroundCloud->points[i].y;
                        pt.z = nongroundCloud->points[i].z;
                        pt.intensity = nongroundCloud->points[i].intensity;
                        pt.label = nongroundCloud->points[i].label;
                        pt.id = id;
                        clusterj.cluster_points_.push_back(pt);
                        clusterj.id = id;
                    }
                }
                (*cluster_vector_ptr)[id] = clusterj;
                // cluster_vector_ptr->push_back(clusterj);
                id++;
            }

            //! filter id = 0
            objectCluster(cluster_vector_ptr);

            ObjectClusterRefine(cluster_vector_ptr, cluster_vector_ptr_curr);
        }
        // partially dependent on priori
        else if (segmentation_strategy == 1)
        {
            ClusterVectorPtr cluster_vector_ptr_priori = std::make_shared<ClusterVector>();
            cluster_vector_ptr_priori->resize(currentBboxArragMsg.boxes.size() + 1);

            vector<float> param(5, 0);
            param[0] = 2;
            param[1] = 0.5;
            param[2] = 1;
            param[3] = 1.0;
            param[4] = 50;

            CVC Cluster(param);
            std::vector<PointAPR> capr;
            Cluster.calculateAPR(*nongroundCloud, capr);
            std::unordered_map<int, Voxel> hash_table;
            Cluster.build_hash_table(capr, hash_table);
            vector<int> cluster_indices;
            cluster_indices = Cluster.cluster(hash_table, capr);
            vector<int> cluster_id;
            Cluster.most_frequent_value(cluster_indices, cluster_id);
            ClusterVectorPtr cluster_vector_ptr = std::make_shared<ClusterVector>();
            cluster_vector_ptr->resize(cluster_id.size() + 1);
            cluster_vector_ptr_curr->resize(cluster_id.size() + 1);
            int id = 1;

            for (int j = 0; j < cluster_id.size(); ++j)
            {
                ObjectCluster clusterj;
                for (int i = 0; i < cluster_indices.size(); ++i)
                {
                    if (cluster_indices[i] == cluster_id[j]) // nongroundCloud index map cluster id
                    {
                        nongroundCloud->points[i].id = id;
                        PointType pt;
                        pt.x = nongroundCloud->points[i].x;
                        pt.y = nongroundCloud->points[i].y;
                        pt.z = nongroundCloud->points[i].z;
                        pt.intensity = nongroundCloud->points[i].intensity;
                        pt.label = nongroundCloud->points[i].label;
                        pt.id = id;
                        clusterj.cluster_points_.push_back(pt);
                        clusterj.id = id;
                    }
                }
                (*cluster_vector_ptr)[id] = clusterj;
                // cluster_vector_ptr->push_back(clusterj);
                id++;
            }

            //! filter id = 0
            objectCluster(cluster_vector_ptr);

            // ObjectClusterRefine(cluster_vector_ptr, cluster_vector_ptr_curr);
            for (int j = 0; j < currentBboxArragMsg.boxes.size(); j++)
            {
                int id = currentBboxArragMsg.boxes[j].label;

                if (currentBboxArragMsg.boxes[j].value > minBboxscore)
                {
                    float min_x = std::numeric_limits<float>::max();
                    float max_x = -std::numeric_limits<float>::max();
                    float min_y = std::numeric_limits<float>::max();
                    float max_y = -std::numeric_limits<float>::max();
                    float min_z = std::numeric_limits<float>::max();
                    float max_z = -std::numeric_limits<float>::max();
                    (*cluster_vector_ptr_priori)[id].id = id;
                    (*cluster_vector_ptr_priori)[id].bounding_box_ = currentBboxArragMsg.boxes[j];
                    if (bbox_2d_is_avaliable)
                        (*cluster_vector_ptr_priori)[id].bounding_box2d_ = bbox2dArragMsg.boxes[j];
                    (*cluster_vector_ptr_priori)[id].box_.size.width = (*cluster_vector_ptr_priori)[id].bounding_box_.dimensions.x;
                    (*cluster_vector_ptr_priori)[id].box_.size.height = (*cluster_vector_ptr_priori)[id].bounding_box_.dimensions.y;
                    (*cluster_vector_ptr_priori)[id].box_.center.x = (*cluster_vector_ptr_priori)[id].bounding_box_.pose.position.x;
                    (*cluster_vector_ptr_priori)[id].box_.center.y = (*cluster_vector_ptr_priori)[id].bounding_box_.pose.position.y;
                    (*cluster_vector_ptr_priori)[id].box_initial = true;
                    (*cluster_vector_ptr_priori)[id].max_x_ = (*cluster_vector_ptr_priori)[id].bounding_box_.pose.position.x + (*cluster_vector_ptr_priori)[id].bounding_box_.dimensions.x / 2.0;
                    (*cluster_vector_ptr_priori)[id].max_y_ = (*cluster_vector_ptr_priori)[id].bounding_box_.pose.position.y + (*cluster_vector_ptr_priori)[id].bounding_box_.dimensions.y / 2.0;
                    (*cluster_vector_ptr_priori)[id].max_z_ = (*cluster_vector_ptr_priori)[id].bounding_box_.pose.position.z + (*cluster_vector_ptr_priori)[id].bounding_box_.dimensions.z / 2.0;
                    (*cluster_vector_ptr_priori)[id].min_x_ = (*cluster_vector_ptr_priori)[id].bounding_box_.pose.position.x - (*cluster_vector_ptr_priori)[id].bounding_box_.dimensions.x / 2.0;
                    (*cluster_vector_ptr_priori)[id].min_y_ = (*cluster_vector_ptr_priori)[id].bounding_box_.pose.position.y - (*cluster_vector_ptr_priori)[id].bounding_box_.dimensions.y / 2.0;
                    (*cluster_vector_ptr_priori)[id].min_z_ = (*cluster_vector_ptr_priori)[id].bounding_box_.pose.position.z - (*cluster_vector_ptr_priori)[id].bounding_box_.dimensions.z / 2.0;
                }
            }
            ObjectClusterRefineFromPriori(cluster_vector_ptr_priori, cluster_vector_ptr, cluster_vector_ptr_curr);
        }
        // completely dependent on priori
        else if (segmentation_strategy == 2)
        {
            cluster_vector_ptr_curr->resize(currentBboxArragMsg.boxes.size() + 1);

            for (int i = 0; i < nongroundCloud->size(); i++)
            {
                if (nongroundCloud->points[i].id > 0 && nongroundCloud->points[i].score > minBboxscore)
                {
                    int id = nongroundCloud->points[i].id; // 1，2，3，....
                    id_nongroundCloudindex[id].push_back(i);
                    (*cluster_vector_ptr_curr)[id].cluster_points_.push_back(nongroundCloud->points[i]);
                    (*cluster_vector_ptr_curr)[id].alpha = nongroundCloud->points[i].alpha;
                }
                else
                {
                    int id = nongroundCloud->points[i].id;
                    nongroundCloud->points[i].id = 0;
                    (*cluster_vector_ptr_curr)[id].cluster_points_.resize(0);
                }
            }

            for (int id = 1; id < cluster_vector_ptr_curr->size(); id++)
            {
                // cluster by point.id, which is from perior
                (*cluster_vector_ptr_curr)[id].id = id;
                for (int j = 0; j < currentBboxArragMsg.boxes.size(); j++)
                {
                    if (currentBboxArragMsg.boxes[j].label == id && currentBboxArragMsg.boxes[j].value > minBboxscore)
                    {
                        (*cluster_vector_ptr_curr)[id].bounding_box_ = currentBboxArragMsg.boxes[j];
                        if (bbox_2d_is_avaliable)
                            (*cluster_vector_ptr_curr)[id].bounding_box2d_ = bbox2dArragMsg.boxes[j];
                        break;
                    }
                }
                if ((*cluster_vector_ptr_curr)[id].cluster_points_.size() < minBboxpointnum)
                {
                    for (int j = 0; j < id_nongroundCloudindex[id].size(); j++)
                    {
                        nongroundCloud->points[j].id = 0;
                    }
                    (*cluster_vector_ptr_curr)[id].cluster_points_.resize(0);
                    continue;
                }
                else
                {
                    (*cluster_vector_ptr_curr)[id].id = id;
                    for (int j = 0; j < currentBboxArragMsg.boxes.size(); j++)
                    {
                        if (currentBboxArragMsg.boxes[j].label == id && currentBboxArragMsg.boxes[j].value > minBboxscore)
                        {
                            (*cluster_vector_ptr_curr)[id].bounding_box_ = currentBboxArragMsg.boxes[j];
                            if (bbox_2d_is_avaliable)
                                (*cluster_vector_ptr_curr)[id].bounding_box2d_ = bbox2dArragMsg.boxes[j];
                            break;
                        }
                    }
                }
            }
            objectClusterByBoundingBox(cluster_vector_ptr_curr);
        }
        else
        {
        }

        CloudinCamera->points.clear();
        // for (int i = 0; i < nongroundCloud->points.size(); i++)
        // {
        //     if (nongroundCloud->points[i].id > 0 || nongroundCloud->points[i].x < 0)
        //         continue;
        //     PointType pin = nongroundCloud->points[i];
        //     PointType pout;
        //     if (velo_to_img(pin, pout))
        //     {
        //         CloudinCamera->points.push_back(pin);
        //     }
        // }
        // for (int i = 0; i < groundCloud->points.size(); i++)
        // {
        //     if (groundCloud->points[i].x < 0)
        //         continue;
        //     PointType pin = groundCloud->points[i];
        //     PointType pout;
        //     if (velo_to_img(pin, pout))
        //     {
        //         CloudinCamera->points.push_back(pin);
        //     }
        // }

        for (int i = 0; i < laserCloudIn->points.size(); i++)
        {
            if (laserCloudIn->points[i].x < 0)
                continue;
            PointType pin = laserCloudIn->points[i];
            PointType pout;
            if (velo_to_img(pin, pout))
            {
                CloudinCamera->points.push_back(pin);
            }
        }
        // std:cout << "cloud in camera size: " << CloudinCamera->points.size() << std::endl;

        

        return 0;
    }

    bool velo_to_img(const PointType &pin, PointType &pout)
    {
        pout = pin;
        Eigen::Matrix<double, 1, 4> mat;
        Eigen::Matrix4d R0;
        Eigen::Matrix4d vtc_mat;
        Eigen::Matrix<double, 3, 4> P2;
        mat << pin.x, pin.y, pin.z, 1.;

        if (sequence == 0)
        {
            // sequence 0000
            P2 << 7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01,
                0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01,
                0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03;

            R0 << 9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03, 0.,
                -9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03, 0.,
                7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01, 0.,
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00;

            vtc_mat << 7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04, -4.069766000000e-03,
                1.480249000000e-02, 7.280733000000e-04, -9.998902000000e-01, -7.631618000000e-02,
                9.998621000000e-01, 7.523790000000e-03, 1.480755000000e-02, -2.717806000000e-01,
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00;
        }
        else if (sequence == 20)
        {
            // sequence 0020
            P2 << 7.188560e+02, 0.000000e+00, 6.071928e+02, 4.538225e+01,
                0.000000e+00, 7.188560e+02, 1.852157e+02, -1.130887e-01,
                0.000000e+00, 0.000000e+00, 1.000000e+00, 3.779761e-03;

            R0 << 0.9999454, 0.00725913, -0.00751955, 0.,
                -0.00729221, 0.99996382, -0.00438173, 0.,
                0.00748747, 0.00443632, 0.99996209, 0.,
                0., 0., 0., 1.;

            vtc_mat << 7.96751399e-03, -9.99967873e-01, -8.46226409e-04, -1.37776900e-02,
                -2.77105300e-03, 8.24171002e-04, -9.99995828e-01, -5.54211698e-02,
                9.99964416e-01, 7.96982460e-03, -2.76439707e-03, -2.91858912e-01,
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00;
        }
        else
        {
            // TODO
            P2 << 7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01,
                0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01,
                0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03;

            R0 << 9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03, 0.,
                -9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03, 0.,
                7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01, 0.,
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00;

            vtc_mat << 7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04, -4.069766000000e-03,
                1.480249000000e-02, 7.280733000000e-04, -9.998902000000e-01, -7.631618000000e-02,
                9.998621000000e-01, 7.523790000000e-03, 1.480755000000e-02, -2.717806000000e-01,
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00;
        }

        vtc_mat = R0 * vtc_mat;

        Eigen::Matrix<double, 3, 4> normal = vtc_mat.block<3, 4>(0, 0);
        Eigen::Matrix<double, 3, 1> transformed_mat;
        transformed_mat = normal * mat.transpose();

        mat << transformed_mat(0, 0), transformed_mat(1, 0), transformed_mat(2, 0), 1.;

        Eigen::Matrix<double, 1, 3> img_pts;
        img_pts = mat * P2.transpose();

        pout.x = img_pts(0, 0) / img_pts(0, 2);
        pout.y = img_pts(0, 1) / img_pts(0, 2);
        pout.z = 1.0;

        if (pout.x > 0 && pout.x < 1242 && pout.y > 0 && pout.y < 375)
            return true;
        else
            return false;
    }

    void ObjectClusterRefine(ClusterVectorPtr &cluster_vector_ptr, ClusterVectorPtr &final_cluster_vector_ptr)
    {
        if (cluster_vector_ptr->size() <= 1)
            return;

        vector<bool> match_flag(cluster_vector_ptr->size(), false);
        for (int i = 1; i < cluster_vector_ptr->size() - 1; i++)
        {
            if ((*cluster_vector_ptr)[i].box_.size.height > 6 || (*cluster_vector_ptr)[i].box_.size.width > 6)
                continue;
            if (match_flag[i])
                continue;
            for (int j = i + 1; j < cluster_vector_ptr->size(); j++)
            {
                if ((*cluster_vector_ptr)[j].box_.size.height > 6 || (*cluster_vector_ptr)[j].box_.size.width > 6)
                {
                    match_flag[j] = true;
                    continue;
                }
                if ((*cluster_vector_ptr)[i].min_z_ - (*cluster_vector_ptr)[j].max_z_ > 0.5 ||
                    (*cluster_vector_ptr)[j].min_z_ - (*cluster_vector_ptr)[i].max_z_ > 0.5)
                    continue;
                if ((*cluster_vector_ptr)[i].box_initial && (*cluster_vector_ptr)[j].box_initial)
                {
                    if (computeIou((*cluster_vector_ptr)[i].box_, (*cluster_vector_ptr)[j].box_, true) > 0)
                    {
                        match_flag[j] = true;
                        clusterMerge((*cluster_vector_ptr)[i], (*cluster_vector_ptr)[j]);
                    }
                }
            }
            if ((*cluster_vector_ptr)[i].box_.size.height > 6 || (*cluster_vector_ptr)[i].box_.size.width > 6)
                continue;

            if (!(*cluster_vector_ptr)[i].object_classify())
            {
                (*cluster_vector_ptr)[i].id = 0;
                for (int k = 0; k < (*cluster_vector_ptr)[i].cluster_points_.size(); ++k)
                {
                    (*cluster_vector_ptr)[i].cluster_points_[k].id = 0;
                }
            }
            (*final_cluster_vector_ptr)[i] = (*cluster_vector_ptr)[i];

            cluster_vector_ptr_map[(*cluster_vector_ptr)[i].id] = (*cluster_vector_ptr)[i];
        }
    }

    void ObjectClusterRefineFromPriori(ClusterVectorPtr &cluster_vector_ptr_priori, ClusterVectorPtr &cluster_vector_ptr, ClusterVectorPtr &final_cluster_vector_ptr)
    {
        if (cluster_vector_ptr->size() <= 1)
            return;

        for (int i = 1; i < cluster_vector_ptr_priori->size(); i++)
        {
            if ((*cluster_vector_ptr_priori)[i].box_.size.height > 6 || (*cluster_vector_ptr_priori)[i].box_.size.width > 6)
                continue;
            for (int j = 0; j < cluster_vector_ptr->size(); j++)
            {
                if ((*cluster_vector_ptr)[j].box_.size.height > 6 || (*cluster_vector_ptr)[j].box_.size.width > 6)
                {
                    continue;
                }
                if ((*cluster_vector_ptr_priori)[i].min_z_ - (*cluster_vector_ptr)[j].max_z_ > 0.5 ||
                    (*cluster_vector_ptr)[j].min_z_ - (*cluster_vector_ptr_priori)[i].max_z_ > 0.5)
                    continue;
                if ((*cluster_vector_ptr_priori)[i].box_initial && (*cluster_vector_ptr)[j].box_initial)
                {
                    if (computeIou((*cluster_vector_ptr_priori)[i].box_, (*cluster_vector_ptr)[j].box_, false) > 0)
                    {
                        clusterMerge((*cluster_vector_ptr_priori)[i], (*cluster_vector_ptr)[j]);
                    }
                }
            }
            if ((*cluster_vector_ptr_priori)[i].box_.size.height > 6 || (*cluster_vector_ptr_priori)[i].box_.size.width > 6)
                continue;

            if (!(*cluster_vector_ptr_priori)[i].object_classify())
            {
                (*cluster_vector_ptr_priori)[i].id = 0;
                for (int k = 0; k < (*cluster_vector_ptr_priori)[i].cluster_points_.size(); ++k)
                {
                    (*cluster_vector_ptr_priori)[i].cluster_points_[k].id = 0;
                }
            }
            (*final_cluster_vector_ptr)[i] = (*cluster_vector_ptr_priori)[i];

            cluster_vector_ptr_map[(*cluster_vector_ptr_priori)[i].id] = (*cluster_vector_ptr_priori)[i];
        }
    }

    double computeIou(const cv::RotatedRect rect_1, const cv::RotatedRect rect_2, bool flag)
    {
        cv::RotatedRect rect1 = rect_1;
        cv::RotatedRect rect2 = rect_2;
        if (rect1.size.width > 6 || rect1.size.height > 6 || rect2.size.width > 6 || rect2.size.height > 6)
            return 0.0;
        if (flag)
        {
            rect1.size.width = rect1.size.width + 0.25;
            rect1.size.height = rect1.size.height + 0.25;
            rect2.size.width = rect2.size.width + 0.25;
            rect2.size.height = rect2.size.height + 0.25;
        }

        double area1 = rect1.size.width * rect1.size.height;
        double area2 = rect2.size.width * rect2.size.height;

        std::vector<cv::Point2f> vertices;
        int intersectionType = cv::rotatedRectangleIntersection(rect1, rect2, vertices);
        if (vertices.size() < 3)
        {
            return 0.0;
        }
        else
        {
            std::vector<cv::Point2f> order_pts;
            cv::convexHull(cv::Mat(vertices), order_pts, true);
            double inter_area = cv::contourArea(order_pts);
            if (inter_area == (area1 + area2))
                return 1.0;
            else
                return inter_area * 1.0 / (area1 + area2 - inter_area);
        }
    }

    cv::Point2f get2linePoint(const cv::Vec4f line1, const cv::Vec4f line2)
    {
        cv::Point2f result;
        float cos_theta1 = line1[0];
        float sin_theta1 = line1[1];
        float x1 = line1[2];
        float y1 = line1[3];

        float cos_theta2 = line2[0];
        float sin_theta2 = line2[1];
        float x2 = line2[2];
        float y2 = line2[3];

        if (cos_theta1 == 0)
        {
            result.y = y2;
            result.x = x1;
        }
        else if (cos_theta2 == 0)
        {
            result.y = y1;
            result.x = x2;
        }
        else
        {
            float slope1 = sin_theta1 / cos_theta1;
            float b1 = y1 - slope1 * x1;

            float slope2 = sin_theta2 / cos_theta2;
            float b2 = y2 - slope2 * x2;

            result.x = (b2 - b1) / (slope1 - slope2);
            result.y = slope1 * result.x + b1;
        }
        return result;
    }

    cv::RotatedRect fitBox(std::vector<Eigen::Vector3d> points, double sigma)
    {
        std::vector<cv::Point2f> points_2d;
        for (int i = 0; i < points.size(); ++i)
        {
            points_2d.push_back(cv::Point2f(points[i].x(), points[i].y()));
        }

        std::vector<cv::Point2f> hull;
        cv::convexHull(points_2d, hull);
        cv::RotatedRect box = minAreaRect(hull);

        cv::RNG rng;
        int iterations = 500;
        double bestScore = -1.;
        // cv::Vec4f line;
        std::vector<Eigen::Vector3d> point_on_plane_vec_best;
        Eigen::Vector3d max_point_best;
        Eigen::Vector3d norm_best;
        double max_d_best;
        Eigen::Vector3d norm_z(0, 0, 1);
        Eigen::VectorXd plane_para(6);
        for (int k = 0; k < iterations; k++)
        {
            int i1 = 0, i2 = 0, i3 = 0;
            while (i1 == i2 || i1 == i3 || i2 == i3)
            {
                i1 = rng(points.size() - 1);
                i2 = rng(points.size() - 1);
                i3 = rng(points.size() - 1);
            }
            const Eigen::Vector3d &p1 = points[i1];
            const Eigen::Vector3d &p2 = points[i2];
            const Eigen::Vector3d &p3 = points[i3];

            Eigen::Vector3d p12 = p2 - p1; // 直线的方向向量
            Eigen::Vector3d p13 = p3 - p1; // 直线的方向向量

            Eigen::Vector3d norm = p13.cross(p12);
            norm *= 1. / norm.norm();
            double cos_val = norm_z.dot(norm);
            double angle_val = acos(cos_val) * 180 / M_PI;

            if (fabs(angle_val - 90) > 10)
                continue;

            double score = 0;
            double max_d = -std::numeric_limits<float>::max();
            std::vector<Eigen::Vector3d> point_on_plane_vec;
            Eigen::Vector3d max_point;

            for (int i = 0; i < points.size(); i++)
            {
                double d = (points[i] - p1).dot(norm); // 向量a与b x乘/向量b的摸.||b||=1./norm(dp)
                // score += exp(-0.5*d*d/(sigma*sigma));//误差定义方式的一种
                if (fabs(d) < sigma)
                {
                    score += 1;
                    point_on_plane_vec.push_back(points[i]);
                }
                if (fabs(d) > max_d)
                {
                    max_d = fabs(d);
                    max_point = points[i];
                }
            }
            if (score > bestScore)
            {
                norm_best = norm;
                max_d_best = max_d;
                max_point_best = max_point;
                point_on_plane_vec_best = point_on_plane_vec;
                // line = cv::Vec4f(dp.x, dp.y, p1.x, p1.y);
                plane_para << norm.x(), norm.y(), norm.z(), p1.x(), p1.y(), p1.z();
                bestScore = score;
            }
        }
        if (bestScore < 10)
        {
            return box;
        }

        Eigen::VectorXd plane_vertical(6);
        plane_vertical << norm_best.y(), -norm_best.x(), 0, max_point_best.x(), max_point_best.y(), max_point_best.z();
        Eigen::Vector3d norm_vertical(norm_best.y(), -norm_best.x(), 0);
        double max_dis = -std::numeric_limits<float>::max();

        Eigen::Vector3d max_point_vertical;
        for (int i = 0; i < points.size(); ++i)
        {
            double dis = (points[i] - max_point_best).dot(norm_vertical); // 向量a与b x乘/向量b的摸.||b||=1./norm(dp)
                                                                          //  score += exp(-0.5*d*d/(sigma*sigma));//误差定义方式的一种
            if (fabs(dis) > max_dis)
            {
                max_dis = fabs(dis);
                max_point_vertical = points[i];
            }
        }

        Eigen::VectorXd plane_vertical_else(6);
        plane_vertical_else << norm_best.y(), -norm_best.x(), 0, max_point_vertical.x(), max_point_vertical.y(), max_point_vertical.z();

        Eigen::VectorXd plane_else(6);
        plane_else << norm_best.x(), norm_best.y(), norm_best.z(), max_point_best.x(), max_point_best.y(), max_point_best.z();

        cv::Vec4f line = cv::Vec4f(norm_best.y(), -norm_best.x(), plane_para[3], plane_para[4]);
        cv::Vec4f line_else = cv::Vec4f(norm_best.y(), -norm_best.x(), max_point_best.x(), max_point_best.y());

        cv::Vec4f line_vertical = cv::Vec4f(norm_best.x(), norm_best.y(), max_point_best.x(), max_point_best.y());
        cv::Vec4f line_vertical_else = cv::Vec4f(norm_best.x(), norm_best.y(), max_point_vertical.x(), max_point_vertical.y());

        cv::Point2f max_point_parallel = get2linePoint(line, line_vertical_else);
        cv::Point2f point_parallel = get2linePoint(line_else, line_vertical_else);

        cv::Point2f point_1 = get2linePoint(line, line_vertical);
        cv::Point2f point_2 = get2linePoint(line_else, line_vertical);

        std::vector<cv::Point2f> point_final;
        points_2d.push_back(max_point_parallel);
        points_2d.push_back(point_parallel);
        points_2d.push_back(point_1);
        points_2d.push_back(point_2);

        cv::convexHull(points_2d, hull);
        cv::RotatedRect box_tmp = minAreaRect(hull);

        if (fabs(box_tmp.angle - box.angle) < 3)
        {

            return box;
        }
        else
        {

            return box_tmp;
        }
    }

    void clusterMerge(ObjectCluster &clusteri, ObjectCluster &clusterj)
    {
        int point_num_i = clusteri.cluster_points_.size();
        int point_num_j = clusterj.cluster_points_.size();
        // point update
        clusteri.average_x_ = (clusteri.average_x_ * point_num_i + clusterj.average_x_ * point_num_j) / (point_num_i + point_num_j);
        clusteri.average_y_ = (clusteri.average_y_ * point_num_i + clusterj.average_y_ * point_num_j) / (point_num_i + point_num_j);
        clusteri.average_z_ = (clusteri.average_z_ * point_num_i + clusterj.average_z_ * point_num_j) / (point_num_i + point_num_j);

        clusteri.max_x_ = clusteri.max_x_ > clusterj.max_x_ ? clusteri.max_x_ : clusterj.max_x_;
        clusteri.min_x_ = clusteri.min_x_ > clusterj.min_x_ ? clusterj.min_x_ : clusteri.min_x_;
        clusteri.max_y_ = clusteri.max_y_ > clusterj.max_y_ ? clusteri.max_y_ : clusterj.max_y_;
        clusteri.min_y_ = clusteri.min_y_ > clusterj.min_y_ ? clusterj.min_y_ : clusteri.min_y_;
        clusteri.max_z_ = clusteri.max_z_ > clusterj.max_z_ ? clusteri.max_z_ : clusterj.max_z_;
        clusteri.min_z_ = clusteri.min_z_ > clusterj.min_z_ ? clusterj.min_z_ : clusteri.min_z_;

        for (int i = 0; i < clusterj.cluster_points_.size(); ++i)
        {
            PointType pt;
            pt.x = clusterj.cluster_points_[i].x;
            pt.y = clusterj.cluster_points_[i].y;
            pt.z = clusterj.cluster_points_[i].z;
            pt.intensity = clusterj.cluster_points_[i].intensity;
            pt.label = clusterj.cluster_points_[i].label;
            pt.id = clusteri.id;
            clusteri.cluster_points_.push_back(pt);
        }

        if (segmentation_strategy == 0)
        {
            // update Bbox
            clusteri.bounding_box_.pose.position.x = (clusteri.average_x_ * point_num_i + clusterj.average_x_ * point_num_j) / (point_num_i + point_num_j);
            clusteri.bounding_box_.pose.position.y = (clusteri.average_y_ * point_num_i + clusterj.average_y_ * point_num_j) / (point_num_i + point_num_j);
            clusteri.bounding_box_.pose.position.z = clusteri.min_z_ + (clusteri.max_z_ - clusteri.min_z_) / 2.0;

            float length_ = clusteri.max_x_ - clusteri.min_x_;
            float width_ = clusteri.max_y_ - clusteri.min_y_;
            float height_ = clusteri.max_z_ - clusteri.min_z_;

            clusteri.bounding_box_.dimensions.x = ((length_ < 0) ? -1 * length_ : length_);
            clusteri.bounding_box_.dimensions.y = ((width_ < 0) ? -1 * width_ : width_);
            clusteri.bounding_box_.dimensions.z = ((height_ < 0) ? -1 * height_ : height_);

            clusteri.convex_hull_.insert(clusteri.convex_hull_.end(), clusterj.convex_hull_.begin(), clusterj.convex_hull_.end());
            clusteri.origin_point2f_.insert(clusteri.origin_point2f_.end(), clusterj.origin_point2f_.begin(), clusterj.origin_point2f_.end());
            clusteri.origin_point3f_.insert(clusteri.origin_point3f_.end(), clusterj.origin_point3f_.begin(), clusterj.origin_point3f_.end());

            std::vector<cv::Point2f> order_pts;
            cv::convexHull(cv::Mat(clusteri.convex_hull_), order_pts, true);
            if (order_pts.size() < 3)
                return;
            cv::RotatedRect box = fitBox(clusteri.origin_point3f_, 0.1);
            clusteri.box_ = box;
            clusteri.convex_hull_ = order_pts;
            clusteri.bounding_box_.pose.position.x = box.center.x;
            clusteri.bounding_box_.pose.position.y = box.center.y;
            clusteri.bounding_box_.dimensions.x = box.size.width;
            clusteri.bounding_box_.dimensions.y = box.size.height;
            double rz = box.angle * M_PI / 180.0;
            tf::Quaternion quat = tf::createQuaternionFromRPY(0, 0, rz);
            tf::quaternionTFToMsg(quat, clusteri.bounding_box_.pose.orientation);
        }
    }

    void objectClusterByBoundingBox(ClusterVectorPtr &cluster_vector_ptr)
    {
        stableBboxArragMsg.header = currentBboxArragMsg.header;
        stableBboxArragMsg.boxes.resize(0);
        for (int id = 1; id < cluster_vector_ptr->size(); id++)
        {
            if ((*cluster_vector_ptr)[id].cluster_points_.size() < minBboxpointnum)
            {
                continue;
            }

            (*cluster_vector_ptr)[id].length_ = (*cluster_vector_ptr)[id].bounding_box_.dimensions.x;
            (*cluster_vector_ptr)[id].width_ = (*cluster_vector_ptr)[id].bounding_box_.dimensions.y;
            (*cluster_vector_ptr)[id].height_ = (*cluster_vector_ptr)[id].bounding_box_.dimensions.z;

            (*cluster_vector_ptr)[id].average_x_ = (*cluster_vector_ptr)[id].bounding_box_.pose.position.x;
            (*cluster_vector_ptr)[id].average_y_ = (*cluster_vector_ptr)[id].bounding_box_.pose.position.y;
            (*cluster_vector_ptr)[id].average_z_ = (*cluster_vector_ptr)[id].bounding_box_.pose.position.z;
            (*cluster_vector_ptr)[id].score = (*cluster_vector_ptr)[id].cluster_points_[0].score;
            stableBboxArragMsg.boxes.push_back((*cluster_vector_ptr)[id].bounding_box_);
            cluster_vector_ptr_map[id] = (*cluster_vector_ptr)[id];
        }
    }

    void objectCluster(ClusterVectorPtr &cluster_vector_ptr)
    {
        for (int id = 1; id < cluster_vector_ptr->size(); id++)
        {
            float min_x = std::numeric_limits<float>::max();
            float max_x = -std::numeric_limits<float>::max();
            float min_y = std::numeric_limits<float>::max();
            float max_y = -std::numeric_limits<float>::max();
            float min_z = std::numeric_limits<float>::max();
            float max_z = -std::numeric_limits<float>::max();
            float average_x = 0, average_y = 0, average_z = 0, average_intensity = 0;
            std::vector<cv::Point2f> points;
            std::vector<Eigen::Vector3d> points_3d;
            for (int j = 0; j < (*cluster_vector_ptr)[id].cluster_points_.size(); j++)
            {
                PointType p;
                p.x = (*cluster_vector_ptr)[id].cluster_points_[j].x;
                p.y = (*cluster_vector_ptr)[id].cluster_points_[j].y;
                p.z = (*cluster_vector_ptr)[id].cluster_points_[j].z;

                average_x += p.x;
                average_y += p.y;
                average_z += p.z;

                if (p.x < min_x)
                    min_x = p.x;
                if (p.y < min_y)
                    min_y = p.y;
                if (p.z < min_z)
                    min_z = p.z;
                if (p.x > max_x)
                    max_x = p.x;
                if (p.y > max_y)
                    max_y = p.y;
                if (p.z > max_z)
                    max_z = p.z;

                cv::Point2f pt;
                pt.x = p.x;
                pt.y = p.y;
                points.push_back(pt);

                points_3d.push_back(Eigen::Vector3d(p.x, p.y, p.z));
            }
            // calculate centroid, average
            if ((*cluster_vector_ptr)[id].cluster_points_.size() > 0)
            {
                average_x /= (*cluster_vector_ptr)[id].cluster_points_.size();
                average_y /= (*cluster_vector_ptr)[id].cluster_points_.size();
                average_z /= (*cluster_vector_ptr)[id].cluster_points_.size();
            }
            // calculate bounding box
            float length_ = max_x - min_x;
            float width_ = max_y - min_y;
            float height_ = max_z - min_z;

            (*cluster_vector_ptr)[id].bounding_box_.header = cloudHeader;

            (*cluster_vector_ptr)[id].bounding_box_.pose.position.x = average_x;
            (*cluster_vector_ptr)[id].bounding_box_.pose.position.y = average_y;
            (*cluster_vector_ptr)[id].bounding_box_.pose.position.z = min_z + height_ / 2;

            (*cluster_vector_ptr)[id].bounding_box_.dimensions.x = ((length_ < 0) ? -1 * length_ : length_);
            (*cluster_vector_ptr)[id].bounding_box_.dimensions.y = ((width_ < 0) ? -1 * width_ : width_);
            (*cluster_vector_ptr)[id].bounding_box_.dimensions.z = ((height_ < 0) ? -1 * height_ : height_);

            (*cluster_vector_ptr)[id].average_x_ = average_x;
            (*cluster_vector_ptr)[id].average_y_ = average_y;
            (*cluster_vector_ptr)[id].average_z_ = average_z;
            (*cluster_vector_ptr)[id].max_x_ = max_x;
            (*cluster_vector_ptr)[id].min_x_ = min_x;
            (*cluster_vector_ptr)[id].max_y_ = max_y;
            (*cluster_vector_ptr)[id].min_y_ = min_y;
            (*cluster_vector_ptr)[id].max_z_ = max_z;
            (*cluster_vector_ptr)[id].min_z_ = min_z;

            // pose estimation
            if (points.size() < 3)
                return;
            std::vector<cv::Point2f> hull;
            cv::convexHull(points, hull);

            cv::RotatedRect box = fitBox(points_3d, 0.1);
            (*cluster_vector_ptr)[id].box_ = box;
            (*cluster_vector_ptr)[id].convex_hull_ = hull;
            (*cluster_vector_ptr)[id].origin_point2f_ = points;
            (*cluster_vector_ptr)[id].origin_point3f_ = points_3d;
            (*cluster_vector_ptr)[id].box_initial = true;

            (*cluster_vector_ptr)[id].bounding_box_.pose.position.x = box.center.x;
            (*cluster_vector_ptr)[id].bounding_box_.pose.position.y = box.center.y;
            (*cluster_vector_ptr)[id].bounding_box_.dimensions.x = box.size.width;
            (*cluster_vector_ptr)[id].bounding_box_.dimensions.y = box.size.height;

            double rz = box.angle * M_PI / 180.0;
            tf::Quaternion quat = tf::createQuaternionFromRPY(0, 0, rz);
            tf::quaternionTFToMsg(quat, (*cluster_vector_ptr)[id].bounding_box_.pose.orientation);
        }
    }

    void projectPointCloud()
    {
        float verticalAngle, horizonAngle, range;

        size_t rowIdn, columnIdn, index, gcloudSize = groundCloud->points.size();
        PointType thisPoint;

        bool halfPassed = false;

        for (size_t i = 0; i < gcloudSize; ++i)
        {

            thisPoint.x = groundCloud->points[i].x;
            thisPoint.y = groundCloud->points[i].y;
            thisPoint.z = groundCloud->points[i].z;
            thisPoint.id = 0;

            verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
            rowIdn = (verticalAngle + ang_bottom) / ang_res_y;

            if (rowIdn < 0 || rowIdn >= N_SCANS)
                continue;

            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCANS / 2;
            if (columnIdn >= Horizon_SCANS)
                columnIdn -= Horizon_SCANS;

            if (columnIdn < 0 || columnIdn >= Horizon_SCANS)
                continue;

            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            groundCloud->points[i].intensity = range;

            if (range < sensorMinimumRange)
                continue;

            rangeMat.at<float>(rowIdn, columnIdn) = range;

            float ori = -atan2(thisPoint.y, thisPoint.x);
            if (!halfPassed)
            {
                if (ori < cloudInfo.startOrientation - M_PI / 2)
                    ori += 2 * M_PI;
                else if (ori > cloudInfo.startOrientation + M_PI * 3 / 2)
                    ori -= 2 * M_PI;

                if (ori - cloudInfo.startOrientation > M_PI)
                    halfPassed = true;
            }
            else
            {
                ori += 2 * M_PI;
                if (ori < cloudInfo.endOrientation - M_PI * 3 / 2)
                    ori += 2 * M_PI;
                else if (ori > cloudInfo.endOrientation + M_PI / 2)
                    ori -= 2 * M_PI;
            }

            float relTime = (ori - cloudInfo.startOrientation) / cloudInfo.orientationDiff;
            thisPoint.intensity = (float)relTime * 0.1;

            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;
            index = columnIdn + rowIdn * Horizon_SCANS;
            fullCloud->points[index] = thisPoint;

            groundMat.at<int8_t>(rowIdn, columnIdn) = 1;
        }

        // object
        halfPassed = false;
        for (int id = 1; id < cluster_vector_ptr_curr->size(); id++)
        {

            ObjectCluster clusteri = (*cluster_vector_ptr_curr)[id];
            if (clusteri.cluster_points_.size() == 0)
            {
                continue;
            }
            for (int j = 0; j < clusteri.cluster_points_.size(); ++j)
            {
                thisPoint = clusteri.cluster_points_[j];
                verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
                rowIdn = (verticalAngle + ang_bottom) / ang_res_y;

                if (rowIdn < 0 || rowIdn >= N_SCANS)
                    continue;

                horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

                columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCANS / 2;
                if (columnIdn >= Horizon_SCANS)
                    columnIdn -= Horizon_SCANS;

                if (columnIdn < 0 || columnIdn >= Horizon_SCANS)
                    continue;

                range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
                nongroundCloud->points[id].intensity = range;

                if (range < sensorMinimumRange)
                    continue;

                rangeMat.at<float>(rowIdn, columnIdn) = range;

                float ori = -atan2(thisPoint.y, thisPoint.x);
                if (!halfPassed)
                {
                    if (ori < cloudInfo.startOrientation - M_PI / 2)
                        ori += 2 * M_PI;
                    else if (ori > cloudInfo.startOrientation + M_PI * 3 / 2)
                        ori -= 2 * M_PI;

                    if (ori - cloudInfo.startOrientation > M_PI)
                        halfPassed = true;
                }
                else
                {
                    ori += 2 * M_PI;
                    if (ori < cloudInfo.endOrientation - M_PI * 3 / 2)
                        ori += 2 * M_PI;
                    else if (ori > cloudInfo.endOrientation + M_PI / 2)
                        ori -= 2 * M_PI;
                }

                float relTime = (ori - cloudInfo.startOrientation) / cloudInfo.orientationDiff;
                thisPoint.intensity = (float)relTime * 0.1;

                thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;
                index = columnIdn + rowIdn * Horizon_SCANS;
                fullCloud->points[index] = thisPoint;

                long img_index = columnIdn * 1000 + rowIdn;

                imgIndex[img_index] = clusteri.id;

                groundMat.at<int8_t>(rowIdn, columnIdn) = 2;
            }
        }
    }

    void cloudSegmentation()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        for (size_t i = 0; i < N_SCANS; ++i)
        {

            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (size_t j = 0; j < Horizon_SCANS; ++j)
            {
                if (groundMat.at<int8_t>(i, j) > 0 && rangeMat.at<float>(i, j) != FLT_MAX)
                {
                    if (fullCloud->points[j + i * Horizon_SCANS].id > 0)
                    {
                        long img_index = j * 1000 + i;
                        int id = imgIndex[img_index];
                        objIndexMap[id].push_back(count); // each object's index in segmentedCloud
                    }
                    // mark ground points so they will not be considered as edge features later
                    cloudInfo.groundFlag[count] = (size_t)groundMat.at<int8_t>(i, j);
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i, j);
                    // image_index[make_pair(i,j)]
                    // save seg cloud
                    segmentedCloud->push_back(fullCloud->points[j + i * Horizon_SCANS]);
                    // size of seg cloud
                    ++count;
                }
            }

            cloudInfo.endRingIndex[i] = count - 1 - 5;
        }

        // cloudInfo.objIndex.assign(count, 0);
        std::map<int, vector<int>>::iterator obj_ind_iter;
        for (obj_ind_iter = objIndexMap.begin(); obj_ind_iter != objIndexMap.end(); ++obj_ind_iter)
        {
            fgo_mot::detect_object detect_obj;
            fgo_mot::index_vector obj_index_vector;
            vector<int> obj_vec = obj_ind_iter->second;
            for (int i = 0; i < obj_vec.size(); ++i)
            {
                obj_index_vector.objIndex.push_back(obj_vec[i]);
                detect_obj.objPointIndex.push_back(obj_vec[i]);
            }
            cloudInfo.objIndexVec.push_back(obj_index_vector); // can remove

            int obj_id = obj_ind_iter->first;
            ObjectCluster object_cluster = cluster_vector_ptr_map[obj_id];

            if (object_cluster.cluster_points_.size() < minBboxpointnum)
                continue;
            detect_obj.average_x = object_cluster.average_x_;
            detect_obj.average_y = object_cluster.average_y_;
            detect_obj.average_z = object_cluster.average_z_;
            detect_obj.score = object_cluster.score;
            detect_obj.bounding_box = object_cluster.bounding_box_;


            pcl::PointCloud<PointType> cloud;
            for (int i = 0; i < cluster_vector_ptr_map[obj_id].convex_hull_.size(); ++i)
            {
                PointType p;
                p.x = cluster_vector_ptr_map[obj_id].convex_hull_[i].x;
                p.y = cluster_vector_ptr_map[obj_id].convex_hull_[i].y;
                p.z = object_cluster.bounding_box_.pose.position.z;
                cloud.push_back(p);
            }
            detect_obj.bounding_box = object_cluster.bounding_box_;
            detect_obj.alpha = object_cluster.alpha;

            if (bbox_2d_is_avaliable)
                detect_obj.bounding_box2d = object_cluster.bounding_box2d_;

            pcl::toROSMsg(cloud, detect_obj.pointcloudConvexhull);

            cloudInfo.objArray.push_back(detect_obj);
        }
    }

    void publishCloud()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.frame_id = frame_ind;
        cloudInfo.seq_id = seq_ind;
        cloudInfo.full_cloud_projection = publishPointCloud(pubFullCloud, fullCloud, cloudHeader);
        cloudInfo.segment_cloud = publishPointCloud(pubSegmentedCloud, segmentedCloud, cloudHeader);
        cloudInfo.ground_cloud = publishPointCloud(pubGroundCloud, groundCloud, cloudHeader);
        cloudInfo.nonground_cloud = publishPointCloud(pubSegmentedCloudPure, nongroundCloud, cloudHeader);
        cloudInfo.Cloud_in_Camera = publishPointCloud(pubCloudinCamera, CloudinCamera, cloudHeader);
        // temp_file << cloudInfo.frame_id << std::endl;
        pubBboxArray.publish(stableBboxArragMsg);
        pubLaserCloudInfo.publish(cloudInfo);
        // publishPointCloud(pubCloudinCamera, CloudinCamera, cloudHeader);
    }

    //    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        mBuf.lock();
        cloudQueue.push_back(*laserCloudMsg);
        mBuf.unlock();
    }

    void solveRotation(double dt, Eigen::Vector3d angular_velocity)
    {
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity);
        qIMU *= deltaQ(un_gyr * dt);
        gyr_0 = angular_velocity;
    }

    void processIMU(double t_cur)
    {
        double rx = 0, ry = 0, rz = 0;
        int i = idxImu;
        if (i >= imuBuf.size())
            i--;
        while (imuBuf[i]->header.stamp.toSec() < t_cur)
        {

            double t = imuBuf[i]->header.stamp.toSec();
            if (currentTimeImu < 0)
                currentTimeImu = t;
            double dt = t - currentTimeImu;
            currentTimeImu = imuBuf[i]->header.stamp.toSec();

            rx = imuBuf[i]->angular_velocity.x;
            ry = imuBuf[i]->angular_velocity.y;
            rz = imuBuf[i]->angular_velocity.z;
            solveRotation(dt, Eigen::Vector3d(rx, ry, rz));
            i++;
            if (i >= imuBuf.size())
                break;
        }

        if (i < imuBuf.size())
        {
            double dt1 = t_cur - currentTimeImu;
            double dt2 = imuBuf[i]->header.stamp.toSec() - t_cur;

            double w1 = dt2 / (dt1 + dt2);
            double w2 = dt1 / (dt1 + dt2);

            rx = w1 * rx + w2 * imuBuf[i]->angular_velocity.x;
            ry = w1 * ry + w2 * imuBuf[i]->angular_velocity.y;
            rz = w1 * rz + w2 * imuBuf[i]->angular_velocity.z;
            solveRotation(dt1, Eigen::Vector3d(rx, ry, rz));
        }
        currentTimeImu = t_cur;
        idxImu = i;
    }

    void imuHandler(const sensor_msgs::ImuConstPtr &imu_in)
    {
        imuBuf.push_back(imu_in);

        if (imuBuf.size() > 600)
            imuBuf[imuBuf.size() - 601] = nullptr;

        if (currentTimeImu < 0)
            currentTimeImu = imu_in->header.stamp.toSec();

        if (!firstImu)
        {
            firstImu = true;
            double rx = 0, ry = 0, rz = 0;
            rx = imu_in->angular_velocity.x;
            ry = imu_in->angular_velocity.y;
            rz = imu_in->angular_velocity.z;
            Eigen::Vector3d angular_velocity(rx, ry, rz);
            gyr_0 = angular_velocity;
        }
    }

    void run()
    {
        // if (cloudQueue.size() <= 2)
        //     return;
        if (cloudQueue.size() == 0)
            return;
        currentCloudMsg = cloudQueue.front();
        pcl::PointCloud<PointXYZIKITTI> kitti_cloud_input;
        pcl::fromROSMsg(currentCloudMsg, kitti_cloud_input);
        // std::cout << "======================run_img_frame_id : " << kitti_cloud_input.points.front().frame_id << "  ==================" << std::endl;

        cloudHeader = currentCloudMsg.header;
        cloudHeader.frame_id = frame_id;

        double timecloudMsg = cloudHeader.stamp.toSec();
        bool cloudSyns = false;

        // std::cout << "objBboxVecBuf.empty(): " << objBboxVecBuf.size() << std::endl;
        while (!objBboxVecBuf.empty())
        {
            jsk_recognition_msgs::BoundingBoxArray objBboxVec = objBboxVecBuf.front();
            double timeBboxVec = objBboxVec.header.stamp.toSec();
            if (timecloudMsg < timeBboxVec - 0.05)
            {
                // message too old
                mBuf.lock();
                cloudQueue.pop_front();
                mBuf.unlock();
                ROS_WARN("--------cloud too old!!!---------");
                return;
            }
            else if (fabs(timecloudMsg - timeBboxVec) <= 0.05)
            {
                currentBboxArragMsg = objBboxVec;
                cloudSyns = true;
                mBuf.lock();
                objBboxVecBuf.pop_front();
                mBuf.unlock();
                break;
            }
            else
            {
                mBuf.lock();
                objBboxVecBuf.pop_front();
                mBuf.unlock();
                ROS_WARN("--------box too old!!!---------");
                // break;
            }
        }

        while (!objBbox2DVecBuf.empty())
        {
            fgo_mot::boundingBox2DArray objBboxVec = objBbox2DVecBuf.front();
            double timeBboxVec = objBboxVec.header.stamp.toSec();
            if (timecloudMsg < timeBboxVec - 0.05)
            {
                ROS_WARN("--------NO  BBOX2D!!!---------");
                break;
            }
            else if (fabs(timecloudMsg - timeBboxVec) <= 0.05)
            {
                bbox2dArragMsg = objBboxVec;
                mBuf.lock();
                objBbox2DVecBuf.pop_front();
                mBuf.unlock();
                break;
            }
            else
            {
                mBuf.lock();
                objBbox2DVecBuf.pop_front();
                mBuf.unlock();
                ROS_WARN("--------box2d too old!!!---------");
                // break;
            }
        }

        if (!cloudSyns && segmentation_strategy != 0)
        {
            if (!objBboxVecBuf.empty())
            {
                ROS_WARN("--------NO BOXES!!!---------");
            }
            return;
        }

        if (bbox2dArragMsg.boxes.size() != currentBboxArragMsg.boxes.size())
        {
            bbox_2d_is_avaliable = false;
            ROS_WARN("================================2d box num != 3d box num !!!================================");
        }
        else
        {
            bbox_2d_is_avaliable = true;
        }

        Timer imageProjection_t("imageProjection");
        imageProjection_t.tic();
        mBuf.lock();
        cloudQueue.pop_front();
        mBuf.unlock();

        timeScanNext = cloudQueue.front().header.stamp.toSec();

        // ROS_INFO("new laser frame coming!!!");
        // 1. Convert ros message to pcl point cloud
        if (!copyPointCloud())
            return;

        // 2. Start and end angle of a scan
        findStartEndAngle();
        // imuCompensation();


        groundSegAndCluster();


        // 3. Range image projection
        projectPointCloud();
        // 4. Mark ground points


        // groundRemoval();

        // 5. Point cloud segmentation
        cloudSegmentation();

        // 6. Publish all clouds
        publishCloud();

        // 7. Reset parameters for next iteration
        resetParameters();

        imageProjection_timer.push_back(imageProjection_t.toc());
    }
};

int main(int argc, char **argv)
{

    ros::init(argc, argv, "fgo_mot");

    ImageProjection IP;
    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");
    ros::Rate rate(200);

    while (ros::ok())
    {
        ros::spinOnce();
        IP.run();
        rate.sleep();
    }

    ros::spin();
    return 0;
}
