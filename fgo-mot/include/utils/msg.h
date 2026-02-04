#pragma once
#include "utils/common.h"
#include <queue>



namespace sensor_msgs
{
    class obj_box_msg
    {
    public:
        int status = 1;
        std::string label;
        float alpha;
        cv::Rect bbox2d;
        jsk_recognition_msgs::BoundingBox bbox3d_in_local;
        fgo_mot::boundingBox2D bounding_box_2d;
        float prob;
        std::vector<Eigen::Vector3d> corners_in_local;
        std::vector<Eigen::Vector3d> corners_in_local_predict;
        // std::vector<cv::Point2f> corners_in_uv;
        // cv::Mat mask;
        int frame_id_for_kitti;
        int obj_id_for_kitti;
    };

    class ObjectPerFrameObs
    {
    public:
        std_msgs::Header header;

        bool isUpdateInW = false;
        bool isStatic = false;

        int seq_id;
        int frame_id;
        int viewPoint; // 0,1,2,3,4,5,6,7
        double time;
        double score;
        double alpha;

        cv::Point3d center;

        // 运动模型：匀加速度直线运动
        Eigen::Vector3d Pi;
        Eigen::Quaterniond Qi;
        Eigen::Vector3d Vi;
        Eigen::Vector3d Ai;
        Eigen::Vector3d Wi;
        Eigen::Vector3d Dimensions;
        Eigen::Vector3d centroid;

        Eigen::Vector3d Pi_obs;
        Eigen::Quaterniond Qi_obs;
        Eigen::Vector3d Vi_obs;
        Eigen::Vector3d Pi_cov = Eigen::Vector3d(0.1,0.1,0.1);
        Eigen::Vector3d Qi_cov = Eigen::Vector3d(0.1,0.1,0.1);

        Eigen::Matrix<double, 15, 15> covariance;

        obj_box_msg feature_box;

        pcl::PointCloud<PointType> pointcloudFull;
        pcl::PointCloud<PointType> pointcloudCorner;
        pcl::PointCloud<PointType> pointcloudSurf;

        pcl::PointCloud<PointType> pointcloudFullInObj;
        pcl::PointCloud<PointType> pointcloudCornerInObj;
        pcl::PointCloud<PointType> pointcloudSurfInObj;

        pcl::PointCloud<PointType> currSurfFeaturePoints;
        pcl::PointCloud<PointType> mapSurfFeaturePointsNormal;
        pcl::PointCloud<PointType> mapSurfFeaturePointsCenter;
        pcl::PointCloud<PointType> mapSurfFeaturePoints;
        pcl::PointCloud<PointType> currCornerFeaturePoints;
        pcl::PointCloud<PointType> mapCornerFeaturePoints;
        pcl::PointCloud<PointType> mapCornerFeaturePointsA;
        pcl::PointCloud<PointType> mapCornerFeaturePointsB;

        pcl::PointCloud<PointType> currFeaturePoints;
        pcl::PointCloud<PointType> mapFeaturePoints;

        Eigen::Vector3d delta_p_to_first_frame;
        Eigen::Quaterniond delta_q_to_first_frame;
        bool relative_measurement = false;
    };
    typedef std::shared_ptr<::sensor_msgs::ObjectPerFrameObs> ObjectPerFrameObsPtr;
    typedef std::shared_ptr<::sensor_msgs::ObjectPerFrameObs const> ObjectPerFrameObsConstPtr;

    // trackingĿ���feature����
    class FeaturePerFrame
    {
    public:
        FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
        {
            point.x() = _point(0);
            point.y() = _point(1);
            point.z() = _point(2);
            uv.x() = _point(3);
            uv.y() = _point(4);
            velocity.x() = _point(5);
            velocity.y() = _point(6);
            cur_td = td;
            is_stereo = false;
        }
        void rightObservation(const Eigen::Matrix<double, 7, 1> &_point)
        {
            pointRight.x() = _point(0);
            pointRight.y() = _point(1);
            pointRight.z() = _point(2);
            uvRight.x() = _point(3);
            uvRight.y() = _point(4);
            velocityRight.x() = _point(5);
            velocityRight.y() = _point(6);
            is_stereo = true;
        }

        Eigen::Vector3d point, pointRight;
        Eigen::Vector2d uv, uvRight;
        Eigen::Vector2d velocity, velocityRight;
        double depth = -1;
        bool is_stereo;
        double cur_td;

        cv::Mat descriptor;
        cv::KeyPoint keyPoint;

        int frame_id;
    };

    class FeaturePerId
    {
    public:
        std_msgs::Header header;
        int feature_id;
        int start_frame;

        double estimated_depth;
        double laser_point_depth;

        bool isHasPredict = false;
        bool isInBoxCheck = false;
        cv::Point2f predict_pt;

        Eigen::Vector3d pos_in_obj;
        vector<FeaturePerFrame> feature_per_frame;

        FeaturePerId(int _feature_id, int _start_frame) : feature_id(_feature_id), start_frame(_start_frame), estimated_depth(-1.0), laser_point_depth(-1.0)
        {
        }

        FeaturePerId &operator=(const FeaturePerId &perFeatureId)
        {
            if (this != &perFeatureId)
            {
                this->feature_id = perFeatureId.feature_id;
                this->start_frame = perFeatureId.start_frame;
                this->estimated_depth = perFeatureId.estimated_depth;
                this->laser_point_depth = perFeatureId.laser_point_depth;
                this->isHasPredict = perFeatureId.isHasPredict;
                this->isInBoxCheck = perFeatureId.isInBoxCheck;
                this->predict_pt = perFeatureId.predict_pt;
                this->pos_in_obj = perFeatureId.pos_in_obj;
                this->feature_per_frame = perFeatureId.feature_per_frame;
            }
            return *this;
        }
    };

    class ObjectTrack
    {
    public:
        std_msgs::Header header;
        bool isFailTracking = false;
        bool isCheckStatus = false;
        bool isOpt = false;
        bool hasPredict = false;
        bool objMapInit = false;

        bool start_frame_valid = true;
        int start_valid_frame;
        int id;
        int loss_box_num = 0;
        int track_box_num = 0;
        int first_obs_frame;

        double l, w, h;
        double l_predict, w_predict, h_predict;

        Eigen::Vector3d Pi_predict;
        Eigen::Quaterniond Qi_predict;
        Eigen::Vector3d Vi_predict;
        bool map_init = false;

        Eigen::Vector3d position_map;
        Eigen::Quaterniond quater_map;

        std::map<int, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> traj_history;
        std::map<int, pcl::PointCloud<PointType>> vehicle_corner_cloud_total;
        std::map<int, pcl::PointCloud<PointType>> vehicle_surf_cloud_total;
        std::map<int, pcl::PointCloud<PointType>> vehicle_full_cloud_total;
        std::deque<pcl::PointCloud<PointType>> recentSurfFrames;
        std::deque<pcl::PointCloud<PointType>> recentCornerFrames;

        list<FeaturePerId> feature_points;
        std::vector<ObjectPerFrameObs> object_perframes;

        int current_map_frame;

        pcl::PointCloud<PointType> vehiclePointCloud;
        pcl::PointCloud<PointType> vehiclePointCloudInWorld;
        pcl::PointCloud<PointType> vehicleCornerCloudMap;
        pcl::PointCloud<PointType> vehicleSurfCloudMap;
        pcl::PointCloud<PointType> vehicleFullCloudMap;
        pcl::KdTreeFLANN<PointType> kdtreeCornerLast;
        pcl::KdTreeFLANN<PointType> kdtreeSurfLast;
        pcl::KdTreeFLANN<PointType> kdtreeFullLast;
    };

    struct ObjsPerFrameDetection
    {
        std_msgs::Header header;
        std::vector<obj_box_msg> objects;
    };
    typedef std::shared_ptr<::sensor_msgs::ObjsPerFrameDetection> ObjsPerFrameDetectionPtr;
    typedef std::shared_ptr<::sensor_msgs::ObjsPerFrameDetection const> ObjsPerFrameDetectionConstPtr;

    struct Kitti_GT
    {
        std_msgs::Header header;
        double px, py, pz, yaw, pitch, roll;
    };
    typedef std::shared_ptr<::sensor_msgs::Kitti_GT> Kitti_GTPtr;
    typedef std::shared_ptr<::sensor_msgs::Kitti_GT const> Kitti_GTConstPtr;

}
