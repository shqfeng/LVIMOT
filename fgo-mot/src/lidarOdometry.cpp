#include "utils/common.h"
#include "utils/math_tools.h"

class LidarOdometry
{
private:
    ros::NodeHandle nh;

    ros::Subscriber subCloudInfo;
    ros::Subscriber subTFInfo;

    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdomCloudInfo;
    ros::Publisher pubRecentKeyFrame;

    std_msgs::Header cloudHeader;
    nav_msgs::Odometry odom;

    fgo_mot::cloud_info cloudInfo;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    pcl::PointCloud<PointType>::Ptr laserCloudFullLast;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;

    double timeNewCloudInfo = 0;

    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;

    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterSurfMap;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterCornerMap;

    string frame_id = "fgo_mot";

    std::mutex mBuf;
    std::deque<nav_msgs::Odometry> odomMsgBuff;
    std::deque<fgo_mot::cloud_infoConstPtr> cloudInfoBuff;
    nav_msgs::Odometry currOdomMsg;

    std::vector<double> odometry_timer;

public:
    LidarOdometry() : nh("~")
    {
        initializeParameters();
        allocateMemory();

        subCloudInfo = nh.subscribe<fgo_mot::cloud_info>("/feature/cloud_info", 10000, &LidarOdometry::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subTFInfo = nh.subscribe<tf2_msgs::TFMessage>("/lidar_odometry", 10000, &LidarOdometry::TFInfoHandler, this);
        pubLaserOdomCloudInfo = nh.advertise<fgo_mot::cloud_info>("/odometry/cloud_info", 10000);
        pubRecentKeyFrame = nh.advertise<sensor_msgs::PointCloud2>("/odometry/map_local", 1);
    }

    virtual ~LidarOdometry() 
    {
        std::pair<double, double> mean_std_timer;
        mean_std_timer = calVarStdev(odometry_timer);
        printf("\033[1;32mlidarOdometry     Time[ms] : %0.2f ± %0.2f, %0.0f FPS. \033[0m \n", mean_std_timer.first, mean_std_timer.second, floor(1000.0 / mean_std_timer.first));
    }

    void initializeParameters()
    {
        if (!getParameter("/common/frame_id", frame_id))
        {
            ROS_WARN("frame_id not set, use default value: fgo_mot");
            frame_id = "fgo_mot";
        }

        odom.header.frame_id = frame_id;

        downSizeFilterSurf.setLeafSize(1, 1, 1);
        downSizeFilterSurfMap.setLeafSize(0.4, 0.4, 0.4);

        downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilterCornerMap.setLeafSize(0.2, 0.2, 0.2);
    }

    void allocateMemory()
    {
        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudFullLast.reset(new pcl::PointCloud<PointType>());

        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());

        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>());
    }

    void laserCloudInfoHandler(const fgo_mot::cloud_infoConstPtr &msgIn)
    {
        mBuf.lock();
        cloudInfoBuff.push_back(msgIn);
        mBuf.unlock();
    }

    void TFInfoHandler(const tf2_msgs::TFMessageConstPtr &msg_tf)
    {
        odom.header.frame_id = frame_id;
        odom.header.stamp = msg_tf->transforms[0].header.stamp;
        odom.pose.pose.position.x = msg_tf->transforms[0].transform.translation.x;
        odom.pose.pose.position.y = msg_tf->transforms[0].transform.translation.y;
        odom.pose.pose.position.z = msg_tf->transforms[0].transform.translation.z;
        odom.pose.pose.orientation.w = msg_tf->transforms[0].transform.rotation.w;
        odom.pose.pose.orientation.x = msg_tf->transforms[0].transform.rotation.x;
        odom.pose.pose.orientation.y = msg_tf->transforms[0].transform.rotation.y;
        odom.pose.pose.orientation.z = msg_tf->transforms[0].transform.rotation.z;
        pubLaserOdometryGlobal.publish(odom);

        mBuf.lock();
        odomMsgBuff.push_back(odom);
        mBuf.unlock();
    }

    void run()
    {
        if (!cloudInfoBuff.empty())
        {
            cloudInfo = *cloudInfoBuff.front();
            timeNewCloudInfo = cloudInfo.header.stamp.toSec();
            bool odomSyns = false;
            if (odomMsgBuff.empty())
                return;
            while (!odomMsgBuff.empty())
            {
                nav_msgs::Odometry thisOdom = odomMsgBuff.front();
                double timeOdom = thisOdom.header.stamp.toSec();
                if (timeOdom < timeNewCloudInfo - 0.05)
                {
                    mBuf.lock();
                    odomMsgBuff.pop_front();
                    mBuf.unlock();
                    ROS_WARN("--------odom too old!!!---------");
                }
                else if (fabs(timeOdom - timeNewCloudInfo) <= 0.05)
                {
                    currOdomMsg = thisOdom;
                    odomSyns = true;
                    break;
                }
                else
                {
                    mBuf.lock();
                    cloudInfoBuff.pop_front();
                    mBuf.unlock();
                    ROS_WARN("--------cloud info too old!!!---------");
                    return;
                }
            }
            if (!odomSyns)
            {
                ROS_WARN("--------NO ODOMETRY!!!---------");
                return;
            }

            Timer odometry_t("lidarOdometry");
            odometry_t.tic();
            mBuf.lock();
            cloudInfoBuff.pop_front();
            odomMsgBuff.pop_front();
            mBuf.unlock();

            timeNewCloudInfo = cloudInfo.header.stamp.toSec();
            cloudHeader = cloudInfo.header;

            pcl::fromROSMsg(cloudInfo.cloud_corner, *laserCloudCornerLast);
            pcl::fromROSMsg(cloudInfo.cloud_surface, *laserCloudSurfLast);
            pcl::fromROSMsg(cloudInfo.cloud_feature_origin, *laserCloudFullLast);
            
            downSampleCloud();

            publishFrames();
            publishCloudInfoLast();
            clearCloud();
            odometry_timer.push_back(odometry_t.toc());
        }
    }

    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, Eigen::Quaterniond quaternion, Eigen::Vector3d transition)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int numPts = cloudIn->points.size();
        cloudOut->resize(numPts);

        for (int i = 0; i < numPts; ++i)
        {
            Eigen::Vector3d ptIn(cloudIn->points[i].x, cloudIn->points[i].y, cloudIn->points[i].z);
            Eigen::Vector3d ptOut = quaternion * ptIn + transition;

            PointType pt;
            pt.x = ptOut.x();
            pt.y = ptOut.y();
            pt.z = ptOut.z();
            pt.intensity = cloudIn->points[i].intensity;
            pt.label = cloudIn->points[i].label;
            pt.id = cloudIn->points[i].id;

            cloudOut->points[i] = pt;
        }

        return cloudOut;
    }

    void clearCloud()
    {
        laserCloudSurfFromMap->clear();
        laserCloudSurfFromMapDS->clear();
        laserCloudCornerFromMap->clear();
        laserCloudCornerFromMapDS->clear();
        laserCloudCornerLast->clear();
        laserCloudSurfLast->clear();
        laserCloudFullLast->clear();
    }

    void downSampleCloud()
    {
        downSizeFilterSurfMap.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurfMap.filter(*laserCloudSurfFromMapDS);

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);

        downSizeFilterCornerMap.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCornerMap.filter(*laserCloudCornerFromMapDS);

        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
    }

    void publishCloudInfoLast()
    {
        cloudInfo.pose_lidar_odom = currOdomMsg;
        pubLaserOdomCloudInfo.publish(cloudInfo);
    }

    void publishFrames()
    {
        cloudHeader.frame_id = "map";
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            Eigen::Quaterniond q_w_l = Eigen::Quaterniond(currOdomMsg.pose.pose.orientation.w, currOdomMsg.pose.pose.orientation.x, currOdomMsg.pose.pose.orientation.y, currOdomMsg.pose.pose.orientation.z);
            Eigen::Vector3d t_w_l = Eigen::Vector3d(currOdomMsg.pose.pose.position.x, currOdomMsg.pose.pose.position.y, currOdomMsg.pose.pose.position.z);
            *cloudOut += *transformCloud(laserCloudCornerLastDS, q_w_l, t_w_l);
            *cloudOut += *transformCloud(laserCloudSurfLastDS, q_w_l, t_w_l);
            publishPointCloud(pubRecentKeyFrame, cloudOut, cloudHeader);
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fgo_mot");

    LidarOdometry LO;

    ROS_INFO("\033[1;32m---->\033[0m Lidar Odometry Started.");

    ros::Rate rate(200);

    while (ros::ok())
    {
        ros::spinOnce();
        LO.run();
        rate.sleep();
    }

    ros::spin();
    return 0;
}
