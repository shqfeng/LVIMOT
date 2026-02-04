
#include "utils/common.h"
#include "utils/math_tools.h"
#include "fgo_mot/cloud_info.h"

struct smoothness_t
{
    double value;
    size_t ind;
};

struct by_value
{
    bool operator()(smoothness_t const &left, smoothness_t const &right)
    {
        return left.value < right.value;
    }
};

class FeatureExtraction
{

public:
    ros::NodeHandle nh;
    ros::Subscriber subLaserCloudInfo;

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;
    ros::Publisher pubPointCloudTest;
    ros::Publisher pubObjectCornerPoints;
    ros::Publisher pubObjectSurfacePoints;
    ros::Publisher pubFeaturePointsOrigin;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr staticCornerCloud;
    pcl::PointCloud<PointType>::Ptr staticSurfaceCloud;
    std::vector<pcl::PointCloud<PointType>> objCloudVec;
    std::vector<pcl::PointCloud<PointType>> objCornerCloudVec;
    std::vector<pcl::PointCloud<PointType>> objSurfaceCloudVec;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudCluster;       // 初始化
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr objectCornerCloud;  // 初始化
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr objectSurfaceCloud; // 初始化

    pcl::VoxelGrid<PointType> downSizeFilter;

    fgo_mot::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    double *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    std::string frame_id;
    int N_SCANS;
    int Horizon_SCANS;
    double surfLeafSize;
    double edgeThreshold;
    double surfThreshold;
    int k_correspondences;
    double gicp_epsilon;
    fstream temp_file;
    std::vector<double> feature_timer;

    // std::vector<Eigen::Matrix3d> covMatVec;

    FeatureExtraction() : nh("~")
    {
        initializeParameters();

        subLaserCloudInfo = nh.subscribe<fgo_mot::cloud_info>("/image_project/cloud_info", 10000, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

        pubLaserCloudInfo = nh.advertise<fgo_mot::cloud_info>("/feature/cloud_info", 10000);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("/feature/cloud_corner", 1);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("/feature/cloud_surface", 1);

        pubPointCloudTest = nh.advertise<sensor_msgs::PointCloud2>("/feature/cluster", 1);
        pubObjectCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("/feature/cloud_object_corner", 1);
        pubObjectSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("/feature/cloud_object_surface", 1);

        pubFeaturePointsOrigin = nh.advertise<sensor_msgs::PointCloud2>("/feature/cloud_feature_origin", 1);

    }

    virtual ~FeatureExtraction()
    {
        std::pair<double, double> mean_std_timer;
        mean_std_timer = calVarStdev(feature_timer);
        printf("\033[1;32mfeatureExtraction Time[ms] : %0.2f ± %0.2f, %0.0f FPS. \033[0m \n", mean_std_timer.first, mean_std_timer.second, floor(1000.0 / mean_std_timer.first));
    }
    void initializeParameters()
    {
        if (!getParameter("/common/frame_id", frame_id))
        {
            ROS_WARN("frame_id not set, use default value: fgo_mot");
            frame_id = "fgo_mot";
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

        if (!getParameter("/common/surf_leaf_size", surfLeafSize))
        {
            ROS_WARN("surfLeafSize not set, use default value: 2000");
            surfLeafSize = 0.4;
        }

        if (!getParameter("/feature/edge_threshold", edgeThreshold))
        {
            ROS_WARN("edge_threshold not set, use default value: 2");
            edgeThreshold = 0.4;
        }

        if (!getParameter("/feature/surf_threshold", surfThreshold))
        {
            ROS_WARN("surf_threshold not set, use default value: 0.1");
            surfThreshold = 0.4;
        }

        if (!getParameter("/feature/k_correspondences", k_correspondences))
        {
            ROS_WARN("k_correspondences not set, use default value: 5");
            k_correspondences = 5;
        }

        if (!getParameter("/feature/gicp_epsilon", gicp_epsilon))
        {
            ROS_WARN("gicp_epsilon not set, use default value: 1e-3");
            gicp_epsilon = 1e-3;
        }

 
        cloudSmoothness.resize(N_SCANS * Horizon_SCANS);

        downSizeFilter.setLeafSize(surfLeafSize, surfLeafSize, surfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        staticCornerCloud.reset(new pcl::PointCloud<PointType>());
        staticSurfaceCloud.reset(new pcl::PointCloud<PointType>());
        cloudCluster.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
        objectCornerCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
        objectSurfaceCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());

        cloudCurvature = new double[N_SCANS * Horizon_SCANS];
        cloudNeighborPicked = new int[N_SCANS * Horizon_SCANS];
        cloudLabel = new int[N_SCANS * Horizon_SCANS];
    }

    void laserCloudInfoHandler(const fgo_mot::cloud_infoConstPtr &msgIn)
    {
        Timer feature_t("featureExtraction");
        feature_t.tic();
        cloudInfo = *msgIn;          // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        cloudHeader.frame_id = frame_id;
        pcl::fromROSMsg(msgIn->segment_cloud, *extractedCloud); // new cloud for extraction

        // calculateCovariances();

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        publishFeatureCloud();

        feature_timer.push_back(feature_t.toc());
    }

    void calculateCovariances()
    {
        pcl::KdTreeFLANN<PointType> kdtreeInput;
        kdtreeInput.setInputCloud(extractedCloud);

        if (extractedCloud->size() < k_correspondences)
        {
            ROS_ERROR("extractedCloud size < k_correspondences, EXIT!");
            return;
        }
        // covMatVec.clear();
        // #pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
        for (int i = 0; i < extractedCloud->size(); i++)
        {
            // 均值
            Eigen::Vector3d mean;
            std::vector<int> pointSearchInd;
            pointSearchInd.reserve(k_correspondences);
            std::vector<float> pointSearchSqDis;
            pointSearchSqDis.reserve(k_correspondences);

            Eigen::Matrix3d cov;
            mean.setZero();
            cov.setZero();
            kdtreeInput.nearestKSearch(extractedCloud->points[i], k_correspondences, pointSearchInd, pointSearchSqDis);

            int validCorrespondenceCnt = 0;
            for (int j = 0; j < k_correspondences; j++)
            {
                if (pointSearchSqDis[j] < 0.5)
                {
                    PointType &pt = extractedCloud->points[pointSearchInd[j]];
                    mean[0] += pt.x;
                    mean[1] += pt.y;
                    mean[2] += pt.z;

                    cov(0, 0) += pt.x * pt.x;

                    cov(1, 0) += pt.y * pt.x;
                    cov(1, 1) += pt.y * pt.y;

                    cov(2, 0) += pt.z * pt.x;
                    cov(2, 1) += pt.z * pt.y;
                    cov(2, 2) += pt.z * pt.z;

                    validCorrespondenceCnt++;
                }
            }

            if (validCorrespondenceCnt < 2)
                continue;

            // 计算实际的均值和协方差
            mean /= static_cast<double>(validCorrespondenceCnt);
            // Get the actual covariance
            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l <= k; l++)
                {
                    cov(k, l) /= static_cast<double>(validCorrespondenceCnt);
                    cov(k, l) -= mean[k] * mean[l];
                    cov(l, k) = cov(k, l);
                }
            }

            // Compute the SVD (covariance matrix is symmetric so U = V')
            // 对应论文注释1
            // 协方差矩阵是对称的，所以SVD分解后，其U=V’
            // 对协方差矩阵进行SVD分解
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU);
            cov.setZero();
            Eigen::Matrix3d U = svd.matrixU();
            // Reconstitute the covariance matrix with modified singular values using the column
            // // vectors in V.
            for (int k = 0; k < 3; k++)
            { // 重构协方差矩阵
                Eigen::Vector3d col = U.col(k);
                double v = 1.;        // biggest 2 singular values replaced by 1
                if (k == 2)           // smallest singular value replaced by gicp_epsilon
                    v = gicp_epsilon; // 表示拟合平面法向量的不确定性
                cov += v * col * col.transpose();
                // 前两个最大的奇异值被替换成1
                // 最小的奇异值被替换成gicp_epsilon
            }
            // std::cout << "cov: \n" << cov << std::endl;
            extractedCloud->points[i].cov00 = cov(0, 0);
            extractedCloud->points[i].cov01 = cov(0, 1);
            extractedCloud->points[i].cov02 = cov(0, 2);
            extractedCloud->points[i].cov10 = cov(1, 0);
            extractedCloud->points[i].cov11 = cov(1, 1);
            extractedCloud->points[i].cov12 = cov(1, 2);
            extractedCloud->points[i].cov20 = cov(2, 0);
            extractedCloud->points[i].cov21 = cov(2, 1);
            extractedCloud->points[i].cov22 = cov(2, 2);

            // Eigen::Matrix3d cov_target;
            // cov_target << extractedCloud->points[i].cov00, extractedCloud->points[i].cov01, extractedCloud->points[i].cov02,
            //             extractedCloud->points[i].cov10, extractedCloud->points[i].cov11, extractedCloud->points[i].cov12,
            //             extractedCloud->points[i].cov20, extractedCloud->points[i].cov21, extractedCloud->points[i].cov22;

            // std::cout << "cov_target: \n" << cov_target << std::endl;
            // covMatVec.push_back(cov);
        }
    }

    void calculateSmoothness()
    {

        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {
            double diffRange = cloudInfo.pointRange[i - 5] + cloudInfo.pointRange[i - 4] + cloudInfo.pointRange[i - 3] + cloudInfo.pointRange[i - 2] + cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i] * 10 + cloudInfo.pointRange[i + 1] + cloudInfo.pointRange[i + 2] + cloudInfo.pointRange[i + 3] + cloudInfo.pointRange[i + 4] + cloudInfo.pointRange[i + 5];

            cloudCurvature[i] = diffRange * diffRange; // diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
            extractedCloud->points[i].smooth = cloudCurvature[i];
            // std::cout << "smooth: " << extractedCloud->points[i].smooth << std::endl;
        }
    }

    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            double depth1 = cloudInfo.pointRange[i];
            double depth2 = cloudInfo.pointRange[i + 1];
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i + 1] - cloudInfo.pointColInd[i]));

            if (columnDiff < 10)
            {
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3)
                {
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }
                else if (depth2 - depth1 > 0.3)
                {
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            double diff1 = std::abs(double(cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i]));
            double diff2 = std::abs(double(cloudInfo.pointRange[i + 1] - cloudInfo.pointRange[i]));

            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()
    {
        staticCornerCloud->clear();
        staticSurfaceCloud->clear();

        objCornerCloudVec.clear();
        objSurfaceCloudVec.clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < N_SCANS; i++)
        {
            surfaceCloudScan->clear();

            for (int j = 0; j < 6; j++)
            {

                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, by_value());

                int largestPickedNum = 0;

                //* static corner 只从background中选取
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;
                    // std::cout << "corner extractedCloud->points[ind].id " << extractedCloud->points[ind].id << std::endl;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold && extractedCloud->points[ind].id <= 0)
                    // if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= 20)
                        {
                            cloudLabel[ind] = 1;
                            extractedCloud->points[ind].label = CORNER;
                            staticCornerCloud->push_back(extractedCloud->points[ind]);
                        }
                        else
                        {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10 || extractedCloud->points[ind + l].id != 1) // filter object
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10 || extractedCloud->points[ind + l].id != 1) // filter object
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                //* static surface 首先从地面点中提取部分曲率较小的点，然后提取不是线特征的背景点作为面特征
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    // std::cout << "surf extractedCloud->points[ind].id " << extractedCloud->points[ind].id << std::endl;
                    // 首先从地面点中提取一部分surf面特征，标记cloudLabel[ind] = -1
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold && extractedCloud->points[ind].id <= 0)
                    {
                        cloudLabel[ind] = -1;

                        // //! 4.提取部分地面点作为面特征
                        extractedCloud->points[ind].label = SURFACE;

                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++)
                        {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10 || extractedCloud->points[ind + l].id > 1)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10 || extractedCloud->points[ind + l].id > 1)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++) // obj corner surf
                {
                    int ind = cloudSmoothness[k].ind;
                    // std::cout << "surf extractedCloud->points[ind].id " << extractedCloud->points[ind].id << std::endl;
                    if (cloudCurvature[ind] > edgeThreshold && extractedCloud->points[ind].id > 0)
                    {
                        cloudNeighborPicked[ind] = 1;
                        cloudLabel[ind] = extractedCloud->points[ind].id;
                        // std::cout << "cloudLabel[ind]: " << cloudLabel[ind] << std::endl;
                        extractedCloud->points[ind].label = CORNER;
                    }
                    else if (cloudCurvature[ind] <= edgeThreshold && extractedCloud->points[ind].id > 0)
                    {
                        cloudNeighborPicked[ind] = 1;
                        cloudLabel[ind] = extractedCloud->points[ind].id;
                        // std::cout << "cloudLabel[ind]: " << cloudLabel[ind] << std::endl;
                        extractedCloud->points[ind].label = SURFACE;
                    }
                }

                for (int k = sp; k <= ep; k++)
                {

                    if (cloudLabel[k] <= 0)
                    {
                        extractedCloud->points[k].label = SURFACE;
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *staticSurfaceCloud += *surfaceCloudScanDS;
        }

        objCloudVec.clear();
        objCornerCloudVec.clear();
        objSurfaceCloudVec.clear();
        for (int i = 0; i < cloudInfo.objIndexVec.size(); ++i)
        {
            pcl::PointCloud<PointType> obj_cloud;
            pcl::PointCloud<PointType> obj_cloud_corner;
            pcl::PointCloud<PointType> obj_cloud_surface;
            for (int j = 0; j < cloudInfo.objIndexVec[i].objIndex.size(); ++j)
            {
                obj_cloud.push_back(extractedCloud->points[cloudInfo.objIndexVec[i].objIndex[j]]);
                if (extractedCloud->points[cloudInfo.objIndexVec[i].objIndex[j]].label == CORNER)
                {
                    obj_cloud_corner.push_back(extractedCloud->points[cloudInfo.objIndexVec[i].objIndex[j]]);
                }
                else if (extractedCloud->points[cloudInfo.objIndexVec[i].objIndex[j]].label == SURFACE)
                {
                    obj_cloud_surface.push_back(extractedCloud->points[cloudInfo.objIndexVec[i].objIndex[j]]);
                }
            }
            objCloudVec.push_back(obj_cloud);
            objCornerCloudVec.push_back(obj_cloud_corner);
            objSurfaceCloudVec.push_back(obj_cloud_surface);
        }

        cloudCluster->clear();
        objectCornerCloud->clear();  // 初始化
        objectSurfaceCloud->clear(); // 初始化

        cv::RNG rng(12345);
        for (int i = 0; i < objCloudVec.size(); ++i)
        {
            int r = rng.uniform(20, 255);
            int g = rng.uniform(20, 255);
            int b = rng.uniform(20, 255);

            for (int j = 0; j < objCloudVec[i].points.size(); ++j)
            {
                pcl::PointXYZRGB point;
                point.x = objCloudVec[i].points[j].x;
                point.y = objCloudVec[i].points[j].y;
                point.z = objCloudVec[i].points[j].z;
                point.r = r;
                point.g = g;
                point.b = b;
                cloudCluster->points.push_back(point);
                if (objCloudVec[i].points[j].label == CORNER)
                    objectCornerCloud->points.push_back(point);
                else if (objCloudVec[i].points[j].label == SURFACE)
                    objectSurfaceCloud->points.push_back(point);
            }
        }

        if (pubPointCloudTest.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 laserCloudTemp;
            pcl::toROSMsg(*cloudCluster, laserCloudTemp);
            laserCloudTemp.header = cloudHeader;
            pubPointCloudTest.publish(laserCloudTemp);
        }
        if (pubObjectCornerPoints.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 laserCloudTemp;
            pcl::toROSMsg(*objectCornerCloud, laserCloudTemp);
            laserCloudTemp.header = cloudHeader;
            pubObjectCornerPoints.publish(laserCloudTemp);
        }
        if (pubObjectSurfacePoints.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 laserCloudTemp;
            pcl::toROSMsg(*objectSurfaceCloud, laserCloudTemp);
            laserCloudTemp.header = cloudHeader;
            pubObjectSurfacePoints.publish(laserCloudTemp);
        }
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();

        cloudInfo.endRingIndex.clear();

        cloudInfo.pointColInd.clear();

        cloudInfo.pointRange.clear();

        cloudInfo.groundFlag.clear();

        cloudInfo.objIndexVec.clear();
    }

    void publishFeatureCloud()
    {
        cloudInfo.cloud_corner = publishPointCloud(pubCornerPoints, staticCornerCloud, cloudHeader);
        cloudInfo.cloud_surface = publishPointCloud(pubSurfacePoints, staticSurfaceCloud, cloudHeader);
        cloudInfo.cloud_feature_origin = publishPointCloud(pubFeaturePointsOrigin, extractedCloud, cloudHeader);

        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
        // temp_file << cloudInfo.frame_id << std::endl;

        // free cloud info memory
        freeCloudInfoMemory();
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fgo_mot");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m---->\033[0m Feature Extraction Started.");

    ros::spin();

    return 0;
}
