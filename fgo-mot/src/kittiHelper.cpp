#include "utils/common.h"
#include "utils/math_tools.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <string>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <tf/tf.h>
#include <ros/ros.h>
#include <ros/time.h>
#include <ros/duration.h>
#include <rosbag/bag.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Transform.h>
#include <std_msgs/Header.h>
#include <tf2_msgs/TFMessage.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>


int time_count = 0;

class kitti2bag
{
private:
    int seq_id;
    ros::NodeHandle nh;
    ros::Publisher pubLaserCloud;
    ros::Publisher pubTFInfo;
    ros::Publisher pubBbArray;
    ros::Publisher pubBb2DArray;
    image_transport::Publisher pubImageLeft;

    std::map<int, std::vector<std::vector<float>>> detection_data_map;

    std::map<int, std::vector<std::vector<float>>> detection2d_data_map;
    std::string bage_file_path;
    std::string out_bag_path;

    bool to_bag;
    rosbag::Bag bag_out;

    std::string lidar_topic;
    std::string Bbox_topic;
    std::string Image2_topic;
    std::string Image3_topic;
    std::string tf_topic;
    std::string world_frame_id;

    std::string base_path;
    std::string lidar_path;
    std::string image2_path;
    std::string image3_path;
    std::string detection_path;
    std::string oxts_path;

    std::string pose_name;
    std::string calib_name;
    std::string timestamp_name;

    std::vector<std::string> lidar_names;
    std::vector<std::string> image2_names;
    std::vector<std::string> image3_names;
    std::vector<std::string> detection_names;

    Eigen::Matrix<double, 4, 4> P2;
    Eigen::Matrix<double, 4, 4> R_rect;
    Eigen::Matrix<double, 4, 4> Tr_velo_cam;
    Eigen::Matrix<double, 4, 4> Tr_imu_velo;
    std::vector<Eigen::Matrix<double, 4, 4>> Lidar_pose;
    // std::vector<float> timestamps;
    double initime;
    double data_pub_rate;

    bool OneFile = true;
    bool useoxts = false;

public:
    std::string sequence_num;

    kitti2bag(int sequence, bool &flag)
    {
        std::stringstream ss;
        seq_id = sequence;
        ss << setw(4) << setfill('0') << sequence;
        ss >> sequence_num;

        initializeParameters();

        flag = true;

        if (!read_calib_data() )
        {
          std::cout << "!1" << std::endl;
        }
        if (!read_pose_data() )
        {
          std::cout << "!2" << std::endl;
        }
        if (!Integrity_test() )
        {
          std::cout << "3" << std::endl;
        }

        if (!read_calib_data() || !read_pose_data() || !Integrity_test())
        {
            std::cout << "somethong wrong!" << std::endl;
            flag = false;
        }
        if (flag && to_bag)
        {
            bage_file_path = out_bag_path + "/kitti_data_tracking_lidar_sequence_" + sequence_num + ".bag";
            bag_out.open(bage_file_path, rosbag::bagmode::Write);
        }
        initime = 1668248484.080204;

        pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>(lidar_topic, 50);
        pubBb2DArray = nh.advertise<fgo_mot::boundingBox2DArray>("/detection2d_array", 50);
        pubTFInfo = nh.advertise<tf2_msgs::TFMessage>(tf_topic, 50);
        pubBbArray = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>(Bbox_topic, 50);
        image_transport::ImageTransport it(nh);
        pubImageLeft = it.advertise(Image2_topic, 20);
    }
    virtual ~kitti2bag()
    {
        if (to_bag)
        {
            bag_out.close();
        }
    }

    void initializeParameters()
    {
        if (!getParameter("/kitti_helper/base_path", base_path))
        {
            ROS_WARN("base_path not set, use default value: /home/hickeytom/Workspace/F-MOT-V2_WS/tracking/training/");
            base_path = "/home/hickeytom/Workspace/F-MOT-V2_WS/tracking/training/";
        }

        if (!getParameter("/kitti_helper/detection_in_one_file", OneFile))
        {
            ROS_WARN("detection_in_one_file not set, use default value: true");
            OneFile = true;
        }

        if (!getParameter("/kitti_helper/to_bag", to_bag))
        {
            ROS_WARN("to_bag not set, use default value: false");
            to_bag = false;
        }

        // extrinsic parameters
        if (!getParameter("/kitti_helper/out_bag_path", out_bag_path) && to_bag)
        {
            ROS_WARN("out_bag_path not set, use default value: /home/hickeytom/Workspace/F-MOT-V2_WS/output/rosbag/");
            out_bag_path = "/home/hickeytom/Workspace/F-MOT-V2_WS/output/rosbag/";
        }

        if (!getParameter("/kitti_helper/data_pub_rate", data_pub_rate))
        {
            ROS_WARN("data_pub_rate not set, use default value: 10.0");
            data_pub_rate = 10.0;
        }

        if (!getParameter("/kitti_helper/OneFile", OneFile))
        {
            ROS_WARN("OneFile not set, use default value: true");
            OneFile = true;
        }
        if (!getParameter("/kitti_helper/useoxts", useoxts))
        {
            ROS_WARN("useoxts not set, use default value: false");
            useoxts = false;
        }

        lidar_path = base_path + "velodyne/" + sequence_num + "/";
        image2_path = base_path + "image_02/" + sequence_num + "/";
        image3_path = base_path + "image_03/" + sequence_num + "/";
        oxts_path = base_path + "oxts/" + sequence_num + ".txt";

        if (OneFile)
        {
            detection_path = base_path + "pointrcnn_Car_val/" + sequence_num + ".txt";
            // detection_path = base_path + "pointrcnn_pc3t/" + sequence_num + ".txt";
        }
        else
        {
            detection_path = base_path + "detection/" + sequence_num + "/";
        }

        pose_name = base_path + "pose/" + sequence_num + "/" + "pose.txt";
        calib_name = base_path + "calib/" + sequence_num + ".txt";
        timestamp_name = base_path + "/times.txt";

        lidar_topic = "/points_raw";
        Bbox_topic = "/detection";
        Image2_topic = "/camera_color_left";
        Image3_topic = "/camera_color_right";
        tf_topic = "/lidar_odometry";
        world_frame_id = "map";

        lidar_names = get_subfile_name(lidar_path);
        image2_names = get_subfile_name(image2_path);
        image3_names = get_subfile_name(image3_path);

        if (!OneFile)
        {
            detection_names = get_subfile_name(detection_path);
        }
        else
        {
            detection_data_map = read_detection_file_inonefile(detection_path);
        }
    }

    void load_velodyne_tf_image_BoxArray()
    {
        std::vector<std::vector<float>> lidar_data;
        std::vector<std::vector<float>> detection_data;
        std::vector<std::vector<float>> detection2d_data;  // 2d detection in point rcnn
        std::vector<float> alpha_data;
        // 等待1s，保证系统完全启动
        ros::Rate rate1(1.0);
        ros::Rate rate2(data_pub_rate);
        rate1.sleep();

        for (int i = 0; i < lidar_names.size(); i++)
        {
            alpha_data.clear();
            std::string lidar_name = lidar_names[i];
            std::string detection_name = lidar_name.substr(0, lidar_name.length() - 3) + "txt";
            std::string iamge2_name = image2_names[i];
            double times = time_count * 0.1 + initime;
            time_count++;

            lidar_data = read_lidar_file(lidar_name);

            if (!OneFile)
            {
                detection_data = read_detection_file(detection_name);
            }
            else
            {
                int fram_id = i;
                std::map<int, std::vector<std::vector<float>>>::iterator map_iter = detection_data_map.find(fram_id);
                if (map_iter == detection_data_map.end())
                {
                    std::vector<std::vector<float>> detection_buffer;
                    detection_data = detection_buffer;
                    alpha_data.clear();
                }
                else
                {
                    detection_data = map_iter->second;

                    detection2d_data = detection2d_data_map[fram_id];

                    for (int k = 0; k < detection_data.size(); k++)
                    {
                        alpha_data.push_back(detection_data[k][8]);  // alpha is 8
                    }
                }
            }

            cam_to_velo(detection_data);

            convert_box_type(detection_data);

            // get pointcloud for filter
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in{new pcl::PointCloud<pcl::PointXYZI>};
            pcl::PointCloud<PointXYZIKITTI> laser_cloud;
            for (std::size_t j = 0; j < lidar_data.size(); j++)
            {
                pcl::PointXYZI point;

                point.x = lidar_data[j][0];
                point.y = lidar_data[j][1];
                point.z = lidar_data[j][2];
                // index -> intensity
                point.intensity = j;
                cloud_in->push_back(point);

                PointXYZIKITTI pointkitti;
                pointkitti.x = lidar_data[j][0];
                pointkitti.y = lidar_data[j][1];
                pointkitti.z = lidar_data[j][2];
                pointkitti.intensity = lidar_data[j][3];
                pointkitti.id = 0;
                pointkitti.score = -999.0;
                pointkitti.alpha = -999.0;
                pointkitti.frame_id = i;
                pointkitti.seq_id = seq_id;
                laser_cloud.push_back(pointkitti);
            }

            // write BoungBoxArray msgs
            jsk_recognition_msgs::BoundingBoxArray bbox_arr;
            std::vector<std::vector<int>> obj_cloud_index;
            bbox_arr.header.frame_id = "fgo_mot";
            bbox_arr.header.stamp = ros::Time().fromSec(times);

            fgo_mot::boundingBox2DArray bbox2d_arr;
            bbox2d_arr.header.frame_id = "fgo_mot";
            bbox2d_arr.header.stamp = ros::Time().fromSec(times);
 
            for (int j = 0; j < detection_data.size(); j++)
            {
                jsk_recognition_msgs::BoundingBox bbox;
                bbox.header.frame_id = "fgo_mot";
                bbox.header.stamp = ros::Time().fromSec(times);
                bbox.pose.position.x = float(detection_data[j][0]);
                bbox.pose.position.y = float(detection_data[j][1]);
                bbox.pose.position.z = float(detection_data[j][2]);
                // if (bbox.pose.position.x > 60) continue;
                // if (bbox.pose.position.y < -8.0) continue;
                // if (bbox.pose.position.y > 4.0) continue;

                bbox.dimensions.x = float(detection_data[j][3]);
                bbox.dimensions.y = float(detection_data[j][4]);
                bbox.dimensions.z = float(detection_data[j][5]);
                bbox.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0.0, 0.0, detection_data[j][6]);
                bbox.value = detection_data[j][7];
                bbox.label = j + 1;
                bbox_arr.boxes.push_back(bbox);
                std::vector<int> cloud_index;
                Bbox_filter_cloud(detection_data[j], cloud_in, cloud_index);
                obj_cloud_index.push_back(cloud_index);

                fgo_mot::boundingBox2D bbox2d;
                bbox2d.header.frame_id = "fgo_mot";
                bbox2d.header.stamp = ros::Time().fromSec(times);
                bbox2d.xmin = detection2d_data[j][0];
                bbox2d.ymin = detection2d_data[j][1];
                bbox2d.xmax = detection2d_data[j][2];
                bbox2d.ymax = detection2d_data[j][3];
                bbox2d.score = detection2d_data[j][4];
                bbox2d_arr.boxes.push_back(bbox2d);
            }

            for (int j = 0; j < obj_cloud_index.size(); j++)
            {
                for (int k = 0; k < obj_cloud_index[j].size(); k++)
                {
                    int index = obj_cloud_index[j][k];
                    laser_cloud.points[index].id = j + 1;
                    laser_cloud.points[index].score = bbox_arr.boxes[j].value;
                    laser_cloud.points[index].alpha = alpha_data[j];
                }
            }

            sensor_msgs::PointCloud2 laser_cloud_msg;
            pcl::toROSMsg(laser_cloud, laser_cloud_msg);
            laser_cloud_msg.header.stamp = ros::Time().fromSec(times);
            laser_cloud_msg.header.frame_id = "fgo_mot";

            cv::Mat left_image = cv::imread(iamge2_name, cv::IMREAD_COLOR);
            std_msgs::Header Header;
            Header.frame_id = "fgo_mot";
            Header.stamp = ros::Time().fromSec(times);
            sensor_msgs::ImagePtr image_left_msg = cv_bridge::CvImage(Header, "bgr8", left_image).toImageMsg();

            Eigen::Matrix<double, 4, 4> pose = Lidar_pose[i];
            Eigen::Matrix<double, 3, 1> t = pose.block<3, 1>(0, 3);
            Eigen::Matrix<double, 3, 3> r = pose.block<3, 3>(0, 0);
            tf2_msgs::TFMessage tf_msg;
            geometry_msgs::TransformStamped tf_stamped;
            tf_stamped.header.stamp = ros::Time().fromSec(times);
            tf_stamped.header.frame_id = "map";
            tf_stamped.child_frame_id = "fgo_mot";
            tf_stamped.transform.translation.x = t(0);
            tf_stamped.transform.translation.y = t(1);
            tf_stamped.transform.translation.z = t(2);
            Eigen::Quaterniond quaternion(r);
            tf_stamped.transform.rotation.w = quaternion.w();
            tf_stamped.transform.rotation.x = quaternion.x();
            tf_stamped.transform.rotation.y = quaternion.y();
            tf_stamped.transform.rotation.z = quaternion.z();
            tf_msg.transforms.push_back(tf_stamped);

            if (to_bag)
            {
                bag_out.write(lidar_topic, ros::Time().fromSec(times), laser_cloud_msg);
                bag_out.write(Bbox_topic, ros::Time().fromSec(times), bbox_arr);
                bag_out.write(tf_topic, ros::Time().fromSec(times), tf_msg);
                bag_out.write(Image2_topic, ros::Time().fromSec(times), image_left_msg);
            }

            pubLaserCloud.publish(laser_cloud_msg);
            pubBbArray.publish(bbox_arr);
            pubBb2DArray.publish(bbox2d_arr);
            pubTFInfo.publish(tf_msg);
            pubImageLeft.publish(image_left_msg);
            rate2.sleep();
        }
    }

    std::vector<std::string> get_subfile_name(std::string path)
    {
        std::vector<std::string> file_names;
        cv::glob(path, file_names);
        for (int i = 0; i < file_names.size(); i++)
        {
            std::string file_num;
            std::stringstream ss;
            int num1 = std::stoi(get_file_name_without_extension(file_names[i]));
            char pointscloud[200];
            sprintf(pointscloud, "%06d", num1);
            file_num = pointscloud;
            size_t found1 = file_names[i].find_last_of('/');
            size_t found2 = file_names[i].find_last_of('.');
            std::string temp_name = file_names[i].substr(0, found1 + 1) + file_num + file_names[i].substr(found2, file_names[i].length() - found2);
            file_names[i] = temp_name;
        }
        sort(file_names.begin(), file_names.end());
        return file_names;
    }

    template <typename T>
    void printvector(std::vector<T> sv)
    {
        std::for_each(sv.begin(), sv.end(), [](T i)
                      { std::cout << i << std::endl; });
        std::cout << std::endl;
    }

    bool Integrity_test()
    {
        std::vector<unsigned long> num{lidar_names.size(), image2_names.size(), image3_names.size(), detection_names.size()};
        sort(num.begin(), num.end());
        if (lidar_names.size() < num.back() || image2_names.size() < num.back() || image3_names.size() < num.back())
        {
            std::cerr << "lidar_names.size() " << lidar_names.size() << std::endl;
            std::cerr << "image2_names.size() " << image2_names.size() << std::endl;
            std::cerr << "image3_names.size() " << image3_names.size() << std::endl;
            // std::cerr << "detection_names.size() " << detection_names.size() << std::endl;
            std::cerr << "Failed to pass the integrity test. Please check the integrity of the file!" << std::endl;
            return false;
        }
        return true;
    }

    std::string get_file_name_without_extension(std::string file_abs_name_with_extension)
    {
        size_t found1 = file_abs_name_with_extension.find_last_of('/');
        size_t found2 = file_abs_name_with_extension.find_last_of('.');
        return file_abs_name_with_extension.substr(found1 + 1, found2 - found1 - 1);
    }

    bool read_calib_data()
    {
        std::ifstream calib_data_file(calib_name, std::ifstream::in);

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
            else if (count == 6)
            {
                std::stringstream calib_stream(line);
                std::string s;
                std::getline(calib_stream, s, ' ');
                if (s != "Tr_imu_velo")
                    return false;
                for (std::size_t i = 0; i < 3; ++i)
                {
                    for (std::size_t j = 0; j < 4; ++j)
                    {
                        std::getline(calib_stream, s, ' ');
                        Tr_imu_velo(i, j) = stof(s);
                    }
                }
                Tr_imu_velo(3, 0) = Tr_imu_velo(3, 1) = Tr_imu_velo(3, 2) = 0.0;
                Tr_imu_velo(3, 3) = 1.0;
            }

            count++;
        }
        Tr_velo_cam = R_rect * Tr_velo_cam;
        calib_data_file.close();
        return true;
    }

    bool read_pose_data()
    {
        if (useoxts)
        {
            std::ifstream oxts_data_file(oxts_path, std::ifstream::in);
            if (!oxts_data_file)
                cerr << "oxts_data_file does not exist " << std::endl;
            std::string line;
            bool flag = false;
            double er = 6378137.0;
            double scale = 0.0;
            Eigen::Vector3d t_0, t;
            double tx, ty, tz, lat, lon;
            while (getline(oxts_data_file, line))
            {
                std::stringstream oxts_stream(line);
                std::string s;
                std::vector<double> oxts_data;
                while (oxts_stream >> s)
                {
                    oxts_data.push_back(stof(s));
                }
                if (oxts_data.size() < 2)
                    break;
                if (!flag)
                {
                    scale = cos(oxts_data[0] * M_PI / 180.0);
                    tx = scale * oxts_data[1] * M_PI * er / 180.0;
                    ty = scale * er * log(tan((90.0 + oxts_data[0]) * M_PI / 360.0));
                    tz = oxts_data[2];
                    t_0 << tx, ty, tz;
                    flag = true;
                }
                tx = scale * oxts_data[1] * M_PI * er / 180.0;
                ty = scale * er * log(tan((90.0 + oxts_data[0]) * M_PI / 360.0));
                tz = oxts_data[2];
                t << tx, ty, tz;
                tf2::Quaternion q_temp;
                q_temp.setRPY(oxts_data[3], oxts_data[4], oxts_data[5]);
                Eigen::Quaterniond q_odom_curr_tmp;
                q_odom_curr_tmp.x() = q_temp.x();
                q_odom_curr_tmp.y() = q_temp.y();
                q_odom_curr_tmp.z() = q_temp.z();
                q_odom_curr_tmp.w() = q_temp.w();
                Eigen::Matrix3d R;
                R = q_odom_curr_tmp.normalized().toRotationMatrix();
                Eigen::Matrix4d T = Eigen::Matrix4d::Identity(4, 4);
                T.block<3, 3>(0, 0) = R;
                T.block<3, 1>(0, 3) = t - t_0;

                Eigen::Matrix<double, 4, 4> gt_pose;
                gt_pose = Tr_imu_velo * T * Tr_imu_velo.inverse();
                Lidar_pose.push_back(gt_pose);
            }
            oxts_data_file.close();
        }
        else
        {
            std::ifstream pose_data_file(pose_name, std::ifstream::in);
            if (!pose_data_file)
                cerr << "pose_data_file does not exist " << std::endl;
            std::string line;
            while (getline(pose_data_file, line))
            {
                std::stringstream pose_stream(line);
                std::string s;
                Eigen::Matrix<double, 4, 4> gt_pose;
                for (std::size_t i = 0; i < 3; ++i)
                {
                    for (std::size_t j = 0; j < 4; ++j)
                    {
                        std::getline(pose_stream, s, ' ');
                        gt_pose(i, j) = stof(s);
                    }
                }
                gt_pose(3, 0) = gt_pose(3, 1) = gt_pose(3, 1) = 0.0;
                gt_pose(3, 3) = 1.0;
                Lidar_pose.push_back(gt_pose);
            }
            pose_data_file.close();
        }

        return true;
    }

    std::vector<std::vector<float>> read_lidar_file(const std::string lidar_data_path)
    {
        std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
        if (!lidar_data_file)
            cerr << "lidar_data_file does not exist " << std::endl;
        lidar_data_file.seekg(0, std::ios::end);
        const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
        lidar_data_file.seekg(0, std::ios::beg);

        std::vector<float> lidar_data_buffer(num_elements);
        std::vector<std::vector<float>> lidar_data;
        std::vector<float> cloud;
        lidar_data_file.read(reinterpret_cast<char *>(&lidar_data_buffer[0]), num_elements * sizeof(float));
        int count = 0;
        for (int i = 0; i < lidar_data_buffer.size(); i++)
        {
            cloud.push_back(lidar_data_buffer[i]);
            count++;
            if (count % 4 == 0)
            {
                lidar_data.push_back(cloud);
                cloud.clear();
            }
        }
        lidar_data_file.close();
        return lidar_data;
    }

    std::map<int, std::vector<std::vector<float>>> read_detection_file_inonefile(const std::string detection_data_path)
    {
        std::map<int, std::vector<std::vector<float>>> detection_data_map;
        std::ifstream detection_data_file(detection_data_path, std::ifstream::in);
        if (!detection_data_file)
            return detection_data_map;

        std::string line;

        while (getline(detection_data_file, line))
        {
            std::vector<float> nums;
            char *s_input = (char *)line.c_str();
            const char *split = ",";
            char *p = strtok(s_input, split);
            float a;
            while (p != NULL)
            {
                a = atof(p);
                nums.push_back(a);
                p = strtok(NULL, split);
            }
            if (nums[1] != 2.0)
                continue;
            std::vector<float> detection_data_buffer;
            for (int b = 7; b < nums.size(); b++)
            {
                detection_data_buffer.push_back(nums[b]); // 7,8,9,10,11,12,13,14 -> 0,1,2, 3,4,5, 6(r_y),7(alpha)
            }
            detection_data_buffer.back() = nums[6]; // 7(alpha) -> scores
            detection_data_buffer.push_back(nums[14]); // alpha 8  final 0,1,2, 3,4,5, 6(r_y),7(scores), 8 alpha

            std::vector<float> detection2d_data_buffer;
            for (int b = 2; b <= 5; b++)
            {
                detection2d_data_buffer.push_back(nums[b]); // 2d detection in pointrcnn
            }

            detection2d_data_buffer.push_back(nums[6]); // 2d detection score in pointrcnn

            detection_data_map[(int)nums[0]].push_back(detection_data_buffer);
            detection2d_data_map[(int)nums[0]].push_back(detection2d_data_buffer);
        }
        detection_data_file.close();
        return detection_data_map;
    }

    std::vector<std::vector<float>> read_detection_file(const std::string detection_data_path)
    {
        std::vector<std::vector<float>> detection_data_buffer;
        std::ifstream detection_data_file(detection_data_path, std::ifstream::in);
        if (!detection_data_file)
            return detection_data_buffer;
        // cerr << "detection_data_file does not exist " << std::endl;
        std::string line;
        while (getline(detection_data_file, line))
        {
            std::stringstream detection_stream(line);
            std::string s;
            int count = 0;
            std::vector<float> obj_data;
            bool flag = false;
            while (detection_stream >> s)
            {
                if (count == 0)
                {
                    if (s != "Car" && s != "car" && s != "Van" && s != "van")
                    {
                        flag = true;
                        break;
                    }
                }
                else if (count >= 8)
                {
                    obj_data.push_back(stof(s));
                }

                count++;
            }
            if (!flag)
                detection_data_buffer.push_back(obj_data);
        }
        detection_data_file.close();
        return detection_data_buffer;
    }

    void cam_to_velo(std::vector<std::vector<float>> &pose)
    {

        Eigen::Matrix<double, 3, 4> ctv = Tr_velo_cam.inverse().block<3, 4>(0, 0);
        for (int i = 0; i < pose.size(); i++)
        {
            Eigen::Matrix<double, 4, 1> Pose{pose[i][3], pose[i][4], pose[i][5], 1.0};
            Eigen::Matrix<double, 3, 1> Pose_new = ctv * Pose;
            pose[i][3] = Pose_new(0, 0);
            pose[i][4] = Pose_new(1, 0);
            pose[i][5] = Pose_new(2, 0);
        }
    }

    void convert_box_type(std::vector<std::vector<float>> &detection_data)
    {
        // (h,w,l,x,y,z,yaw,socre) -> (x,y,z,l,w,h,yaw,score)
        std::vector<std::vector<float>> box = detection_data;
        for (int i = 0; i < detection_data.size(); i++)
        {
            detection_data[i][0] = box[i][3];
            detection_data[i][1] = box[i][4];
            detection_data[i][2] = box[i][5];
            detection_data[i][3] = box[i][2];
            detection_data[i][4] = box[i][1];
            detection_data[i][5] = box[i][0];
            detection_data[i][6] = (M_PI - box[i][6]) + M_PI / 2.0;
            detection_data[i][2] += box[i][0] / 2.0;
        }
    }
    void get_box_corner(std::vector<float> bbox, Eigen::Vector4f &min_corner, Eigen::Vector4f &max_corner)
    {
        float x = bbox[0];
        float y = bbox[1];
        float z = bbox[2];
        float l = bbox[3];
        float w = bbox[4];
        float h = bbox[5];
        float yaw = bbox[6];
        Eigen::Matrix4d transform_mat;
        transform_mat << cos(yaw), -sin(yaw), 0, x,
            sin(yaw), cos(yaw), 0, y,
            0, 0, 1, z,
            0, 0, 0, 1;
        Eigen::Matrix<double, 8, 4> corner_points;
        corner_points << -l / 2, -w / 2, -h / 2, 1,
            -l / 2, -w / 2, h / 2, 1,
            -l / 2, w / 2, h / 2, 1,
            -l / 2, w / 2, -h / 2, 1,
            l / 2, w / 2, -h / 2, 1,
            l / 2, w / 2, h / 2, 1,
            l / 2, -w / 2, h / 2, 1,
            l / 2, -w / 2, -h / 2, 1;
        corner_points = corner_points * transform_mat.transpose();

        for (int i = 0; i < 3; i++)
        {
            min_corner(i) = corner_points.col(i).minCoeff();
            max_corner(i) = corner_points.col(i).maxCoeff();
        }
        min_corner(3) = max_corner(3) = 1.0;
    }

    void Bbox_filter_cloud(std::vector<float> bbox, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in, std::vector<int> &cloud_index)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_final{new pcl::PointCloud<pcl::PointXYZI>};
        pcl::CropBox<pcl::PointXYZI> clipper;
        Eigen::Vector4f min_corner, max_corner;
        get_box_corner(bbox, min_corner, max_corner);
        clipper.setMin(min_corner);
        clipper.setMax(max_corner);
        clipper.setInputCloud(cloud_in);
        clipper.filter(*cloud_final);
        clipper.setNegative(false);
        cloud_index.clear();

        for (int i = 0; i < cloud_final->points.size(); i++)
        {
            cloud_index.push_back(cloud_final->points[i].intensity);
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fgo_mot");

    ROS_INFO("\033[1;32m---->\033[0m Kitti Helper Started.");

    std::vector<int> sequences;

    if (!getParameter("/kitti_helper/sequences", sequences))
    {
        ROS_WARN("sequence not set, use default value: [1,6,8,10,12,13,14,15,16,18,19]");
        sequences = {1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19};
    }

    for (int i = 0; i < sequences.size(); i++)
    {
        int sequence = sequences[i];
        bool flag;
        kitti2bag k2b(sequence, flag);

        if (!flag)
            continue;

        k2b.load_velodyne_tf_image_BoxArray();

        // 每个序列开始前休息（第一个序列也休息）
        if (i > 0) {
            ROS_INFO("Waiting 10 seconds before starting sequence %d...", sequences[i]);
            ros::Duration(10.0).sleep();
        }
    }
}