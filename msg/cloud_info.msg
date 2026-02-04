# Cloud Info
Header header 

int32 seq_id
int32 frame_id

int32[] startRingIndex
int32[] endRingIndex

int32[]  pointColInd # point column index in range image
float32[] pointRange # point range 
int32[] groundFlag

int32[] objIndex

float32 startOrientation
float32 endOrientation
float32 orientationDiff

index_vector[] objIndexVec
detect_object[] objArray

int64 imuAvailable
int64 odomAvailable

# Attitude for LOAM initialization
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Initial guess from imu pre-integration
float32 initialGuessX
float32 initialGuessY
float32 initialGuessZ
float32 initialGuessRoll
float32 initialGuessPitch
float32 initialGuessYaw

# Point cloud messages
sensor_msgs/PointCloud2 full_cloud_projection  # original cloud
sensor_msgs/PointCloud2 segment_cloud  # segment_cloud
sensor_msgs/PointCloud2 ground_cloud  # ground_cloud
sensor_msgs/PointCloud2 nonground_cloud  # nonground_cloud
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature
sensor_msgs/PointCloud2 cloud_feature_origin   # extracted origin feature

sensor_msgs/PointCloud2 cloud_corner_after_odom    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface_after_odom   # extracted surface feature
sensor_msgs/PointCloud2 full_cloud_after_odom  # original cloud
sensor_msgs/PointCloud2 Cloud_in_Camera  # original cloud

nav_msgs/Odometry pose_lidar_odom
nav_msgs/Odometry odom_after_odom
nav_msgs/Odometry incremental_odom_after_odom


# 3rd party messages
sensor_msgs/PointCloud2 key_frame_cloud
sensor_msgs/PointCloud2 key_frame_color
sensor_msgs/PointCloud2 key_frame_poses
sensor_msgs/PointCloud2 key_frame_map

float32[] P2
float32[] R_rect
