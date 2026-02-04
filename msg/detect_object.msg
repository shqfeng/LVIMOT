std_msgs/Header                 header

uint32                          id
string                          label
float32                         score   #Score as defined by the detection, Optional
int32[]                         objPointIndex 
float32                         average_x
float32                         average_y
float32                         average_z
float32                         alpha

jsk_recognition_msgs/BoundingBox bounding_box
boundingBox2D bounding_box2d

sensor_msgs/PointCloud2         pointcloudFull
sensor_msgs/PointCloud2         pointcloudCorner
sensor_msgs/PointCloud2         pointcloudSurf

sensor_msgs/PointCloud2         pointcloudConvexhull