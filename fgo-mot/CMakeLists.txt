cmake_minimum_required(VERSION 3.10.2) 
project(fgo_mot)

# 设置编译选项 - 使用现代CMake方式
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE Debug)

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rosconsole
  rospy
  cv_bridge
  image_transport
  pcl_ros
  pcl_conversions
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  jsk_recognition_msgs
)


add_message_files(
  DIRECTORY msg
  FILES
  boundingBox2D.msg
  boundingBox2DArray.msg
  index_vector.msg
  cloud_info.msg
  detect_object.msg
)
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
  jsk_recognition_msgs
)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
find_package(GTSAM REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  DEPENDS PCL
)

# 包含目录
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# 链接目录
link_directories(
  ${catkin_LIBRARY_DIRS}
  ${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${CERES_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
)

# 可执行文件
add_executable(kittiHelper src/kittiHelper.cpp)
add_dependencies(kittiHelper ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(kittiHelper 
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES} 
    ${OpenCV_LIBRARIES}
)

add_executable(imageProjection src/imageProjection.cpp)
add_dependencies(imageProjection ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(imageProjection 
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES} 
    ${OpenCV_LIBRARIES}
)

add_executable(featureExtraction src/featureExtraction.cpp)
add_dependencies(featureExtraction ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(featureExtraction 
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES} 
    ${OpenCV_LIBRARIES}
)

add_executable(lidarOdometry src/lidarOdometry.cpp)
add_dependencies(lidarOdometry ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(lidarOdometry 
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES} 
    ${OpenCV_LIBRARIES} 
    ${CERES_LIBRARIES}
)

# objectTracker 
add_executable(objectTracker src/objectTracker.cpp)
add_dependencies(objectTracker ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(objectTracker
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${CERES_LIBRARIES}
    ${GTSAM_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
)


