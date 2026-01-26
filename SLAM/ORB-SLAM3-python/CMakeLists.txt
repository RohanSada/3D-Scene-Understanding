cmake_minimum_required(VERSION 3.10)
project(orbslam3)

# 1. SETUP: Force C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. FORCE CONDA PATHS & IGNORE HOMEBREW
set(CONDA_PATH $ENV{CONDA_PREFIX})
set(CMAKE_PREFIX_PATH ${CONDA_PATH})
set(CMAKE_FIND_ROOT_PATH ${CONDA_PATH})

# CRITICAL: Tell CMake to ignore Homebrew explicitly
list(APPEND CMAKE_IGNORE_PATH "/opt/homebrew")
list(APPEND CMAKE_IGNORE_PATH "/usr/local")

# CRITICAL: Tell the linker to look in Conda/lib
link_directories(${CONDA_PATH}/lib)

# 3. FIND PACKAGES
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)

# CRITICAL: Force MODULE mode for GLEW. 
# This prevents picking up 'glew-config.cmake' from Homebrew 
# and forces it to look for the library files in Conda using GLEW_ROOT.
set(GLEW_ROOT ${CONDA_PATH})
find_package(GLEW REQUIRED MODULE)

# 4. SUBDIRECTORIES
add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/ORB_SLAM3)

# 5. BUILD MODULE
pybind11_add_module(orbslam3 
    src/ORBSLAM3Wrapper.cpp
    src/NDArrayConverter.cpp
)

# 6. LINK LIBRARIES
target_link_libraries(orbslam3
    PRIVATE
    ORB_SLAM3
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${Boost_LIBRARIES}
    ${GLEW_LIBRARIES}
)

# 7. INCLUDE DIRECTORIES
target_include_directories(orbslam3
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    
    # Internal Paths
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/ORB_SLAM3
    ${CMAKE_SOURCE_DIR}/third_party/ORB_SLAM3/include
    ${CMAKE_SOURCE_DIR}/third_party/ORB_SLAM3/include/CameraModels
    ${CMAKE_SOURCE_DIR}/third_party/ORB_SLAM3/Thirdparty/Sophus
)

target_compile_definitions(orbslam3 
    PRIVATE 
    VERSION_INFO=${EXAMPLE_VERSION_INFO}
)