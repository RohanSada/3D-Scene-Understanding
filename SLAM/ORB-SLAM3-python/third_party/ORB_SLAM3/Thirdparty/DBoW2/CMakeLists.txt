cmake_minimum_required(VERSION 3.10)
project(DBoW2)

# --- START: CONDA CONFIGURATION ---
# This forces DBoW2 to find OpenCV and other libs inside your active Conda env
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "DBoW2: Found Conda Environment: $ENV{CONDA_PREFIX}")
    # Tell CMake to look for packages (like OpenCV) here first
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    # Add includes and lib paths manually to be safe
    include_directories("$ENV{CONDA_PREFIX}/include")
    link_directories("$ENV{CONDA_PREFIX}/lib")
else()
    message(WARNING "DBoW2: CONDA_PREFIX is not set. Please activate your conda environment!")
endif()
# --- END: CONDA CONFIGURATION ---

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

set(HDRS_DBOW2
  DBoW2/BowVector.h
  DBoW2/FORB.h 
  DBoW2/FClass.h       
  DBoW2/FeatureVector.h
  DBoW2/ScoringObject.h   
  DBoW2/TemplatedVocabulary.h)

set(SRCS_DBOW2
  DBoW2/BowVector.cpp
  DBoW2/FORB.cpp      
  DBoW2/FeatureVector.cpp
  DBoW2/ScoringObject.cpp)

set(HDRS_DUTILS
  DUtils/Random.h
  DUtils/Timestamp.h)

set(SRCS_DUTILS
  DUtils/Random.cpp
  DUtils/Timestamp.cpp)

# Because we set CMAKE_PREFIX_PATH above, this will now find the Conda OpenCV
find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 3.0 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 3.0 not found.")
   endif()
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(DBoW2 SHARED ${SRCS_DBOW2} ${SRCS_DUTILS})
target_link_libraries(DBoW2 ${OpenCV_LIBS})