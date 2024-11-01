cmake_minimum_required(VERSION 3.10)
project(my_project)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置TensorFlow路径
set(TENSORFLOW_ROOT "/home/z2/tensorflow")
set(TFLITE_BUILD_PATH "${TENSORFLOW_ROOT}/bazel-bin/tensorflow/lite")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# 设置Flatbuffers路径（使用bazel缓存中的路径）
set(FLATBUFFERS_INCLUDE_DIR "/home/zyi/.cache/bazel/_bazel_zyi/2e52e7c451514eb6f8d9cfc1ea4915ea/external/flatbuffers/include")

# 查找OpenCV包
find_package(OpenCV REQUIRED)

# TensorFlow Lite 头文件路径
set(TFLITE_INCLUDE_DIRS
    ${TENSORFLOW_ROOT}
    ${TENSORFLOW_ROOT}/tensorflow/lite/tools/pip_package/gen/tflite_pip/tensorflow/lite/
    ${TENSORFLOW_ROOT}/tensorflow/lite/tools/pip_package/gen/tflite_pip/tensorflow/include/
    ${TENSORFLOW_ROOT}/tensorflow/lite/c
    ${TENSORFLOW_ROOT}/tensorflow/lite/core/api
    ${TENSORFLOW_ROOT}/tensorflow/lite/kernels
    ${FLATBUFFERS_INCLUDE_DIR}  # 添加Flatbuffers头文件路径
    ${TENSORFLOW_ROOT}/bazel-tensorflow/external/flatbuffers/include  # 添加额外的Flatbuffers路径
)

# 添加所有源文件
file(GLOB SOURCE_FILES 
    "*.cpp"
    "*.cc"
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# 包含目录
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TFLITE_INCLUDE_DIRS}
    ${FLATBUFFERS_INCLUDE_DIR}
    /home/zyi/.cache/bazel/_bazel_zyi/2e52e7c451514eb6f8d9cfc1ea4915ea/execroot/org_tensorflow/bazel-out/k8-opt/bin/external/flatbuffers/_virtual_includes/flatbuffers  # 添加额外的Flatbuffers包含路径
)

# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${OpenCV_LIBS}
    ${TFLITE_BUILD_PATH}/libtensorflowlite.so
    flatbuffers  # 系统Flatbuffers库
    dl
    pthread
)

# 编译选项
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -O2
    -fPIC
)

# 确保运行时能找到库
set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH "${TFLITE_BUILD_PATH}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# 打印一些调试信息
message(STATUS "FLATBUFFERS_INCLUDE_DIR: ${FLATBUFFERS_INCLUDE_DIR}")
message(STATUS "TFLITE_BUILD_PATH: ${TFLITE_BUILD_PATH}")