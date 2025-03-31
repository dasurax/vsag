# Fetch gflags
FetchContent_Declare(
    gflags
    URL https://github.com/gflags/gflags/archive/v2.2.2.tar.gz
    URL_HASH SHA256=34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf
)
# Make sure the dependency is downloaded

#set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries only" FORCE) # 选择静态库
#set(GFLAGS_BUILD_TESTING OFF CACHE BOOL "Disable gflags tests" FORCE) # 禁用测试
FetchContent_MakeAvailable(gflags)

# Fetch glog
FetchContent_Declare(
    glog
    #URL file:///home/suguan.dx/vecbase_code/third_party/vsag/build-release/_deps/glog-subbuild/glog-populate-prefix/src/v0.7.1.zip
    URL https://github.com/google/glog/archive/v0.7.1.zip
    URL_HASH SHA256=c17d85c03ad9630006ef32c7be7c65656aba2e7e2fbfc82226b7e680c771fc88
)
# Make sure the dependency is downloaded
FetchContent_MakeAvailable(glog)

message("gflags_SOURCE_DIR = ${CMAKE_BINARY_DIR}/_deps/gflags-build/include/gflags")
message("glog_SOURCE_DIR = ${glog_SOURCE_DIR}/src/glog")
include_directories(${CMAKE_BINARY_DIR}/_deps/gflags-build/include/gflags)
include_directories(${glog_SOURCE_DIR}/src/glog)