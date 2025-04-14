
add_subdirectory (simd)
add_subdirectory (io)
add_subdirectory (quantization)

file (GLOB CPP_SRCS "*.cpp")
list (FILTER CPP_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_CONJUGATE_GRAPH_SRCS "impl/*.cpp")
list (FILTER CPP_CONJUGATE_GRAPH_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_INDEX_SRCS "index/*.cpp")
list (FILTER CPP_INDEX_SRCS EXCLUDE REGEX "_test.cpp")

file (GLOB CPP_DATA_CELL_SRCS "data_cell/*.cpp")
list (FILTER CPP_DATA_CELL_SRCS EXCLUDE REGEX "_test.cpp")

file(GLOB CPP_ALGORITHM_PUCK_SRCS "algorithm/puck/*.cpp")
set_source_files_properties(${CPP_ALGORITHM_PUCK_SRCS} PROPERTIES COMPILE_FLAGS "-msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 -mavx -mf16c -mavx2 -mfma -mf16c -mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -mavx512vbmi -fopenmp -fopenmp-simd")

file (GLOB CPP_ALGORITHM_SRCS "algorithm/*.cpp")
file (GLOB CPP_ALGORITHM_EXTRA_SRCS "algorithm/**/*.cpp")
list (APPEND CPP_ALGORITHM_SRCS ${CPP_ALGORITHM_EXTRA_SRCS})
list (FILTER CPP_ALGORITHM_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_ALGORITHM_SRCS EXCLUDE REGEX "algorithm/puck/.*\.cpp$")
list (APPEND CPP_ALGORITHM_SRCS ${CPP_ALGORITHM_PUCK_SRCS})

file (GLOB CPP_UTILS_SRCS "utils/*.cpp")
list (FILTER CPP_UTILS_SRCS EXCLUDE REGEX "_test.cpp")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fopenmp-simd")
set (VSAG_SRCS ${CPP_SRCS} ${CPP_INDEX_SRCS} ${CPP_CONJUGATE_GRAPH_SRCS}
        ${CPP_DATA_CELL_SRCS} ${CPP_ALGORITHM_SRCS} ${CPP_UTILS_SRCS})

add_library (vsag SHARED ${VSAG_SRCS})
add_library (vsag_static STATIC ${VSAG_SRCS})

set (VSAG_DEP_LIBS diskann pthread m dl simd io quantizer fmt::fmt-header-only nlohmann_json::nlohmann_json roaring glog gflags)
target_link_libraries (vsag ${VSAG_DEP_LIBS} coverage_config)
target_link_libraries (vsag_static ${VSAG_DEP_LIBS} coverage_config)

maybe_add_dependencies (vsag spdlog roaring openblas boost mkl)
maybe_add_dependencies (vsag_static spdlog roaring openblas boost mkl)
