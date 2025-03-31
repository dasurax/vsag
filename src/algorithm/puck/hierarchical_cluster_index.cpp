//Copyright (c) 2023 Baidu, Inc.  All Rights Reserved.
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
/**
 * @file hierarchical_cluster.cpp
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2022/9/02 11:35
 * @brief
 *
 **/

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <thread>
#include <unordered_set>
#include <functional>
#include <random>
#include <memory>
#include <set>
#include <omp.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <glog/logging.h>
#include "algorithm/puck/search_context.h"
#include "algorithm/puck/imitative_heap.h"
#include "algorithm/puck/max_heap.h"
#include "algorithm/puck/hierarchical_cluster_index.h"
#include "algorithm/puck/hnsw_distfunc_opt_impl_inline.h"
#include "algorithm/puck/puck_gflags.h"
#include "algorithm/puck/search_context.h"
#include "algorithm/puck/md5.h"


extern "C" {

    /* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

    int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace puck {
using base::FunctionTimer;

long readProcStatus2(const std::string& name) {
    std::ifstream file("/proc/self/status");
    if (!file.is_open()) {
        return -1;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.substr(0, name.size()) == name) {
            // 假设格式为 "name: value"
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                return std::stol(line.substr(pos + 2));
            }
        }
    }
    return -1; // 如果没有找到对应的行，则返回-1
}

DEFINE_bool(kmeans_init_berkeley, true, "using kmeans_init_berkeley");
DEFINE_int32(kmeans_iterations_count, 2, "iterations count");
DEFINE_int32(thread_chunk_size, 10000, "chunk size of each thread");
DEFINE_int32(train_points_count, 500000, "used for HierarchicalClusterIndex train clusters");
DEFINE_string(train_fea_file_name, "mid-data/train_clusters.dat",
              "random sampling for HierarchicalClusterIndex train clusters");
DEFINE_int32(single_build_max_points, 1, "during single build process, max points can be saved in memory");

void matrix_multiplication(const float* left, const float* right,
                           FINTEGER m, FINTEGER n, FINTEGER k,
                           const char* transp,
                           float* result) {
    float alpha = 1;
    float beta = 0;
    FINTEGER lda = transp[0] == 'N' ? m : k;
    FINTEGER ldb = transp[1] == 'N' ? k : n;
    FINTEGER ldc = m;
    sgemm_((char*)transp, (char*)(transp + 1), &m, &n, &k,
           &alpha, left, &lda, right, &ldb, &beta, result, &ldc);

}

//检查文件长度
bool check_file_length_info(const std::string& file_name,
                            const uint64_t file_length) {
    int fd = -1;
    fd = open(file_name.c_str(), O_RDONLY);
    struct stat st;

    if (fd == -1 || -1 == fstat(fd, &st) || (file_length != (uint64_t)st.st_size)) {
        LOG(ERROR) << "check file length has error, file name = " << file_name 
                << " expected file_length: " << file_length << " st_size:" << st.st_size;
        close(fd);
        return false;
    }
    close(fd);
    return true;
}

//写码本
int write_fvec_format(const char* file_name, const uint32_t dim, const uint64_t n, const float* fea_vocab) {
    LOG(INFO) << "start write_fvec_format " << file_name;
    FILE* out_fvec_init = fopen(file_name, "wb");

    if (!out_fvec_init) {
        LOG(ERROR) << "cannot open " << file_name << " for writing";
        fclose(out_fvec_init);
        return -1;
    }

    int ret = 0;

    for (uint64_t i = 0; i < n; ++i) {
        if (fwrite(&dim, sizeof(int), 1, out_fvec_init) != 1) {
            LOG(ERROR) << "write error";
            ret = -1;
            break;
        }

        if (fwrite(fea_vocab + (size_t)i * dim, sizeof(float), dim, out_fvec_init) != (size_t)dim) {
            LOG(ERROR) << "write error";
            ret = -1;
            break;
        }
    }

    fclose(out_fvec_init);
    return ret;
}

int read_fvec_format(FILE* fvec_init, uint32_t dim, uint64_t n, float* v) {

    for (uint64_t i = 0; i < n; ++i) {
        int cur_dim = 0;

        if (fread(&cur_dim, sizeof(int), 1, fvec_init) != 1 || cur_dim != (int)dim) {
            LOG(ERROR) << "read_fvec_format : dim error";
            return -1;
        }

        if (fread((void*)(v + i * dim), sizeof(v[0]), dim, fvec_init) != dim) {
            LOG(ERROR) << "read_fvec_format : read feature error";
            return -1;
        }
    }

    return n;
}


int read_fvec_format(const char* file_name, uint32_t dim, uint64_t n, float* v) {
    size_t expect_len = n * (sizeof(int) + sizeof(v[0]) * dim);

    if (!check_file_length_info(file_name, expect_len)) {
        return -1;
    }

    FILE* fvec_init = fopen(file_name, "rb");

    if (!fvec_init) {
        LOG(ERROR) << "read_fvec_format: cannot open for writing";
        return -1;
    }

    int ret = read_fvec_format(fvec_init, dim, n, v);
    fclose(fvec_init);
    return ret;
}

void InitializeLogger(int choice) {
    if (!google::IsGoogleLoggingInitialized()){
        google::InitGoogleLogging("puck");
    }

    if (FLAGS_need_log_file) {
        std::string log_file_name = puck::FLAGS_puck_log_file;
        google::SetLogDestination(static_cast<google::LogSeverity>(choice), log_file_name.c_str());
        google::SetStderrLogging(google::GLOG_INFO);
    }
}

void HierarchicalClusterIndex::init_params_value() {
    _coarse_clusters = nullptr;
    _coarse_vocab = nullptr;
    _coarse_norms = nullptr;
    _fine_vocab = nullptr;
    _fine_norms = nullptr;
    _model = nullptr;
    _memory_to_local = nullptr;
    _all_feature = nullptr;
    _all_feature_int8 = nullptr;
    _int8_multipliers = nullptr;
    _dp_norms = nullptr;
}

HierarchicalClusterIndex::HierarchicalClusterIndex() {
    _conf.index_type = IndexType::HIERARCHICAL_CLUSTER;
    init_params_value();
    omp_set_num_threads(FLAGS_threads_count);
    LOG(INFO) << "FLAGS_threads_count: " << FLAGS_threads_count; 
    omp_set_dynamic(true);
}

HierarchicalClusterIndex::~HierarchicalClusterIndex() {
    if (_model != nullptr) {
        free(_model);
        _model = nullptr;
    }
    if (_all_feature != nullptr) {
        free(_all_feature);
        _all_feature = nullptr;
    }
    
    if (_all_feature_int8 != nullptr) {
        free(_all_feature_int8);
        _all_feature_int8 = nullptr;
    }
    init_params_value();
}

int HierarchicalClusterIndex::init() {
    LOG(INFO) << "start init index.";

    LOG(INFO) << "MEM0: " << readProcStatus2("VmRSS:");
    //从文件获取配置信息
    if (read_model_file() != 0) {
        LOG(ERROR) << "read_model_file has error.";
        return -1;
    }
    LOG(INFO) << "MEM00: " << readProcStatus2("VmRSS:") << _conf.brutal_search;
    if (_conf.brutal_search) {
        LOG(INFO) << "MEM000: " << readProcStatus2("VmRSS:") << _conf.brutal_search;
        int ret = init_on_brutal_serving();
        LOG(INFO) << "MEM001: " << readProcStatus2("VmRSS:") << _conf.brutal_search;
        if (ret != 0) {
            return ret;
        }
        init_context_pool();
        return _conf.adaptive_search_param();
    }

    LOG(INFO) << "MEM01: " << readProcStatus2("VmRSS:") << _conf.brutal_search;

    LOG(INFO) << "MEM1: " << readProcStatus2("VmRSS:");
    if (check_index_type() != 0) {
        LOG(ERROR) << "check_index_type has error.";
        return -1;
    }
    LOG(INFO) << "MEM2: " << readProcStatus2("VmRSS:");
    //初始化内存
    if (init_model_memory() != 0) {
        LOG(ERROR) << "init_model_memory has error.";
        return -1;
    }
    LOG(INFO) << "MEM3: " << readProcStatus2("VmRSS:");
    //读码本
    if (read_codebooks() != 0) {
        LOG(ERROR) << "read_codebooks has error.";
        return -1;
    }
    LOG(INFO) << "MEM4: " << readProcStatus2("VmRSS:");
    LOG(INFO) << "start init index.";
    //读与样本有关的索引部分

    if (read_index() != 0) {
        LOG(ERROR) << "read_index has error.";
        return -1;
    }
    LOG(INFO) << "MEM5: " << readProcStatus2("VmRSS:");
    //if (true) return 0;
    LOG(INFO) << "start init index.";
    init_context_pool();
    LOG(INFO) << "start init index.";
    LOG(INFO) << "MEM6: " << readProcStatus2("VmRSS:");
    //调整默认的检索参数 & 检索参数检查
    auto ret = _conf.adaptive_search_param();
    LOG(INFO) << "conf.filter_topk: " << _conf.filter_topk;
    return ret;
}

int HierarchicalClusterIndex::read_codebooks() {
    LOG(INFO) << "start load index file " << _conf.coarse_codebook_file_name;
    //一级聚类中心文件长度检查
    uint64_t coarse_vocab_length = (uint64_t)_conf.coarse_cluster_count * _conf.feature_dim * sizeof(
                                       float);
    uint64_t expect_length = (uint64_t)_conf.coarse_cluster_count * sizeof(int) + coarse_vocab_length;
    bool is_ok = check_file_length_info(_conf.coarse_codebook_file_name, expect_length);

    if (is_ok == false) {
        return -1;
    }

    int ret = read_fvec_format(_conf.coarse_codebook_file_name.c_str(), _conf.feature_dim,
                               _conf.coarse_cluster_count, _coarse_vocab);
    if (ret != (int)_conf.coarse_cluster_count) {
        LOG(FATAL) << "load file error, file : " << _conf.coarse_codebook_file_name << " feature_dim : " <<
                   _conf.feature_dim << " number : " << _conf.coarse_cluster_count << " return code : " << ret;
        return -2;
    }

    for (uint32_t i = 0; i < _conf.coarse_cluster_count; ++i) {
        _coarse_norms[i] = cblas_sdot(_conf.feature_dim, _coarse_vocab + _conf.feature_dim * i, 1,
                                      _coarse_vocab + _conf.feature_dim * i, 1) / 2;
    }

    LOG(INFO) << "start load index file " << _conf.fine_codebook_file_name;
    //二级聚类中心文件长度检查
    uint64_t fine_vocab_length = (uint64_t)_conf.fine_cluster_count * _conf.feature_dim * sizeof(float);
    expect_length = (uint64_t)_conf.fine_cluster_count * sizeof(int) + fine_vocab_length;
    is_ok = check_file_length_info(_conf.fine_codebook_file_name, expect_length);

    if (is_ok == false) {
        return -1;
    }

    ret = read_fvec_format(_conf.fine_codebook_file_name.c_str(), _conf.feature_dim,
                           _conf.fine_cluster_count, _fine_vocab);

    if (ret != (int)_conf.fine_cluster_count) {
        LOG(FATAL) << "load file error, file : " << _conf.fine_codebook_file_name << " feature_dim : " <<
                   _conf.feature_dim << " number : " << _conf.coarse_cluster_count << " return code : " << ret;
        return -3;
    }

    std::vector<float> fine_norms(_conf.fine_cluster_count);

    for (uint32_t i = 0; i < _conf.fine_cluster_count; ++i) {
        fine_norms[i] = cblas_sdot(_conf.feature_dim, _fine_vocab + _conf.feature_dim * i, 1,
                                   _fine_vocab + _conf.feature_dim * i, 1) / 2;
    }

    std::vector<float> coarse_fine_products(_conf.fine_cluster_count * _conf.coarse_cluster_count);
    //矩阵乘
    matrix_multiplication(_fine_vocab, _coarse_vocab, _conf.fine_cluster_count, _conf.coarse_cluster_count,
                          _conf.feature_dim, "TN", coarse_fine_products.data());

    for (uint32_t i = 0; i < _conf.fine_cluster_count * _conf.coarse_cluster_count; ++i) {
        int coarse_id = i / _conf.fine_cluster_count;
        int fine_id = i % _conf.fine_cluster_count;
        FineCluster& cur_fine_cluster = _coarse_clusters[coarse_id].fine_cell_list[fine_id];
        cur_fine_cluster.memory_idx_start = 0;
        cur_fine_cluster.stationary_cell_dist = fine_norms[fine_id] + coarse_fine_products[i];
    }   

    if (_conf.spilled_copy_num > 0) {
        // CC
        matrix_multiplication(_coarse_vocab, _coarse_vocab, 
                                _conf.coarse_cluster_count, _conf.coarse_cluster_count,
                                _conf.feature_dim, "TN", _coarse_inner_products);

        // FF
        matrix_multiplication(_fine_vocab, _fine_vocab, 
                                _conf.fine_cluster_count, _conf.fine_cluster_count,
                                _conf.feature_dim, "TN", _fine_inner_products);
        
        // CF
        matrix_multiplication(_fine_vocab, _coarse_vocab, 
                                _conf.fine_cluster_count, _conf.coarse_cluster_count,
                                _conf.feature_dim, "TN", _coarse_fine_products);
    }

    LOG(INFO) << "HierarchicalClusterIndex::read_codebooks Suc.";
    return 0;
}

int HierarchicalClusterIndex::fill_codebooks() {
    for (uint32_t i = 0; i < _conf.coarse_cluster_count; ++i) {
        _coarse_norms[i] = cblas_sdot(_conf.feature_dim, _coarse_vocab + _conf.feature_dim * i, 1,
                                      _coarse_vocab + _conf.feature_dim * i, 1) / 2;
    }

    std::vector<float> fine_norms(_conf.fine_cluster_count);

    for (uint32_t i = 0; i < _conf.fine_cluster_count; ++i) {
        fine_norms[i] = cblas_sdot(_conf.feature_dim, _fine_vocab + _conf.feature_dim * i, 1,
                                   _fine_vocab + _conf.feature_dim * i, 1) / 2;
    }

    std::vector<float> coarse_fine_products(_conf.fine_cluster_count * _conf.coarse_cluster_count);
    //矩阵乘
    matrix_multiplication(_fine_vocab, _coarse_vocab, _conf.fine_cluster_count, _conf.coarse_cluster_count,
                          _conf.feature_dim, "TN", coarse_fine_products.data());

    for (uint32_t i = 0; i < _conf.fine_cluster_count * _conf.coarse_cluster_count; ++i) {
        int coarse_id = i / _conf.fine_cluster_count;
        int fine_id = i % _conf.fine_cluster_count;
        FineCluster& cur_fine_cluster = _coarse_clusters[coarse_id].fine_cell_list[fine_id];
        cur_fine_cluster.memory_idx_start = 0;
        cur_fine_cluster.stationary_cell_dist = fine_norms[fine_id] + coarse_fine_products[i];
    }   
    return 0;
}

int HierarchicalClusterIndex::save_codebooks() const {
    LOG(INFO) << "HierarchicalClusterIndex start save index";
    save_model_file();
    //写一级聚类中心索引文件
    int ret = write_fvec_format(_conf.coarse_codebook_file_name.c_str(), _conf.feature_dim,
                                _conf.coarse_cluster_count, _coarse_vocab);

    if (ret != 0) {
        LOG(ERROR) << "write codebooks has error, ret = " << ret;
        return -1;
    }

    //写二级聚类中心索引文件
    ret = write_fvec_format(_conf.fine_codebook_file_name.c_str(), _conf.feature_dim, _conf.fine_cluster_count,
                            _fine_vocab);

    if (ret != 0) {
        LOG(ERROR) << "write codebooks has error, ret = " << ret;
        return -1;
    }

    return 0;
}

int HierarchicalClusterIndex::read_index() {
    uint32_t cell_cnt = _conf.coarse_cluster_count * _conf.fine_cluster_count;
    auto total_point_count = _conf.total_point_count;
    if (_conf.spilled_copy_num > 0) {
        total_point_count *= 2;
    }
    std::vector<uint32_t> cell_start_memory_idx(cell_cnt + 1, total_point_count);
    std::vector<uint32_t> local_to_memory_idx(total_point_count);
    //if (true) return 0;
    //读取local idx，计算与memory idx的映射关系
    LOG(INFO) << "MEM4.1: " << readProcStatus2("VmRSS:");
    
    if (FLAGS_convert_local_to_memory_while_build) {
        LOG(INFO) << "MEM4.10: " << readProcStatus2("VmRSS:");
        if (convert_local_to_memory_idx(cell_start_memory_idx.data(), local_to_memory_idx.data()) != 0) {
            return -1;
        }
        LOG(INFO) << "MEM4.11: " << readProcStatus2("VmRSS:");
        if (read_feature_index(nullptr) != 0) {
            return -1;
        }
        LOG(INFO) << "MEM4.12: " << readProcStatus2("VmRSS:");
    } else {
        LOG(INFO) << "MEM4.20: " << readProcStatus2("VmRSS:");
        if (convert_local_to_memory_idx(cell_start_memory_idx.data(), local_to_memory_idx.data()) != 0) {
            return -1;
        }
        LOG(INFO) << "MEM4.2: " << readProcStatus2("VmRSS:");
        if (read_feature_index(local_to_memory_idx.data()) != 0) {
            return -1;
        }
        LOG(INFO) << "MEM4.3: " << readProcStatus2("VmRSS:");
    }

    return 0;
}

int HierarchicalClusterIndex::read_src_feature_index(char * fileBaseAddress, uint64_t fileLength) {
    // 原始特征文件长度检查
    u_int64_t all_feature_length = (u_int64_t)_conf.total_point_count * _conf.feature_dim * sizeof(float);
    if (all_feature_length != fileLength) {
        LOG(ERROR) << "Unexpected source feature length: " << fileLength << ", expected: " << all_feature_length << ".";
        return -1;
    }
    // NOTE: In order to be consistent with other `_all_feature` occurrances,
    // we allocate memory instead of using the fileBaseAddress directly.
    _all_feature = (float*)malloc(all_feature_length);
    memcpy((void *)_all_feature, (const void *)fileBaseAddress, all_feature_length);
    return 0;
}

int HierarchicalClusterIndex::read_feature_index(uint32_t* local_to_memory_idx, bool reset_quantized_feature) {
    LOG(INFO) << "start load index file " << _conf.feature_file_name 
            << " " << FLAGS_whether_pq << " " << FLAGS_use_int8;
    
    if (false) {
        //原始特征文件长度检查
        u_int64_t all_feature_length = (u_int64_t)_conf.total_point_count * _conf.feature_dim * sizeof(float);
        u_int64_t expect_length = (u_int64_t)_conf.total_point_count * sizeof(int) + all_feature_length;
        bool is_ok = check_file_length_info(_conf.feature_file_name, expect_length);

        if (is_ok == false) {
            return -1;
        }

        //申请内存
        {
            void* memb = nullptr;
            int32_t pagesize = getpagesize();
            //size_t all_feature_length = (size_t)_conf.total_point_count * _conf.feature_dim * sizeof(float);
            size_t size = all_feature_length + (pagesize - all_feature_length % pagesize);
            int err = posix_memalign(&memb, pagesize, size);

            if (err != 0) {
                std::runtime_error("alloc_aligned_mem_failed errno=" + errno);
                return -1;
            }

            _all_feature = reinterpret_cast<float*>(memb);
        }
        LOG(INFO) << "_all_feature: " << (_all_feature == nullptr);
        FILE* init_feature_stream = fopen(_conf.feature_file_name.c_str(), "r");

        //原始特征加载至内存，按memory idx的顺序
        for (uint32_t idx = 0; idx < _conf.total_point_count; ++idx) {
            uint64_t memory_offset = (u_int64_t)local_to_memory_idx[idx] * (u_int64_t)_conf.feature_dim;
            read_fvec_format(init_feature_stream, _conf.feature_dim, 1, _all_feature + memory_offset);
        }

        fclose(init_feature_stream);
    }

    if (!FLAGS_use_int8) {
    //if (true) {
        //原始特征文件长度检查
        u_int64_t all_feature_length = (u_int64_t)_conf.total_point_count * _conf.feature_dim * sizeof(float);
        u_int64_t expect_length = (u_int64_t)_conf.total_point_count * sizeof(int) + all_feature_length;
        bool is_ok = check_file_length_info(_conf.feature_file_name, expect_length);

        if (is_ok == false) {
            return -1;
        }

        //申请内存
        {
            void* memb = nullptr;
            int32_t pagesize = getpagesize();
            //size_t all_feature_length = (size_t)_conf.total_point_count * _conf.feature_dim * sizeof(float);
            size_t size = all_feature_length + (pagesize - all_feature_length % pagesize);
            int err = posix_memalign(&memb, pagesize, size);

            if (err != 0) {
                std::runtime_error("alloc_aligned_mem_failed errno=" + errno);
                return -1;
            }

            _all_feature = reinterpret_cast<float*>(memb);
        }
        LOG(INFO) << "_all_feature: " << (_all_feature == nullptr);
        FILE* init_feature_stream = fopen(_conf.feature_file_name.c_str(), "r");

        //原始特征加载至内存，按memory idx的顺序
        for (uint32_t idx = 0; idx < _conf.total_point_count; ++idx) {
            uint64_t memory_idx = idx;
            if (local_to_memory_idx != nullptr) {
                memory_idx = (u_int64_t)local_to_memory_idx[idx];
            }
            uint64_t memory_offset = memory_idx * (u_int64_t)_conf.feature_dim;
            if (_conf.spilled_copy_num > 0) {
                memory_offset = (u_int64_t)idx * (u_int64_t)_conf.feature_dim;
            }
            read_fvec_format(init_feature_stream, _conf.feature_dim, 1, _all_feature + memory_offset);
        }

        fclose(init_feature_stream);
    } else {
        //原始特征文件长度检查
        u_int64_t int8_dataset_length = (u_int64_t)_conf.total_point_count * _conf.feature_dim * sizeof(int8_t);
        u_int64_t int8_multipliers_length  = _conf.feature_dim * sizeof(float);
        u_int64_t dp_norms_length = (u_int64_t)_conf.total_point_count * sizeof(float);
        u_int64_t all_feature_length = int8_dataset_length + int8_multipliers_length + dp_norms_length;
        u_int64_t expect_length = all_feature_length;
        bool is_ok = check_file_length_info(_conf.int8_dataset_file_name, int8_dataset_length);
        is_ok = is_ok && check_file_length_info(_conf.int8_multipliers_file_name, int8_multipliers_length);
        is_ok = is_ok && check_file_length_info(_conf.dp_norms_file_name, dp_norms_length);

        if (is_ok == false) {
            return -1;
        }
        
        //申请内存
        {
            void* memb = nullptr;
            int32_t pagesize = getpagesize();
            //size_t all_feature_length = (size_t)_conf.total_point_count * _conf.feature_dim * sizeof(float);
            size_t size = all_feature_length + (pagesize - all_feature_length % pagesize);
            int err = posix_memalign(&memb, pagesize, size);

            if (err != 0) {
                std::runtime_error("alloc_aligned_mem_failed errno=" + errno);
                return -1;
            }
            char* mem_ptr = reinterpret_cast<char*>(memb);
            _all_feature_int8 = reinterpret_cast<int8_t*>(mem_ptr);
            mem_ptr += (u_int64_t)_conf.total_point_count * _conf.feature_dim * sizeof(int8_t);
            _int8_multipliers = reinterpret_cast<float*>(mem_ptr);
            mem_ptr += _conf.feature_dim * sizeof(float);
            _dp_norms = reinterpret_cast<float*>(mem_ptr);
        }
        
        std::ifstream fin(_conf.int8_dataset_file_name, std::ofstream::binary);
        std::ifstream fin2(_conf.int8_multipliers_file_name, std::ofstream::binary);
        std::ifstream fin3(_conf.dp_norms_file_name, std::ofstream::binary);

        fin.read(reinterpret_cast<char*>(_all_feature_int8), int8_dataset_length);
        fin2.read(reinterpret_cast<char*>(_int8_multipliers), int8_multipliers_length);
        fin3.read(reinterpret_cast<char*>(_dp_norms), dp_norms_length);

        uint64_t memory_offset = (u_int64_t)local_to_memory_idx[0] * (u_int64_t)_conf.feature_dim;
        for (size_t i = 0; i < _conf.feature_dim; i++) {
            //OG(INFO) << "int8_val: " << _all_feature[memory_offset + i] << " " << (int)_all_feature_int8[i] << " " << _int8_multipliers[i] << " " << _all_feature_int8[i] / _int8_multipliers[i];
        }
        LOG(INFO) << "read int8_dataset Suc.";

    }

    LOG(INFO) << "HierarchicalClusterIndex::read_quantization_index Suc.";
    return 0;
}

void load_int_array_by_idx_thread_fun(const std::string& file_name,
                                      const int start_id, const int count, uint32_t* point_id) {
    std::ifstream input_file(file_name.c_str(), std::ios::binary | std::ios::in);
    input_file.seekg(start_id * sizeof(int));

    for (int i = 0; i < count; ++i) {
        input_file.read((char*) & (point_id[start_id + i]), sizeof(int));
    }

    input_file.close();
}

int HierarchicalClusterIndex::load_cell_assign(uint32_t* cell_assign) {
    auto cell_assign_file_name = _conf.cell_assign_file_name;
    auto total_point_count = _conf.total_point_count;
    if (_conf.spilled_copy_num > 0) {
        // cell_assign_file_name = _conf.cell_assign_bak_file_name;
        total_point_count = _conf.total_point_count * 2;
    }
    
    LOG(INFO) << "start load index file " << cell_assign_file_name << " " << total_point_count;
    base::Timer all_cost;
    all_cost.start();
    //文件长度检查
    u_int64_t cell_assign_length = (u_int64_t)total_point_count * sizeof(int);
    bool is_ok = check_file_length_info(cell_assign_file_name, cell_assign_length);

    if (is_ok == false) {
        return -1;
    }

    if (cell_assign == nullptr) {
        LOG(FATAL) << "malloc cell_edges momery error " << cell_assign_length << " error.";
        return -2;
    }

    //多线程加载文件
    int per_thread_count = ceil(1.0 * total_point_count / (float)_conf.threads_count);
    std::vector<std::thread> threads;

    for (uint32_t thread_id = 0; thread_id < _conf.threads_count; ++thread_id) {
        int start_id = thread_id * per_thread_count;
        int count = std::min(per_thread_count, (int)total_point_count - start_id);
        threads.push_back(std::thread(std::bind(load_int_array_by_idx_thread_fun,
                                                cell_assign_file_name, start_id, count, cell_assign)));
    }

    for (uint32_t thread_id = 0; thread_id < _conf.threads_count; ++thread_id) {
        threads[thread_id].join();
    }

    all_cost.stop();
    LOG(INFO) << "load index file " << cell_assign_file_name << " cost " << all_cost.m_elapsed() << " ms";
    return 0;
}

int HierarchicalClusterIndex::load_int_array_by_idx(
        uint32_t* memory, 
        std::string& file_name,
        size_t total_point_count) {

    LOG(INFO) << "start load index file " << file_name << " " << total_point_count;
    base::Timer all_cost;
    all_cost.start();
    //文件长度检查
    u_int64_t file_length = (u_int64_t)total_point_count * sizeof(int);
    bool is_ok = check_file_length_info(file_name, file_length);

    if (is_ok == false) {
        return -1;
    }

    if (memory == nullptr) {
        LOG(FATAL) << "malloc cell_edges momery error " << file_length << " error.";
        return -2;
    }

    //多线程加载文件
    int per_thread_count = ceil(1.0 * total_point_count / (float)_conf.threads_count);
    std::vector<std::thread> threads;

    for (uint32_t thread_id = 0; thread_id < _conf.threads_count; ++thread_id) {
        int start_id = thread_id * per_thread_count;
        int count = std::min(per_thread_count, (int)total_point_count - start_id);
        threads.push_back(std::thread(std::bind(load_int_array_by_idx_thread_fun,
                                                file_name, start_id, count, memory)));
    }

    for (uint32_t thread_id = 0; thread_id < _conf.threads_count; ++thread_id) {
        threads[thread_id].join();
    }

    all_cost.stop();
    LOG(INFO) << "load index file " << file_name << " cost " << all_cost.m_elapsed() << " ms";
    return 0;
}

int HierarchicalClusterIndex::convert_local_to_memory_idx(uint32_t* cell_start_memory_idx,
        uint32_t* local_to_memory_idx) {
    std::unique_ptr<uint32_t[]> cell_assign(new uint32_t[_conf.total_point_count]);

    if (load_cell_assign(cell_assign.get()) != 0) {
        return -1;
    }

    typedef std::pair<uint32_t, std::pair<float, uint32_t> > MemoryOrder;
    std::vector<MemoryOrder> point_reorder(_conf.total_point_count);

    for (uint32_t i = 0; i < _conf.total_point_count; ++i) {
        point_reorder[i].first = cell_assign[i];
        point_reorder[i].second.first = 0;
        point_reorder[i].second.second = i;
    }

    std::stable_sort(point_reorder.begin(), point_reorder.end());

    uint32_t cellcount = 0;
    cell_start_memory_idx[cellcount] = 0;

    for (uint32_t i = 0; i < _conf.total_point_count; ++i) {
        uint32_t local_idx = point_reorder[i].second.second;
        local_to_memory_idx[local_idx] = i;
        _memory_to_local[i] = local_idx;

        while (cellcount + 1 <= point_reorder[i].first) {
            ++cellcount;
            cell_start_memory_idx[cellcount] = i;
        }
    }

    for (uint32_t i = 0; i < _conf.fine_cluster_count * _conf.coarse_cluster_count; ++i) {
        int start_point_id = cell_start_memory_idx[i];
        int end_point_id = cell_start_memory_idx[i + 1];
        int coarse_id = i / _conf.fine_cluster_count;
        int fine_id = i % _conf.fine_cluster_count;

        if (start_point_id > end_point_id) {
            LOG(INFO) << "load index error";
            return -1;
        }

        FineCluster& cur_fine_cluster = _coarse_clusters[coarse_id].fine_cell_list[fine_id];
        cur_fine_cluster.memory_idx_start = start_point_id;

        if (start_point_id == end_point_id) {
            cur_fine_cluster.stationary_cell_dist = std::sqrt(std::numeric_limits<float>::max());
        }

        _coarse_clusters[coarse_id].min_dist_offset = std::min(_coarse_clusters[coarse_id].min_dist_offset,
                cur_fine_cluster.stationary_cell_dist);
    }

    LOG(INFO) << "convert_local_to_memory_idx cost ";
    return 0;
}

template <typename T>
char* SetValueAndIncPtr(char* ptr, const T& val) {
    *((T*)(ptr)) = val;
    return ptr + sizeof(T);
}

size_t HierarchicalClusterIndex::save_model_config(char* ptr) const {
    char* temp_ptr = ptr;
    //索引数据相关
    temp_ptr = SetValueAndIncPtr<IndexType>(temp_ptr, _conf.index_type);
    temp_ptr = SetValueAndIncPtr<uint32_t>(temp_ptr, _conf.feature_dim);
    temp_ptr = SetValueAndIncPtr<bool>(temp_ptr, _conf.whether_norm);
    temp_ptr = SetValueAndIncPtr<uint32_t>(temp_ptr, _conf.total_point_count);
    //filter
    temp_ptr = SetValueAndIncPtr<bool>(temp_ptr, _conf.whether_filter);
    temp_ptr = SetValueAndIncPtr<uint32_t>(temp_ptr, _conf.filter_nsq);
    temp_ptr = SetValueAndIncPtr<uint32_t>(temp_ptr, _conf.filter_ks);
    //pq
    temp_ptr = SetValueAndIncPtr<bool>(temp_ptr, _conf.whether_pq);
    temp_ptr = SetValueAndIncPtr<uint32_t>(temp_ptr, _conf.nsq);
    temp_ptr = SetValueAndIncPtr<uint32_t>(temp_ptr, _conf.ks);
    //ip2cos
    temp_ptr = SetValueAndIncPtr<uint32_t>(temp_ptr, _conf.ip2cos);
    //cluster
    temp_ptr = SetValueAndIncPtr<uint32_t>(temp_ptr, _conf.coarse_cluster_count);
    temp_ptr = SetValueAndIncPtr<uint32_t>(temp_ptr, _conf.fine_cluster_count);
    temp_ptr = SetValueAndIncPtr<uint32_t>(temp_ptr, _conf.spilled_copy_num);
    // brutal_search flag
    temp_ptr = SetValueAndIncPtr<bool>(temp_ptr, _conf.brutal_search);
    return temp_ptr - ptr;
}

template <typename T>
char* GetValueAndIncPtr(char* ptr, T& val) {
    val = *((T*)(ptr));
    return ptr + sizeof(T);
}

char* HierarchicalClusterIndex::load_model_config(char* ptr, size_t total_size) {
    LOG(INFO) << " load_model_config: " <<  _conf.filter_ks << " " << _conf.spilled_copy_num;
    char* temp_ptr = ptr;
    temp_ptr = GetValueAndIncPtr<IndexType>(temp_ptr, _conf.index_type);
    temp_ptr = GetValueAndIncPtr<uint32_t>(temp_ptr, _conf.feature_dim);
    temp_ptr = GetValueAndIncPtr<bool>(temp_ptr, _conf.whether_norm);
    temp_ptr = GetValueAndIncPtr<uint32_t>(temp_ptr, _conf.total_point_count);
    temp_ptr = GetValueAndIncPtr<bool>(temp_ptr, _conf.whether_filter);
    temp_ptr = GetValueAndIncPtr<uint32_t>(temp_ptr, _conf.filter_nsq);
    
    // temp_ptr = GetValueAndIncPtr<uint32_t>(temp_ptr, _conf.filter_ks);
    // TODO: do not update filter_ks due to old train lib not write this field
    temp_ptr = GetValueAndIncPtr<uint32_t>(temp_ptr, _conf.ks);
    LOG(INFO) << " _conf.filter_ks: " <<  _conf.filter_ks << " " << _conf.spilled_copy_num;
    temp_ptr = GetValueAndIncPtr<bool>(temp_ptr, _conf.whether_pq);
    temp_ptr = GetValueAndIncPtr<uint32_t>(temp_ptr, _conf.nsq);
    temp_ptr = GetValueAndIncPtr<uint32_t>(temp_ptr, _conf.ks);
    temp_ptr = GetValueAndIncPtr<uint32_t>(temp_ptr, _conf.ip2cos);
    temp_ptr = GetValueAndIncPtr<uint32_t>(temp_ptr, _conf.coarse_cluster_count);
    temp_ptr = GetValueAndIncPtr<uint32_t>(temp_ptr, _conf.fine_cluster_count);

    size_t remain_size = (total_size - (temp_ptr - ptr));
    LOG(INFO) << " _conf.filter_ks: " <<  _conf.filter_ks << " " << _conf.spilled_copy_num << " " << remain_size;
    if (remain_size == 5) {
        temp_ptr = GetValueAndIncPtr<uint32_t>(temp_ptr, _conf.spilled_copy_num);
        temp_ptr = GetValueAndIncPtr<bool>(temp_ptr, _conf.brutal_search);
    }
    //_conf.filter_topk = 20;
    LOG(INFO) << " _conf.filter_ks: " <<  _conf.filter_ks << " " << _conf.spilled_copy_num;
    _conf.use_lut16 = _conf.filter_ks == 16;
    return temp_ptr;
}

int HierarchicalClusterIndex::init_model_memory() {
    TIME_FUNCTION();
    if (_model != nullptr) {
        free(_model);
        _model = nullptr;
    }

    size_t model_len = 0;
    size_t coarse_vocab_length = (size_t)_conf.coarse_cluster_count * _conf.feature_dim * sizeof(float);
    model_len += coarse_vocab_length;

    size_t coarse_norms_length = _conf.coarse_cluster_count * sizeof(float);
    model_len += coarse_norms_length;

    size_t fine_vocab_length = (size_t)_conf.fine_cluster_count * _conf.feature_dim * sizeof(float);
    model_len += fine_vocab_length;

    size_t coarse_clusters_length = sizeof(CoarseCluster) * _conf.coarse_cluster_count;
    model_len += coarse_clusters_length;
    LOG(INFO) << "coarse_clusters_length " << coarse_clusters_length << " " << sizeof(CoarseCluster) << " " << _conf.coarse_cluster_count;
    uint32_t cell_cnt = _conf.coarse_cluster_count * _conf.fine_cluster_count;
    size_t fine_clusters_length = sizeof(FineCluster) * (cell_cnt + 1);
    model_len += fine_clusters_length;
    LOG(INFO) << "fine_clusters_length " << cell_cnt << " " << sizeof(FineCluster);

    size_t memory_to_local_len = sizeof(uint32_t) * _conf.total_point_count;
    
    
    LOG(INFO) << "_conf.spilled_copy_num: " << _conf.spilled_copy_num;
    size_t buffer_num = 1;  
    if (_conf.spilled_copy_num > 0) {
        buffer_num = 2;
        // _memory_to_ori_cell_id
        model_len += memory_to_local_len;
        model_len += _conf.coarse_cluster_count * _conf.coarse_cluster_count * sizeof(float);
        model_len += _conf.fine_cluster_count * _conf.fine_cluster_count * sizeof(float);
        model_len += _conf.coarse_cluster_count * _conf.fine_cluster_count * sizeof(float);
    }

    // _cell_assign & memory_to_local
    model_len += 2 * buffer_num * memory_to_local_len;

    // _local_to_memory
    model_len += memory_to_local_len;

    void* memb = nullptr;

    int32_t pagesize = getpagesize();
    size_t size = model_len + (pagesize - model_len % pagesize);
    

    LOG(ERROR) << "model_len: " << size;
    //exit(0);
    int err = posix_memalign(&memb, pagesize, size);

    if (err != 0) {
        std::runtime_error("alloc_aligned_mem_failed errno=" + errno);
        return -1;
    }

    _model = reinterpret_cast<char*>(memb);

    char* temp_data = _model;
    _coarse_vocab = (float*)temp_data;
    temp_data += coarse_vocab_length;

    _coarse_norms = (float*)temp_data;
    temp_data += coarse_norms_length;

    _fine_vocab = (float*)temp_data;
    temp_data += fine_vocab_length;

    _coarse_clusters = (CoarseCluster*)temp_data;
    temp_data += coarse_clusters_length;

    FineCluster* fine_cluster_list = (FineCluster*)temp_data;
    temp_data += fine_clusters_length;

    for (uint32_t i = 0; i < _conf.coarse_cluster_count; ++i) {
        _coarse_clusters[i].init();
        _coarse_clusters[i].fine_cell_list = fine_cluster_list + i * _conf.fine_cluster_count;
        
    }
    
    fine_cluster_list[cell_cnt].memory_idx_start = _conf.total_point_count;

    if (_conf.spilled_copy_num > 0) {
        fine_cluster_list[cell_cnt].memory_idx_start += _conf.total_point_count;
        _coarse_inner_products = (float*)temp_data;
        temp_data += _conf.coarse_cluster_count * _conf.coarse_cluster_count * sizeof(float);
        _fine_inner_products = (float*)temp_data;
        temp_data += _conf.fine_cluster_count * _conf.fine_cluster_count * sizeof(float);
        _coarse_fine_products = (float*)temp_data;
        temp_data += _conf.coarse_cluster_count * _conf.fine_cluster_count * sizeof(float);

    }

    LOG(INFO) << "init_cell_assign " << memory_to_local_len << " " << _conf.total_point_count;
    _cell_assign = (uint32_t*)temp_data;
    temp_data += buffer_num * memory_to_local_len;

    _local_to_memory = (uint32_t*)temp_data;
    temp_data += 1 * memory_to_local_len;

    _memory_to_local = (uint32_t*)temp_data;
    temp_data += buffer_num * memory_to_local_len;

    _memory_to_ori_cell_id = (uint32_t*)temp_data;
    // temp_data += 1 * memory_to_local_len;


    //_dp_norms = (float*)temp_data;
    return 0;
}

int HierarchicalClusterIndex::init_model_memory(char * fileBaseAddress, uint64_t fileLength) {
    TIME_FUNCTION();
    size_t model_len = 0;
    size_t coarse_vocab_length = (size_t)_conf.coarse_cluster_count * _conf.feature_dim * sizeof(float);
    model_len += coarse_vocab_length;

    size_t coarse_norms_length = _conf.coarse_cluster_count * sizeof(float);
    model_len += coarse_norms_length;

    size_t fine_vocab_length = (size_t)_conf.fine_cluster_count * _conf.feature_dim * sizeof(float);
    model_len += fine_vocab_length;

    size_t coarse_clusters_length = sizeof(CoarseCluster) * _conf.coarse_cluster_count;
    model_len += coarse_clusters_length;
    LOG(INFO) << "coarse_clusters_length " << coarse_clusters_length << " " << sizeof(CoarseCluster) << " " << _conf.coarse_cluster_count;
    uint32_t cell_cnt = _conf.coarse_cluster_count * _conf.fine_cluster_count;
    size_t fine_clusters_length = sizeof(FineCluster) * (cell_cnt + 1);
    model_len += fine_clusters_length;
    LOG(INFO) << "fine_clusters_length " << cell_cnt << " " << sizeof(FineCluster);

    size_t memory_to_local_len = sizeof(uint32_t) * _conf.total_point_count;
    
    
    LOG(INFO) << "_conf.spilled_copy_num: " << _conf.spilled_copy_num;
    size_t buffer_num = 1;  
    if (_conf.spilled_copy_num > 0) {
        buffer_num = 2;
        // model_len += memory_to_local_len;
    }

    // _cell_assign & memory_to_local
    model_len += 2 * buffer_num * memory_to_local_len;

    // _local_to_memory
    model_len += memory_to_local_len;

    void* memb = nullptr;

    if(fileLength != model_len) {
        LOG(ERROR) << "file length( " << fileLength << " ) is not equal to expected( " << model_len << " )";
        return -1;
    }

    char * temp_data = fileBaseAddress;

    _coarse_vocab = (float*)temp_data;
    temp_data += coarse_vocab_length;

    _coarse_norms = (float*)temp_data;
    temp_data += coarse_norms_length;

    _fine_vocab = (float*)temp_data;
    temp_data += fine_vocab_length;

    _coarse_clusters = (CoarseCluster*)temp_data;
    temp_data += coarse_clusters_length;

    FineCluster* fine_cluster_list = (FineCluster*)temp_data;
    temp_data += fine_clusters_length;

    for (uint32_t i = 0; i < _conf.coarse_cluster_count; ++i) {
        _coarse_clusters[i].init();
        _coarse_clusters[i].fine_cell_list = fine_cluster_list + i * _conf.fine_cluster_count;
        
    }
    fine_cluster_list[cell_cnt].memory_idx_start = _conf.total_point_count;
    if (_conf.spilled_copy_num > 0) {
        fine_cluster_list[cell_cnt].memory_idx_start += _conf.total_point_count;
    }

    LOG(INFO) << "init_cell_assign " << memory_to_local_len << " " << _conf.total_point_count;
    _cell_assign = (uint32_t*)temp_data;
    temp_data += buffer_num * memory_to_local_len;

    _local_to_memory = (uint32_t*)temp_data;
    temp_data += 1 * memory_to_local_len;

    _memory_to_local = (uint32_t*)temp_data;
    temp_data += buffer_num * memory_to_local_len;

    _memory_to_ori_cell_id = (uint32_t*)temp_data;

    return 0;
}

int HierarchicalClusterIndex::read_model_file() {
    int fd = -1;
    fd = open(_conf.index_file_name.c_str(), O_RDONLY);
    struct stat st;
    if (fd == -1 || -1 == fstat(fd, &st)) {
        LOG(ERROR) << "model file " << _conf.index_file_name << " stat error";
        return -1;
    }

    size_t size = st.st_size;
    ssize_t file_size = static_cast<ssize_t>(st.st_size);
    ssize_t read_size = 0;
    std::unique_ptr<char[]> buffer(new char[st.st_size]);
    do {
        ssize_t ret = read(fd, buffer.get() + read_size, size - read_size);

        if (ret < 0) {
            LOG(ERROR) << "read error errno:" << errno;
            return -1;
        }

        read_size += ret;

        //LOG(INFO)<<"read_size = "<<read_size;
        if (read_size == file_size || ret == 0) {
            break;
        }
    } while (true);

    close(fd);

    size_t part_size = 0;
    char* temp_buffer = buffer.get();
    temp_buffer = GetValueAndIncPtr<size_t>(temp_buffer, part_size);
 
    LOG(INFO) << "part_size=" << part_size << " st.st_size= " << st.st_size << " sizeof(size_t) = " << sizeof(
                  size_t);
    temp_buffer = load_model_config(temp_buffer, part_size);
    //_conf.filter_ks = 16;
    //save_model_file();
    //exit(0);
    _conf.whether_pq = FLAGS_whether_pq;
    LOG(INFO) << "show conf after read_model_file.";
    _conf.show();
    return 0;
}

int HierarchicalClusterIndex::check_index_type() {
    if (_conf.index_type != IndexType::HIERARCHICAL_CLUSTER) {
        LOG(ERROR) << "index_type is not HIERARCHICAL_CLUSTER";
        return -1;
    }

    return 0;
}

int HierarchicalClusterIndex::save_model_file() const {
    /// index_conf
    size_t conf_struct_size = sizeof(_conf);
    std::unique_ptr<char[]> temp_ptr(new char[conf_struct_size]);
    size_t conf_size = save_model_config(temp_ptr.get());
    std::ofstream out_fvec_init;
    out_fvec_init.open(_conf.index_file_name.c_str(), std::ios::binary | std::ios::out);
    out_fvec_init.write((char*)&conf_size, sizeof(size_t));
    out_fvec_init.write((char*)temp_ptr.get(), conf_size);
    out_fvec_init.close();
    return 0;
}

int HierarchicalClusterIndex::search_nearest_coarse_cluster(SearchContext* context, const float* feature,
        const uint32_t top_coarse_cnt) {
    //base::Timer tm_cost;
    //tm_cost.start();
    SearchCellData& search_cell_data = context->get_search_cell_data();
    float* cluster_inner_product = search_cell_data.cluster_inner_product;
    matrix_multiplication(_coarse_vocab, feature, _conf.coarse_cluster_count, 1, _conf.feature_dim, "TN",
                          cluster_inner_product);


    //计算一级聚类中心的距离,使用最大堆
    float* coarse_distance = search_cell_data.coarse_distance;
    uint32_t* coarse_tag = search_cell_data.coarse_tag;
    //初始化最大堆。
    PuckMaxHeap max_heap(top_coarse_cnt, coarse_distance, coarse_tag);

    for (uint32_t c =  0; c < _conf.coarse_cluster_count; ++c) {
        float temp_dist = _coarse_norms[c] - cluster_inner_product[c];
        //if (c == 665) {
            //LOG(INFO) << "temp_dist: " << temp_dist << " " << _coarse_norms[c] << " " << cluster_inner_product[c];
        //}
        if (temp_dist < coarse_distance[0]) {
            max_heap.max_heap_update(temp_dist, c);
        }
    }

    //堆排序
    max_heap.reorder();
    return 0;
}

int HierarchicalClusterIndex::search_nearest_fine_cluster(SearchContext* context, const float* feature) {
    //base::Timer tm_cost;
    //tm_cost.start();
    SearchCellData& search_cell_data = context->get_search_cell_data();
    float* cluster_inner_product = search_cell_data.cluster_inner_product;
    matrix_multiplication(_fine_vocab, feature, _conf.fine_cluster_count, 1, _conf.feature_dim,
                          "TN", cluster_inner_product);

    PuckMaxHeap max_heap(_conf.fine_cluster_count, search_cell_data.fine_distance,
                     search_cell_data.fine_tag);

    for (uint32_t k = 0; k < _conf.fine_cluster_count; ++k) {
        max_heap.max_heap_update(-cluster_inner_product[k], k);
    }

    max_heap.reorder();

    //计算一级聚类中心的距离,使用最大堆
    float* coarse_distance = search_cell_data.coarse_distance;
    uint32_t* coarse_tag = search_cell_data.coarse_tag;
    ImitativeHeap imitative_heap(_conf.neighbors_count, search_cell_data.cell_distance);
    imitative_heap.set_pivot(coarse_distance[_conf.search_coarse_count - 1]);

    for (uint32_t l = 0; l < _conf.search_coarse_count; ++l) {
        int coarse_id = coarse_tag[l];
        //计算query与当前一级聚类中心下cell的距离
        FineCluster* cur_fine_cluster_list = _coarse_clusters[coarse_id].fine_cell_list;
        float min_dist = _coarse_clusters[coarse_id].min_dist_offset + coarse_distance[l];
        float max_stationary_dist = imitative_heap.get_pivot() - coarse_distance[l] -
                                    search_cell_data.fine_distance[0];

        for (uint32_t idx = 0; idx < _conf.fine_cluster_count; ++idx) {
            if (search_cell_data.fine_distance[idx] + min_dist >= imitative_heap.get_pivot()) {
                //LOG(INFO)<<l<<" "<<idx<<" break;";
                break;
            }

            uint32_t k = search_cell_data.fine_tag[idx];

            if (cur_fine_cluster_list[k].stationary_cell_dist >= max_stationary_dist) {
                continue;
            }

            float temp_dist = coarse_distance[l] + cur_fine_cluster_list[k].stationary_cell_dist +
                              search_cell_data.fine_distance[idx];
            int ret = imitative_heap.push(temp_dist, cur_fine_cluster_list + k,
                                          (cur_fine_cluster_list + k)->get_point_cnt());

            if (ret == 0) {
                max_stationary_dist = imitative_heap.get_pivot() - coarse_distance[l] - search_cell_data.fine_distance[idx];
            }
        }
    }

    uint32_t cell_cnt = imitative_heap.get_top_idx();
    std::sort(search_cell_data.cell_distance.begin(), search_cell_data.cell_distance.begin() + cell_cnt);
    return cell_cnt;
}

int HierarchicalClusterIndex::search(const Request* request, Response* response) {
    if (request->topk > _conf.topk || request->feature == nullptr) {
        LOG(ERROR) << "topk should <= topk, topk = " << _conf.topk << ", or feature is nullptr";
        return -1;
    }

    DataHandler<SearchContext> context(_context_pool);

    if (0 != context->reset(_conf)) {
        LOG(ERROR) << "init search context has error.";
        return -1;
    }

    context->set_request(request);
    const float* feature = normalization(context.get(), request->feature);
    //输出query与一级聚类中心的top-search-cell个ID和距离
    int ret = search_nearest_coarse_cluster(context.get(), feature,
                                            _conf.search_coarse_count);

    if (ret != 0) {
        LOG(ERROR) << "search nearest coarse cluster has error.";
        return ret;
    }

    //计算query与二级聚类中心的距离并排序
    int search_cell_cnt = search_nearest_fine_cluster(context.get(), feature);

    if (search_cell_cnt < 0) {
        LOG(ERROR) << "search nearest fine cluster has error.";
        response->result_num = 0;
        return search_cell_cnt;
    }

    PuckMaxHeap max_heap(request->topk, response->distance, response->local_idx);
    ret = flat_topN_points(context.get(), feature, search_cell_cnt, max_heap);

    if (ret == 0) {
        response->result_num = max_heap.get_heap_size();
    }

    return ret;
}

int HierarchicalClusterIndex::compute_exhaustive_distance_with_points(SearchContext* context,
        const int cell_idx,
        const float* feature, PuckMaxHeap& result_heap) {
    const SearchCellData& search_cell_data = context->get_search_cell_data();
    const FineCluster* cur_fine_cluster = search_cell_data.cell_distance[cell_idx].second.first;
    float* result_distance = result_heap.get_top_addr();
    size_t qty_16 = 16;
    float PORTABLE_ALIGN32 TmpRes[8];
    //经验值，剪枝掉的比例通常在40%-60%+
    _mm_prefetch((char*)(feature), _MM_HINT_T0);
    uint32_t point_list_cnt = cur_fine_cluster->get_point_cnt();

    for (uint32_t i = 0; i < point_list_cnt; ++i) {
        const float* exhaustive_feature = _all_feature + (cur_fine_cluster->memory_idx_start + i) * _conf.feature_dim;
        float temp_dist = 0;

        _mm_prefetch((char*)(exhaustive_feature), _MM_HINT_T0);
        uint32_t m = 0;

        for (; (m + qty_16) <= _conf.feature_dim && temp_dist < result_distance[0];) {
            temp_dist += similarity::L2Sqr16Ext(exhaustive_feature + m, feature + m, qty_16, TmpRes);
            m += qty_16;
        }

        if (temp_dist < result_distance[0]) {
            size_t left_dim = _conf.feature_dim - m;

            if (left_dim > 0) {
                temp_dist += similarity::L2SqrExt(exhaustive_feature + m, feature + m, left_dim, TmpRes);
            }

            result_heap.max_heap_update(temp_dist, _memory_to_local[cur_fine_cluster->memory_idx_start + i]);
        }
    }

    return point_list_cnt;
}

int HierarchicalClusterIndex::flat_topN_points(SearchContext* context, const float* feature,
        const int search_cell_cnt,
        PuckMaxHeap& result_heap) {
    uint32_t found = 0;

    for (int idx = 0; idx < search_cell_cnt && found < _conf.neighbors_count; idx++) {
        found += compute_exhaustive_distance_with_points(context, idx, feature, result_heap);
    }

    result_heap.reorder();
    return 0;
}

int HierarchicalClusterIndex::train(const u_int64_t kmenas_point_cnt, float* kmeans_train_vocab) {
    TIME_FUNCTION();
    base::Timer tm_cost;
    tm_cost.start();

    //point属于的cluster id
    std::unique_ptr<int[]> cluster_assign(new int[kmenas_point_cnt]);
    //记录当前最小的deviation error
    float min_err = std::numeric_limits<float>::max();
    //每次kmeans的训练数据
    std::unique_ptr<float[]> train_vocab(new float[kmenas_point_cnt * _conf.feature_dim]);
    memcpy(train_vocab.get(), kmeans_train_vocab, sizeof(float) * kmenas_point_cnt * _conf.feature_dim);
    //记录每次kmeans产生的聚类中心
    std::unique_ptr<float[]> coarse_init_vocab(new float[_conf.coarse_cluster_count * _conf.feature_dim]);
    //std::unique_ptr<int[]> coarse_vocab_assign(new int[kmenas_point_cnt]);
    std::unique_ptr<float[]> fine_init_vocab(new float[_conf.fine_cluster_count * _conf.feature_dim]);
    std::unique_ptr<int[]> fine_vocab_assign(new int[kmenas_point_cnt]);

    Kmeans kmeans_cluster(FLAGS_kmeans_init_berkeley, (int)(_conf.threads_count));
    //使用默认KMEANS参数
    KmeansParams& params = kmeans_cluster.get_params();
    std::unique_ptr<float[]> dis(new float[kmenas_point_cnt]);
    std::unique_ptr<int[]> assign(new int[kmenas_point_cnt]);
    LOG(INFO) << "kmeas params: threads count = " << params.nt;

    for (int ite = 0; ite < FLAGS_kmeans_iterations_count; ++ite) {
        //kmeans聚类得到一级聚类中心
        //params.niter = 20;
        //if (ite == 0) params.niter = 30;
        float err = kmeans_cluster.kmeans(_conf.feature_dim, kmenas_point_cnt, _conf.coarse_cluster_count,
                                          train_vocab.get(),
                                          coarse_init_vocab.get(), nullptr, cluster_assign.get());
        LOG(INFO) << "deviation error of init coarse clusters is " << err << " when ite = " << ite;

        // float err2 = kmeans_cluster.assign(_conf.feature_dim, kmenas_point_cnt, _conf.coarse_cluster_count,
           //                               kmeans_train_vocab,
             //                             coarse_init_vocab.get(), dis.get(), assign.get());
        //LOG(INFO) << "------ deviation error of init coarse clusters is " << err2 << " when ite = " << ite;
        //计算残差  q - S
        memcpy(train_vocab.get(), kmeans_train_vocab, sizeof(float) * kmenas_point_cnt * _conf.feature_dim);

        for (uint32_t i = 0; i < kmenas_point_cnt; ++i) {
            //每次迭代计算T之后的值作为判断标准，所以每次S的值使用最新计算的
            int assign_id = cluster_assign.get()[i];
            cblas_saxpy(_conf.feature_dim, -1.0, coarse_init_vocab.get() + assign_id * _conf.feature_dim, 1,
                        train_vocab.get() + i * _conf.feature_dim, 1);
        }

        //params.niter = 30;
        //残差向量kmeans聚类得到二级聚类中心T, err = mean(|q - S - T|^2)
        // 500000
        err = kmeans_cluster.kmeans(_conf.feature_dim, kmenas_point_cnt, _conf.fine_cluster_count,
                                    train_vocab.get(),
                                    fine_init_vocab.get(), nullptr, cluster_assign.get());
        LOG(INFO) << ite << " deviation error of init fine clusters is " << err << " when ite = " << ite;
        //err = kmeans_cluster.assign(_conf.feature_dim, kmenas_point_cnt, _conf.fine_cluster_count,
          //                          train_vocab.get(),
            //                        fine_init_vocab.get(), dis.get(), cluster_assign.get());
        //LOG(INFO) << ite << " deviation error of init fine clusters is " << err << " when ite = " << ite;

        //err2 = kmeans_cluster.assign(_conf.feature_dim, kmenas_point_cnt, _conf.coarse_cluster_count,
          //                                kmeans_train_vocab,
            //                              coarse_init_vocab.get(), dis.get(), assign.get());
        //LOG(INFO) << "---------------- deviation error of init coarse clusters is " << err2 << " when ite = " << ite;

        //如果小于当前最小的deviation error，更新记录的S&T
        if ((min_err - err) >= 1e-4) {
            memcpy(_coarse_vocab, coarse_init_vocab.get(),
                   sizeof(float) * _conf.coarse_cluster_count * _conf.feature_dim);
            memcpy(_fine_vocab, fine_init_vocab.get(),
                   sizeof(float) * _conf.fine_cluster_count * _conf.feature_dim);
            memcpy(fine_vocab_assign.get(), cluster_assign.get(), sizeof(int) * kmenas_point_cnt);
            min_err = err;
        } else { //大于等于最小值，开始出现抖动
            LOG(INFO) << "current deviation error > min deviation error : " << err << " / " << min_err <<
                      ", params.niter = " << params.niter;

            //params.niter初值为30，大于80跳出
            if (params.niter > 80) {
                break;
            }

            params.niter += 10;
        }

        //计算残差
        memcpy(train_vocab.get(), kmeans_train_vocab, sizeof(float) * kmenas_point_cnt * _conf.feature_dim);

        for (uint32_t i = 0; i < kmenas_point_cnt; ++i) {
            //计算最优情况下的q-T_best
            int assign_id = fine_vocab_assign.get()[i];
            cblas_saxpy(_conf.feature_dim, -1.0, _fine_vocab + assign_id * _conf.feature_dim, 1,
                        train_vocab.get() + i * _conf.feature_dim, 1);
        }
    }

    // exit(0);

    std::vector<float> fine_reorder_dist(_conf.fine_cluster_count);
    std::vector<uint32_t> fine_reorder_id(_conf.fine_cluster_count);
    PuckMaxHeap max_heap(_conf.fine_cluster_count,
                     fine_reorder_dist.data(),
                     fine_reorder_id.data());

    for (uint32_t j = 0; j < _conf.fine_cluster_count; ++j) {
        //float temp = fvec_norm2sqr(_fine_vocab + _conf.feature_dim * j, _conf.feature_dim) / 2;
        float temp = cblas_sdot(_conf.feature_dim, _fine_vocab + _conf.feature_dim * j, 1,
                                _fine_vocab + _conf.feature_dim * j, 1) / 2;
        max_heap.max_heap_update(0 - std::sqrt(temp), j);
    }

    max_heap.reorder();

    memcpy(fine_init_vocab.get(), _fine_vocab,
           sizeof(float) * _conf.fine_cluster_count * _conf.feature_dim);

    for (uint32_t j = 0; j < _conf.fine_cluster_count; ++j) {
        memcpy(_fine_vocab + j * _conf.feature_dim,
               fine_init_vocab.get() + fine_reorder_id[j] * _conf.feature_dim,
               sizeof(float) * _conf.feature_dim);
    }

    tm_cost.stop();
    LOG(INFO) << "init coarse & fine clusters and alpha vocab cost " << tm_cost.m_elapsed() << " ms";
    return 0;
}


int HierarchicalClusterIndex::build() {
    TIME_FUNCTION();
    if (_conf.brutal_search) {
        return 0;
    }
    //从文件获取配置信息
    if (read_model_file() != 0) {
        LOG(INFO) << "read_model_file error";
        return -1;
    }

    if (check_feature_dim() != 0) {
        return -1;
    }


    //从文件获取配置信息
    if (init_model_memory() != 0) {
        LOG(INFO) << "init_model_memory error";
        return -1;
    }
    
    //读码本
    if (read_codebooks() != 0) {
        LOG(INFO) << "read_codebooks error";
        return -1;
    }

    uint32_t* cell_assign = _cell_assign;
    batch_assign(_conf.total_point_count, _conf.feature_file_name, cell_assign);
  
    if (_conf.spilled_copy_num > 0) {
        batch_assign(_conf.total_point_count, _conf.feature_file_name, cell_assign + _conf.total_point_count, true);
        // batch_assign_bak(_conf.total_point_count, _conf.feature_file_name, cell_assign + _conf.total_point_count);
        // save_index_bak();
    }

    uint32_t cell_cnt = _conf.coarse_cluster_count * _conf.fine_cluster_count;
    auto total_point_count = _conf.total_point_count;
    if (_conf.spilled_copy_num > 0) {
        total_point_count *= 2;
    }

    std::vector<uint32_t> cell_start_memory_idx(cell_cnt + 1, total_point_count);
    std::vector<uint32_t> local_to_memory_idx(total_point_count);
    LOG(INFO) << "scalar_quantize_float_dataset: " << _conf.total_point_count << " " << _conf.feature_file_name;
    scalar_quantize_float_dataset(_conf.total_point_count, _conf.feature_file_name);
    save_index();
   
    
    if (FLAGS_convert_local_to_memory_while_build && FLAGS_ad_hoc) {
        LOG(INFO) << "FLAGS_convert_local_to_memory_while_build";
        if (convert_local_to_memory_idx(cell_start_memory_idx.data(), local_to_memory_idx.data(), cell_assign) != 0) {
            return -1;
        }
        save_index_after_convert();
          // cell_start_memory_idx_file_name
        FILE* f = fopen(_conf.cell_start_memory_idx_file_name.c_str(), "wb");

        if (f == nullptr) {
            LOG(ERROR) << "cannot open " << _conf.cell_start_memory_idx_file_name << " for writing";
            return -1;
        }

        long ret = fwrite(cell_start_memory_idx.data(), sizeof(cell_start_memory_idx[0]), cell_cnt + 1, f);
        fclose(f);

        if (ret != cell_cnt + 1) {
            LOG(ERROR) << "writint to " << _conf.cell_start_memory_idx_file_name << " has error";
            return -1;
        }
    }

    return 0;
}

int HierarchicalClusterIndex::save_index() {
    TIME_FUNCTION();
    save_model_file();
    //ivec_write_raw(_conf.cell_assign_file_name.c_str(), (int*)_memory_to_local, _conf.total_point_count);
    FILE* f = fopen(_conf.cell_assign_file_name.c_str(), "wb");

    if (f == nullptr) {
        LOG(ERROR) << "cannot open " << _conf.cell_assign_file_name << " for writing";
        return -1;
    }

    long ret = fwrite(_cell_assign, sizeof(*_cell_assign), _conf.total_point_count * (1 + _conf.spilled_copy_num), f);
    
    
    fclose(f);

    if (ret != _conf.total_point_count * (1 + _conf.spilled_copy_num)) {
        LOG(ERROR) << "writint to " << _conf.cell_assign_file_name << " has error";
        return -1;
    }

    if (_conf.spilled_copy_num > 0) {
        //HierarchicalClusterIndex::save_index_bak();
    }
    return 0;
}

int HierarchicalClusterIndex::save_index_bak() {
    TIME_FUNCTION();
   
    // cell_assign_bak
    FILE* f = fopen(_conf.cell_assign_bak_file_name.c_str(), "wb");

    if (f == nullptr) {
        LOG(ERROR) << "cannot open " << _conf.cell_assign_bak_file_name << " for writing";
        return -1;
    }

    long ret = fwrite(_cell_assign, sizeof(*_cell_assign), _conf.total_point_count * 2, f);
    fclose(f);

    if (ret != _conf.total_point_count * 2) {
        LOG(ERROR) << "writint to " << _conf.cell_assign_bak_file_name << " has error";
        return -1;
    }
    return 0;
}

int HierarchicalClusterIndex::save_index_after_convert() {

   FILE* f = fopen(_conf.cell_assign_file_name.c_str(), "wb");

    if (f == nullptr) {
        LOG(ERROR) << "cannot open " << _conf.cell_assign_file_name << " for writing";
        return -1;
    }

    long ret = fwrite(_cell_assign, sizeof(*_cell_assign), _conf.total_point_count * (1 + _conf.spilled_copy_num), f);
    
    
    fclose(f);

    if (ret != _conf.total_point_count * (1 + _conf.spilled_copy_num)) {
        LOG(ERROR) << "writint to " << _conf.cell_assign_file_name << " has error";
        return -1;
    }

    // local_to_memory
    f = fopen(_conf.local_to_memory_file_name.c_str(), "wb");

    if (f == nullptr) {
        LOG(ERROR) << "cannot open " << _conf.local_to_memory_file_name << " for writing";
        return -1;
    }
    ret = fwrite(_local_to_memory, sizeof(*_local_to_memory), _conf.total_point_count, f);
    fclose(f);

    if (ret != _conf.total_point_count) {
        LOG(ERROR) << "writint to " << _conf.local_to_memory_file_name << " has error";
        return -1;
    }
    return 0;
}

//point距离最近的cell信息
int HierarchicalClusterIndex::nearest_cell_assign(const float* coarse_distance,
        const float* fine_distance,
        const float query_norm,
        NearestCell& nearest_cell,
        uint32_t doc_id) const {
    //计算一级聚类中心并排序
    std::vector<std::pair<float, int>> coarse_distance_list(_conf.coarse_cluster_count);

    for (u_int64_t i = 0; i < _conf.coarse_cluster_count; ++i) {
        coarse_distance_list[i].second = i;
        coarse_distance_list[i].first = _coarse_norms[i] - coarse_distance[i] + query_norm;
    }

    std::sort(coarse_distance_list.begin(), coarse_distance_list.end());
    //遍历每个coarse下的所有fine cluster
    nearest_cell.init();

    for (uint32_t i = 0; i < _conf.coarse_cluster_count; ++i) {
        int current_coarse_id = coarse_distance_list[i].second;
        float current_coarse_term = coarse_distance_list[i].first;

        for (uint32_t idx = 0; idx < _conf.fine_cluster_count; ++idx) {
            //三角不等式
            //if (std::sqrt(nearest_cell.distance) > 0.6) {
              //  LOG(INFO) << "distance: " << std::sqrt(nearest_cell.distance) << " " << std::sqrt(current_coarse_term) + _fine_norms[idx];
            //}
            if (std::sqrt(nearest_cell.distance) - _fine_norms[idx] < std::sqrt(current_coarse_term)) {
                //记录跳过的cell个数占比
                //LOG(INFO)<<nearest_cell.distance<<" "<<i<<" "<<idx;
                nearest_cell.pruning_computation += (1.0 - 1.0 * (idx) / _conf.fine_cluster_count);
                break;
            }

            uint32_t cur_fine_id  = idx;
            int cur_cell_id = current_coarse_id * _conf.fine_cluster_count + cur_fine_id;
            float score = current_coarse_term -
                          fine_distance[cur_fine_id] +
                          get_fine_cluster(cur_cell_id)->stationary_cell_dist;

            if (score < nearest_cell.distance) {
                nearest_cell.distance = score;
                nearest_cell.cell_id = cur_cell_id;
            }
        }
    }
    
    return 0;
}

float cosine(float* a, float* b, int dim) {
    // a * b / |a||b|
    float a_norms_sq = cblas_sdot(dim, a, 1, a, 1);
    float b_norms_sq = cblas_sdot(dim, b, 1, b, 1);
    float ab = cblas_sdot(dim, a, 1, b, 1);
    return ab / (std::sqrt(a_norms_sq) * std::sqrt(b_norms_sq));
}

int HierarchicalClusterIndex::bak_cell_assign2(const float* coarse_distance,
        const float* fine_distance,
        const float query_norm,
        NearestCell& nearest_cell,
        uint32_t gt_cell_id,
        uint32_t doc_id,
        float* cur_point_fea) const {
    //计算一级聚类中心并排序
    std::vector<std::pair<float, int>> coarse_distance_list(_conf.coarse_cluster_count);

    for (u_int64_t i = 0; i < _conf.coarse_cluster_count; ++i) {
        coarse_distance_list[i].second = i;
        coarse_distance_list[i].first = _coarse_norms[i] - coarse_distance[i] + query_norm;
    }

    std::sort(coarse_distance_list.begin(), coarse_distance_list.end());

    auto gt_coarse_id = gt_cell_id / _conf.fine_cluster_count;
    auto gt_fine_id = gt_cell_id % _conf.fine_cluster_count;

    nearest_cell.init();

    float fix_term = 2 * query_norm - coarse_distance[gt_coarse_id] - fine_distance[gt_fine_id];

    auto coarse_sample_num = _conf.spilled_coarse_min_sample_num;
    if (_conf.coarse_cluster_count / _conf.spilled_coarse_sample_factor > coarse_sample_num) {
        coarse_sample_num = _conf.coarse_cluster_count / _conf.spilled_coarse_sample_factor;
    }
    coarse_sample_num = std::min(_conf.coarse_cluster_count, coarse_sample_num);
    // LOG(INFO) << "coarse_sample_num: " << coarse_sample_num;
    for (uint32_t ii = 0; ii < coarse_sample_num; ++ii) {
        int i = coarse_distance_list[ii].second;
        float current_coarse_term = coarse_distance_list[ii].first;

        float cur_coarse_term = -coarse_distance[i] 
                + _coarse_inner_products[gt_coarse_id * _conf.coarse_cluster_count + i] 
                + _coarse_fine_products[i * _conf.fine_cluster_count + gt_fine_id];

        for (uint32_t idx = 0; idx < _conf.fine_cluster_count; ++idx) {
            if (i == gt_coarse_id && idx == gt_fine_id) {
                continue;
            }
            int cur_cell_id = i * _conf.fine_cluster_count + idx;
            float half_rnorm_sq = current_coarse_term //_coarse_norms[i] - coarse_distance[i] + query_norm 
                    + get_fine_cluster(cur_cell_id)->stationary_cell_dist - fine_distance[idx];
            float rnorm_sq = half_rnorm_sq * 2;

            if (rnorm_sq > nearest_cell.distance) {
                continue;
            }
            
            float cur_fine_term = -fine_distance[idx] 
                    + _fine_inner_products[gt_fine_id * _conf.fine_cluster_count + idx] 
                    + _coarse_fine_products[gt_coarse_id * _conf.fine_cluster_count + idx];

            float rip = fix_term + cur_coarse_term + cur_fine_term;
            float rip_sq = rip * rip;
            
            float score = rnorm_sq + rip_sq / rnorm_sq;

            if (score < nearest_cell.distance) {
                nearest_cell.distance = score;
                nearest_cell.cell_id = cur_cell_id;
            }
        }
    }

    return 0;
}

int HierarchicalClusterIndex::bak_cell_assign(const float* coarse_distance,
        const float* fine_distance,
        const float query_norm,
        NearestCell& nearest_cell,
        uint32_t gt_cell_id,
        uint32_t doc_id) const {
    //计算一级聚类中心并排序
    std::vector<std::pair<float, int>> coarse_distance_list(_conf.coarse_cluster_count);

    for (u_int64_t i = 0; i < _conf.coarse_cluster_count; ++i) {
        coarse_distance_list[i].second = i;
        coarse_distance_list[i].first = _coarse_norms[i] - coarse_distance[i] + query_norm;
    }

    std::sort(coarse_distance_list.begin(), coarse_distance_list.end());
    //遍历每个coarse下的所有fine cluster
    nearest_cell.init();

    float second_min = 999.0;
    uint32_t second_idx = 0;
    for (uint32_t i = 0; i < _conf.coarse_cluster_count; ++i) {
        int current_coarse_id = coarse_distance_list[i].second;
        float current_coarse_term = coarse_distance_list[i].first;

        for (uint32_t idx = 0; idx < _conf.fine_cluster_count; ++idx) {
            //三角不等式
            //if (std::sqrt(nearest_cell.distance) > 0.6) {
              //  LOG(INFO) << "distance: " << std::sqrt(nearest_cell.distance) << " " << std::sqrt(current_coarse_term) + _fine_norms[idx];
            //}
            if (std::sqrt(second_min) - _fine_norms[idx] < std::sqrt(current_coarse_term)) {
                //记录跳过的cell个数占比
                //LOG(INFO)<<nearest_cell.distance<<" "<<i<<" "<<idx;
                nearest_cell.pruning_computation += (1.0 - 1.0 * (idx) / _conf.fine_cluster_count);
                break;
            }

            uint32_t cur_fine_id  = idx;
            int cur_cell_id = current_coarse_id * _conf.fine_cluster_count + cur_fine_id;
            float score = current_coarse_term -
                          fine_distance[cur_fine_id] +
                          get_fine_cluster(cur_cell_id)->stationary_cell_dist;

            if (score < nearest_cell.distance) {
                if (nearest_cell.distance < second_min) {
                    second_min = nearest_cell.distance;
                    second_idx = nearest_cell.cell_id;
                }
                nearest_cell.distance = score;
                nearest_cell.cell_id = cur_cell_id;
                
            } else {
                if (score < second_min) {
                    second_min = score;
                    second_idx = cur_cell_id;
                }
            }
        }
    }

    nearest_cell.distance = second_min;
    nearest_cell.cell_id = second_idx;

    return 0;
}

int HierarchicalClusterIndex::assign(const ThreadParams& thread_params, uint32_t* cell_assign,
                                     float* error_distance,
                                     float* pruning_computation,
                                     bool bak_assign,
                                     uint32_t offset) const {
    std::unique_ptr<float[]> points_coarse_terms(new
            float[FLAGS_thread_chunk_size * _conf.coarse_cluster_count]);
    std::unique_ptr<float[]> points_fine_terms(new
            float[FLAGS_thread_chunk_size * _conf.fine_cluster_count]);
    std::unique_ptr<float[]> chunk_points(new
                                          float[FLAGS_thread_chunk_size * _conf.feature_dim]);
    NearestCell nearest_cell;

    for (uint32_t cid = 0; cid < thread_params.chunks_count; ++cid) {

        
        LOG(INFO) << "HierarchicalClusterIndex::assign nearest_cell_assign_batch processing " << cid << "/" <<
                  thread_params.chunks_count << " " << bak_assign;
        int real_thread_chunk_size = std::min(FLAGS_thread_chunk_size,
                                              (int)(thread_params.points_count - cid * FLAGS_thread_chunk_size));
        int read_chunk_size = read_fvec_format(thread_params.learn_stream, _conf.feature_dim, real_thread_chunk_size,
                                               chunk_points.get());

        if (read_chunk_size != real_thread_chunk_size) {
            LOG(ERROR) << "puck assign from " << thread_params.start_id << " read file error at cid = " << cid << " / " <<
                       thread_params.chunks_count << ", ret = " << read_chunk_size;
            throw "read_fvec_format error!";
        }

        if (thread_params.start_id + cid * FLAGS_thread_chunk_size >= 8309927 
            || thread_params.start_id + (cid + 1) * FLAGS_thread_chunk_size <= 8309927) {
            //continue;
        }

        matrix_multiplication(_coarse_vocab, chunk_points.get(), _conf.coarse_cluster_count,
                              real_thread_chunk_size,
                              _conf.feature_dim, "TN", points_coarse_terms.get());

        matrix_multiplication(_fine_vocab, chunk_points.get(), _conf.fine_cluster_count,
                              real_thread_chunk_size,
                              _conf.feature_dim, "TN", points_fine_terms.get());

        //计算最近的cell
        for (int point_id = 0; point_id < real_thread_chunk_size; ++point_id) {
            if (thread_params.start_id + cid * FLAGS_thread_chunk_size + point_id != 8309927) {
                //continue;
            }

            float* cur_point_fea = chunk_points.get() + point_id * _conf.feature_dim;
            float* cur_coarse_distance = points_coarse_terms.get() + point_id * _conf.coarse_cluster_count;
            float* cur_fine_distance = points_fine_terms.get() + point_id * _conf.fine_cluster_count;
            float point_norms = cblas_sdot(_conf.feature_dim, cur_point_fea, 1, cur_point_fea, 1) / 2;

            if (bak_assign) {
                //LOG(INFO) << "offset: " << offset << " " << cell_assign[-5000000] << " " << cell_assign[-4999999] << " " << cell_assign[-1] << " " << -offset + thread_params.start_id + cid * FLAGS_thread_chunk_size + point_id;
                //LOG(INFO) << "old: " << cell_assign [ + thread_params.start_id + cid * FLAGS_thread_chunk_size + point_id];
                if (FLAGS_bak_cell_assign == 1) {
                    bak_cell_assign(cur_coarse_distance, cur_fine_distance,
                                        point_norms,
                                        nearest_cell,
                                        (cell_assign - offset)[thread_params.start_id + cid * FLAGS_thread_chunk_size + point_id],
                                        thread_params.start_id + cid * FLAGS_thread_chunk_size + point_id);
                } else if (FLAGS_bak_cell_assign == 2) {
                    bak_cell_assign2(cur_coarse_distance, cur_fine_distance,
                                        point_norms,
                                        nearest_cell,
                                        (cell_assign - offset)[thread_params.start_id + cid * FLAGS_thread_chunk_size + point_id],
                                        thread_params.start_id + cid * FLAGS_thread_chunk_size + point_id,
                                        cur_point_fea);
                }
            } else {
                nearest_cell_assign(cur_coarse_distance, cur_fine_distance,
                                    point_norms,
                                    nearest_cell,
                                    thread_params.start_id + cid * FLAGS_thread_chunk_size + point_id);
            }
            // 0.5 + 0.31584 + 0.00224505 - 0.589607 - 0.0157079 = 0.21277015
            // 0.212771 133164 s:665 t:164 qs:0.589607 qt:0.0157079 q^2:0.5 s^2:0.31584 t^2:-0.117474 station:0.00224505 cur_point_fea:-0.004957 0.033843 0.008715
            if (thread_params.start_id + cid * FLAGS_thread_chunk_size + point_id == 6395337) {
                LOG(INFO) << "nearest_cell.distance : " << nearest_cell.distance << " " << nearest_cell.cell_id 
                    << " s:" << nearest_cell.cell_id / _conf.fine_cluster_count << " t:" << nearest_cell.cell_id % _conf.fine_cluster_count
                    << " qs:" << cur_coarse_distance[nearest_cell.cell_id / _conf.fine_cluster_count]
                    << " qt:" << cur_fine_distance[nearest_cell.cell_id % _conf.fine_cluster_count] << " q^2:" << point_norms
                    << " s^2:" << _coarse_norms[nearest_cell.cell_id / _conf.fine_cluster_count] 
                    << " t^2:" << _fine_norms[nearest_cell.cell_id % _conf.fine_cluster_count]
                    << " station:" << get_fine_cluster(nearest_cell.cell_id)->stationary_cell_dist
                    << " cur_point_fea:" << cur_point_fea[0] << " " << cur_point_fea[1] << " " << cur_point_fea[767];
                
            }
            cell_assign[thread_params.start_id + cid * FLAGS_thread_chunk_size + point_id] =
                nearest_cell.cell_id;
            *error_distance += nearest_cell.distance;
            *pruning_computation += nearest_cell.pruning_computation * 1.0 /
                                    _conf.coarse_cluster_count;
        }
    }

    return 0;
}

//打开文件
int ThreadParams::open_file(const char* train_fea_file_name, uint32_t feature_dim) {
    close_file();
    learn_stream = fopen(train_fea_file_name, "r");

    if (!learn_stream) {
        LOG(FATAL) << "open file " << train_fea_file_name << " has error.";
        return -1;
    }

    u_int64_t offset = (u_int64_t)start_id * feature_dim * sizeof(float) +
                       (u_int64_t)start_id * sizeof(int);
    //文件句柄指向要处理的point块初始地址
    int ret = fseek(learn_stream, offset, SEEK_SET);

    if (ret != 0) {
        fpos_t ps;
        fgetpos(learn_stream, &ps);
        LOG(FATAL) << "seek file " << train_fea_file_name << " error, need offset = " << offset << " cur pos = " <<
                   ps.__pos <<
                   " when init thread params (start_id = " << start_id << ")";
        return -1;
    }

    return 0;
}

int ThreadParams::close_file() {
    if (learn_stream) {
        fclose(learn_stream);
        learn_stream = nullptr;
    }

    return 0;
}

void HierarchicalClusterIndex::batch_assign(const uint32_t total_cnt, const std::string& feature_file_name,
        uint32_t* cell_assign, bool bak_assign) {
    LOG(INFO) << "HierarchicalClusterIndex::batch_assign";
    std::vector<std::thread> threads;
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;
    //int points_count = 10000;
    std::vector<float> error_distance(_conf.threads_count, 0);
    std::vector<float> pruning_computation(_conf.threads_count, 0);

    if (_fine_norms == nullptr) {
        _fine_norms = new float[_conf.fine_cluster_count];

        for (uint32_t j = 0; j < _conf.fine_cluster_count; ++j) {
            //_fine_norms[j] = fvec_norm2sqr(_fine_vocab + _conf.feature_dim * j, _conf.feature_dim) / 2;
            _fine_norms[j] = cblas_sdot(_conf.feature_dim, _fine_vocab + _conf.feature_dim * j, 1,
                                        _fine_vocab + _conf.feature_dim * j, 1) / 2;
            _fine_norms[j] = 0 - std::sqrt(_fine_norms[j]);
        }
    }
    
    //线程个数由gflags参数threads_count指定，默认等于CPU核数
    for (uint32_t threadId = 0; threadId < _conf.threads_count; ++threadId) {
        threads.push_back(std::thread([&, threadId] {
            {
                ThreadParams thread_params;
                thread_params.points_count = std::ceil(1.0 * total_cnt / _conf.threads_count);
                thread_params.start_id = threadId* thread_params.points_count;
                thread_params.points_count = std::min(thread_params.points_count, (int)(total_cnt - thread_params.start_id));

                if (thread_params.points_count > 0) {
                    try {
                        thread_params.chunks_count = std::ceil(1.0 * thread_params.points_count / FLAGS_thread_chunk_size);
                        int param_stat = thread_params.open_file(feature_file_name.c_str(), _conf.feature_dim);

                        if (param_stat != 0) {
                            throw "open file has error.";
                        }

                        LOG(INFO) << "assign, thread_params.start_id = " << thread_params.start_id << " points_count = " <<
                                  thread_params.points_count << " feature_file_name = " << feature_file_name << " threadId = " << threadId;
                        assign(thread_params, cell_assign, error_distance.data() + threadId,  pruning_computation.data() + threadId, bak_assign, total_cnt);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                    }
                }
            }
        }));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    if (lastException) {
        std::rethrow_exception(lastException);
    }
    
    float total_error = 0;
    float total_pruning_computation = 0;

    for (uint32_t i = 0; i < _conf.threads_count; ++i) {
        total_error += error_distance[i];
        total_pruning_computation += pruning_computation[i];
    }
    // 16:20:36.760354  16:21:57.385944
    delete [] _fine_norms;
    _fine_norms = nullptr;
    LOG(INFO) << "batch_assign succeeded, deviation error = " <<
              total_error / total_cnt << ", pruning computation = " << total_pruning_computation / total_cnt;

    std::vector<size_t> count(_conf.coarse_cluster_count * _conf.fine_cluster_count, 0);
    for (size_t i = 0; i < total_cnt; i++) {
        // LOG(INFO) << "cell_assign[i]: " << i << " " << cell_assign[i] << " " << _conf.total_point_count;
        count[cell_assign[i]] += 1;
    }
    for (size_t i = 0; i < _conf.coarse_cluster_count * _conf.fine_cluster_count; i++) {
        int c = i / _conf.fine_cluster_count;
        int f = i % _conf.fine_cluster_count;
        if (i < 0)
        LOG(INFO) << "count: " << c << " " << f << " " << count[i];
    }
}

void HierarchicalClusterIndex::batch_assign_bak(const uint32_t total_cnt, const std::string& feature_file_name,
        uint32_t* cell_assign) {
    LOG(INFO) << "HierarchicalClusterIndex::batch_assign_bak";
    std::vector<std::thread> threads;
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;
    //int points_count = 10000;
    std::vector<float> error_distance(_conf.threads_count, 0);
    std::vector<float> pruning_computation(_conf.threads_count, 0);

    if (_fine_norms == nullptr) {
        _fine_norms = new float[_conf.fine_cluster_count];

        for (uint32_t j = 0; j < _conf.fine_cluster_count; ++j) {
            //_fine_norms[j] = fvec_norm2sqr(_fine_vocab + _conf.feature_dim * j, _conf.feature_dim) / 2;
            _fine_norms[j] = cblas_sdot(_conf.feature_dim, _fine_vocab + _conf.feature_dim * j, 1,
                                        _fine_vocab + _conf.feature_dim * j, 1) / 2;
            _fine_norms[j] = 0 - std::sqrt(_fine_norms[j]);
        }
    }
    
    //线程个数由gflags参数threads_count指定，默认等于CPU核数
    for (uint32_t threadId = 0; threadId < _conf.threads_count; ++threadId) {
        threads.push_back(std::thread([&, threadId] {
            {
                ThreadParams thread_params;
                thread_params.points_count = std::ceil(1.0 * total_cnt / _conf.threads_count);
                thread_params.start_id = threadId* thread_params.points_count;
                thread_params.points_count = std::min(thread_params.points_count, (int)(total_cnt - thread_params.start_id));

                if (thread_params.points_count > 0) {
                    try {
                        thread_params.chunks_count = std::ceil(1.0 * thread_params.points_count / FLAGS_thread_chunk_size);
                        int param_stat = thread_params.open_file(feature_file_name.c_str(), _conf.feature_dim);

                        if (param_stat != 0) {
                            throw "open file has error.";
                        }

                        LOG(INFO) << "assign, thread_params.start_id = " << thread_params.start_id << " points_count = " <<
                                  thread_params.points_count << " feature_file_name = " << feature_file_name << " threadId = " << threadId;
                        assign(thread_params, cell_assign, error_distance.data() + threadId,  pruning_computation.data() + threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                    }
                }
            }
        }));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    if (lastException) {
        std::rethrow_exception(lastException);
    }
    
    float total_error = 0;
    float total_pruning_computation = 0;

    for (uint32_t i = 0; i < _conf.threads_count; ++i) {
        total_error += error_distance[i];
        total_pruning_computation += pruning_computation[i];
    }
    // 16:20:36.760354  16:21:57.385944
    delete [] _fine_norms;
    _fine_norms = nullptr;
    LOG(INFO) << "batch_assign succeeded, deviation error = " <<
              total_error / total_cnt << ", pruning computation = " << total_pruning_computation / total_cnt;

    std::vector<size_t> count(_conf.coarse_cluster_count * _conf.fine_cluster_count, 0);
    for (size_t i = 0; i < total_cnt; i++) {
        // LOG(INFO) << "cell_assign[i]: " << i << " " << cell_assign[i] << " " << _conf.total_point_count;
        count[cell_assign[i]] += 1;
    }
    for (size_t i = 0; i < _conf.coarse_cluster_count * _conf.fine_cluster_count; i++) {
        int c = i / _conf.fine_cluster_count;
        int f = i % _conf.fine_cluster_count;
        // LOG(INFO) << "count: " << c << " " << f << " " << count[i];
    }
}

int HierarchicalClusterIndex::compute_max_quantization_multipliers_and_norms(
                                    const ThreadParams& thread_params, 
                                    std::vector<float>& multipliers,
                                    std::vector<float>& dp_norms) const {
    std::unique_ptr<float[]> chunk_points(new
                                          float[FLAGS_thread_chunk_size * _conf.feature_dim]);
  
    for (uint32_t cid = 0; cid < thread_params.chunks_count; ++cid) {
        //LOG(INFO) << "HierarchicalClusterIndex::assign nearest_cell_assign_batch processing " << cid << "/" <<
                  // thread_params.chunks_count;
        int real_thread_chunk_size = std::min(FLAGS_thread_chunk_size,
                                              (int)(thread_params.points_count - cid * FLAGS_thread_chunk_size));
        int read_chunk_size = read_fvec_format(thread_params.learn_stream, _conf.feature_dim, real_thread_chunk_size,
                                               chunk_points.get());

        if (read_chunk_size != real_thread_chunk_size) {
            LOG(ERROR) << "puck assign from " << thread_params.start_id << " read file error at cid = " << cid << " / " <<
                       thread_params.chunks_count << ", ret = " << read_chunk_size;
            throw "read_fvec_format error!";
        }

    
        //计算最近的cell
        for (int point_id = 0; point_id < real_thread_chunk_size; ++point_id) {
            float* cur_point_fea = chunk_points.get() + point_id * _conf.feature_dim;
            float point_norms = cblas_sdot(_conf.feature_dim, cur_point_fea, 1, cur_point_fea, 1);
            dp_norms[thread_params.start_id + cid * FLAGS_thread_chunk_size + point_id] = point_norms;
            for (size_t j = 0; j < _conf.feature_dim; j++) {
                multipliers[j] = std::max<float>(multipliers[j], std::abs(cur_point_fea[j]));
            }

        }
    }
    //LOG(INFO) << "multipliers[j]: " << multipliers[0] << " " << multipliers[1];

    return 0;
}

int8_t int8_quantize(float value) {
    const float fp_val = std::round(value);
    if (fp_val > std::numeric_limits<int8_t>::max()) {
        return std::numeric_limits<int8_t>::max();
    }
    if (fp_val < std::numeric_limits<int8_t>::min()) {
        return std::numeric_limits<int8_t>::min();
    }
    return fp_val;
}

int HierarchicalClusterIndex::scalar_quantize_float_dataset_with_multipliers(
                                    const ThreadParams& thread_params, 
                                    std::vector<float>& multipliers,
                                    std::vector<int8_t>& quantized_dp) const {
    std::unique_ptr<float[]> chunk_points(new
                                          float[FLAGS_thread_chunk_size * _conf.feature_dim]);
  
    for (uint32_t cid = 0; cid < thread_params.chunks_count; ++cid) {
        //LOG(INFO) << "HierarchicalClusterIndex::assign nearest_cell_assign_batch processing " << cid << "/" <<
                  thread_params.chunks_count;
        int real_thread_chunk_size = std::min(FLAGS_thread_chunk_size,
                                              (int)(thread_params.points_count - cid * FLAGS_thread_chunk_size));
        int read_chunk_size = read_fvec_format(thread_params.learn_stream, _conf.feature_dim, real_thread_chunk_size,
                                               chunk_points.get());

        if (read_chunk_size != real_thread_chunk_size) {
            LOG(ERROR) << "puck assign from " << thread_params.start_id << " read file error at cid = " << cid << " / " <<
                       thread_params.chunks_count << ", ret = " << read_chunk_size;
            throw "read_fvec_format error!";
        }

    
        //计算最近的cell
        for (int point_id = 0; point_id < real_thread_chunk_size; ++point_id) {
            float* cur_point_fea = chunk_points.get() + point_id * _conf.feature_dim;
            auto global_point_id = thread_params.start_id + cid * FLAGS_thread_chunk_size + point_id;
            int8_t* cur_quantized_dp = quantized_dp.data() + (size_t)global_point_id * _conf.feature_dim;
            
            float point_norms = cblas_sdot(_conf.feature_dim, cur_point_fea, 1, cur_point_fea, 1);
            for (size_t j = 0; j < _conf.feature_dim; j++) {
                cur_quantized_dp[j] =
                        int8_quantize(cur_point_fea[j] * multipliers[j]);
            }

        }
    }

    return 0;
}

void HierarchicalClusterIndex::scalar_quantize_float_dataset(const uint32_t total_cnt, const std::string& feature_file_name) {
    LOG(INFO) << "HierarchicalClusterIndex::scalar_quantize_float_dataset";
    std::vector<std::thread> threads;
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;
    //int points_count = 10000;
    std::vector<int8_t> quantized_dp((size_t)_conf.feature_dim * _conf.total_point_count);
    std::vector<float> multipliers(_conf.feature_dim, 0.0f);
    std::vector<std::vector<float>> max_quantization_multipliers_vec(_conf.threads_count);
    for (auto& v: max_quantization_multipliers_vec) {
        v.resize(_conf.feature_dim);
    }

    std::vector<float> dp_norms(_conf.total_point_count);

    // 计算各个维度最大值
    for (uint32_t threadId = 0; threadId < _conf.threads_count; ++threadId) {
        threads.push_back(std::thread([&, threadId] {
            {
                ThreadParams thread_params;
                thread_params.points_count = std::ceil(1.0 * total_cnt / _conf.threads_count);
                thread_params.start_id = threadId* thread_params.points_count;
                thread_params.points_count = std::min(thread_params.points_count, (int)(total_cnt - thread_params.start_id));

                if (thread_params.points_count > 0) {
                    try {
                        thread_params.chunks_count = std::ceil(1.0 * thread_params.points_count / FLAGS_thread_chunk_size);
                        int param_stat = thread_params.open_file(feature_file_name.c_str(), _conf.feature_dim);

                        if (param_stat != 0) {
                            throw "open file has error.";
                        }

                        LOG(INFO) << "assign, thread_params.start_id = " << thread_params.start_id << " points_count = " <<
                                  thread_params.points_count << " feature_file_name = " << feature_file_name << " threadId = " << threadId;
                        compute_max_quantization_multipliers_and_norms(
                                thread_params, max_quantization_multipliers_vec[threadId], dp_norms);
                                
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                    }
                }
            }
        }));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    float max_multipliers = 0.0;
    
    for (auto& v: max_quantization_multipliers_vec) {
        for (size_t j = 0; j < _conf.feature_dim; j++) {
            // LOG(INFO) << "j: " << multipliers[j] << " " << std::abs(v[j]);
            multipliers[j] = std::max<float>(multipliers[j], std::abs(v[j]));
        }
    }
    
    /*
    for (auto& v: max_quantization_multipliers_vec) {
        for (size_t j = 0; j < _conf.feature_dim; j++) {
            max_multipliers = std::max<float>(max_multipliers, std::abs(v[j]));
        }
    }
    for (size_t j = 0; j < _conf.feature_dim; j++) {
        multipliers[j] = max_multipliers;
    }
    */


    for (float& f : multipliers) {
        if (f == 0.0f) {
            f = 1.0f;
        } else {
            f = std::numeric_limits<int8_t>::max() / f;
        }
    }

    if (lastException) {
        std::rethrow_exception(lastException);
    }

    threads.resize(0);
    for (uint32_t threadId = 0; threadId < _conf.threads_count; ++threadId) {
        threads.push_back(std::thread([&, threadId] {
            {
                ThreadParams thread_params;
                thread_params.points_count = std::ceil(1.0 * total_cnt / _conf.threads_count);
                thread_params.start_id = threadId* thread_params.points_count;
                thread_params.points_count = std::min(thread_params.points_count, (int)(total_cnt - thread_params.start_id));

                if (thread_params.points_count > 0) {
                    try {
                        thread_params.chunks_count = std::ceil(1.0 * thread_params.points_count / FLAGS_thread_chunk_size);
                        int param_stat = thread_params.open_file(feature_file_name.c_str(), _conf.feature_dim);

                        if (param_stat != 0) {
                            throw "open file has error.";
                        }

                        //LOG(INFO) << "assign, thread_params.start_id = " << thread_params.start_id << " points_count = " <<
                          //        thread_params.points_count << " feature_file_name = " << feature_file_name << " threadId = " << threadId;
                        scalar_quantize_float_dataset_with_multipliers(
                                thread_params, multipliers, quantized_dp);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                    }
                }
            }
        }));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    LOG(WARNING) << "quantized_dp: " << (int)quantized_dp[0] << " " << (int)quantized_dp[0]; // << " " << (int)quantized_dp[6713829 * 1024 + 1];
    LOG(WARNING) << "quantized_dp: " << quantized_dp[0] / multipliers[0] << " " << quantized_dp[0] / multipliers[1];
    LOG(WARNING) << "quantized_dp: " << quantized_dp.size() << " " << (size_t)_conf.feature_dim * _conf.total_point_count << " " << _conf.feature_dim << " " << _conf.total_point_count;
    if (lastException) {
        std::rethrow_exception(lastException);
    }

    std::ofstream fout(FLAGS_index_path + "/int8_dataset.dat", std::ofstream::binary);
    std::ofstream fout2(FLAGS_index_path + "/dp_norms.dat", std::ofstream::binary);
    std::ofstream fout3(FLAGS_index_path + "/int8_multipliers.dat", std::ofstream::binary);
    const char* ptr = reinterpret_cast<const char*>(quantized_dp.data());
    fout.write(ptr, quantized_dp.size());
    const char* ptr2 = reinterpret_cast<const char*>(dp_norms.data());
    fout2.write(ptr2, dp_norms.size() * sizeof(float));
    const char* ptr3 = reinterpret_cast<const char*>(multipliers.data());
    fout3.write(ptr3, multipliers.size() * sizeof(float));
    LOG(INFO) << "scalar_quantize_float_dataset succeeded";
}

///随机采样，去重
int random_sampling(const std::string& init_file_name, const u_int64_t total_cnt,
                    const u_int64_t sampling_cnt, const uint32_t feature_dim, float* sampling_vocab) {
    if (!sampling_vocab) {
        LOG(FATAL) << "sampling_vocab is nullptr";
        return -1;
    }

    //随机抽样
    std::mt19937 rnd(time(0));
    std::uniform_int_distribution<> dis(0, total_cnt-1);
    std::vector<bool> filter(total_cnt, false);

    std::ifstream learn_stream;
    learn_stream.open(init_file_name.c_str(), std::ios::binary);

    if (!learn_stream.good()) {
        learn_stream.close();
        LOG(FATAL) << "read all data file error : " << init_file_name;
        return -1;
    }

    uint32_t available_sample = 0;
    uint32_t filtered_cnt = 0;
    std::unordered_set<std::string> md5_str_set;
    base::MD5Digest md5_digest;

    while (available_sample < sampling_cnt && filtered_cnt < total_cnt) {
        std::vector<int> sampled_points;
        while(sampled_points.size() + available_sample < sampling_cnt && filtered_cnt < total_cnt){
            uint32_t rnd_int = dis(rnd);
            if (filter[rnd_int]){
                continue;
            } 
            filter[rnd_int] = true;
            filtered_cnt++;
            sampled_points.push_back(rnd_int);
        }

        std::sort(sampled_points.begin(), sampled_points.end());
        for(uint32_t i = 0; i < sampled_points.size(); ++i){

            int true_point_idx = sampled_points[i];
            u_int64_t offset = (u_int64_t)true_point_idx * feature_dim * sizeof(float) +
                            (u_int64_t)true_point_idx * sizeof(
                                int);
            learn_stream.seekg(offset, std::ios::beg);
            uint32_t cur_dim = -1;
            //feature长度检查，如果出错说明文件格式有问题
            learn_stream.read((char*)&cur_dim, sizeof(int));

            if (cur_dim != feature_dim) {
                LOG(FATAL) << true_point_idx << " feature dim error, " << cur_dim << " != " << feature_dim << init_file_name;
                    learn_stream.close();
                return -1;
            }

            u_int64_t vocab_offset = (u_int64_t)available_sample * feature_dim;
            learn_stream.read((char*)(sampling_vocab + vocab_offset), sizeof(float) * feature_dim);

            //判断是否是数字
            if (!isfinite((sampling_vocab + vocab_offset)[0])) {
                continue;
            }

            //重复数据检查
            base::MD5Sum((void*)(sampling_vocab + vocab_offset), sizeof(float) * feature_dim, &md5_digest);
            std::string md5_str = base::MD5DigestToBase16(md5_digest);

            if (md5_str_set.find(md5_str) != md5_str_set.end()) {
                //LOG(INFO)<<"duplicate:"<<available_sample<<"\t"<<md5_str;
                continue;
            }

            md5_str_set.insert(md5_str);
            ++available_sample;

            if (available_sample % 100000 == 0) {
                LOG(INFO) << "read sample " << available_sample << " from source file " << init_file_name;
            }
        }
    }

    learn_stream.close();
    return available_sample;
}

int HierarchicalClusterIndex::check_feature_dim() {
    int fd = -1;
    fd = open(_conf.feature_file_name.c_str(), O_RDONLY);
    struct stat st;
    size_t per_point_len = sizeof(uint32_t) + sizeof(float) * _conf.feature_dim;

    if (fd == -1 || -1 == fstat(fd, &st) || st.st_size % per_point_len != 0) {
        LOG(ERROR) << "model file " << _conf.feature_file_name << " stat error";
        return -1;
    }

    uint32_t feature_dim;

    if (read(fd, (char*)&feature_dim, sizeof(uint32_t)) < 0) {
        close(fd);
        LOG(ERROR) << "read file " << _conf.feature_file_name << " error.";
        return -1;
    }

    if (feature_dim != _conf.feature_dim) {
        close(fd);
        LOG(ERROR) << "feature_dim of file " << _conf.feature_file_name << " is " << feature_dim <<
                   ", feature_dim in GFLAGS is  _conf.feature_dim";
        return -1;
    }
    close(fd);
    _conf.total_point_count = st.st_size / per_point_len;
    return 0;
}

int HierarchicalClusterIndex::train() {
    if (check_feature_dim() != 0) {
        LOG(ERROR) << "check " << _conf.feature_file_name << " has error.";
        return -1;
    }

    LOG(INFO) << "total_point_count for train is " << _conf.total_point_count;

    if (init_model_memory() != 0) {
        return -1;
    }

    // reset train_points_count and pq_train_points_count
    if (_conf.train_points_count > 0) {
        std::string train_points_count_str = std::to_string(_conf.train_points_count);
        gflags::SetCommandLineOption("train_points_count", train_points_count_str.c_str());
        LOG(INFO) << "HierarchicalClusterIndex set train_points_count to " << _conf.train_points_count;
    }

    if (_conf.pq_train_points_count > 0) {
        std::string pq_train_points_count_str = std::to_string(_conf.pq_train_points_count);
        gflags::SetCommandLineOption("pq_train_points_count", pq_train_points_count_str.c_str());
        LOG(INFO) << "HierarchicalClusterIndex set pq_train_points_count to " << _conf.pq_train_points_count;
    }

    if (_conf.total_point_count < (uint32_t)FLAGS_train_points_count) {
        gflags::SetCommandLineOption("train_points_count", std::to_string(_conf.total_point_count).c_str());
    }

    //随机抽样kmeans聚类的训练数据
    u_int64_t train_vocab_len = (u_int64_t)FLAGS_train_points_count * _conf.feature_dim;
    std::unique_ptr<float[]> kmeans_train_vocab(new float[train_vocab_len]);

    int train_points_count = random_sampling(_conf.feature_file_name, _conf.total_point_count,
                            FLAGS_train_points_count, _conf.feature_dim, kmeans_train_vocab.get());

    // no need write tempfile
    //if (_conf.total_point_count <= (int)FLAGS_train_points_count) {
      //  gflags::SetCommandLineOption("train_points_count", std::to_string(train_points_count).c_str());
        //gflags::SetCommandLineOption("train_fea_file_name", 
          //      _conf.feature_file_name.c_str());
    //} else {
        if (train_points_count <= 0) {
            LOG(ERROR) << "sampling data has error.";
            return -1;
        }

        if (train_points_count < (int)FLAGS_train_points_count) {
            gflags::SetCommandLineOption("train_points_count", std::to_string(train_points_count).c_str());
            LOG(INFO) << "true point cnt for kmeans = " << train_points_count;
        }

        std::string cur_index_path = FLAGS_train_fea_file_name;
        cur_index_path = cur_index_path.substr(0, cur_index_path.rfind('/'));
        LOG(INFO) << "cur_index_path = " << cur_index_path;
        LOG(INFO) << "FLAGS_train_fea_file_name = " << FLAGS_train_fea_file_name;
        LOG(INFO) << "train_points_count: " << train_points_count;
        
        mkdir(cur_index_path.c_str(), 0777);
        //写文件，训练使用这批抽样数据
        int ret = write_fvec_format(FLAGS_train_fea_file_name.c_str(), _conf.feature_dim, train_points_count,
                                    kmeans_train_vocab.get());

        if (FLAGS_only_sample) {
            exit(0);
        }

        if (ret != 0) {
            return -1;
        }
    //}
    
    LOG(INFO) << "FLAGS_only_train_pq: " << FLAGS_only_train_pq;
    // exit(0);
    if (FLAGS_only_train_pq) {
        return 0;
    }

    if (train(FLAGS_train_points_count, kmeans_train_vocab.get()) != 0) {
        return 1;
    }

    return this->HierarchicalClusterIndex::save_codebooks();
}

int HierarchicalClusterIndex::init_single_build() {
    TIME_FUNCTION();

    //从文件获取配置信息
    if (read_model_file() != 0) {
        LOG(ERROR) << "read_model_file error";
        return -1;
    }

    _conf.total_point_count = FLAGS_single_build_max_points;

    //从文件获取配置信息
    if (init_model_memory() != 0) {
        LOG(ERROR) << "init_model_memory error";
        return -1;
    }

    //读码本
    if (read_codebooks() != 0) {
        LOG(ERROR) << "read_codebooks error";
        return -1;
    }

    if (_fine_norms == nullptr) {
        _fine_norms = new float[_conf.fine_cluster_count];

        for (uint32_t j = 0; j < _conf.fine_cluster_count; ++j) {
            //_fine_norms[j] = fvec_norm2sqr(_fine_vocab + _conf.feature_dim * j, _conf.feature_dim) / 2;
            _fine_norms[j] = cblas_sdot(_conf.feature_dim, _fine_vocab + _conf.feature_dim * j, 1,
                                        _fine_vocab + _conf.feature_dim * j, 1) / 2;
            _fine_norms[j] = 0 - std::sqrt(_fine_norms[j]);
        }
    }

    return 0;
}

int HierarchicalClusterIndex::single_build(BuildInfo* build_info) {
    if (_model == nullptr) {
        LOG(ERROR) << "should call init_model_memory() first to init builder";
        return -1;
    }

    build_info->nearest_cell.resize(1 + _conf.spilled_copy_num);
    build_info->nearest_cell_id.resize(1 + _conf.spilled_copy_num);

    float* chunk_points = build_info->feature.data();

    if (_conf.whether_norm) {
        //fvec_normalize(chunk_points, _conf.feature_dim, 2);
        float norm = cblas_snrm2(_conf.feature_dim, chunk_points, 1);

        if (norm < 1e-6) {
            LOG(ERROR) << "query norm is " << norm << ", could not be normalize";
            return -1;
        }

        cblas_sscal(_conf.feature_dim, 1.0 / norm, chunk_points, 1);
    }

    std::unique_ptr<float[]> points_coarse_terms(new float[_conf.coarse_cluster_count]);
    std::unique_ptr<float[]> points_fine_terms(new float[_conf.fine_cluster_count]);
    
    for(size_t i = 0; i < _conf.coarse_cluster_count; ++i){
        points_coarse_terms[i] = cblas_sdot(_conf.feature_dim, chunk_points, 1, _coarse_vocab + i * _conf.feature_dim, 1);
    }
    for(size_t i = 0; i < _conf.fine_cluster_count; ++i){
        points_fine_terms[i] = cblas_sdot(_conf.feature_dim, chunk_points, 1, _fine_vocab + i * _conf.feature_dim, 1);
    }
    
    float point_norms = cblas_sdot(_conf.feature_dim, chunk_points, 1, chunk_points, 1) / 2;
    nearest_cell_assign(points_coarse_terms.get(), points_fine_terms.get(), point_norms,
                        build_info->nearest_cell[0], 0);

    int cell_id = build_info->nearest_cell[0].cell_id;
    build_info->nearest_cell_id[0] = cell_id;
    if (cell_id < 0) {
        LOG(ERROR) << "get nearest cell id has error, error cell id = " << cell_id;
        return cell_id;
    }

    if (_conf.spilled_copy_num == 1) {
        bak_cell_assign2(
                points_coarse_terms.get(), 
                points_fine_terms.get(), 
                point_norms,
                build_info->nearest_cell[1],
                cell_id,
                0,
                chunk_points);
        int cell_id = build_info->nearest_cell[1].cell_id;
        build_info->nearest_cell_id[1] = cell_id;
    }

    return 0;
}

void HierarchicalClusterIndex::init_context_pool() {
    //初始化cpu逻辑内核数个context
    
    //LOG(INFO) << "init_context_pool";
    std::vector<SearchContext*> init_pool_vect(FLAGS_context_initial_pool_size, nullptr);
    //LOG(INFO) << "init_context_pool";
    for (int i = 0; i < (int)init_pool_vect.size(); ++i) {
        init_pool_vect[i] = _context_pool.Borrow();
        //LOG(INFO) << "init_context_pool " << i;
        while (init_pool_vect[i]->reset(_conf) != 0) {}
    }
    // if (true) return;
    //LOG(INFO) << "init_context_pool";
    for (int i = 0; i < (int)init_pool_vect.size(); ++i) {
        if (init_pool_vect[i]) {
            _context_pool.Return(init_pool_vect[i]);
        }
    }
}

const float* HierarchicalClusterIndex::normalization(SearchContext* context, const float* feature) {
    SearchCellData& search_cell_data = context->get_search_cell_data();

    if (_conf.ip2cos == 1) {
        uint32_t dim = _conf.feature_dim - 1;
        memset(search_cell_data.query_norm, 0, _conf.feature_dim);
        memcpy(search_cell_data.query_norm, feature, sizeof(float) * dim);
        return search_cell_data.query_norm;
    } else if (_conf.whether_norm) {
        memcpy(search_cell_data.query_norm, feature, sizeof(float) * _conf.feature_dim);
        //fvec_normalize(search_cell_data.query_norm, _conf.feature_dim, 2);
        float norm = cblas_snrm2(_conf.feature_dim, search_cell_data.query_norm, 1);

        if (norm < 1e-6) {
            LOG(ERROR) << "query norm is " << norm << ", could not be normalize";
            return nullptr;
        }

        cblas_sscal(_conf.feature_dim, 1.0 / norm, search_cell_data.query_norm, 1);
        return search_cell_data.query_norm;
    }

    return feature;
}

IndexConf load_index_conf_file() {
    HierarchicalClusterIndex index;
    index.read_model_file();
    return index._conf;
}

IndexType load_index_type() {
    //此处实现后续应该改成索引无关的,不论是不是基于hcluster都应该能从固定为主比如前4字节读到索引类型
    HierarchicalClusterIndex index;
    index.read_model_file();
    return index._conf.index_type;
}

//获取文件行数，index初始化时候通过key file确定样本总个数
int getFileLineCnt(const char* fileName) {
    struct stat st;

    if (stat(fileName, &st) != 0) {
        LOG(ERROR) << "checking file stat has error, file name = " << fileName;
        return -1;
    }

    char buff[1024];
    sprintf(buff, "wc -l %s", fileName);
    FILE* fstream = nullptr;
    fstream = popen(buff, "r");
    int total_line_cnt = -1;

    if (fstream) {
        memset(buff, 0x00, sizeof(buff));

        if (fgets(buff, sizeof(buff), fstream)) {
            int index = strchr((const char*)buff, ' ') - buff;
            buff[index] = '\0';
            total_line_cnt =  atoi(buff);
        }
    } else {
        LOG(ERROR) << "checking file line cnt has error, file name = " << fileName;
    }

    if (fstream) {
        pclose(fstream);
    }

    return total_line_cnt;
}

}//puck
