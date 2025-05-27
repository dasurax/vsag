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
 * @file puck_index.cpp
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2022-09-27 15:56
 * @brief
 *
 **/
#include <thread>
#include <omp.h>
#include <functional>
#include <malloc.h>
#include <numeric>
#include <unistd.h>
#include <memory>
#include <glog/logging.h>
#include <unordered_set>
#include <cassert>
#ifdef __SSE__
#include <immintrin.h>
#endif
#include "algorithm/puck/puck_gflags.h"
#include "algorithm/puck/max_heap.h"
#include "algorithm/puck/search_context.h"
#include "algorithm/puck/hnsw_distfunc_opt_impl_inline.h"
#include "algorithm/puck/puck_index.h"
#include "algorithm/puck/lut16_avx2.h"

namespace puck {
DEFINE_int32(pq_train_points_count, 1000000, "used for puck train pq codebooks");
DEFINE_string(train_pq_file_name, "mid-data/train_pq.dat", "random sampling for puck train pq codebooks");

#ifdef __SSE__
static inline __m128 masked_table_read(int d, const unsigned char* assign, const float* x, const size_t dim) {
    assert(0 <= d && d <= 4);
    __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};

    switch (d) {
    case 4:
        buf[0] = x[(int)assign[0]];
        buf[1] = (x + dim)[(int)assign[1]];
        buf[2] = (x + 2 * dim)[(int)assign[2]];
        buf[3] = (x + 3 * dim)[(int)assign[3]];
        break;

    case 3:
        buf[0] = x[(int)assign[0]];
        buf[1] = (x + dim)[(int)assign[1]];
        buf[2] = (x + 2 * dim)[(int)assign[2]];
        break;

    case 2:
        buf[0] = x[(int)assign[0]];
        buf[1] = (x + dim)[(int)assign[1]];
        break;

    case 1:
        buf[0] = x[(int)assign[0]];
        break;
    }

    return _mm_load_ps(buf);
}
#endif

float lookup_dist_table(const unsigned char* assign,
                        const float* dist_table, size_t dim, size_t nsq) {
    __m128 msum1 = _mm_setzero_ps();
    const unsigned char* x = assign;
    const float* y = dist_table;
    size_t d = nsq;

    while (d >= 8) {
        __m128 m1 = masked_table_read(4, x, y, dim);
        msum1 += m1;
        x += 4;
        y += 4 * dim;
        d -= 4;

        __m128 m2 = masked_table_read(4, x, y, dim);
        msum1 += m2;
        x += 4;
        y += 4 * dim;
        d -= 4;
        //LOG(INFO)<<_mm_cvtss_f32(msum1);
    }

    if (d >= 4) {
        __m128 m1 = masked_table_read(4, x, y, dim);
        msum1 += m1;
        x += 4;
        y += 4 * dim;
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_table_read(d, x, y, dim);
        msum1 += mx;
    }

    msum1 = _mm_hadd_ps(msum1, msum1);
    msum1 = _mm_hadd_ps(msum1, msum1);
    return  _mm_cvtss_f32(msum1);
}


float lookup_dist_table3(const unsigned char* assign,
                        const float* dist_table, size_t dim, size_t nsq) {
    __m128 msum1 = _mm_setzero_ps();
    const unsigned char* x = assign;
    const float* y = dist_table;
    size_t d = nsq;

    while (d >= 8) {
        __m128 m1 = masked_table_read(4, x, y, dim);
        msum1 += m1;
        x += 4;
        y += 4 * dim;
        d -= 4;

        __m128 m2 = masked_table_read(4, x, y, dim);
        msum1 += m2;
        x += 4;
        y += 4 * dim;
        d -= 4;
        //LOG(INFO)<<_mm_cvtss_f32(msum1);
    }

    if (d >= 4) {
        __m128 m1 = masked_table_read(4, x, y, dim);
        msum1 += m1;
        x += 4;
        y += 4 * dim;
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_table_read(d, x, y, dim);
        msum1 += mx;
    }

    msum1 = _mm_hadd_ps(msum1, msum1);
    msum1 = _mm_hadd_ps(msum1, msum1);
    return  _mm_cvtss_f32(msum1);
}

static inline __m256 masked_table_read_256(int d, const unsigned char* assign, const float* x, const size_t dim) {
    __attribute__((__aligned__(32))) float buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    buf[0] = x[(int)assign[0]];
    buf[1] = (x + dim)[(int)assign[1]];
    buf[2] = (x + 2 * dim)[(int)assign[2]];
    buf[3] = (x + 3 * dim)[(int)assign[3]];
    buf[4] = (x + 4 * dim)[(int)assign[4]];
    buf[5] = (x + 5 * dim)[(int)assign[5]];
    buf[6] = (x + 6 * dim)[(int)assign[6]];
    buf[7] = (x + 7 * dim)[(int)assign[7]];
       
    

    return _mm256_load_ps(buf);
}

float lookup_dist_table2(const unsigned char* assign,
                        const float* dist_table, size_t dim, size_t nsq) {
    __m256 msum1 = _mm256_setzero_ps();
    const unsigned char* x = assign;
    const float* y = dist_table;
    size_t d = nsq;

    while (d >= 8) {
        __m256 m1 = masked_table_read_256(4, x, y, dim);
        msum1 += m1;
        x += 8;
        y += 8 * dim;
        d -= 8;

        //LOG(INFO)<<_mm_cvtss_f32(msum1);
    }


    msum1 = _mm256_hadd_ps(msum1, msum1);
    msum1 = _mm256_hadd_ps(msum1, msum1);

    __m128 low = _mm256_extractf128_ps(msum1, 0);
    // 提取高 128 位
    __m128 high = _mm256_extractf128_ps(msum1, 1);

    // 提取低位浮点数（0到31位对应）
    float low_val = _mm_cvtss_f32(low); // 提取索引0
    // 提取高位浮点数（128到159位对应）
    float high_val = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 0)); // 提取索引0

    
    return low_val + high_val;
}


PuckIndex::PuckIndex() {
    _conf.index_type = IndexType::PUCK;
    _datapoints_by_token.resize(FLAGS_coarse_cluster_count * FLAGS_fine_cluster_count);
}

PuckIndex::~PuckIndex() {
}

void save_quantization_codebooks(const Quantization* quantization, const std::string& file_name) {
    quantization->save_codebooks(file_name);
}

long readProcStatus3(const std::string& name) {
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

int PuckIndex::save_codebooks() const {
    LOG(INFO) << "PuckIndex save codebooks";
    this->HierarchicalClusterIndex::save_codebooks();
    std::vector<std::thread> writers;

    if (_filter_quantization != nullptr) {
        std::thread write_codebook(save_quantization_codebooks, _filter_quantization.get(),
                                   _conf.filter_codebook_file_name);
        writers.push_back(std::move(write_codebook));
    }

    if (_pq_quantization != nullptr) {
        std::thread write_codebook(save_quantization_codebooks, _pq_quantization.get(), _conf.pq_codebook_file_name);
        writers.push_back(std::move(write_codebook));
    }

    for (auto& t : writers) {
        t.join();
    }

    return 0;
}

void save_quantization_index(const Quantization* quantization, const std::string& file_name) {
    quantization->save_index(file_name);
}

void save_quantization_index_bak(const Quantization* quantization, const std::string& file_name) {
    quantization->save_index_bak(file_name);
}

int PuckIndex::save_index() {
    LOG(INFO) << "---------------------------------------------- PuckIndex::save_index";
    this->HierarchicalClusterIndex::save_index();
    
    std::vector<std::thread> writers;
    LOG(INFO) << "get_quantized_feature: ";
    LOG(INFO) << "get_quantized_feature: " << ((float*)_filter_quantization->get_quantized_feature(0))[0];
    if (_filter_quantization != nullptr) {
        std::thread write_index(save_quantization_index, _filter_quantization.get(), _conf.filter_data_file_name);
        writers.push_back(std::move(write_index));
    }

    if (_pq_quantization != nullptr) {
        std::thread write_index(save_quantization_index, _pq_quantization.get(), _conf.pq_data_file_name);
        writers.push_back(std::move(write_index));
    }

    for (auto& t : writers) {
        t.join();
    }

    if (_conf.spilled_copy_num > 0) {
        //save_index_bak();
    }


    LOG(INFO) << "---------------------------------------------- end save_index";
    return 0;
}

int PuckIndex::save_index_bak() {
    LOG(INFO) << "---------------------------------------------- save_index_bak " << _conf.filter_data_bak_file_name;
    std::vector<std::thread> writers;

    if (_filter_quantization != nullptr) {
        std::thread write_index(save_quantization_index_bak, _filter_quantization.get(), _conf.filter_data_bak_file_name);
        writers.push_back(std::move(write_index));
    }

    for (auto& t : writers) {
        t.join();
    }

    LOG(INFO) << "---------------------------------------------- end save_index_bak";
    return 0;
}

int PuckIndex::save_index_after_convert() {
    
    LOG(INFO) << "---------------------------------------------- save_index_after_convert";
    this->HierarchicalClusterIndex::save_index_after_convert();
    std::vector<std::thread> writers;
    if (_filter_quantization != nullptr) {
        std::thread write_index(save_quantization_index, _filter_quantization.get(), _conf.filter_data_file_name);
        writers.push_back(std::move(write_index));
    }

    if (_pq_quantization != nullptr) {
        std::thread write_index(save_quantization_index, _pq_quantization.get(), _conf.pq_data_file_name);
        writers.push_back(std::move(write_index));
    }

    for (auto& t : writers) {
        t.join();
    }

    FILE* f = fopen(_conf.memory_to_local_file_name.c_str(), "wb");

    if (f == nullptr) {
        LOG(ERROR) << "cannot open " << _conf.memory_to_local_file_name << " for writing";
        return -1;
    }

    long ret = fwrite(_memory_to_local, sizeof(*_memory_to_local), _conf.total_point_count * (1 + _conf.spilled_copy_num), f);
    fclose(f);

    if (ret != _conf.total_point_count * (1 + _conf.spilled_copy_num)) {
        LOG(ERROR) << "writint to " << _conf.memory_to_local_file_name << " has error";
        return -1;
    }

    /*
    if (_conf.spilled_copy_num > 0) {
        // save_index_bak();

        FILE* f = fopen(_conf.memory_to_local_file_name.c_str(), "wb");

        if (f == nullptr) {
            LOG(ERROR) << "cannot open " << _conf.memory_to_local_file_name << " for writing";
            return -1;
        }

        long ret = fwrite(_memory_to_local, sizeof(*_memory_to_local), _conf.total_point_count * 2, f);
        fclose(f);

        if (ret != _conf.total_point_count * 2) {
            LOG(ERROR) << "writint to " << _conf.memory_to_local_file_name << " has error";
            return -1;
        }

    }
    */

    LOG(INFO) << "---------------------------------------------- end save_index_after_convert";
    return 0;
}

int PuckIndex::read_codebooks() {
    LOG(INFO) << "PuckIndex read_codebooks Start.";

    if (this->HierarchicalClusterIndex::read_codebooks() != 0) {
        LOG(ERROR) << "read HierarchicalClusterIndex codebook fail";
        return -1;
    }

    LOG(INFO) << "_filter_quantization->load_codebooks";
    if (_filter_quantization->load_codebooks(_conf.filter_codebook_file_name) != 0) {
        LOG(ERROR) << "read filter quantization codebook fail";
        return -2;
    }

    LOG(INFO) << "_pq_quantization->load_codebooks";
    //PQ量化的时候
    if (_conf.whether_pq) {
        if (_pq_quantization->load_codebooks(_conf.pq_codebook_file_name) != 0) {
            LOG(ERROR) << "read pq quantization codebook fail";
            return -2;
        }
    }

    LOG(INFO) << "PuckIndex read_codebooks Suc.";
    return 0;
}

int PuckIndex::read_feature_index(uint32_t* local_to_memory_idx, bool reset_quantized_feature) {
   
    if (!FLAGS_use_pq_method_1) {
        auto filter_data_file_name = _conf.filter_data_file_name;
        if (_conf.spilled_copy_num > 0) {
            //filter_data_file_name = _conf.filter_data_bak_file_name;
        }
        if (FLAGS_packed_cluster) {
            if (_filter_quantization->load(_conf.filter_codebook_file_name, filter_data_file_name,
                                        _datapoints_by_token, local_to_memory_idx, _min_offset) != 0) {
                LOG(ERROR) << "read_feature_index filter quantization Error.";
                return -1;
            }
        } else {
            if (_filter_quantization->load(_conf.filter_codebook_file_name, filter_data_file_name,
                                        local_to_memory_idx, _min_offset) != 0) {
                LOG(ERROR) << "read_feature_index filter quantization Error.";
                return -1;
            }
        }
        LOG(INFO) << "MEM4.2.5: " << readProcStatus3("VmRSS:");
        LOG(INFO) << "read_feature_index filter quantization Suc.";
    
        if (_conf.whether_pq) {
            auto local_to_memory_idx_ptr = _conf.spilled_copy_num > 0 ? nullptr : local_to_memory_idx;
            // LOG(ERROR) << "local_to_memory_idx: " << local_to_memory_idx[0] << " " << local_to_memory_idx[1];
            if (_pq_quantization->load(_conf.pq_codebook_file_name, _conf.pq_data_file_name,
                                    local_to_memory_idx_ptr) != 0) {
                LOG(ERROR) << "read_feature_index PQ quantization Error.";
                return -1;
            }

            LOG(INFO) << "read_feature_index PQ quantization Suc.";
        } else {
            //HierarchicalClusterIndex负责原始特征内存申请和释放
            if (this->HierarchicalClusterIndex::read_feature_index(local_to_memory_idx) != 0) {
                LOG(ERROR) << "read_feature_index HierarchicalClusterIndex Error.";
                return -1;
            }

            LOG(INFO) << "read_feature_index HierarchicalClusterIndex Suc";
        }
    } else {
        if (_pq_quantization->load(_conf.pq_codebook_file_name, _conf.pq_data_file_name,
                                    local_to_memory_idx) != 0) {
            LOG(ERROR) << "read_feature_index PQ quantization Error.";
            return -1;
        }

        if (_filter_quantization->load(_conf.filter_codebook_file_name, _conf.filter_data_file_name,
                                        local_to_memory_idx, _min_offset, 
                                        _pq_quantization->get_quantized_feature(0),
                                        _pq_quantization->get_per_fea_len()) != 0) {
                LOG(ERROR) << "read_feature_index filter quantization Error.";
                return -1;
            }

    }

    /*
    if (this->HierarchicalClusterIndex::read_feature_index(local_to_memory_idx) != 0) {
                LOG(ERROR) << "read_feature_index HierarchicalClusterIndex Error.";
                return -1;
            }
    */
    LOG(INFO) << "MEM4.2.7: " << readProcStatus3("VmRSS:");
    if (_conf.use_lut16 && reset_quantized_feature) {
        _filter_quantization->reset_quantized_feature();
    }
    LOG(INFO) << "MEM4.2.8: " << readProcStatus3("VmRSS:");
    //float* offset = (float*)_filter_quantization->get_quantized_feature(1);
    //LOG(INFO) << "offset: " << *offset;

    LOG(INFO) << "PuckIndex::read_quantization_index Suc.";
    return 0;
}

int PuckIndex::convert_local_to_memory_idx(uint32_t* cell_start_memory_idx, uint32_t* local_to_memory_idx) {
    LOG(INFO) << "convert_local_to_memory_idx";
    auto total_point_count = _conf.total_point_count;
    if (_conf.spilled_copy_num > 0) {
        total_point_count *= 2;
    }
    LOG(INFO) << "convert_local_to_memory_while_build: " << FLAGS_convert_local_to_memory_while_build << " " << _conf.memory_to_local_file_name;
    if (FLAGS_convert_local_to_memory_while_build) {
        if (load_cell_assign(_cell_assign) != 0) {
            LOG(INFO) << "load_cell_assign has error.";
            return -1;
        }  

        load_int_array_by_idx(_memory_to_local, _conf.memory_to_local_file_name, total_point_count);
        if (_conf.spilled_copy_num > 0) {
            //load_int_array_by_idx(_memory_to_ori_cell_id, _conf.memory_to_ori_cell_id_file_name, _conf.total_point_count);
            load_int_array_by_idx(_local_to_memory, _conf.local_to_memory_file_name, _conf.total_point_count);
        }

        auto cell_count = _conf.coarse_cluster_count * _conf.fine_cluster_count + 1;
        if (load_int_array_by_idx(cell_start_memory_idx, _conf.cell_start_memory_idx_file_name, cell_count) != 0) {
            LOG(INFO) << "load_cell_start_memory_idx has error.";
            return -1;
        }  

        for (uint32_t i = 0; i < _conf.fine_cluster_count * _conf.coarse_cluster_count; ++i) {
            int start_point_id = cell_start_memory_idx[i];
            int end_point_id = cell_start_memory_idx[i + 1];
            //if (end_point_id > start_point_id)
            
            int coarse_id = i / _conf.fine_cluster_count;
            int fine_id = i % _conf.fine_cluster_count;

            FineCluster& cur_fine_cluster = _coarse_clusters[coarse_id].fine_cell_list[fine_id];

            //if (point_reorder[end_point_id].second.first > 0)
            //LOG(INFO) << "min_offset: " << point_reorder[start_point_id].second.first << " " <<  point_reorder[end_point_id - 1].second.first;
            
            if (start_point_id > end_point_id) {
                LOG(INFO) << "start_point_id = " << start_point_id << ", end_point_id = " << end_point_id;
                LOG(INFO) << "load index error";
                return -1;
            }

            cur_fine_cluster.memory_idx_start = start_point_id;

        
            if (start_point_id == end_point_id) {
                cur_fine_cluster.stationary_cell_dist = std::sqrt(std::numeric_limits<float>::max());
            }

            _coarse_clusters[coarse_id].min_dist_offset = std::min(_coarse_clusters[coarse_id].min_dist_offset,
                    cur_fine_cluster.stationary_cell_dist);

        }
    } else {
        LOG(INFO) << "convert_local_to_memory_idx";

        std::unique_ptr<uint32_t[]> cell_assign(new uint32_t[total_point_count]);

        if (load_cell_assign(cell_assign.get()) != 0) {
            LOG(INFO) << "load_cell_assign has error.";
            return -1;
        }  

        // LOG(INFO) << "8309927: " << cell_assign[8309927] << " " << cell_assign[_conf.total_point_count + 8309927];

        std::unique_ptr<Quantization> filter_quantization;
        std::unique_ptr<Quantization> pq_quantization;
        LOG(INFO) << "QuantizationParams 0";
        QuantizationParams q_param;
        q_param.init(_conf, true);

        if (_conf.use_lut16) {
            q_param.ks = 16;
            q_param.spilled_copy_num = _conf.spilled_copy_num;
            filter_quantization.reset(new Quantization4BitPacked(q_param, _conf.total_point_count * (1 + _conf.spilled_copy_num)));
        } else {
            filter_quantization.reset(new Quantization(q_param, _conf.total_point_count));
        }
        LOG(INFO) << "QuantizationParams 1";
        QuantizationParams q_param2;
        q_param2.init(_conf);
        pq_quantization.reset(new Quantization(q_param2, _conf.total_point_count));
        LOG(INFO) << "MEM4.1.2: " << readProcStatus3("VmRSS:");
        LOG(INFO) << "convert_local_to_memory_idx filter: " << _conf.filter_codebook_file_name << " " << _conf.filter_data_file_name;
        
        if (!FLAGS_use_pq_method_1) {
            auto filter_data_file_name = _conf.filter_data_file_name;
            if (_conf.spilled_copy_num > 0) {
                filter_data_file_name = _conf.filter_data_bak_file_name;
            }
            if (filter_quantization->load(_conf.filter_codebook_file_name, filter_data_file_name) != 0) {
                LOG(ERROR) << "load " << _conf.filter_codebook_file_name << " error.";
                return -1;
            }
        } else {
            if (pq_quantization->load(_conf.pq_codebook_file_name, _conf.pq_data_file_name) != 0) {
                LOG(ERROR) << "read_feature_index PQ quantization Error.";
                return -1;
            }

            if (filter_quantization->load(_conf.filter_codebook_file_name, _conf.filter_data_file_name,
                                            nullptr, 0, 
                                            pq_quantization->get_quantized_feature(0),
                                            pq_quantization->get_per_fea_len()) != 0) {
                LOG(ERROR) << "read_feature_index filter quantization Error.";
                return -1;
            }
        }
        LOG(INFO) << "MEM4.1.3: " << readProcStatus3("VmRSS:");
        typedef std::pair<uint32_t, std::pair<float, uint32_t> > MemoryOrder;

        std::vector<MemoryOrder> point_reorder(total_point_count);

        for (uint32_t i = 0; i < total_point_count; ++i) {
            point_reorder[i].first = cell_assign[i];

            if (cell_assign[i] >= _conf.coarse_cluster_count * _conf.fine_cluster_count) {
                LOG(ERROR) << i << " " << cell_assign[i] << " " << _conf.coarse_cluster_count * _conf.fine_cluster_count;
                return -1;
            }

            point_reorder[i].second.first = 0;
            point_reorder[i].second.second = i;

            if (filter_quantization) {
                float* offset = (float*)filter_quantization->get_quantized_feature(i);
                point_reorder[i].second.first = *offset;
            }
        }
        LOG(INFO) << "MEM4.1.4: " << readProcStatus3("VmRSS:");
        std::stable_sort(point_reorder.begin(), point_reorder.end());

        uint32_t cellcount = 0;
        cell_start_memory_idx[cellcount] = 0;

        for (uint32_t i = 0; i < total_point_count; ++i) {
            uint32_t local_idx = point_reorder[i].second.second;

            if (i < 10) {
                //LOG(INFO) << "sorted local_idx: " << i << " " << local_idx << " " << point_reorder[i].second.first << " " << point_reorder[i].first;
            }
            //LOG(INFO) << "min_offset: " << point_reorder[i].second.first;
            local_to_memory_idx[local_idx] = i;
            
            if (local_idx >= _conf.total_point_count && local_idx <= _conf.total_point_count + 10) {
                LOG(INFO) << "local_idx: " << local_idx;
            }
            _memory_to_local[i] = local_idx % _conf.total_point_count;
            
            if (_conf.spilled_copy_num > 0) {
                
                if (local_idx < _conf.total_point_count) {
                    _local_to_memory[local_idx] = i;
                    //_memory_to_ori_cell_id[local_idx] = point_reorder[i].first;
                }

                //} else {
                    //_memory_to_ori_cell_id[i] = _memory_to_ori_cell_id[_local_to_memory[_memory_to_local[i]]];
                //}
            }

            

            while (cellcount + 1 <= point_reorder[i].first) {
                ++cellcount;
                cell_start_memory_idx[cellcount] = i;
            }
        }
        LOG(INFO) << "MEM4.1.5: " << readProcStatus3("VmRSS:");
        float min_offset = 1000.0;

        size_t total = 0;
        for (uint32_t i = 0; i < _conf.fine_cluster_count * _conf.coarse_cluster_count; ++i) {
            int start_point_id = cell_start_memory_idx[i];
            int end_point_id = cell_start_memory_idx[i + 1];
            int coarse_id = i / _conf.fine_cluster_count;
            int fine_id = i % _conf.fine_cluster_count;

            FineCluster& cur_fine_cluster = _coarse_clusters[coarse_id].fine_cell_list[fine_id];
            cur_fine_cluster.min_offset = point_reorder[start_point_id].second.first;
            min_offset = std::min(min_offset, cur_fine_cluster.min_offset);
            //if (point_reorder[end_point_id].second.first > 0)
            //LOG(INFO) << "min_offset: " << point_reorder[start_point_id].second.first << " " <<  point_reorder[end_point_id - 1].second.first;
            
            if (start_point_id > end_point_id) {
                LOG(INFO) << "start_point_id = " << start_point_id << ", end_point_id = " << end_point_id;
                LOG(INFO) << "load index error";
                return -1;
            }

            cur_fine_cluster.memory_idx_start = start_point_id;

        
            if (start_point_id == end_point_id) {
                cur_fine_cluster.stationary_cell_dist = std::sqrt(std::numeric_limits<float>::max());
                cur_fine_cluster.min_offset = 1000.0;
            }

            float total = 0.0;
            for (size_t j = start_point_id; j < end_point_id; j++) {
                total += point_reorder[j].second.first;
                _datapoints_by_token[i].push_back(_memory_to_local[j]);
                
            }   

            if (FLAGS_use_avg_offset) {
                if (start_point_id != end_point_id) {
                    cur_fine_cluster.min_offset = total / (end_point_id - start_point_id);
                }
            }


            _coarse_clusters[coarse_id].min_dist_offset = std::min(_coarse_clusters[coarse_id].min_dist_offset,
                    cur_fine_cluster.stationary_cell_dist);
        }

        for (uint32_t i = 0; i < _conf.fine_cluster_count * _conf.coarse_cluster_count; ++i) {
            int start_point_id = cell_start_memory_idx[i];
            int end_point_id = cell_start_memory_idx[i + 1];
            int coarse_id = i / _conf.fine_cluster_count;
            int fine_id = i % _conf.fine_cluster_count;

            FineCluster& cur_fine_cluster = _coarse_clusters[coarse_id].fine_cell_list[fine_id];
            total += cur_fine_cluster.get_point_cnt();

        }
        LOG(INFO) << "min_offset: " << min_offset << " " << total << " " << _conf.fine_cluster_count * _conf.coarse_cluster_count;
        min_offset = 10000;
        if (min_offset < 0) {
            for (uint32_t i = 0; i < _conf.fine_cluster_count * _conf.coarse_cluster_count; ++i) {
                int start_point_id = cell_start_memory_idx[i];
                int end_point_id = cell_start_memory_idx[i + 1];
                int coarse_id = i / _conf.fine_cluster_count;
                int fine_id = i % _conf.fine_cluster_count;

                FineCluster& cur_fine_cluster = _coarse_clusters[coarse_id].fine_cell_list[fine_id];

                if (start_point_id < end_point_id) {
                    cur_fine_cluster.min_offset -= min_offset;
                }

                if (cur_fine_cluster.min_offset < 0) {
                    LOG(INFO) << "---------min_offset";
                }
            }

            _min_offset = min_offset;
        }

        LOG(INFO) << "MEM4.1.4: " << readProcStatus3("VmRSS:");
        filter_quantization->reset();
        LOG(INFO) << "MEM4.1.5: " << readProcStatus3("VmRSS:");
        LOG(INFO) << "convert_local_to_memory_idx Suc.";
    }

    _converted = true;

    //22 33
    return 0;
}

int PuckIndex::convert_local_to_memory_idx(
        uint32_t* cell_start_memory_idx, 
        uint32_t* local_to_memory_idx,
        uint32_t* cell_assign,
        float* fea_offset,
        bool need_init) {
    LOG(INFO) << "convert_local_to_memory_idx";

    if (fea_offset != nullptr && need_init) {
        init_model_memory();
    }

    auto total_point_count = _conf.total_point_count * (1 + _conf.spilled_copy_num);

    typedef std::pair<uint32_t, std::pair<float, uint32_t> > MemoryOrder;

    std::vector<MemoryOrder> point_reorder(total_point_count);
    for (uint32_t i = 0; i < total_point_count; ++i) {
        point_reorder[i].first = cell_assign[i];
        if (cell_assign[i] >= _conf.coarse_cluster_count * _conf.fine_cluster_count) {
            LOG(ERROR) << i << " " << cell_assign[i];
            return -1;
        }

        point_reorder[i].second.first = 0;
        point_reorder[i].second.second = i;

        if (_filter_quantization) {
            if (fea_offset != nullptr) {
                point_reorder[i].second.first = fea_offset[i];
            } else {
                float* offset = (float*)_filter_quantization->get_quantized_feature(i);

                point_reorder[i].second.first = *offset;
            }
            
        }
    }
    std::stable_sort(point_reorder.begin(), point_reorder.end());

    uint32_t cellcount = 0;
    cell_start_memory_idx[cellcount] = 0;
    for (uint32_t i = 0; i < total_point_count; ++i) {
        uint32_t local_idx = point_reorder[i].second.second;

        local_to_memory_idx[local_idx] = i;
        _cell_assign[i] = point_reorder[i].first;

        // ori feature
        if (FLAGS_ad_hoc) {
            _memory_to_local[i] = local_idx % _conf.total_point_count;
            if (local_idx < _conf.total_point_count) {
                _local_to_memory[local_idx] = i;
            }
        } else {
            _memory_to_local[i] = local_idx / (1 + _conf.spilled_copy_num);
            if (local_idx % (1 + _conf.spilled_copy_num) == 0) {
                _local_to_memory[local_idx / (1 + _conf.spilled_copy_num)] = i;

            }
        }


        while (cellcount + 1 <= point_reorder[i].first) {
            ++cellcount;
            cell_start_memory_idx[cellcount] = i;
        }
    }

    read_feature_index(local_to_memory_idx, false);

    LOG(INFO) << "convert_local_to_memory_idx Suc.";

    // save_index();
    save_index_after_convert();
    _converted = true;

    return 0;
}

int PuckIndex::check_index_type() {
    if (_conf.index_type != IndexType::PUCK) {
        LOG(ERROR) << "index_type is not PUCK";
        return -1;
    }

    return 0;
}

int PuckIndex::init_model_memory() {
    LOG(INFO) << "init_model_memory " << _conf.total_point_count;
    this->HierarchicalClusterIndex::init_model_memory();
    QuantizationParams q_param;

    if (q_param.init(_conf, true) != 0) {
        LOG(ERROR) << "QuantizationParams of filter has Error.";
        return -1;
    }

    if (_conf.use_lut16) {
        q_param.ks = 16;
        if (_conf.spilled_copy_num > 0) {
            q_param.spilled_copy_num = _conf.spilled_copy_num;
            LOG(INFO) << "_conf.spilled_copy_num: " << _conf.spilled_copy_num;
            _filter_quantization.reset(new Quantization4BitPacked(q_param, (1 + _conf.spilled_copy_num) * _conf.total_point_count));
        } else {
            _filter_quantization.reset(new Quantization4BitPacked(q_param, _conf.total_point_count));
        }
    } else {
        _filter_quantization.reset(new Quantization(q_param, _conf.total_point_count));
    }
    LOG(WARNING) << "_conf.whether_pq: " << _conf.whether_pq;
    //PQ量化的时候
    if (_conf.whether_pq) {
        QuantizationParams q_param;
        if (q_param.init(_conf, false) != 0) {
            LOG(ERROR) << "QuantizationParams of PQ has Error.";
            return 1;
        }
        q_param.ks = 256;
        _pq_quantization.reset(new Quantization(q_param, _conf.total_point_count));
    }

    return 0;
}

int PuckIndex::compute_quantized_distance(SearchContext* context, const FineCluster* cur_fine_cluster,
        const float cell_dist, PuckMaxHeap& result_heap, int& total, size_t& total_cost) {
    float* result_distance = result_heap.get_top_addr();
    const float* pq_dist_table = context->get_search_point_data().pq_dist_table;
    const uint8_t* pq_dist_table_int8 = context->get_search_point_data().pq_dist_table_int8;
    
    //point info存储的量化特征对应的参数
    auto& quantization_params = _filter_quantization->get_quantization_params();
    uint32_t* query_sorted_tag = context->get_search_point_data().query_sorted_tag;
    auto point_cnt = cur_fine_cluster->get_point_cnt();
    uint32_t updated_cnt = 0;

    //base::Timer tm_cost;
    //tm_cost.start();
    for (uint32_t i = 0; i < point_cnt; ++i) {
        const unsigned char* feature = _filter_quantization->get_quantized_feature(
                                           cur_fine_cluster->memory_idx_start + i);
        float temp_dist = 2.0 * cell_dist + ((float*)feature)[0];
        if (temp_dist >= result_distance[0]) {
            break;
        }

        const unsigned char* pq_feature = (unsigned char*)feature + _filter_quantization->get_fea_offset();
#ifdef __SSE__
        //LOG(ERROR) << "lookup_dist_table";
        auto dist_ori = lookup_dist_table(pq_feature, pq_dist_table, quantization_params.ks, quantization_params.nsq);
        temp_dist += dist_ori;
        //auto dist_int8 = lookup_dist_table_16(pq_feature, pq_dist_table_int8, quantization_params.ks, quantization_params.nsq);
        //temp_dist += dist_ori;
#else

        for (uint32_t m = 0; m < (uint32_t)quantization_params.nsq; ++m) {
            uint32_t idx = query_sorted_tag[m];
            temp_dist += (pq_dist_table + idx * quantization_params.ks)[pq_feature[idx]];

            //当PQ子空间累计距离已经大于当前最大值，不再计算
            if (temp_dist > result_distance[0]) {
                break;
            }
        }

#endif
        
            /*
            auto vvv = _memory_to_local[cur_fine_cluster->memory_idx_start + i];
            //LOG(INFO) << vvv;
            if (vvv == 4549955 || vvv == 2904076 || vvv == 2043490 ) {
                LOG(INFO) << "vvv: " << vvv << " " << temp_dist << " " << result_distance[0];
            }
            */
        
            //if (_memory_to_local[cur_fine_cluster->memory_idx_start + i] == 4143571 || _memory_to_local[cur_fine_cluster->memory_idx_start + i] == 3047306 || _memory_to_local[cur_fine_cluster->memory_idx_start + i] == 169252) {
              //  LOG(INFO) << _memory_to_local[cur_fine_cluster->memory_idx_start + i] << " " << temp_dist << " " << result_distance[0];
            //}
        total += 1;
        if (temp_dist < result_distance[0]) {
            //LOG(INFO) << "temp_dist: " << temp_dist << " " << result_distance[0];
            result_heap.max_heap_update(temp_dist, cur_fine_cluster->memory_idx_start + i);
            ++updated_cnt;
        }
    }
    //tm_cost.stop();
    //total_cost += tm_cost.n_elapsed();
    return updated_cnt;
}

int lookup_dist_table_16_naive(std::vector<float>& dist, const uint8_t* data_start,
                        const uint8_t* lookups, size_t num_blocks, float multiplier) {

    constexpr uint8_t bias = static_cast<uint8_t>(1) << ((sizeof(uint8_t) * 8) - 1);
    for (size_t i = 0; i < 32; i++) {
        float d = 0.0;
        for (size_t j = 0; j < num_blocks; j++) {
            auto lookups_j = lookups + j * 16;
            auto idx = i % 16;
            auto v = (data_start + j * 16)[idx];
            int hi = i / 16;
            uint8_t newv = 0;
            if (hi) {
                newv = v >> 4;
            } else {
                newv = v & 0x0F;
            }
            // d += (float)lookups_j[(size_t)newv];
            d += (lookups_j[(size_t)newv] - bias) / multiplier;
            //if (i == 0)
              //  LOG(INFO) << "iiI: " << j << " " << (uint32_t)newv;
        }
        //if (std::fabs(d - dist[i]) > 1e-5)
          //  LOG(INFO) << "d: " << d << " " << dist[i] << " multiplier: " << multiplier; 
        dist[i] = d;
        //LOG(INFO) << "i: " << i << " " << dist[i];
    }
    return 0;
}

/*
void print_m256i(const __m256i& vec) {  
    // 遍历向量中的每个8位整数  
    for (int i = 0; i < 32; ++i) {  
        // 使用_mm256_extract_epi8提取第i个8位整数  
        int8_t val = _mm256_extract_epi8(vec, i);  
        // 打印这个值（注意：这里以int打印，但实际上它是int8_t）  
        std::cout << static_cast<int>(val) << " ";  
    }  
    std::cout << std::endl;  
}  
*/


uint32_t lookup_dist_table_16_clip(std::vector<float>& dist, const uint8_t* data_start,
                        const uint8_t* lookups, size_t num_blocks, 
                        float multiplier, __m256& inv_mults, __m256i& simd_thresholds, __m256& simd_biases, float* fea_offset, __m256i* results, bool debug) {

    constexpr uint8_t bias = static_cast<uint8_t>(1) << ((sizeof(uint8_t) * 8) - 1);
    __m256i int16_accums[4];
    for (size_t i = 0; i < 4; i++) {
        int16_accums[i] = _mm256_setzero_si256();
    }
    __m256i sign7 = _mm256_set1_epi8(0x0F);

    size_t total_block = 0;
    size_t num_unroll_iter = num_blocks / 2;
    for (; num_unroll_iter != 0; --num_unroll_iter) {
        constexpr uint32_t kPointsPerIter = 32;
        __m256i mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data_start));
        data_start += kPointsPerIter;
        
        __m256i mask0 = _mm256_and_si256(mask, sign7);
        __m256i mask1 = _mm256_and_si256(_mm256_srli_epi16(mask, 4), sign7);

        __m256i dict = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lookups));
        lookups += kPointsPerIter;
        __m256i res0 = _mm256_shuffle_epi8(dict, mask0);
        __m256i res1 = _mm256_shuffle_epi8(dict, mask1);

        int16_accums[0] = _mm256_add_epi16(int16_accums[0], res0);
        int16_accums[1] = _mm256_add_epi16(int16_accums[1], _mm256_srli_epi16(res0, 8));
        int16_accums[2] = _mm256_add_epi16(int16_accums[2], res1);
        int16_accums[3] = _mm256_add_epi16(int16_accums[3], _mm256_srli_epi16(res1, 8));
        
        total_block += 2;
        
        if (total_block > 170) {
            results[0] =
                    PostprocessAccumulatorPair(int16_accums[0], int16_accums[1]);
            results[1] =
                    PostprocessAccumulatorPair(int16_accums[2], int16_accums[3]);
            //if (debug) {
            //  print_m256i_as_16bits(results[0]);
            //}

            __m256i total_bias = _mm256_set1_epi16(total_block * 128);
            results[0] = _mm256_sub_epi16(results[0], total_bias);
            results[1] = _mm256_sub_epi16(results[1], total_bias);

            __m256i masks[2];
            masks[0] = _mm256_cmpgt_epi16(simd_thresholds, results[0]);
            masks[1] = _mm256_cmpgt_epi16(simd_thresholds, results[1]);

            uint32_t push_mask = get_comparison_mask(masks[0], masks[1]);
            if (!push_mask) {
                //LOG(INFO) << "clip " << total_block << " " << num_blocks;
                //print_m256i_as_16bits(simd_thresholds);
                //print_m256i_as_16bits(results[0]);
                //print_m256i_as_16bits(results[1]);
                return push_mask;
            }
        }

    }

    //__m256i results[2];
    results[0] =
            PostprocessAccumulatorPair(int16_accums[0], int16_accums[1]);
    results[1] =
            PostprocessAccumulatorPair(int16_accums[2], int16_accums[3]);

    //if (debug) {
      //  print_m256i_as_16bits(results[0]);
    //}

    __m256i total_bias = _mm256_set1_epi16(num_blocks * 128);
    results[0] = _mm256_sub_epi16(results[0], total_bias);
    results[1] = _mm256_sub_epi16(results[1], total_bias);

    __m256i masks[2];
    masks[0] = _mm256_cmpgt_epi16(simd_thresholds, results[0]);
    masks[1] = _mm256_cmpgt_epi16(simd_thresholds, results[1]);

    uint32_t push_mask = get_comparison_mask(masks[0], masks[1]);

    //_mm256_storeu_ps(distances_buffer + 8, _mm256_add_ps(_mm256_mul_ps(fvals[1], inv_mults), simd_biases));
    //_mm256_storeu_ps(distances_buffer + 16, _mm256_add_ps(_mm256_mul_ps(fvals[2], inv_mults), simd_biases));
    //_mm256_storeu_ps(distances_buffer + 24, _mm256_add_ps(_mm256_mul_ps(fvals[3], inv_mults), simd_biases));


    //if (push_mask) {
      //  LOG(INFO) << "push_mask: " << push_mask;
    //}
    
    //if (true) {
    if (push_mask) {
       // multiplier = max_integer_value / max_abs_lookup_element
        __m256i expand_results[4];

        __m128i hi = _mm256_extractf128_si256(results[0], 1);
        __m128i lo = _mm256_castsi256_si128(results[0]);

        expand_results[0] = _mm256_cvtepi16_epi32(lo);
        expand_results[1] = _mm256_cvtepi16_epi32(hi);

        hi = _mm256_extractf128_si256(results[1], 1);
        lo = _mm256_castsi256_si128(results[1]);

        expand_results[2] = _mm256_cvtepi16_epi32(lo);
        expand_results[3] = _mm256_cvtepi16_epi32(hi);

        __m256 fvals[4];
        fvals[0] = _mm256_cvtepi32_ps(expand_results[0]);
        fvals[1] = _mm256_cvtepi32_ps(expand_results[1]);
        fvals[2] = _mm256_cvtepi32_ps(expand_results[2]);
        fvals[3] = _mm256_cvtepi32_ps(expand_results[3]);

        // fvals * inv_mults[j] + simd_biases[j]
        // std::vector<float> dist2(32);
        float* distances_buffer = dist.data();
        for (size_t i = 0; i < 4; i++) {
            _mm256_storeu_ps(distances_buffer + i * 8,
                    _mm256_add_ps(
                        _mm256_add_ps(
                            _mm256_mul_ps(fvals[i], inv_mults), 
                            simd_biases), 
                        _mm256_load_ps(fea_offset + i * 8)));
        }
        
    }
    return push_mask;
}

/*
template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX2_INLINE Avx2<int16_t, kNumQueries, 2> Avx2LUT16BottomLoop(
        const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
        const DimensionIndex num_blocks, const uint8_t* prefetch_start) {
    static_assert(kNumQueries <= 3,
                    "Register spilling happens when kNumQueries > 3");
    Avx2<int16_t, kNumQueries, 4> int16_accums = avx2::Zeros();
    const Avx2<uint8_t> sign7 = 0x0F;

    DimensionIndex num_unroll_iter = num_blocks / 2;
    for (; num_unroll_iter != 0; --num_unroll_iter) {
        constexpr uint32_t kPointsPerIter = 32;
        if constexpr (kPrefetch == PrefetchStrategy::kSeq) {
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
            data_start + kPrefetchBytesAhead);
        } else if constexpr (kPrefetch == PrefetchStrategy::kSmart) {
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_NTA>(
            prefetch_start);
        prefetch_start += kPointsPerIter;
        }

        auto mask = Avx2<uint8_t>::Load(data_start);
        data_start += kPointsPerIter;

        Avx2<uint8_t> mask0 = mask & sign7;

        Avx2<uint8_t> mask1 = Avx2<uint8_t>((Avx2<uint16_t>(mask) >> 4)) & sign7;

        for (size_t j : Seq(kNumQueries)) {
        const Avx2<uint8_t> dict = Avx2<uint8_t>::Load(lookup_starts[j]);
        lookup_starts[j] += kPointsPerIter;
        const Avx2<uint8_t> res0 = _mm256_shuffle_epi8(*dict, *mask0);
        const Avx2<uint8_t> res1 = _mm256_shuffle_epi8(*dict, *mask1);

        int16_accums[j][0] += Avx2<int16_t>(Avx2<uint16_t>(res0));
        int16_accums[j][1] += Avx2<int16_t>(Avx2<uint16_t>(res0) >> 8);
        int16_accums[j][2] += Avx2<int16_t>(Avx2<uint16_t>(res1));
        int16_accums[j][3] += Avx2<int16_t>(Avx2<uint16_t>(res1) >> 8);
        }
    }

    Avx2<int16_t, kNumQueries, 2> results;
    for (size_t j : Seq(kNumQueries)) {
        results[j][0] =
            PostprocessAccumulatorPair(int16_accums[j][0], int16_accums[j][1]);
        results[j][1] =
            PostprocessAccumulatorPair(int16_accums[j][2], int16_accums[j][3]);
    }

    const bool has_odd_block = num_blocks & 1;
    if (has_odd_block) {
        const Sse4<uint8_t> sign7 = 0x0F;
        const Sse4<uint8_t> mask = Sse4<uint8_t>::Load(data_start);
        const Sse4<uint8_t> mask0 = mask & sign7;
        const Sse4<uint8_t> mask1 =
            Sse4<uint8_t>((Sse4<uint16_t>(mask) >> 4)) & sign7;

        for (size_t j : Seq(kNumQueries)) {
        auto dict = Sse4<uint8_t>::Load(lookup_starts[j]);
        Sse4<uint8_t> val0 = _mm_shuffle_epi8(*dict, *mask0);
        Sse4<uint8_t> val1 = _mm_shuffle_epi8(*dict, *mask1);
        results[j][0] += Avx2<int16_t>(_mm256_cvtepu8_epi16(*val0));
        results[j][1] += Avx2<int16_t>(_mm256_cvtepu8_epi16(*val1));
        }
    }

    const Avx2<int16_t> total_bias = num_blocks * 128;
    for (size_t j : Seq(kNumQueries)) {
        results[j] -= total_bias;
    }
    return results;
}
*/

int PuckIndex::compute_quantized_distance_lut16(SearchContext* context, const FineCluster* cur_fine_cluster,
        const float cell_dist, PuckMaxHeap& result_heap, FastMaxHeap& result_heap2, int& total, float multiplier, size_t& total_cost) {
    float* result_distance = result_heap.get_top_addr();
    const float* pq_dist_table = context->get_search_point_data().pq_dist_table;
    const uint8_t* pq_dist_table_int8 = context->get_search_point_data().pq_dist_table_int8;

    auto& quantization_params = _filter_quantization->get_quantization_params();
    uint32_t* query_sorted_tag = context->get_search_point_data().query_sorted_tag;
    auto point_cnt = cur_fine_cluster->get_point_cnt();
    uint32_t updated_cnt = 0;   
    
    
    uint32_t iters_start = cur_fine_cluster->memory_idx_start / 32;
    uint32_t first_j = cur_fine_cluster->memory_idx_start - iters_start * 32;
    uint32_t iters_end = (cur_fine_cluster->memory_idx_start + point_cnt + 31) / 32;
    uint32_t last_j = cur_fine_cluster->memory_idx_start + point_cnt - iters_end * 32 + 32;
    uint32_t r = (cur_fine_cluster->memory_idx_start + point_cnt) % 32;
    uint32_t idx_start = cur_fine_cluster->memory_idx_start;
    uint32_t idx_end = cur_fine_cluster->memory_idx_start + point_cnt;
    uint32_t num_32dp_simd_iters2 = (point_cnt + 31) / 32;
    uint32_t num_32dp_simd_iters = iters_end - iters_start;
    //LOG(INFO) << "idx_start: " << idx_start << " " << idx_end;
    std::vector<float> dist(32, 0.0);
    __m256 simd_biases = _mm256_set1_ps(2.0 * cell_dist);
    __m256 inv_mults = _mm256_set1_ps(1.0 / multiplier);
    __m256i results[2];
    for (uint32_t k = 0; k < num_32dp_simd_iters; k++) {
        total += 32;
        // 这里的 fea_offset 是有序的，从小到大
        float* fea_offset = _filter_quantization->get_fea_offset_packed(iters_start + k);

        int j_start = k == 0 ? first_j : 0;
        int j_end = k == num_32dp_simd_iters - 1 ? last_j : 32;
        uint32_t end_mask = (k == num_32dp_simd_iters - 1 && last_j != 32) ? ((1U << last_j) - 1) : 4294967295;
        //uint32_t end_mask = (k == num_32dp_simd_iters - 1) ? ((1U << last_j) - 1) : 4294967295;
        // float float_threshold = result_heap2.epsilon() - (2.0 * cell_dist + fea_offset[j_start]);
        float float_threshold = result_distance[0] - (2.0 * cell_dist + fea_offset[j_start]);
        
        //LOG(INFO) << "float_threshold: " << k << " " << num_32dp_simd_iters << " " << float_threshold;
        int16_t int16_threshold = get_int16_threshold(float_threshold * multiplier);
        __m256i simd_thresholds = _mm256_set1_epi16(int16_threshold);

        const uint8_t* data_start = (const uint8_t*)_filter_quantization->get_quantized_feature_packed(iters_start + k);
        bool debug = false;
        /*
        for (size_t j = 0; j < 32; j++) {
            auto vvv = _memory_to_local[(iters_start + k) * 32 + j];
            // LOG(INFO) << push_mask;
            if (vvv == 6668151
                    || vvv == 7128161|| vvv == 6774047|| vvv == 4847143|| vvv == 2389110|| vvv == 955695
                    || vvv == 3283536|| vvv == 6966618|| vvv == 4294400|| vvv == 5303765|| vvv == 1543531) {
                debug = true;
            }
        } 
        */
        uint32_t push_mask = lookup_dist_table_16(dist, data_start, pq_dist_table_int8, quantization_params.nsq, 
                multiplier, inv_mults, simd_thresholds, simd_biases, fea_offset, results, debug);
        
        //std::vector<float> dist2(32, 0.0);
        
        //lookup_dist_table_16_naive(dist2, data_start, pq_dist_table_int8, quantization_params.nsq, multiplier);
        push_mask &= ((uint32_t)-1 << j_start);
      
        push_mask &= end_mask;
        
        /*
        for (size_t j = 0; j < 32; j++) {
            auto vvv = _memory_to_local[(iters_start + k) * 32 + j];
            // LOG(INFO) << push_mask;
            if (vvv == 6668151
                    || vvv == 7128161|| vvv == 6774047|| vvv == 4847143|| vvv == 2389110|| vvv == 955695
                    || vvv == 3283536|| vvv == 6966618|| vvv == 4294400|| vvv == 5303765|| vvv == 1543531) {
                LOG(INFO) << "vvv: " << k << " j: " << j << " " << vvv << " " << dist[j] << " " << cell_dist << " " << result_distance[0] << " " << push_mask;
            }
        } 
        */

        if (!push_mask) {
            continue;
            // break;
        }

        while (push_mask) {
            const int j = find_lsb_set_non_zero(push_mask);

            push_mask &= (push_mask - 1);

            if (dist[j] < result_distance[0]) {
                result_heap.max_heap_update(dist[j], 2.0 * cell_dist + fea_offset[j], (iters_start + k) * 32 + j);
                ++updated_cnt;
            }
        }
    }
    return updated_cnt;
}

int PuckIndex::compute_quantized_distance_lut16_fast_heap(SearchContext* context, const FineCluster* cur_fine_cluster,
        const float cell_dist, PuckMaxHeap& result_heap, FastMaxHeap& result_heap2, int& total, float multiplier, size_t& total_cost, size_t token, std::vector<float>& dist, float thr) {
    float* result_distance = result_heap.get_top_addr();
    const float* pq_dist_table = context->get_search_point_data().pq_dist_table;
    const uint8_t* pq_dist_table_int8 = context->get_search_point_data().pq_dist_table_int8;

    auto& quantization_params = _filter_quantization->get_quantization_params();
    uint32_t* query_sorted_tag = context->get_search_point_data().query_sorted_tag;
    auto point_cnt = cur_fine_cluster->get_point_cnt();
    uint32_t updated_cnt = 0;   
    
    uint32_t num_32dp_simd_iters = (point_cnt + 31) / 32;

    __m256 simd_biases = _mm256_set1_ps(2.0 * cell_dist);
    __m256 inv_mults = _mm256_set1_ps(1.0 / multiplier);
    __m256i results[2];
    uint32_t last_j = point_cnt % 32;
    //LOG(INFO) << "token: " << token << " " << point_cnt;
    for (uint32_t k = 0; k < num_32dp_simd_iters; k++) {
        //LOG(INFO) << "k: " << k;
        total += 32;
        // 这里的 fea_offset 是有序的，从小到大
        float* fea_offset = _filter_quantization->get_fea_offset_packed(token, k);

    
        uint32_t end_mask = last_j != 0 ? ((1U << last_j) - 1) : 4294967295;
        //uint32_t end_mask = (k == num_32dp_simd_iters - 1) ? ((1U << last_j) - 1) : 4294967295;
        // float float_threshold = result_heap2.epsilon() - (2.0 * cell_dist + fea_offset[j_start]);
        float float_threshold = result_heap2.epsilon() - (2.0 * cell_dist + fea_offset[0]);
        // float float_threshold = result_heap2.epsilon() - (2.0 * cell_dist + fea_offset[j_start]);

        if (float_threshold < thr) {
            //LOG(INFO) << "float_threshold: " << k << " " << num_32dp_simd_iters << " " << float_threshold << " " << push_mask;
            
            break;
        }
        
        int16_t int16_threshold = get_int16_threshold(float_threshold * multiplier);
        __m256i simd_thresholds = _mm256_set1_epi16(int16_threshold);

        const uint8_t* data_start = (const uint8_t*)_filter_quantization->get_quantized_feature_packed(token, k);
        bool debug = false;
        /*
        bool debug = false;
        for (size_t j = 0; j < 32; j++) {
            auto vvv = _memory_to_local[(iters_start + k) * 32 + j];
            // LOG(INFO) << push_mask;
            if (vvv == 6668151
                    || vvv == 7128161|| vvv == 6774047|| vvv == 4847143|| vvv == 2389110|| vvv == 955695
                    || vvv == 3283536|| vvv == 6966618|| vvv == 4294400|| vvv == 5303765|| vvv == 1543531) {
                debug = true;
            }
        } 
        */
        

        uint32_t push_mask = lookup_dist_table_16(dist, data_start, pq_dist_table_int8, quantization_params.nsq, 
                multiplier, inv_mults, simd_thresholds, simd_biases, fea_offset, results, debug);

        if (k == num_32dp_simd_iters - 1) {
        
            push_mask &= end_mask;
        }

        //LOG(INFO) << "---------" << point_cnt << " " << k << " " << __builtin_popcount(push_mask) << " " << float_threshold << " " << push_mask << " " << end_mask;
        
        /*
        for (size_t j = 0; j < 32; j++) {
            auto vvv = _memory_to_local[(iters_start + k) * 32 + j];
            // LOG(INFO) << push_mask;
            if (vvv == 6668151
                    || vvv == 7128161|| vvv == 6774047|| vvv == 4847143|| vvv == 2389110|| vvv == 955695
                    || vvv == 3283536|| vvv == 6966618|| vvv == 4294400|| vvv == 5303765|| vvv == 1543531) {
                LOG(INFO) << "vvv: " << k << " j: " << j << " " << vvv << " " << dist[j] << " " << cell_dist << " " << result_distance[0] << " " << push_mask;
            }
        } 
        */
        /*
        std::vector<float> dist2(32, 0.0);
        if (float_threshold < 1110.9) {
            //LOG(INFO) << "float_threshold: " << k << " " << num_32dp_simd_iters << " " << float_threshold << " " << push_mask;
            
            //break;
            
            lookup_dist_table_16_naive(dist2, data_start, pq_dist_table_int8, quantization_params.nsq, multiplier);
        }
        */
        if (!push_mask) {
            continue;
            // break;
        }
        
        /*
        if (float_threshold < 0.8) {
            LOG(INFO) << "float_threshold: " << k << " " << num_32dp_simd_iters << " " << float_threshold << " " << push_mask;
            std::vector<float> dist2(32, 0.0);
            
            // lookup_dist_table_16_naive(dist2, data_start, pq_dist_table_int8, quantization_params.nsq, multiplier);
        }
        */

        /*
        for (int j = 0; j < 32; j++) {
            auto vvv = _memory_to_local[(iters_start + k) * 32 + j];
            // LOG(INFO) << push_mask;
            if (vvv == 645396 || vvv == 1328825 || vvv == 6542130 || vvv == 3863826 || vvv == 1674400
                    || vvv == 4020063 || vvv == 4869310 || vvv == 3323252 || vvv == 677891
                    || vvv == 582536 || vvv == 525656 || vvv == 5992872 || vvv == 6589606) {
                //LOG(INFO) << "vvv: " << vvv << " " << dist[j] << " " << result_distance[0];
            }
        }
        */

        while (push_mask) {
            const int j = find_lsb_set_non_zero(push_mask);
            
            //LOG(INFO) << j << " " << cur_fine_cluster->memory_idx_start + k * 32 + j << " " << _memory_to_local[cur_fine_cluster->memory_idx_start + k * 32 + j] << " " << dist2[j] << " " << push_mask;
            /*
            auto vvv = _memory_to_local[(iters_start + k) * 32 + j];

            if (vvv == 645396 || vvv == 1328825 || vvv == 6542130 || vvv == 3863826 || vvv == 1674400
                    || vvv == 4020063 || vvv == 4869310 || vvv == 3323252 || vvv == 677891
                    || vvv == 582536 || vvv == 525656 || vvv == 5992872 || vvv == 6589606) {
                //LOG(INFO) << "vvv: " << vvv << " " << dist[j] << " " << result_distance[0];
            }
            */

            push_mask &= (push_mask - 1);

            //auto vvv = _memory_to_local[(iters_start + k) * 32 + j];
            // LOG(INFO) << push_mask;
            //if (vvv == 228
              //      || vvv == 72) {
                //LOG(INFO) << "vvv: " << vvv << " " << dist[j] << " " << result_distance[0];
            //}

            if (dist[j] < result_heap2.epsilon()) {
                //LOG(INFO) << "dist: " << dist[j];
                bool needs_gc = result_heap2.max_heap_update(dist[j], 2.0 * cell_dist + fea_offset[j], cur_fine_cluster->memory_idx_start + k * 32 + j);
                //LOG(INFO) << "dist: " << dist[j];
                if (needs_gc) {
                    if (_conf.use_lut16) {
                        //LOG(INFO) << "garbage_collect_aux";
                        result_heap2.garbage_collect_aux();
                        //LOG(INFO) << "garbage_collect_aux2";
                    } else {
                        result_heap2.garbage_collect();
                    }
                    float_threshold = result_heap2.epsilon() - (2.0 * cell_dist + fea_offset[0]);

                    int16_threshold = get_int16_threshold(float_threshold * multiplier);
                    simd_thresholds = _mm256_set1_epi16(int16_threshold);

                    __m256i masks[2];
                    masks[0] = _mm256_cmpgt_epi16(simd_thresholds, results[0]);
                    masks[1] = _mm256_cmpgt_epi16(simd_thresholds, results[1]);

                    push_mask &= get_comparison_mask(masks[0], masks[1]);
                }
                //result_heap.max_heap_update(dist[j], 0, (iters_start + k) * 32 + j);
                ++updated_cnt;
            }
        }
    }
    return updated_cnt;
}

int PuckIndex::compute_quantized_distance_lut16_fast_heap2(SearchContext* context, const FineCluster* cur_fine_cluster,
        const float cell_dist, PuckMaxHeap& result_heap, FastMaxHeap& result_heap2, int& total, float multiplier, size_t& total_cost, size_t token, size_t l,
        std::vector<float>& dist, float thr, double& max_q, size_t total_updated_cnt) {
    //LOG(INFO) << "compute_quantized_distance_lut16_fast_heap2";
    float* result_distance = result_heap.get_top_addr();
    const float* pq_dist_table = context->get_search_point_data().pq_dist_table;
    const uint8_t* pq_dist_table_int8 = context->get_search_point_data().pq_dist_table_int8;

    auto& quantization_params = _filter_quantization->get_quantization_params();
    uint32_t* query_sorted_tag = context->get_search_point_data().query_sorted_tag;
    auto point_cnt = cur_fine_cluster->get_point_cnt();
    uint32_t updated_cnt = 0;   
    
    // max_q = 100.0;
    uint32_t iters_start = cur_fine_cluster->memory_idx_start / 32;
    uint32_t first_j = cur_fine_cluster->memory_idx_start - iters_start * 32;
    uint32_t iters_end = (cur_fine_cluster->memory_idx_start + point_cnt + 31) / 32;
    uint32_t last_j = cur_fine_cluster->memory_idx_start + point_cnt - iters_end * 32 + 32;
    uint32_t r = (cur_fine_cluster->memory_idx_start + point_cnt) % 32;
    uint32_t idx_start = cur_fine_cluster->memory_idx_start;
    uint32_t idx_end = cur_fine_cluster->memory_idx_start + point_cnt;
    uint32_t num_32dp_simd_iters2 = (point_cnt + 31) / 32;
    uint32_t num_32dp_simd_iters = iters_end - iters_start;
    //LOG(INFO) << "idx_start: " << idx_start << " " << idx_end;

    __m256 simd_biases = _mm256_set1_ps(2.0 * cell_dist);
    __m256 inv_mults = _mm256_set1_ps(1.0 / multiplier);
    __m256i results[2];
    uint32_t push_mask = 0;
   
    for (uint32_t k = 0; k < num_32dp_simd_iters; k++) {
        //LOG(INFO) << "k: " << k << " " << num_32dp_simd_iters << " " << cur_fine_cluster->memory_idx_start << " " << point_cnt;
        total += 32;
        // 这里的 fea_offset 是有序的，从小到大
        float* fea_offset = _filter_quantization->get_fea_offset_packed(iters_start + k);

        int j_start = k == 0 ? first_j : 0;
        int j_end = k == num_32dp_simd_iters - 1 ? last_j : 32;
        uint32_t end_mask = (k == num_32dp_simd_iters - 1 && last_j != 32) ? ((1U << last_j) - 1) : 4294967295;
        //uint32_t end_mask = (k == num_32dp_simd_iters - 1) ? ((1U << last_j) - 1) : 4294967295;
        // float float_threshold = result_heap2.epsilon() - (2.0 * cell_dist + fea_offset[j_start]);
        float float_threshold = result_heap2.epsilon() - (2.0 * cell_dist + fea_offset[j_start]);
        
        int16_t int16_threshold = get_int16_threshold(float_threshold * multiplier);
        __m256i simd_thresholds = _mm256_set1_epi16(int16_threshold);

        const uint8_t* data_start = (const uint8_t*)_filter_quantization->get_quantized_feature_packed(iters_start + k);

        if (FLAGS_enable_lut_clip) {
            push_mask = lookup_dist_table_16_clip(dist, data_start, pq_dist_table_int8, quantization_params.nsq, 
                multiplier, inv_mults, simd_thresholds, simd_biases, fea_offset, results, false);
        } else {
            push_mask = lookup_dist_table_16(dist, data_start, pq_dist_table_int8, quantization_params.nsq, 
                multiplier, inv_mults, simd_thresholds, simd_biases, fea_offset, results, false);
        }
        
        push_mask &= ((uint32_t)-1 << j_start);

        push_mask &= end_mask;
        if (!push_mask) {
            continue;
        }
        
        /*
        if (evaled_expr) {
            uint32_t filter_mask = evaled_expr->get32((iters_start + k) * 32);
            push_mask &= filter_mask;
        }
        */
        //LOG(INFO) << "asdasdasd";
        while (push_mask) {
            const int j = find_lsb_set_non_zero(push_mask);
           
            push_mask &= (push_mask - 1);

            
            if (dist[j] < result_heap2.epsilon()) {
                
                //LOG(INFO) << "(iters_start + k) * 32 + j: " << (iters_start + k) * 32 + j << " " << dist[j] << " " << cell_dist << " " << fea_offset[j] 
                  //      << " " << (int)data_start[0] << " " << (int)data_start[1]<< " " << (int)data_start[16] << " " << (int)data_start[15];
                /*
                auto vvv = _memory_to_local[(iters_start + k) * 32 + j];
                auto decoded_vec = _filter_quantization->decode_packed((iters_start + k) * 32 + j);
                float norm_sqr = cblas_sdot(_conf.feature_dim, decoded_vec.get(), 1, decoded_vec.get(), 1);
                float norm = std::sqrt(norm_sqr);
                float qQ = dist[j] - 2.0 * cell_dist - fea_offset[j];
                float cos = qQ / (-2 * norm);
                //if (cos > 0.2)
                  //  LOG(INFO) << "vvv: " << l << " " << token / FLAGS_fine_cluster_count << " " << token % FLAGS_fine_cluster_count << " " << vvv << " " << dist[j] << " cos:" << cos << " -2qQ':" << dist[j] - 2.0 * cell_dist - fea_offset[j] << " " << float_threshold;
                */
                /*
                auto vvv = _memory_to_local[(iters_start + k) * 32 + j];
                auto decoded_vec = _filter_quantization->decode_packed((iters_start + k) * 32 + j);
                float norm_sqr = cblas_sdot(_conf.feature_dim, decoded_vec.get(), 1, decoded_vec.get(), 1);
                float norm = std::sqrt(norm_sqr);
                float qQ = dist[j] - 2.0 * cell_dist - fea_offset[j];
                float cos = qQ / (-2 * norm);
                if (vvv == 7222801 || vvv == 3333809 || vvv == 4241032 || vvv == 5098953 || vvv == 5743962
                    || vvv == 1528822 || vvv == 4916321 || vvv == 732763 || vvv == 3121901 || vvv == 7900915
                    || vvv == 6857700 || vvv == 4682241 || vvv == 3012955 || vvv == 3282479 || vvv == 5299461
                    || vvv == 5361757 || vvv == 242793 || vvv == 937085 || vvv == 3815352 || vvv == 7685481
                    || vvv == 3283787 || vvv == 5258356 || vvv == 4604192 || vvv == 5032426 || vvv == 1626188
                    || vvv == 2186212 || vvv == 69163 || vvv == 913050 || vvv == 5815840 || vvv == 301046
                    || vvv == 4063413 || vvv == 5801797 || vvv == 8740021 || vvv == 8447873 || vvv == 3035188 || vvv == 3638855) {
                     LOG(INFO) << "vvv: " << l << " " << token / FLAGS_fine_cluster_count << " " << token % FLAGS_fine_cluster_count << " " << vvv << " dist: " << dist[j] << " 2*cell_dist: " << 2.0 * cell_dist << " cos:" << cos << " -2qQ':" << dist[j] - 2.0 * cell_dist - fea_offset[j] << " " << float_threshold << " " << result_heap2.epsilon();
                }
                */


                //LOG(INFO) << "dist: " << dist[j] << " " << dist[j] - 2.0 * cell_dist - fea_offset[j] << " " << float_threshold;
                //max_q = std::min(max_q, dist[j] - 2.0 * cell_dist - fea_offset[j]);
                // max_q += dist[j] - 2.0 * cell_dist - fea_offset[j];

                //if (_memory_to_local[(iters_start + k) * 32 + j] == 565655) {
                  //  LOG(INFO) << "565655: " << 2.0 * cell_dist << " " << fea_offset[j] << " " << fea_offset[j] << " " << fea_offset[j]; 
                //}

                bool needs_gc = result_heap2.max_heap_update(dist[j], 2.0 * cell_dist + fea_offset[j], (iters_start + k) * 32 + j);
                //LOG(INFO) << "dist: " << dist[j];
                if (needs_gc) {
                    if (_conf.whether_pq && _conf.use_lut16) {
                        //LOG(INFO) << "garbage_collect_aux";
                        result_heap2.garbage_collect_aux();
                        //LOG(INFO) << "garbage_collect_aux2";
                    } else {
                        result_heap2.garbage_collect();
                    }
                    float_threshold = result_heap2.epsilon() - (2.0 * cell_dist + fea_offset[j]);

                    int16_threshold = get_int16_threshold(float_threshold * multiplier);
                    simd_thresholds = _mm256_set1_epi16(int16_threshold);

                    __m256i masks[2];
                    masks[0] = _mm256_cmpgt_epi16(simd_thresholds, results[0]);
                    masks[1] = _mm256_cmpgt_epi16(simd_thresholds, results[1]);

                    push_mask &= get_comparison_mask(masks[0], masks[1]);
                }
                //result_heap.max_heap_update(dist[j], 0, (iters_start + k) * 32 + j);
                ++updated_cnt;
            }
        }
    }
    return updated_cnt;
}


//template <typename AvxFuncs>
float dense_dot_product_int8_float_avx_impl_naive(const int8_t* aptr,
                                                         const float* bptr,
                                                         size_t length) {
    float re = 0.0;
    for (size_t i = 0; i < length; i++) {
        re += aptr[i] * bptr[i];
    }
    return re;
}

static __m256 int8_to_float_lower(__m128i x) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(x));
}

static __m256 int8_to_float_upper(__m128i x) {
    return int8_to_float_lower(_mm_srli_si128(x, 8));
}

static __m256 multiply_add(__m256 mul1, __m256 mul2,
                                              __m256 add) {
    return _mm256_fmadd_ps(mul1, mul2, add);
}

static float sum8(__m256 x) {
    const __m128 upper = _mm256_extractf128_ps(x, 1);
    const __m128 lower = _mm256_castps256_ps128(x);
    __m128 sum = _mm_add_ps(upper, lower);
    sum = _mm_add_ps(
        sum, _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum), 8)));
    return sum[0] + sum[1];
}

// from OneToManyAsymmetricTemplate->ComputeOneToOneScore
float dense_dot_product_int8_float_avx_impl(const int8_t* aptr,
                                                         const float* bptr,
                                                         size_t length) {
    const int8_t* aend = aptr + length;

    auto as_m128i = [](const int8_t* x) -> __m128i* {
        return reinterpret_cast<__m128i*>(const_cast<int8_t*>(x));
    };

    __m256 accumulator0 = _mm256_setzero_ps();
    __m256 accumulator1 = _mm256_setzero_ps();
    for (; aptr + 16 <= aend; aptr += 16, bptr += 16) {
        __m128i avals = _mm_loadu_si128(as_m128i(aptr));
        __m256 avals0 = int8_to_float_lower(avals);
        __m256 bvals0 = _mm256_loadu_ps(bptr);
        accumulator0 = multiply_add(avals0, bvals0, accumulator0);

        __m256 avals1 = int8_to_float_upper(avals);
        __m256 bvals1 = _mm256_loadu_ps(bptr + 8);
        accumulator1 = multiply_add(avals1, bvals1, accumulator1);
    }

    if (aptr + 8 <= aend) {
        __m128i avals = _mm_loadl_epi64(as_m128i(aptr));
        __m256 avals0 = int8_to_float_lower(avals);
        __m256 bvals0 = _mm256_loadu_ps(bptr);
        accumulator0 = multiply_add(avals0, bvals0, accumulator0);
        aptr += 8;
        bptr += 8;
    }

    uint32_t t;
    memcpy(&t, aptr, sizeof t);
    if (aptr + 4 <= aend) {
        __m128i avals = _mm_cvtsi32_si128(t);
        __m128 avals0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(avals));
        __m128 bvals0 = _mm_loadu_ps(bptr);
        __m128 prod = _mm_mul_ps(avals0, bvals0);
        accumulator0 = _mm256_add_ps(
            accumulator0, _mm256_insertf128_ps(_mm256_setzero_ps(), prod, 0));
        aptr += 4;
        bptr += 4;
    }

    float scalar_accumulator =
        sum8(_mm256_add_ps(accumulator0, accumulator1));

    for (; aptr < aend; ++aptr, ++bptr) {
        scalar_accumulator += static_cast<float>(*aptr) * *bptr;
    }

    return static_cast<float>(scalar_accumulator);
}

inline static __m256 Int8ToFloatLower(__m128i x) {
 return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(x));
}

template <typename Int, typename DenomInt>
constexpr Int DivRoundUp3(Int num, DenomInt denom) {
  return (num + static_cast<Int>(denom) - static_cast<Int>(1)) /
         static_cast<Int>(denom);
}

template <typename Int, typename DenomInt>
constexpr Int NextMultipleOf3(Int num, DenomInt denom) {
  return DivRoundUp3(num, denom) * denom;
}

int findFirstLessThan(const std::vector<int>& arr, int val) {
    // 使用lower_bound找到第一个不小于val的元素的位置
    auto it = std::lower_bound(arr.begin(), arr.end(), val);

    // 如果it指向arr的开始位置，说明没有比val小的元素
    if (it == arr.begin()) {
        return 0;
    }

    // 如果it没有超过arr的末尾，则返回前一个元素的下标
    if (it != arr.end()) {
        --it; // 移动迭代器指向前一个元素
        return std::distance(arr.begin(), it);
    }

    // 如果it等于arr.end()，说明整个数组的元素都小于val
    return arr.size() - 1;
}

int PuckIndex::rank_topN_points(SearchContext* context, const float* feature, const uint32_t filter_topk,
                                PuckMaxHeap& result_heap) {
    auto& search_point_data = context->get_search_point_data();
    float* result_distance = search_point_data.result_distance;
    float* result_distance_temp = search_point_data.result_distance_temp;
    uint32_t* result_tag = search_point_data.result_tag;

    float query_norm = cblas_sdot(_conf.feature_dim, feature, 1, feature, 1);
    //堆顶
    float* true_result_distance = result_heap.get_top_addr();
    // LOG(WARNING) << "_conf.whether_pq: " << (_all_feature == nullptr);
    _conf.whether_pq = FLAGS_whether_pq;
    // _conf.whether_pq = false;

    if (_conf.whether_pq) {
        const Quantization* pq_quantization = _pq_quantization.get();
        float* pq_dist_table = context->get_search_point_data().pq_pq_dist_table;
        //float* pq_dist_table = context->get_search_point_data().pq_dist_table;
        float* filter_dist_table = context->get_search_point_data().pq_dist_table;

        auto& quantization_params = _filter_quantization->get_quantization_params();

        //if (FLAGS_use_pq_method_1) {
          //  pq_quantization->get_ip_dist_table(feature, pq_dist_table);
        //} else {
            pq_quantization->get_dist_table(feature, pq_dist_table);
        //}
        std::vector<uint8_t> pq_feature_vec(quantization_params.nsq);
      
        size_t ori_count = 0, bak_count = 0;

        if (_conf.use_lut16) {
            for (uint32_t idx = 0; idx < filter_topk; ++idx) {
                
                int point_id = _memory_to_local[result_tag[idx]];

                if (_conf.spilled_copy_num > 0) {
                    _filter_quantization->get_quantized_feature(_local_to_memory[point_id], pq_feature_vec.data());
                } else {
                    _filter_quantization->get_quantized_feature(result_tag[idx], pq_feature_vec.data());
                }
                const unsigned char* pq_feature = (unsigned char*)pq_feature_vec.data();
                
                //const unsigned char* feature = _filter_quantization->get_quantized_feature(result_tag[idx]);
                //const unsigned char* pq_feature = (unsigned char*)feature + _filter_quantization->get_fea_offset();
                
                auto dist_ori = lookup_dist_table2(pq_feature, filter_dist_table, quantization_params.ks, quantization_params.nsq);
                
                if (result_tag[idx] >= _conf.total_point_count) {
                    bak_count += 1;
                    // LOG(INFO) << "result_tag[idx]: " << result_tag[idx];
                } else {
                    ori_count += 1;
                }
                /*
                auto decoded_vec = _filter_quantization->decode_packed(result_tag[idx]);
                float tempd = -2 * cblas_sdot(_conf.feature_dim, feature, 1, decoded_vec.get(), 1);
                LOG(INFO) << "tempd: " << tempd << " " << dist_ori;
                */
                
                if (_conf.spilled_copy_num > 0) {
                    
                    auto cell_id = _cell_assign[_local_to_memory[point_id]];
                    
                    auto coarse_id = cell_id / _conf.fine_cluster_count;
                    auto fine_id = cell_id % _conf.fine_cluster_count;
                    SearchCellData& search_cell_data = context->get_search_cell_data();
        
                    float* cluster_inner_product = search_cell_data.cluster_inner_product;
                    float* fine_cluster_inner_product = search_cell_data.fine_cluster_inner_product;

                    FineCluster* cur_fine_cluster_list = _coarse_clusters[coarse_id].fine_cell_list;
                    float temp_dist = _coarse_norms[coarse_id] - cluster_inner_product[coarse_id] 
                            + cur_fine_cluster_list[fine_id].stationary_cell_dist
                            - fine_cluster_inner_product[fine_id] + query_norm / 2;

                    temp_dist = temp_dist * 2 + _filter_quantization->get_fea_offset(_local_to_memory[point_id]);

                    
                    result_distance_temp[idx] = temp_dist; // + query_norm;
                    
                }

                dist_ori += result_distance_temp[idx];
                result_distance[idx] = dist_ori;
                
            }
            // LOG(INFO) << "count: " << ori_count << " " << bak_count;
        }
        
        for (uint32_t idx = 0; idx < filter_topk; ++idx) {
            
            int point_id = _memory_to_local[result_tag[idx]];
            float dist_ori = result_distance[idx];
            //LOG(INFO) << idx << " point_id2: " << point_id << " " << dist_ori;
            const unsigned char* pq_feature = (unsigned char*)pq_quantization->get_quantized_feature(result_tag[idx]);
            if (_conf.spilled_copy_num > 0) {
                pq_feature = (unsigned char*)pq_quantization->get_quantized_feature(point_id);
            }
            float temp_dist_app = dist_ori - query_norm + ((float*)pq_feature)[0];
            
            // auto cell_id = _cell_assign[_local_to_memory[point_id]];
            
            //if (FLAGS_use_pq_method_1) {
              //  temp_dist_app = dist_ori;
            //}
            // float temp_dist_app = result_distance[idx] - query_norm + ((float*)pq_feature)[0];
            float temp_dist = temp_dist_app;
           
            pq_feature += pq_quantization->get_fea_offset();
            /*
            if (FLAGS_use_pq_method_1) {
                auto dist = lookup_dist_table(pq_feature, pq_dist_table, pq_quantization->get_quantization_params().ks,
                                           pq_quantization->get_quantization_params().nsq);
                
                // LOG(INFO) << "dist: " << temp_dist << " " << dist;
                temp_dist += dist;
            } else {
#ifdef __SSE__
                temp_dist += lookup_dist_table(pq_feature, pq_dist_table, pq_quantization->get_quantization_params().ks,
                                           pq_quantization->get_quantization_params().nsq);
#else

                for (uint32_t m = 0; m < (uint32_t)pq_quantization->get_quantization_params().nsq
                        && temp_dist < true_result_distance[0]; ++m) {
                    temp_dist += (pq_dist_table + m * pq_quantization->get_quantization_params().ks)[pq_feature[m]];
                }

#endif
            }
            */
            temp_dist += lookup_dist_table(pq_feature, pq_dist_table, pq_quantization->get_quantization_params().ks,
                                           pq_quantization->get_quantization_params().nsq);
            
            if (FLAGS_use_pq_method_0) {
                //temp_dist += query_norm;
            }

            if (temp_dist < true_result_distance[0]) {
                result_heap.max_heap_update(temp_dist, point_id);
            }
        }
    } else {
        //if (true) return 0;
        if (!FLAGS_use_int8) {
            size_t qty_16 = 16;
            _mm_prefetch((char*)(feature), _MM_HINT_T0);

            for (uint32_t idx = 0; idx < filter_topk; ++idx) {
                uint32_t m = 0;
                float PORTABLE_ALIGN32 TmpRes[8];
                const float* exhaustive_feature = _all_feature + (uint64_t)result_tag[idx] * _conf.feature_dim;
                
                if (_conf.spilled_copy_num > 0 || FLAGS_convert_local_to_memory_while_build) {
                    exhaustive_feature = _all_feature + (uint64_t)_memory_to_local[result_tag[idx]] * _conf.feature_dim;
                }
                float temp_dist = 0;

                for (; (m + qty_16) <= _conf.feature_dim && temp_dist < true_result_distance[0];) {
                    temp_dist += similarity::L2Sqr16Ext(exhaustive_feature + m, feature + m, qty_16, TmpRes);
                    m += qty_16;
                }
                
                if (temp_dist < true_result_distance[0]) {
                    size_t left_dim = _conf.feature_dim - m;

                    if (left_dim > 0) {
                        temp_dist += similarity::L2SqrExt(exhaustive_feature + m, feature + m, left_dim, TmpRes);
                    }

                
                    //auto cell_id = _memory_to_ori_cell_id[_memory_to_local[result_tag[idx]]];
                    //LOG(INFO) << "cell_id: " << cell_id / _conf.fine_cluster_count;
                 

                    result_heap.max_heap_update(temp_dist, _memory_to_local[result_tag[idx]]);
                }
            }
        } else {
            size_t array_size = 32;
            std::vector<float> float_array(array_size);
            float increment = 1.0f / (array_size - 1);
            // 填充数组
            for (size_t i = 0; i < array_size; ++i) {
                float_array[i] = static_cast<float>(i) * increment;
            }

            std::vector<float> preprocessed_query(_conf.feature_dim);
            for (size_t i = 0; i < _conf.feature_dim; i++) {
                preprocessed_query[i] = feature[i] / _int8_multipliers[i];
            }

            const size_t num_outer_iters = filter_topk / 3;

            //for (uint32_t idx = 0; idx < filter_topk; ++idx) {
            for (uint32_t idx = filter_topk / 3 * 3; idx < filter_topk; ++idx) {
                size_t point_id = _memory_to_local[result_tag[idx]];

                int8_t* int8_featute = _all_feature_int8 + point_id * _conf.feature_dim;
                // dense_dot_product_int8_float_avx_impl_naive
                float temp_dist = dense_dot_product_int8_float_avx_impl(
                        int8_featute, 
                        preprocessed_query.data(), 
                        _conf.feature_dim);
                temp_dist = query_norm - 2 * temp_dist + _dp_norms[point_id];
            
                if (temp_dist < true_result_distance[0]) {
                    result_heap.max_heap_update(temp_dist, point_id);
                }
            }

            // from OneToManyAsymmetricTemplate->ComputeOneToManyScores
            // num_outer_iters
            for (size_t i = 0; i < num_outer_iters; ++i) {
                size_t point_id = _memory_to_local[result_tag[i]];
                const int8_t* i0 = _all_feature_int8 + point_id * _conf.feature_dim;
                point_id = _memory_to_local[result_tag[i + num_outer_iters]];
                const int8_t* i1 = _all_feature_int8 + point_id * _conf.feature_dim;
                point_id = _memory_to_local[result_tag[i + 2 * num_outer_iters]];
                const int8_t* i2 = _all_feature_int8 + point_id * _conf.feature_dim;

                float result0;
                float result1;
                float result2;

                __m256 a0 = _mm256_setzero_ps();
                __m256 a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps();


                size_t dim = 0;

                for (; dim + 16 <= _conf.feature_dim; dim += 16) {

                    __m256 q = _mm256_loadu_ps(preprocessed_query.data() + dim);
                    __m256 q2 = _mm256_loadu_ps(preprocessed_query.data() + dim + 8);

                    __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(i0 + dim));
                    __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(i1 + dim));
                    __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(i2 + dim));

                        asm("" ::: "memory");


                    a0 = _mm256_fnmadd_ps(q, Int8ToFloatLower(v0), a0);
                    a1 = _mm256_fnmadd_ps(q, Int8ToFloatLower(v1), a1);
                    a2 = _mm256_fnmadd_ps(q, Int8ToFloatLower(v2), a2);
                    a0 = _mm256_fnmadd_ps(q2, Int8ToFloatLower(_mm_srli_si128(v0, 8)), a0);
                    a1 = _mm256_fnmadd_ps(q2, Int8ToFloatLower(_mm_srli_si128(v1, 8)), a1);
                    a2 = _mm256_fnmadd_ps(q2, Int8ToFloatLower(_mm_srli_si128(v2, 8)), a2);
                }
  
                //HorizontalSum3X(accums[0], accums[1], accums[2], &results[0], &results[1],
                  //      &results[2]);

                constexpr int kDestLoEqALo = 0x00;
                constexpr int kDestLoEqAHi = 0x01;
                constexpr int kDestHiEqBLo = 0x20;
                constexpr int kDestHiEqBHi = 0x30;
                auto term0 = _mm256_permute2f128_ps(a0, a2, kDestLoEqALo + kDestHiEqBLo);
                auto term1 = _mm256_permute2f128_ps(a0, a2, kDestLoEqAHi + kDestHiEqBHi);
                auto ac = _mm256_add_ps(term0, term1);
                auto bg = a1 + _mm256_permute2f128_ps(a1, a1, 1);
                auto term00 = _mm256_shuffle_ps(ac, bg, 0b11'10'01'00);
                auto term11 = _mm256_shuffle_ps(ac, bg, 0b01'00'11'10);
                auto abcg = _mm256_add_ps(term00, term11);
                abcg += _mm256_shuffle_ps(abcg, abcg, 0b11'11'01'01);
                

                for (size_t j = 0; j < 3; j++) {
                    size_t point_id = _memory_to_local[result_tag[i + j * num_outer_iters]];
                    auto temp_dist = query_norm + 2 * abcg[2 * j] + _dp_norms[point_id];
  
                    if (temp_dist < true_result_distance[0]) {
                        result_heap.max_heap_update(temp_dist, point_id);
                    }
                }
            }
        }
    }

    result_heap.reorder();
    return 0;
}

int PuckIndex::brute_force(SearchContext* context, const float* feature, const uint32_t filter_topk,
                                PuckMaxHeap& result_heap) {
   
    size_t qty_16 = 16;
    _mm_prefetch((char*)(feature), _MM_HINT_T0);
    float* true_result_distance = result_heap.get_top_addr();
    // " << _conf.total_point_count;
    for (uint32_t idx = 0; idx < _conf.total_point_count; ++idx) {
        uint32_t m = 0;
        float PORTABLE_ALIGN32 TmpRes[8];
        const float* exhaustive_feature = _all_feature + (uint64_t)idx * _conf.feature_dim;
        
        float temp_dist = 0;

        for (; (m + qty_16) <= _conf.feature_dim && temp_dist < true_result_distance[0];) {
            temp_dist += similarity::L2Sqr16Ext(exhaustive_feature + m, feature + m, qty_16, TmpRes);
            m += qty_16;
        }
        
        if (temp_dist < true_result_distance[0]) {
            size_t left_dim = _conf.feature_dim - m;

            if (left_dim > 0) {
                temp_dist += similarity::L2SqrExt(exhaustive_feature + m, feature + m, left_dim, TmpRes);
            }

            result_heap.max_heap_update(temp_dist, idx);
        }
    }
    
    result_heap.reorder();
    return 0;
}

int PuckIndex::pre_filter_search(SearchContext* context, const float* feature) {
    auto& search_point_data = context->get_search_point_data();
    //LOG(ERROR) << "pre_filter_search0";
    /*
#ifndef __SSE__
    LOG(ERROR) << "pre_filter_search1";
    uint32_t* query_sorted_tag = search_point_data.query_sorted_tag;
    float* query_sorted_dist = search_point_data.query_sorted_dist;
    //初始化最大堆。
    auto& params = _filter_quantization->get_quantization_params();
    PuckMaxHeap sorted_heap(params.nsq, query_sorted_dist, query_sorted_tag);

    for (uint32_t i = 0; i < (uint32_t)params.nsq; ++i) {
        float temp_val = 0;

        for (uint32_t m = 0; m < (uint32_t)params.lsq; ++m) {
            temp_val += -fabs(feature[i * params.lsq + m]);
        }

        sorted_heap.max_heap_update(temp_val, i);
    }

    sorted_heap.reorder();
#endif
    */
    //LOG(ERROR) << "pre_filter_search2";
    auto* cur_quantization = _filter_quantization.get();
    float* pq_dist_table = search_point_data.pq_dist_table;

    int ret = 0;
    if (_conf.use_lut16 && FLAGS_use_pq_method_0) {
        ret = cur_quantization->get_ip_dist_table(feature, pq_dist_table);
    } else {
        ret = cur_quantization->get_dist_table(feature, pq_dist_table);
    }

    if (ret != 0) {
        LOG(ERROR) << "get raw dist table failed";
        return -1;
    }
    
    if (cur_quantization->get_quantization_params().ks == 16) {
        uint8_t* pq_dist_table_int8 = search_point_data.pq_dist_table_int8;
        auto ret = cur_quantization->get_dist_table_int8(pq_dist_table, pq_dist_table_int8, &search_point_data.fixed_point_multiplier);
        return ret;
    }
    return 0;
}

float cosine2(const float* a, const float* b, int dim) {
    // a * b / |a||b|
    float a_norms_sq = cblas_sdot(dim, a, 1, a, 1);
    float b_norms_sq = cblas_sdot(dim, b, 1, b, 1);
    float ab = cblas_sdot(dim, a, 1, b, 1);
    return ab / (std::sqrt(a_norms_sq) * std::sqrt(b_norms_sq));
}

int PuckIndex::search_nearest_filter_points(SearchContext* context, const float* feature) {
    if (pre_filter_search(context, feature) != 0) {
        LOG(ERROR) << "cmp filter dist table failed" ;
        return -1;
    }

    SearchCellData& search_cell_data = context->get_search_cell_data();
    float* cluster_inner_product = search_cell_data.cluster_inner_product;
    if (FLAGS_bak_cell_assign) {
        cluster_inner_product = search_cell_data.fine_cluster_inner_product;
    }
    matrix_multiplication(_fine_vocab, feature, _conf.fine_cluster_count, 1, _conf.feature_dim,
                          "TN", cluster_inner_product);

    PuckMaxHeap max_heap(_conf.fine_cluster_count, search_cell_data.fine_distance,
                     search_cell_data.fine_tag);
    for (uint32_t k = 0; k < _conf.fine_cluster_count; ++k) {
        max_heap.max_heap_update(-cluster_inner_product[k], k);
    }

    max_heap.reorder();

    //一级聚类中心的排序结果
    float* coarse_distance = search_cell_data.coarse_distance;
    uint32_t* coarse_tag = search_cell_data.coarse_tag;

    //堆结构
    float* result_distance = context->get_search_point_data().result_distance;
    float* result_distance_temp = context->get_search_point_data().result_distance_temp;
    uint32_t* result_tag = context->get_search_point_data().result_tag;
    float multiplier = context->get_search_point_data().fixed_point_multiplier;
    size_t filter_heap_size = std::max(_conf.filter_topk, context->get_request()->topk);
    
    uint32_t* masks = context->get_search_point_data().masks;
    PuckMaxHeap filter_heap(filter_heap_size, result_distance, result_distance_temp, result_tag);
    FastMaxHeap max_filter_heap(filter_heap_size, result_distance, result_distance_temp, result_tag, masks);

    float query_norm = cblas_sdot(_conf.feature_dim, feature, 1, feature, 1);
    //query_norm = 0.0;
    //过滤阈值
    float pivot = (filter_heap.get_top_addr()[0] - query_norm) / _conf.radius_rate / 2.0;
    size_t n = 0;

    float* filter_dist_table = context->get_search_point_data().pq_dist_table;

    n = 0;
    int cc = 0;
    int fc = 0;
    int fc90 = 0;
    bool ccflag = true;
    int totalf = 0;
    int total_updated_cnt = 0;
    int total_updated_cnt_last = 0;

    base::Timer tm_cost;
    size_t total_cost = 0;
    size_t total_count = 0;
    double max_q = 0;
    std::vector<float> dist(32, 0.0);
    size_t window_size = _conf.window_size;
    size_t sum = 0;

    size_t non_empty_cluster_size = 0;
    if (_conf.search_fine_count > 0) {
        tm_cost.start();
        float* temp_distance = search_cell_data.temp_distance;
        uint32_t* temp_tag = search_cell_data.temp_tag;
        PuckMaxHeap temp_heap(_conf.search_fine_count, temp_distance, temp_tag);
        size_t total_points = 0;
        for (uint32_t l = 0; l < _conf.search_coarse_count; ++l) {
            int coarse_id = coarse_tag[l];
            FineCluster* cur_fine_cluster_list = _coarse_clusters[coarse_id].fine_cell_list;
            
            for (uint32_t idx = 0; idx < _conf.fine_cluster_count; ++idx) {
                uint32_t k = search_cell_data.fine_tag[idx];
                
                if (cur_fine_cluster_list[k].get_point_cnt() == 0) {
                    continue;
                }

                float temp_dist = coarse_distance[l] + cur_fine_cluster_list[k].stationary_cell_dist +
                                search_cell_data.fine_distance[idx]; // + cur_fine_cluster_list[k].min_offset / 2; 

                temp_heap.max_heap_update(temp_dist, coarse_id * _conf.fine_cluster_count + k);
                non_empty_cluster_size += 1;
            }
        }

        temp_heap.reorder();
        tm_cost.stop();
        context->total_cost4 += tm_cost.n_elapsed();

        std::unordered_set<int> coarse_id_set;
        non_empty_cluster_size = std::min(non_empty_cluster_size, (size_t)_conf.search_fine_count);
        for (uint32_t l = 0; l < non_empty_cluster_size; ++l) {
            
            int gid = temp_tag[l];
            int coarse_id = gid / _conf.fine_cluster_count;
            coarse_id_set.insert(coarse_id);
            int k = gid % _conf.fine_cluster_count;
            FineCluster* cur_fine_cluster_list = _coarse_clusters[coarse_id].fine_cell_list;
            float temp_dist = temp_distance[l]; // - cur_fine_cluster_list[k].min_offset / 2;
            if (_conf.use_lut16 && FLAGS_use_pq_method_0) {
                temp_dist += query_norm / 2;
            }
            
            int updated_cnt = 0;
            if (!_conf.use_lut16) {
                updated_cnt = compute_quantized_distance(context, cur_fine_cluster_list + k, temp_dist, filter_heap, totalf, total_cost);
            } else {
                float thr = 0.9; // + std::min(0.1, l * 1.0 / 40000);
                if (FLAGS_packed_cluster) {
                    updated_cnt = compute_quantized_distance_lut16_fast_heap(context, cur_fine_cluster_list + k, temp_dist, filter_heap, max_filter_heap, totalf, multiplier, total_cost, gid, dist, thr);
                } else {
                    updated_cnt = compute_quantized_distance_lut16_fast_heap2(context, cur_fine_cluster_list + k, temp_dist, filter_heap, max_filter_heap, totalf, multiplier, total_cost, gid, l, dist, thr, max_q, total_updated_cnt);
                }
            }

            auto heap_size = max_filter_heap.get_heap_size();
            //LOG(INFO) << l << " heap_size: " << heap_size << " " << updated_cnt;
            temp_tag[l] = updated_cnt;
            sum += updated_cnt;
            if (l >= window_size) {
                sum -= temp_tag[l - window_size];
                if (sum == 0) {
                    break;
                }
            }
            total_updated_cnt += updated_cnt;
            total_count += cur_fine_cluster_list[k].get_point_cnt();
          
            if (FLAGS_search_point_count > 0 && total_count >= FLAGS_search_point_count) {
                context->total_cost6 += l + 1;
                break;
            }
            n += cur_fine_cluster_list[k].get_point_cnt();
        }
    } else {
        
        for (uint32_t l = 0; l < _conf.search_coarse_count; ++l) {
            size_t fidx = 0;
            ccflag = true;
            int coarse_id = coarse_tag[l];
            //计算query与当前一级聚类中心下cell的距离
            FineCluster* cur_fine_cluster_list = _coarse_clusters[coarse_id].fine_cell_list;
            float min_dist = _coarse_clusters[coarse_id].min_dist_offset + coarse_distance[l];
            float max_stationary_dist = pivot
                        - coarse_distance[l] - search_cell_data.fine_distance[0];
            
            for (uint32_t idx = 0; idx < _conf.fine_cluster_count; ++idx) {
                uint32_t k = search_cell_data.fine_tag[idx];
   
                if (search_cell_data.fine_distance[idx] + min_dist >= pivot) {
                    break;
                }
                
                bool cflag = false;
  
                
                if (cur_fine_cluster_list[k].stationary_cell_dist >= max_stationary_dist) {
                    
                    continue;
                }
                total_count += cur_fine_cluster_list[k].get_point_cnt();

                fc += 1;
                if (l >= 90)
                    fc90 += 1;
                if (ccflag) {
                    cc += 1;
                    ccflag = false;
                }
                n += cur_fine_cluster_list[k].get_point_cnt();
                float temp_dist = coarse_distance[l] + cur_fine_cluster_list[k].stationary_cell_dist +
                                search_cell_data.fine_distance[idx];
                
                int updated_cnt = 0;
            

                if (!_conf.use_lut16) {
                    updated_cnt = compute_quantized_distance(context, cur_fine_cluster_list + k, temp_dist, filter_heap, totalf, total_cost);
                } else {
                    updated_cnt = compute_quantized_distance_lut16_fast_heap2(context, cur_fine_cluster_list + k, temp_dist, filter_heap, max_filter_heap, totalf, multiplier, total_cost, 0, l, dist, 0.2, max_q, total_updated_cnt);
                }
                LOG(INFO) << "updated_cnt: " << l << " " << idx << " " << updated_cnt; 
                if (updated_cnt > 0) {
                    pivot = (filter_heap.get_top_addr()[0] - query_norm) / _conf.radius_rate / 2.0;
                }
                
                total_updated_cnt += updated_cnt;

                fidx = idx;
                max_stationary_dist = std::min(max_stationary_dist,
                                            pivot - coarse_distance[l] - search_cell_data.fine_distance[idx]);
                                            
            }
        }
    }
    
    context->total_cost5 += totalf;
    
    context->total_cost3 += total_count;
    
    size_t heap_size = 0;
    // LOG(INFO) << "top dis: " << total_updated_cnt;
    if (!_conf.use_lut16) {
        //LOG(INFO) << "top dis: " << result_distance[0];
        filter_heap.reorder();
        heap_size = filter_heap.get_heap_size();
        
    } else {
        //LOG(INFO) << "top dis: " << max_filter_heap.epsilon();
        if (_conf.whether_pq && _conf.use_lut16) {
            //LOG(INFO) << "reorder_aux 0";
            max_filter_heap.reorder_aux();
            //LOG(INFO) << "reorder_aux 1";
        } else {
            max_filter_heap.reorder();
        }
        heap_size = max_filter_heap.get_heap_size();
       
    }

    return heap_size;
}

int PuckIndex::search(const Request* request, Response* response) {
    if (request->feature == nullptr) {
        LOG(ERROR) << "topk = " << _conf.topk << ", or feature is nullptr";
        return -1;
    }
    base::Timer tm_cost;
    tm_cost.start();
    DataHandler<SearchContext> context(_context_pool);

    auto conf = _conf;
    conf.filter_topk = std::max(_conf.filter_topk, request->topk);
    
    if (0 != context->reset(conf)) {
        LOG(ERROR) << "init search context has error.";
        return -1;
    }
    context->set_request(request);
    const float* feature = normalization(context.get(), request->feature);
    if (_conf.brutal_search || (_conf.freshness_test && request->enable_brutal)) {
        PuckMaxHeap result_heap(request->topk, response->distance, response->local_idx);
        if (int ret = brute_force(context.get(), feature, 0, result_heap); ret != 0) {
            LOG(ERROR) << "rank points after filter has error.";
            return ret;
        }
        response->result_num = result_heap.get_heap_size();
        return 0;
    } else {
        //输出query与一级聚类中心的top-search-cell个ID和距离
        int ret = search_nearest_coarse_cluster(context.get(), feature,
                                                _conf.search_coarse_count);//, coarse_distance, coarse_tag);

        if (ret != 0) {
            LOG(ERROR) << "search nearest coarse cluster error " << ret;
            return ret;
        }
        //计算query与二级聚类中心的距离，并根据filter特征，筛选子集

        tm_cost.stop();
        context.get()->total_cost4 += tm_cost.n_elapsed();

        tm_cost.start();
        int search_point_cnt = search_nearest_filter_points(context.get(), feature);
        if (search_point_cnt < 0) {
            LOG(ERROR) << "search filter points has error.";
            return -1;
        }

        tm_cost.stop();
        context.get()->total_cost += tm_cost.n_elapsed();

        tm_cost.start();
        if (_conf.spilled_copy_num > 0) {
            std::vector<float> distance(request->topk * 2);
            std::vector<uint32_t> local_idx(request->topk * 2);
            PuckMaxHeap result_heap(request->topk * 2, distance.data(), local_idx.data());
            ret = rank_topN_points(context.get(), feature, search_point_cnt, result_heap);
            if (ret == 0) {
                std::unordered_set<uint32_t> id_set;

                for (size_t i = 0; i < result_heap.get_heap_size(); i++) {
                    if (id_set.find(local_idx[i]) != id_set.end()) {
                        continue;
                    }
                    response->distance[id_set.size()] = distance[i];
                    response->local_idx[id_set.size()] = local_idx[i];
                    id_set.insert(local_idx[i]);
                    if (id_set.size() >= request->topk) {
                        break;
                    }
                }
                response->result_num = id_set.size();
            } else {
                LOG(ERROR) << "rank points after filter has error.";
            }
        } else {
            PuckMaxHeap result_heap(request->topk, response->distance, response->local_idx);
            ret = rank_topN_points(context.get(), feature, search_point_cnt, result_heap);
            if (ret == 0) {
                response->result_num = result_heap.get_heap_size();
            } else {
                LOG(ERROR) << "rank points after filter has error.";
            }
        }

        tm_cost.stop();
        context.get()->total_cost2 += tm_cost.n_elapsed();
        context->total_n += 1;

        if (true && (context->total_n % 1000 == 0 || context->total_n == 1))
            LOG(INFO) << "total_costt: " << context->total_n 
                << " " << context.get()->total_cost / context->total_n / 1000 
                << " " << context.get()->total_cost2 / context->total_n / 1000 
                << " " << context.get()->total_cost4 / context->total_n / 1000 
                << " " << context.get()->total_cost3 / context->total_n
                << " " << context.get()->total_cost5 / context->total_n
                << " " << context.get()->total_cost6 / context->total_n
                << " " << _conf.search_coarse_count << " " << _conf.search_fine_count
                << " " << _conf.filter_topk << " " << _conf.window_size << " " << _conf.spilled_copy_num;
        //LOG(INFO) << "total_cost: " << context->total_n << " " << context.get()->total_cost / context->total_n << " " << context.get()->total_cost2 / context->total_n / 1000;
        return ret;
    }
}

int PuckIndex::puck_assign(const ThreadParams& thread_params, uint32_t* cell_assign, uint32_t offset) const {
    base::Timer tm_cost;
    tm_cost.start();
    std::unique_ptr<float[]>  chunk_points(new float[FLAGS_thread_chunk_size * _conf.feature_dim]);
    std::unique_ptr<int[]>  pq_assign(new int[FLAGS_thread_chunk_size]);
    std::unique_ptr<float[]>  pq_distance(new float[FLAGS_thread_chunk_size]);
    int coarse_assign  = -1;
    int fine_assign = -1;

    std::vector<Quantization*> quantizations;

    if (_conf.whether_filter) {
        quantizations.push_back(_filter_quantization.get());
    }

    if (_conf.whether_pq && offset == 0) {
        quantizations.push_back(_pq_quantization.get());
    }

    for (uint32_t cid = 0; cid < thread_params.chunks_count; ++cid) {
        uint32_t real_thread_chunk_size = std::min(FLAGS_thread_chunk_size,
                                          (int)(thread_params.points_count - cid * FLAGS_thread_chunk_size));
        int read_chunk_size = read_fvec_format(thread_params.learn_stream, _conf.feature_dim, real_thread_chunk_size,
                                               chunk_points.get());

        if (read_chunk_size != (int)real_thread_chunk_size) {
            LOG(ERROR) << "puck assign from " << thread_params.start_id << " read file error at cid = " << cid << " / " <<
                       thread_params.chunks_count << ", ret = " << read_chunk_size;
            throw "read_fvec_format error!";
        }

        int cur_start_point_id = thread_params.start_id + cid * FLAGS_thread_chunk_size;

        //计算point的召回特征与所属于的cell的残差向量
        for (uint32_t point_id = 0; point_id < real_thread_chunk_size; ++point_id) {
            int true_point_id = cur_start_point_id + point_id;
            int cell_id = cell_assign[true_point_id];
            coarse_assign = cell_id / _conf.fine_cluster_count;
            fine_assign = cell_id % _conf.fine_cluster_count;
            float* cur_residual = chunk_points.get() + point_id * _conf.feature_dim;
            cblas_saxpy(_conf.feature_dim, -1.0,
                        _coarse_vocab + coarse_assign * _conf.feature_dim, 1,
                        cur_residual, 1);
            cblas_saxpy(_conf.feature_dim, -1.0,
                        _fine_vocab + fine_assign * _conf.feature_dim, 1,
                        cur_residual, 1);
        }

        for (auto quantized : quantizations) {
            auto& cur_params = quantized->get_quantization_params();
            //std::unique_ptr<float[]> sub_residual(new float[FLAGS_thread_chunk_size * cur_params.lsq]);
            std::unique_ptr<float[]> pq_distance_table(new float[cur_params.nsq * cur_params.ks]);
            for (uint32_t i = 0; i < real_thread_chunk_size; i++) {
                float* cur_point_fea = chunk_points.get() + i * _conf.feature_dim;
                quantized->get_dist_table(cur_point_fea, pq_distance_table.get());

                u_int64_t true_point_id = cur_start_point_id + i;
                
                auto* quantized_fea = quantized->get_quantized_feature(true_point_id + offset);
                quantized_fea += quantized->get_fea_offset();
                
                for (uint32_t n = 0; n < (uint32_t)cur_params.nsq; ++n) {
                    float min_distance = std::sqrt(std::numeric_limits<float>::max());
                    const float* sub_dist_table = pq_distance_table.get() + n * cur_params.ks;

                    for (uint32_t k = 0; k < (uint32_t)cur_params.ks; ++k) {
                        if (sub_dist_table[k] < min_distance){
                            min_distance = sub_dist_table[k];
                            quantized_fea[n] = (unsigned char)k;
                        }
                    }
                }
            }
            
            for (uint32_t k = 0; k < (uint32_t)cur_params.nsq; ++k) {
                uint32_t cur_lsq = std::min(cur_params.lsq, cur_params.dim - k * cur_params.lsq);
                float* cur_pq_centroids = quantized->get_sub_codebooks(k);

                for (uint32_t i = 0; i < real_thread_chunk_size; i++) {
                    u_int64_t true_point_id = cur_start_point_id + i;
                    //point i 在第k子空间对应的聚类中心id
                    auto* quantized_fea = quantized->get_quantized_feature(true_point_id + offset);
                    quantized_fea += quantized->get_fea_offset();
                    int cur_assign = (int)quantized_fea[k];
                    float* cur_point_fea = chunk_points.get() + i * _conf.feature_dim;
                    //量化使用残差
                    cblas_saxpy(cur_lsq, -1.0,
                                cur_pq_centroids + (u_int64_t)cur_assign * cur_params.lsq, 1,
                                cur_point_fea + k * cur_params.lsq, 1);
                }
            }
        }

        //计算point在当前量化特征下的offset value
        for (uint32_t point_id = 0; point_id < real_thread_chunk_size; ++point_id) {
            int true_point_id = cur_start_point_id + point_id;
            int cell_id = cell_assign[true_point_id];
            coarse_assign = cell_id / _conf.fine_cluster_count;
            fine_assign = cell_id % _conf.fine_cluster_count;
            std::unique_ptr<float[]> cur_residual(new float[_conf.feature_dim]);
            memcpy(cur_residual.get(), _coarse_vocab + coarse_assign * _conf.feature_dim,
                   _conf.feature_dim * sizeof(float));
            cblas_saxpy(_conf.feature_dim, 1.0,
                        _fine_vocab + fine_assign * _conf.feature_dim, 1,
                        cur_residual.get(), 1);

            for (auto quantized : quantizations) {
                quantized->set_static_value_of_formula(true_point_id + offset, cur_residual.get());
            }
        }
    }

    tm_cost.stop();
    LOG(INFO) << "puck assign from " << thread_params.start_id << " cost " << tm_cost.m_elapsed() << " ms.";
    return 0;
}

int PuckIndex::puck_assign_bak(const ThreadParams& thread_params, uint32_t* cell_assign, uint32_t offset) const {
    base::Timer tm_cost;
    tm_cost.start();
    std::unique_ptr<float[]>  chunk_points(new float[FLAGS_thread_chunk_size * _conf.feature_dim]);
    std::unique_ptr<int[]>  pq_assign(new int[FLAGS_thread_chunk_size]);
    std::unique_ptr<float[]>  pq_distance(new float[FLAGS_thread_chunk_size]);
    int coarse_assign  = -1;
    int fine_assign = -1;

    std::vector<Quantization*> quantizations;

    if (_conf.whether_filter) {
        quantizations.push_back(_filter_quantization.get());
    }

    for (uint32_t cid = 0; cid < thread_params.chunks_count; ++cid) {
        uint32_t real_thread_chunk_size = std::min(FLAGS_thread_chunk_size,
                                          (int)(thread_params.points_count - cid * FLAGS_thread_chunk_size));
        int read_chunk_size = read_fvec_format(thread_params.learn_stream, _conf.feature_dim, real_thread_chunk_size,
                                               chunk_points.get());

        if (read_chunk_size != (int)real_thread_chunk_size) {
            LOG(ERROR) << "puck assign from " << thread_params.start_id << " read file error at cid = " << cid << " / " <<
                       thread_params.chunks_count << ", ret = " << read_chunk_size;
            throw "read_fvec_format error!";
        }

        int cur_start_point_id = thread_params.start_id + cid * FLAGS_thread_chunk_size;

        //计算point的召回特征与所属于的cell的残差向量
        for (uint32_t point_id = 0; point_id < real_thread_chunk_size; ++point_id) {
            int true_point_id = cur_start_point_id + point_id;
            int cell_id = cell_assign[true_point_id];
            coarse_assign = cell_id / _conf.fine_cluster_count;
            fine_assign = cell_id % _conf.fine_cluster_count;
            float* cur_residual = chunk_points.get() + point_id * _conf.feature_dim;
            cblas_saxpy(_conf.feature_dim, -1.0,
                        _coarse_vocab + coarse_assign * _conf.feature_dim, 1,
                        cur_residual, 1);
            cblas_saxpy(_conf.feature_dim, -1.0,
                        _fine_vocab + fine_assign * _conf.feature_dim, 1,
                        cur_residual, 1);
        }

        for (auto quantized : quantizations) {
            auto& cur_params = quantized->get_quantization_params();
            //std::unique_ptr<float[]> sub_residual(new float[FLAGS_thread_chunk_size * cur_params.lsq]);
            std::unique_ptr<float[]> pq_distance_table(new float[cur_params.nsq * cur_params.ks]);
            for (uint32_t i = 0; i < real_thread_chunk_size; i++) {
                float* cur_point_fea = chunk_points.get() + i * _conf.feature_dim;
                quantized->get_dist_table(cur_point_fea, pq_distance_table.get());

                u_int64_t true_point_id = cur_start_point_id + i;
                
                auto* quantized_fea = quantized->get_quantized_feature(true_point_id + offset);
                quantized_fea += quantized->get_fea_offset();
                
                for (uint32_t n = 0; n < (uint32_t)cur_params.nsq; ++n) {
                    float min_distance = std::sqrt(std::numeric_limits<float>::max());
                    const float* sub_dist_table = pq_distance_table.get() + n * cur_params.ks;

                    for (uint32_t k = 0; k < (uint32_t)cur_params.ks; ++k) {
                        if (sub_dist_table[k] < min_distance){
                            min_distance = sub_dist_table[k];
                            quantized_fea[n] = (unsigned char)k;
                        }
                    }
                }
            }
            
            for (uint32_t k = 0; k < (uint32_t)cur_params.nsq; ++k) {
                uint32_t cur_lsq = std::min(cur_params.lsq, cur_params.dim - k * cur_params.lsq);
                float* cur_pq_centroids = quantized->get_sub_codebooks(k);

                for (uint32_t i = 0; i < real_thread_chunk_size; i++) {
                    u_int64_t true_point_id = cur_start_point_id + i;
                    //point i 在第k子空间对应的聚类中心id
                    auto* quantized_fea = quantized->get_quantized_feature(true_point_id + offset);
                    quantized_fea += quantized->get_fea_offset();
                    int cur_assign = (int)quantized_fea[k];
                    float* cur_point_fea = chunk_points.get() + i * _conf.feature_dim;
                    //量化使用残差
                    cblas_saxpy(cur_lsq, -1.0,
                                cur_pq_centroids + (u_int64_t)cur_assign * cur_params.lsq, 1,
                                cur_point_fea + k * cur_params.lsq, 1);
                }
            }
        }

        //计算point在当前量化特征下的offset value
        for (uint32_t point_id = 0; point_id < real_thread_chunk_size; ++point_id) {
            int true_point_id = cur_start_point_id + point_id;
            int cell_id = cell_assign[true_point_id];
            coarse_assign = cell_id / _conf.fine_cluster_count;
            fine_assign = cell_id % _conf.fine_cluster_count;
            std::unique_ptr<float[]> cur_residual(new float[_conf.feature_dim]);
            memcpy(cur_residual.get(), _coarse_vocab + coarse_assign * _conf.feature_dim,
                   _conf.feature_dim * sizeof(float));
            cblas_saxpy(_conf.feature_dim, 1.0,
                        _fine_vocab + fine_assign * _conf.feature_dim, 1,
                        cur_residual.get(), 1);

            for (auto quantized : quantizations) {
                quantized->set_static_value_of_formula(true_point_id + offset, cur_residual.get());
            }
        }
    }

    tm_cost.stop();
    LOG(INFO) << "puck assign from " << thread_params.start_id << " cost " << tm_cost.m_elapsed() << " ms.";
    return 0;
}

void PuckIndex::batch_assign(const uint32_t total_cnt, const std::string& feature_file_name,
                             uint32_t* cell_assign, bool bak_assign) {
    this->HierarchicalClusterIndex::batch_assign(total_cnt, feature_file_name, cell_assign, bak_assign);

    std::vector<std::thread> threads;
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;
    //申请量化特征的内存
    auto offset = total_cnt;
    if (!bak_assign) {
        offset = 0;
        _filter_quantization->init_quantized_feature_memory();

        if (_conf.whether_pq) {
            _pq_quantization->init_quantized_feature_memory();
        }
    }

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

                        if (thread_params.open_file(feature_file_name.c_str(), _conf.feature_dim) != 0) {
                            throw "open file has error.";
                        }

                        LOG(INFO) << "puck_assign, thread_params.start_id = " << thread_params.start_id << " points_count = " <<
                                  thread_params.points_count << " feature_file_name = " << feature_file_name << " threadId = " << threadId;
                        puck_assign(thread_params, cell_assign, offset);
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

    LOG(INFO) << "PuckIndex batch_assign Suc.";
}

void PuckIndex::batch_assign_bak(const uint32_t total_cnt, const std::string& feature_file_name,
                             uint32_t* cell_assign) {
    this->HierarchicalClusterIndex::batch_assign_bak(total_cnt, feature_file_name, cell_assign);

    this->HierarchicalClusterIndex::save_index_bak();
    
    std::vector<std::thread> threads;
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;


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

                        if (thread_params.open_file(feature_file_name.c_str(), _conf.feature_dim) != 0) {
                            throw "open file has error.";
                        }

                        LOG(INFO) << "puck_assign, thread_params.start_id = " << thread_params.start_id << " points_count = " <<
                                  thread_params.points_count << " feature_file_name = " << feature_file_name << " threadId = " << threadId;
                        puck_assign_bak(thread_params, cell_assign, total_cnt);
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

    LOG(INFO) << "PuckIndex batch_assign Suc.";
}

int PuckIndex::train() {
    if (_conf.adaptive_train_param() != 0) {
        LOG(ERROR) << "PuckIndex adaptive train param has error.";
        return -1;
    }

    if (this->HierarchicalClusterIndex::train() != 0) {
        LOG(ERROR) << "HierarchicalClusterIndex train has error.";
        return -1;
    }

    if (this->HierarchicalClusterIndex::read_codebooks() != 0) {
        LOG(ERROR) << "HierarchicalClusterIndex read_codebooks has error.";
        return -1;
    }

    std::string train_pq_file_name = FLAGS_train_pq_file_name;

    if (FLAGS_train_points_count < FLAGS_pq_train_points_count) {
        gflags::SetCommandLineOption("pq_train_points_count", std::to_string(FLAGS_train_points_count).c_str());
    }

    u_int64_t train_vocab_len = (u_int64_t)FLAGS_pq_train_points_count * _conf.feature_dim;
    std::unique_ptr<float[]> kmeans_train_vocab(new float[train_vocab_len]);
    LOG(INFO) << "train_fea_file_name: " << FLAGS_train_fea_file_name;
    uint32_t pq_train_points_count = random_sampling(FLAGS_train_fea_file_name, FLAGS_train_points_count,
                                     FLAGS_pq_train_points_count, _conf.feature_dim, kmeans_train_vocab.get());

    if (pq_train_points_count <= 0) {
        LOG(ERROR) << "sampling data has error.";
        return -1;
    }

    LOG(INFO) << "true point cnt for puck train = " << pq_train_points_count;
    //写文件，训练使用这批抽样数据
    int ret = write_fvec_format(train_pq_file_name.c_str(), _conf.feature_dim, pq_train_points_count,
                                kmeans_train_vocab.get());

    if (ret != 0) {
        LOG(ERROR) << "write sampling data has error.";
        return -1;
    }

    std::unique_ptr<uint32_t[]> cell_assign(new uint32_t[pq_train_points_count]);
    this->HierarchicalClusterIndex::batch_assign(pq_train_points_count, train_pq_file_name, cell_assign.get());

    //算残差
    for (uint32_t i = 0; i < pq_train_points_count; ++i) {
        float* residual = kmeans_train_vocab.get() + i * _conf.feature_dim;
        int coarse_assign = cell_assign[i] / _conf.fine_cluster_count;
        int fine_assign = cell_assign[i] % _conf.fine_cluster_count;
        cblas_saxpy(_conf.feature_dim, -1.0,
                    _coarse_vocab + coarse_assign * _conf.feature_dim, 1,
                    residual, 1);
        cblas_saxpy(_conf.feature_dim, -1.0,
                    _fine_vocab + fine_assign * _conf.feature_dim, 1,
                    residual, 1);
    }

    std::vector<Quantization*> quantizations;
    auto& cur_params = _filter_quantization->get_quantization_params();
    LOG(INFO) << cur_params.nsq << " " << cur_params.lsq << " " << cur_params.ks;
    _filter_quantization->init_codebooks_memory();
    quantizations.push_back(_filter_quantization.get());

    if (_conf.whether_pq) {
        auto& cur_params = _pq_quantization->get_quantization_params();
        LOG(INFO) << cur_params.nsq << " " << cur_params.lsq << " " << cur_params.ks;
        _pq_quantization->init_codebooks_memory();
        quantizations.push_back(_pq_quantization.get());
    }

    std::unique_ptr<int[]> pq_assign(new int[pq_train_points_count]);
    int idx = 0;

    for (auto quantized : quantizations) {
        auto& cur_params = quantized->get_quantization_params();
        LOG(INFO) << cur_params.nsq << " " << cur_params.lsq;
        std::unique_ptr<float[]> sub_resudial(new float[pq_train_points_count * cur_params.lsq]);

        for (uint32_t k = 0; k < (uint32_t)cur_params.nsq; k++) {
            uint32_t cur_lsq = std::min(cur_params.lsq, cur_params.dim - k * cur_params.lsq);
            memset(sub_resudial.get(), 0, sizeof(float) * pq_train_points_count * cur_params.lsq);

            for (uint32_t i = 0; i < pq_train_points_count; ++i) {
                const float* cur_train_point = kmeans_train_vocab.get() + i * _conf.feature_dim;
                memcpy(sub_resudial.get() + i * cur_params.lsq, cur_train_point + k * cur_params.lsq,
                       sizeof(float) * cur_lsq);
            }

            float* cur_pq_centroids = quantized->get_sub_codebooks(k);

            memset(pq_assign.get(), 0, sizeof(int) * pq_train_points_count);
            Kmeans kmeans_cluster(true, (int)(_conf.threads_count));
            LOG(INFO) << "kmeans params: threads count = " << kmeans_cluster.get_params().nt;

            float err = kmeans_cluster.kmeans(cur_params.lsq, pq_train_points_count, cur_params.ks,
                                              sub_resudial.get(),
                                              cur_pq_centroids, nullptr, pq_assign.get());
            LOG(INFO) << "deviation error of init sub " << k << " pq codebook clusters is " << err;

            for (uint32_t i = 0; i < pq_train_points_count; ++i) {
                float* cur_train_point = kmeans_train_vocab.get() + i * _conf.feature_dim;
                int cur_assign = pq_assign.get()[i];
                cblas_saxpy(cur_lsq, -1.0,
                            cur_pq_centroids + (u_int64_t)cur_assign * cur_params.lsq, 1,
                            cur_train_point + k * cur_params.lsq, 1);
            }

            //if (k > 10) exit(0);
        }

        LOG(INFO) << idx << " suc.";
    }

    return save_codebooks();
}

int PuckIndex::init_single_build() {
    if (_conf.brutal_search) {
        LOG(INFO) << "Brutal search enabled, skip single build initialization.";
        return 0;
    }
    if (this->HierarchicalClusterIndex::init_single_build() != 0) {
        return -1;
    }

    if (_conf.whether_filter == false) {
        return -1;
    }

    if (_filter_quantization->init_quantized_feature_memory() != 0) {
        LOG(INFO) << "get_filter_quantization init_quantized_feature_memory error";
        return -1;
    }

    if (_conf.whether_pq) {
        if (_pq_quantization->init_quantized_feature_memory() != 0) {
            LOG(INFO) << "pq_quantization init_quantized_feature_memory error";
            return -1;
        }
    }//不量化时候，会读取原始特征。读原始特征文件时候，会初始化对应的内存

    return 0;
}

int PuckIndex::single_build(BuildInfo* build_info) {
    PuckBuildInfo* puck_build_info = dynamic_cast<PuckBuildInfo*>(build_info);

    if (puck_build_info == nullptr) {
        return -1;
    }

    if (this->HierarchicalClusterIndex::single_build(build_info) != 0) {
        return 1;
    }
    std::vector<Quantization*> quantizations;
    quantizations.push_back(_filter_quantization.get());

    if (_conf.whether_pq) {
        quantizations.push_back(_pq_quantization.get());
    }

    uint32_t local_idx = 0;
    if (puck_single_assign(build_info, quantizations, local_idx) != 0) {
        return 1;
    }
    puck_build_info->quantizated_feature.resize(quantizations.size());

    for (uint32_t i = 0; i < quantizations.size(); ++i) {
        auto* quantized_feature = quantizations[i]->get_quantized_feature(local_idx);
        puck_build_info->quantizated_feature[i].first[0] = ((float*)quantized_feature)[0];
        quantized_feature +=  quantizations[i]->get_fea_offset();
        uint32_t nsq = quantizations[i]->get_quantization_params().nsq;
        puck_build_info->quantizated_feature[i].second.resize(nsq);
        memcpy(puck_build_info->quantizated_feature[i].second.data(), quantized_feature, sizeof(unsigned char) * nsq);
    }

    return 0;
}

int PuckIndex::puck_single_assign(BuildInfo* build_info, std::vector<Quantization*>& quantizations,
                                  uint32_t idx) {
    std::unique_ptr<float[]> residual(new float[_conf.feature_dim]);
    float* chunk_points = build_info->feature.data();
    memcpy(residual.get(), chunk_points, sizeof(float) * _conf.feature_dim);
    int coarse_assign = build_info->nearest_cell[0].cell_id / _conf.fine_cluster_count;
    int fine_assign = build_info->nearest_cell[0].cell_id % _conf.fine_cluster_count;
    cblas_saxpy(_conf.feature_dim, -1.0,
                _coarse_vocab + coarse_assign * _conf.feature_dim, 1,
                residual.get(), 1);
    cblas_saxpy(_conf.feature_dim, -1.0,
                _fine_vocab + fine_assign * _conf.feature_dim, 1,
                residual.get(), 1);
    for (auto quantized : quantizations) {
        auto& cur_params = quantized->get_quantization_params();
        std::unique_ptr<float[]> pq_distance_table(new float[cur_params.nsq * cur_params.ks]);

        quantized->get_dist_table(residual.get(), pq_distance_table.get());

        auto* quantized_fea = quantized->get_quantized_feature(idx);
        quantized_fea += quantized->get_fea_offset();
                
        for (uint32_t n = 0; n < (uint32_t)cur_params.nsq; ++n) {
            float min_distance = std::sqrt(std::numeric_limits<float>::max());
            const float* sub_dist_table = pq_distance_table.get() + n * cur_params.ks;

            for (uint32_t k = 0; k < (uint32_t)cur_params.ks; ++k) {
                if (sub_dist_table[k] < min_distance){
                    min_distance = sub_dist_table[k];
                    quantized_fea[n] = (unsigned char)k;
                }
            }
        }
            
            
        for (uint32_t k = 0; k < (uint32_t)cur_params.nsq; ++k) {
            uint32_t cur_lsq = std::min(cur_params.lsq, cur_params.dim - k * cur_params.lsq);
            float* cur_pq_centroids = quantized->get_sub_codebooks(k);

            //point i 在第k子空间对应的聚类中心id
            auto* quantized_fea = quantized->get_quantized_feature(idx);
            quantized_fea += quantized->get_fea_offset();
            int cur_assign = (int)quantized_fea[k];
            float* cur_point_fea = residual.get();
            //量化使用残差
            cblas_saxpy(cur_lsq, -1.0,
                        cur_pq_centroids + (u_int64_t)cur_assign * cur_params.lsq, 1,
                        cur_point_fea + k * cur_params.lsq, 1);
        }
    }
    
    memcpy(residual.get(), _coarse_vocab + coarse_assign * _conf.feature_dim,
           _conf.feature_dim * sizeof(float));
    cblas_saxpy(_conf.feature_dim, 1.0,
                _fine_vocab + fine_assign * _conf.feature_dim, 1,
                residual.get(), 1);

    for (auto quantized : quantizations) {
        if (idx < 10) {
            LOG(ERROR) << "puck_single_assign";
        }
        quantized->set_static_value_of_formula(idx, residual.get());
    }
        
    return 0;
}
} //namesapce puck
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
