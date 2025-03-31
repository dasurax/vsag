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
 * @file search_context.cpp
 * @author huangben@baidu.com
 * @date 2018/8/13 19:50
 * @brief
 *
 **/
#include <malloc.h>
#include <unistd.h>
#include <glog/logging.h>
#include "algorithm/puck/search_context.h"
//#define _aligned_malloc(size, alignment) aligned_alloc(alignment, size)

namespace puck {

template <typename Int, typename DenomInt>
constexpr Int DivRoundUp2(Int num, DenomInt denom) {
  return (num + static_cast<Int>(denom) - static_cast<Int>(1)) /
         static_cast<Int>(denom);
}

template <typename Int, typename DenomInt>
constexpr Int NextMultipleOf2(Int num, DenomInt denom) {
  return DivRoundUp2(num, denom) * denom;
}

extern const u_int64_t cache_offset_size;
SearchContext::SearchContext() :
    _logid(0),
    //_topk(0),
    _debug(false),
    _inited(false),
    _model(nullptr),
    _log_string("") {
}

SearchContext::~SearchContext() {
    if (_model) {
        free(_model);
        _search_cell_data.init();
        _search_point_data.init();
    }

}

int SearchContext::reset(IndexConf& conf) {
    _logid = 0;
    _log_string = "";
    _debug = false;

    _init_conf.search_coarse_count = std::min(_init_conf.search_coarse_count, conf.coarse_cluster_count);
    conf.search_coarse_count = std::min(conf.search_coarse_count, conf.coarse_cluster_count);
    _init_conf.search_fine_count = std::min(_init_conf.search_fine_count, _init_conf.search_coarse_count * conf.fine_cluster_count);
    conf.search_fine_count = std::min(conf.search_fine_count, conf.search_coarse_count * conf.fine_cluster_count);

    //LOG(INFO) << "reset: " << _init_conf.filter_topk << " " << conf.filter_topk;
    if (_init_conf.filter_topk < conf.filter_topk
            || _init_conf.search_coarse_count < conf.search_coarse_count
            || _init_conf.search_fine_count < conf.search_fine_count
            || _init_conf.neighbors_count < conf.neighbors_count || !_inited) {
        _inited = false;
        _init_conf = conf;

        if (_model) {
            free(_model);
            _search_cell_data.init();
            _search_point_data.init();
        }
    }

    //_topk = conf.topk;
    //二级聚类中心最多需要top-neighbors_count个cell，空间多申请一些可避免频繁更新tag idx
    unsigned int all_cells_cnt = 1;

    //if (conf.index_type == IndexType::PUCK ||  conf.index_type == IndexType::HIERARCHICAL_CLUSTER) {
    if (conf.index_type == IndexType::HIERARCHICAL_CLUSTER) {
        all_cells_cnt = conf.neighbors_count * 1.1;

        if (all_cells_cnt > conf.search_coarse_count * conf.fine_cluster_count) {
            all_cells_cnt = conf.search_coarse_count * conf.fine_cluster_count;
        }

        if (_search_cell_data.cell_distance.size() != all_cells_cnt) {
            _search_cell_data.cell_distance.resize(all_cells_cnt);
        }
    }  

    _total_point_count = conf.total_point_count;
    //LOG(INFO) << "search context: " << conf.total_point_count << " " << _inited;
    uint32_t num_32dp_simd_iters = (conf.total_point_count + 31) / 32;

    if (_inited) {
        // memset(_32dp_simd_iters_visited, 0, num_32dp_simd_iters);
        return 0;
    }

    size_t model_size = 0;
    //每个过程需要的内存
    constexpr size_t kPadding = 96;
    auto capacity_ = NextMultipleOf2(2 * conf.filter_topk, 32);
    size_t coarse_ip_dist = sizeof(float) * conf.coarse_cluster_count;
    size_t fine_ip_dist = sizeof(float) * conf.fine_cluster_count;
    size_t ip_dist = std::max(coarse_ip_dist, fine_ip_dist);

    if (FLAGS_bak_cell_assign) {
        ip_dist = coarse_ip_dist + fine_ip_dist;
    }

    //coarse
    size_t coarse_heap_size = (sizeof(float) + sizeof(uint32_t)) * conf.search_coarse_count;
    size_t stage_coarse = ip_dist + coarse_heap_size;
    model_size = std::max(model_size, stage_coarse);

    //fine
    size_t fine_ip_heap_size = (sizeof(float) + sizeof(uint32_t)) * conf.fine_cluster_count;

    size_t search_cluster_heap_size = (sizeof(float) + sizeof(uint32_t)) * conf.search_fine_count;

    size_t stage_fine = coarse_heap_size + ip_dist + fine_ip_heap_size + search_cluster_heap_size;

    size_t filter_ks = conf.filter_ks;
    //model_size = std::max(model_size, stage_fine);
    //LOG(ERROR) << "---------- conf.ks: " << conf.use_lut16 << " " << filter_ks; 
    size_t fa = 2;
    if (conf.whether_filter) {
        //filter
        size_t filter_dist_table = sizeof(float) * conf.filter_nsq * conf.ks;
        size_t filter_dist_table_int8 = sizeof(uint8_t) * conf.filter_nsq * filter_ks;
        // size_t filter_heap = (sizeof(float) + sizeof(float) + sizeof(uint32_t)) * conf.filter_topk;
        size_t filter_heap = sizeof(float) * 2 * (capacity_ + kPadding)
                + sizeof(uint32_t) * (2 * capacity_ + kPadding)
                + sizeof(uint32_t) * (2 * capacity_ / 32 + 2);
        size_t pq_reorder = (sizeof(float) + sizeof(uint32_t)) * conf.filter_nsq;
        size_t stage_filter = pq_reorder + filter_dist_table + filter_dist_table_int8 + filter_heap;
        //model_size = std::max(model_size, stage_filter);
        //stage_fine += stage_filter;
        if (conf.whether_pq) {
        //if (true) {
            //rank
            size_t pq_dist_table = sizeof(float) * conf.nsq * conf.ks;
            // size_t pq_stage = pq_dist_table + filter_heap + pq_reorder + filter_dist_table + filter_dist_table_int8;
            size_t pq_stage = pq_dist_table + filter_heap + filter_dist_table;
            //model_size = std::max(model_size, pq_stage);
            //LOG(INFO) << "stage_filter: " << stage_filter << " pq_stage: " << pq_stage;
            stage_filter = std::max(stage_filter, pq_stage);
            // stage_fine += pq_stage;
        }
        
        // to avoid memory corrupt
        stage_fine += stage_filter; // + 16 * 1024;
    }

    //LOG(INFO)<<"stage_fine="<<stage_fine;
    model_size = std::max(model_size, stage_fine);
    //LOG(INFO)<<"model_size="<<model_size;

    // query norm
    model_size += sizeof(float) * conf.feature_dim;
    // model_size += sizeof(uint8_t) * num_32dp_simd_iters;

    void* memb = nullptr;
    int32_t pagesize = getpagesize();

    size_t size = model_size + (pagesize - model_size % pagesize);
    //LOG(INFO) << "mem size: " << " " << model_size << " " << size;
    int err = posix_memalign(&memb, pagesize, size);

    if (err != 0) {
        std::runtime_error("alloc_aligned_mem_failed errno=" + errno);
        return -1;
    }

    _model = reinterpret_cast<char*>(memb);
    _search_cell_data.query_norm = (float*)_model;
    char* temp = _model + sizeof(float) * conf.feature_dim;

    _search_cell_data.coarse_distance = (float*)temp;
    temp +=  sizeof(float) * conf.search_coarse_count;

    _search_cell_data.coarse_tag = (uint32_t*)temp;
    temp +=  sizeof(uint32_t) * conf.search_coarse_count;

    _search_cell_data.cluster_inner_product = (float*)temp;

    if (FLAGS_bak_cell_assign) {
        temp += sizeof(float) * conf.coarse_cluster_count;
        _search_cell_data.fine_cluster_inner_product = (float*)temp;
        temp += sizeof(float) * conf.fine_cluster_count;
    } else {
        temp += sizeof(float) * std::max(conf.fine_cluster_count, conf.coarse_cluster_count);
    }

    _search_cell_data.fine_distance = (float*)temp;
    temp += sizeof(float) * conf.fine_cluster_count;

    _search_cell_data.fine_tag = (uint32_t*)temp;
    temp += sizeof(uint32_t) * conf.fine_cluster_count;

    _search_cell_data.temp_distance = (float*)temp;
    temp +=  sizeof(float) * conf.search_fine_count;

    _search_cell_data.temp_tag = (uint32_t*)temp;
    temp +=  sizeof(uint32_t) * conf.search_fine_count;


    //temp = _model + sizeof(float) * conf.feature_dim;
    //temp += sizeof(float) * conf.filter_topk * fa;
    if (conf.whether_filter) {

        _search_point_data.result_distance = (float*)temp;
        temp += sizeof(float) * (capacity_+ kPadding);
        //LOG(INFO) << "result_distance: " << sizeof(float) * (capacity_+ kPadding) << " " << capacity_ << " " << kPadding << " " << _init_conf.filter_topk << " " << conf.filter_topk;
        _search_point_data.result_distance_temp = (float*)temp;
        temp += sizeof(float) * (capacity_+ kPadding);
        //LOG(INFO) << "result_distance: " << sizeof(float) * (capacity_+ kPadding) << " " << capacity_ << " " << kPadding;
        _search_point_data.result_tag = (uint32_t*)temp;
        temp += sizeof(uint32_t) * (2 * capacity_+ kPadding);
        //LOG(INFO) << "result_distance: " << sizeof(uint32_t) * (2 * capacity_+ kPadding) << " " << capacity_ << " " << kPadding;
        _search_point_data.masks = (uint32_t*)temp;
        temp += sizeof(uint32_t) * (2 * capacity_ / 32 + 2);
        //LOG(INFO) << "result_distance: " << sizeof(uint32_t) * (2 * capacity_ / 32 + 2) << " " << capacity_ << " " << kPadding;
        _search_point_data.pq_dist_table = (float*)temp;
        temp += sizeof(float) * conf.filter_nsq * conf.ks;
        
        _search_point_data.pq_dist_table_int8 = (uint8_t*)temp;
        _search_point_data.pq_pq_dist_table= (float*)temp;
        temp += sizeof(uint8_t) * conf.filter_nsq * filter_ks;

        if (conf.whether_pq) {
            //_search_point_data.pq_pq_dist_table = (float*)temp;
            //temp += sizeof(float) * conf.nsq * conf.ks;
        }

        _search_point_data.query_sorted_tag = (uint32_t*)temp;
        temp += sizeof(uint32_t) * conf.filter_nsq;

        _search_point_data.query_sorted_dist = (float*)temp;
        temp += sizeof(uint32_t) * conf.filter_nsq;

    }
    //_32dp_simd_iters_visited = (uint8_t*)temp;
    //memset(_32dp_simd_iters_visited, 0, num_32dp_simd_iters);
    _inited = true;

    return 0;
}


int SearchContext::reset(const size_t feature_dim) {
   
    if (_inited) {
        // memset(_32dp_simd_iters_visited, 0, num_32dp_simd_iters);
        return 0;
    }

    _inited = true;

    return 0;
}


} //namesapce puck
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

