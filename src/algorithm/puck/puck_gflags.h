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
 * @file puck_gflags.h
 * @author huangben(com@baidu.com)
 * @date 2018/7/07 15:59:36
 * @brief
 *
 **/

#pragma once

#include <gflags/gflags.h>

namespace puck {

DECLARE_bool(need_log_file);
DECLARE_string(puck_log_file);

/*****训练&建库参数******/
//通用参数
DECLARE_string(index_path);
DECLARE_string(index_file_name);
DECLARE_string(feature_file_name);
DECLARE_string(sparse_feature_file_name);
DECLARE_string(feature_file_name_udf);
DECLARE_string(coarse_codebook_file_name);
DECLARE_string(fine_codebook_file_name);

DECLARE_string(cell_assign_file_name);
DECLARE_string(cell_start_memory_idx_file_name);
DECLARE_string(cell_assign_bak_file_name);
DECLARE_string(label_file_name);

DECLARE_string(local_to_memory_file_name);
DECLARE_string(memory_to_local_file_name);
DECLARE_string(memory_to_ori_cell_id_file_name);

DECLARE_int32(feature_dim);
DECLARE_bool(whether_norm);

DECLARE_int32(coarse_cluster_count);
DECLARE_int32(fine_cluster_count);
DECLARE_int32(threads_count);

DECLARE_int32(ip2cos);

DECLARE_string(hierarchical_cluster_index_data_file_name);

//puck
DECLARE_bool(whether_pq);
DECLARE_int32(nsq);
DECLARE_string(pq_codebook_file_name);
DECLARE_string(pq_data_file_name);

DECLARE_int32(filter_nsq);
DECLARE_string(filter_codebook_file_name);
DECLARE_string(filter_data_file_name);
DECLARE_string(filter_data_bak_file_name);

//tinker
DECLARE_string(tinker_file_name);
DECLARE_int32(tinker_neighborhood);
DECLARE_int32(tinker_construction);

/***********检索参数*********/
//检索时，初始化内存池的size
DECLARE_int32(context_initial_pool_size);
//检索通用参数
DECLARE_int32(search_coarse_count);
DECLARE_int32(topk);

//HierarchicalClusterIndex
DECLARE_int32(neighbors_count);

//puck
DECLARE_int32(filter_topk);
DECLARE_double(radius_rate);
//tinker
DECLARE_int32(tinker_search_range);

//lut16
DECLARE_bool(use_lut16);
DECLARE_bool(use_int8);
DECLARE_bool(use_fast_heap);
DECLARE_int32(filter_ks);

DECLARE_bool(only_train_pq);
DECLARE_bool(filter_recompute);
DECLARE_int32(search_fine_count);
DECLARE_int32(search_point_count);

DECLARE_bool(packed_cluster);
DECLARE_bool(enable_lut_clip);
DECLARE_int32(window_size);

DECLARE_bool(use_pq_method_0);
DECLARE_bool(use_pq_method_1);
DECLARE_bool(use_avg_offset);
DECLARE_bool(use_bak_cluster);

DECLARE_bool(convert_local_to_memory_while_build);

DECLARE_int32(bak_cell_assign);

DECLARE_bool(ad_hoc);

DECLARE_bool(only_build);

DECLARE_int32(index_type_ad_hoc);

DECLARE_bool(use_brute_force);

DECLARE_bool(spilled_assgin_in_ip_space);
DECLARE_bool(only_sample);

DECLARE_int32(sq_num_bits);

DECLARE_string(filter);

}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100 */
