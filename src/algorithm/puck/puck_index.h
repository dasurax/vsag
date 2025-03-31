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
 * @file puck_index.h
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2022-09-27 15:56
 * @brief
 *
 **/
#pragma once
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <fstream>
#include <stdlib.h>
#include "algorithm/puck/quantization.h"
#include "algorithm/puck/quantization_4bit_packed.h"
#include "algorithm/puck/hierarchical_cluster_index.h"
#include "algorithm/puck/fast_max_heap.h"
//#define _aligned_malloc(size, alignment) aligned_alloc(alignment, size)
namespace puck {
//内存索引结构
class PuckIndex : public puck::HierarchicalClusterIndex {
public:
    /*
     * @brief 默认构造函数，检索配置根据gflag参数确定(推荐使用)
     **/
    PuckIndex();
    virtual ~PuckIndex();
    /*
     * @brief 检索最近的topk个样本
     * @@param [in] request : request
     * @@param [out] response : response
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    virtual int search(const Request* request, Response* response) override;
    /*
    * @brief 初始化内存、训练码本（计算一二级聚类中心）、写码本文件
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int train() override;
    /*
     * @brief 读取索引配置文件（index.dat）、初始化内存、加载码本；第一次调研single_build时，用来初始化
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    virtual int init_single_build() override;
    /*
    * @brief 读取索引配置文件（index.dat）、初始化内存、1个样本建库（MR建库和实时入库时会调用该函数）
    * @@param [in/out] build_info : build_info包含样本建库所需的所有信息
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int single_build(BuildInfo* build_info) override;
protected:
    virtual int check_index_type() override;
    /*
     * @brief 写码本文件
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    int save_codebooks() const override;
    /*
    * @brief 读码本文件
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int read_codebooks() override;
    /*
    * @brief 写索引文件(建库的产出，与建库样本相关)
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int save_index() override;
    virtual int save_index_bak() override;
    virtual int save_index_after_convert() override; 
    /*
     * @brief 初始化内存
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    int init_model_memory() override;
    /*
     * @brief 建库的计算过程
     * @@param [int] total_cnt : 建库样本总数
     * @@param [int] feature_file_name : 建库样本的文件名
     * @@param [out] cell_assign : 存储每个样本最近的cell id
     **/
    virtual void batch_assign(const uint32_t total_cnt, const std::string& feature_file_name,
                              uint32_t* cell_assign, bool bak_assign = false) override;
    virtual void batch_assign_bak(const uint32_t total_cnt, const std::string& feature_file_name,
                              uint32_t* cell_assign) override;
     /*
     * @brief 计算PQ-TABLE等
     * @@param [in\out] context : context由内存池管理
     * @@param [in] feature : query的特征向量
     * @@param [out] 正常返回0，错误返回值<0
     **/
    int pre_filter_search(SearchContext* context, const float* feature);

    /*
     * @brief 计算query与top_coarse_cnt个一级聚类中心的下所有二级聚类中心的距离
     * @@param [in\out] context : context由内存池管理
     * @@param [in] feature : query的特征向量
     * @@return (int) : 正常返回保留的cell个数(>0)，错误返回值<0
     **/
    int search_nearest_filter_points(SearchContext* context, const float* feature);
    /*
     * @brief 计算query与某个cell下所有样本的距离（样本的filter量化特征）
     * @@param [in\out] context : context由内存池管理
     * @@param [in] FineCluster : cell指针
     * @@param [in] cell_dist : query和cell的距离
     * @@param [in] result_heap : 堆结构，存储query与样本的topk
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    virtual int compute_quantized_distance(SearchContext* context, const FineCluster*, const float cell_dist,
                                           PuckMaxHeap& result_heap, int& total, size_t& total_cost);
    virtual int compute_quantized_distance_lut16(SearchContext* context, const FineCluster*, const float cell_dist,
                                           PuckMaxHeap& result_heap, FastMaxHeap& result_heap2, int& total, float multiplier, size_t& total_cost);
    virtual int compute_quantized_distance_lut16_fast_heap(SearchContext* context, const FineCluster*, const float cell_dist,
                                           PuckMaxHeap& result_heap, FastMaxHeap& result_heap2, int& total, float multiplier, size_t& total_cost, size_t token, std::vector<float>& dist, float thr);
    virtual int compute_quantized_distance_lut16_fast_heap2(SearchContext* context, const FineCluster*, const float cell_dist,
                                           PuckMaxHeap& result_heap, FastMaxHeap& result_heap2, int& total, float multiplier, size_t& total_cost, size_t token, size_t l,
                                           std::vector<float>& dist, float thr, double& max_q, size_t total_updated_cnt);
    
    /*
     * @brief 计算query与部分样本的距离(query与filter特征的topN个样本）
     * @@param [in\out] context : context由内存池管理
     * @@param [in] feature : query的特征向量
     * @@param [in] filter_topk : cell指针
     * @@param [in] result_heap : 堆结构，存储query与样本的topk
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    virtual int rank_topN_points(SearchContext* context, const float* feature, const uint32_t filter_topk,
                                 PuckMaxHeap& result_heap);

    virtual int brute_force(SearchContext* context, const float* feature, const uint32_t filter_topk,
                                 PuckMaxHeap& result_heap);
    /*
    * @brief 检索过程中会按某种规则调整样本在内存的顺序（memory_idx），计算对应的信息
    * @@param [out] cell_start_memory_idx : 每个cell下样本中最小的memory_idx
    * @@param [out] local_to_memory_idx : 每个样本local_idx 与 memory_idx的映射关系
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int convert_local_to_memory_idx(uint32_t* cell_start_memory_idx, uint32_t* local_to_memory_idx) override;

    virtual int convert_local_to_memory_idx(
            uint32_t* cell_start_memory_idx, 
            uint32_t* local_to_memory_idx,
            uint32_t* cell_assign,
            float* fea_offset = nullptr,
            bool need_init = true) override;
    /*
    * @brief 加载与样本相关的索引文件
    * @@param [in] local_to_memory_idx : 每个样本local_idx 与 memory_idx的映射关系
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int read_feature_index(uint32_t* local_to_memory_idx = nullptr, bool reset_quantized_feature = true) override;
    /*
    * @brief 计算部分样本的量化特征
    * @@param [in] thread_params : 线程信息
    * @@param [out] cell_assign : 存储样本最近的cell id
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    int puck_assign(const ThreadParams& thread_params, uint32_t* cell_assign, uint32_t offset) const;
    int puck_assign_bak(const ThreadParams& thread_params, uint32_t* cell_assign, uint32_t total_cnt) const;
    /*
    * @brief 计算1个样本的量化特征
    * @@param [in/out] build_info : build_info包含样本建库所需的所有信息
    * @@param [in] quantizations : 需要计算的量化数组
    * @@param [in] idx : 在quantizations中样本的idx
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    int puck_single_assign(BuildInfo* build_info, std::vector<Quantization*>& quantizations, uint32_t idx);
protected:
    DISALLOW_COPY_AND_ASSIGN_AND_MOVE(PuckIndex);
    //通常取1/4量化，对过滤出的小规模样本重新排序时使用
    std::unique_ptr<Quantization> _pq_quantization;
    //大尺度量化，用来过滤
    std::unique_ptr<Quantization> _filter_quantization;
    float _min_offset = 0.0f;
    std::vector<std::vector<uint32_t>> _datapoints_by_token;
    // uint32_t* _memory_to_local;
};

struct PuckBuildInfo : public BuildInfo {
    typedef std::pair<std::vector<float>, std::vector<unsigned char>> QuantizatedFeature;
    std::vector<QuantizatedFeature> quantizated_feature;
    int getSize() {
        int size = BuildInfo::getSize();
        for(int i = 0; i < quantizated_feature.size(); i++) {
            size += quantizated_feature[i].first.size() * 4 + quantizated_feature[i].second.size() * 1;
        }
        return size; 
    }
};
#ifdef __SSE__
float lookup_dist_table(const unsigned char* assign,
                        const float* dist_table, size_t dim, size_t nsq);
#endif

long readProcStatus3(const std::string& name);

} //namesapce puck
