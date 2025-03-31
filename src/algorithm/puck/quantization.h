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
 * @file quantization.h
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2021/7/20 11:17
 * @brief
 *
 **/
#pragma once
#include <string>
#include <iostream>
#include <memory>
#include <functional>
#include <numeric>
#include <math.h>
#include <vector>
#include "algorithm/puck/index_conf.h"
#include <cblas.h>

namespace puck {
struct QuantizationParams {
    uint32_t ks;
    uint32_t dim;
    uint32_t nsq;
    uint32_t lsq;
    uint32_t spilled_copy_num;
    void show();
    int init(const IndexConf& conf, bool is_filter = false) {
        if ((conf.ks != 256 && conf.ks != 16)
                || (is_filter && conf.feature_dim < conf.filter_nsq)
                || (is_filter == false && conf.feature_dim < conf.nsq)) {
            return -1;
        }
        spilled_copy_num = 0;
        ks = conf.ks;
        dim = conf.feature_dim;
        nsq = (is_filter == false) ? conf.nsq : conf.filter_nsq;
        lsq = std::ceil(1.0 * dim / nsq);
        show();
        return 0;
    }
};

///仅支持KS=256 && 分层索引量化后的残差
class Quantization {
public:
    Quantization() : _per_subspace_len(0), _fea_offset(0) {}
    Quantization(const QuantizationParams& params, uint32_t point_count);
    virtual ~Quantization() {
        reset();
    }
    /*
    * @brief 初始化量化特征所需的内存
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int init_quantized_feature_memory();
    /*
    * @brief 计算pq table
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int get_dist_table(const float*  query, float* dist_table) const;
    virtual int get_ip_dist_table(const float*  query, float* dist_table) const;
    virtual int get_dist_table_int8(float* dist_table, uint8_t* dist_table_int8, float* fixed_point_multiplier) const {return 0;}
   
   
   /*
    * @brief 获取量化码本
    * @@return float* : 码本指针
    **/
    virtual inline float* get_codebooks() const {
        return _codebooks.get();
    }
    /*
    * @brief 获取指定空间的量化码本
    * @@return float* : 子空间码本指针
    **/
    virtual inline float* get_sub_codebooks(uint64_t n = 0, uint64_t k = 0) const {
        return _codebooks.get() + (u_int64_t)n * _params.ks * _params.lsq + k * _params.lsq;
    }
    /*
    * @brief 获取某个样本的量化特征（偏移值+PQ量化特征）
    * @@return float* : 量化特征
    **/
    virtual inline unsigned char* get_quantized_feature(uint64_t idx = 0) const {
        // LOG(INFO) << "get_quantized_feature: " << idx << " " << _total_point_count << " " << _per_fea_len;
        return _quantized_feature.get() + (u_int64_t)idx * _per_fea_len;
    }
    
    virtual void get_quantized_feature(uint64_t idx, uint8_t* pq_feature) {}
    
    virtual inline unsigned char* get_quantized_feature_packed(uint64_t k = 0) const {
        return nullptr;
    }

    virtual inline float* get_fea_offset_packed(uint64_t k = 0) const {
        return nullptr;
    }

    virtual inline unsigned char* get_quantized_feature_packed(size_t token, uint64_t k) const {
        return nullptr;
    }

    virtual inline float* get_fea_offset_packed(size_t token, uint64_t k) const {
        return nullptr;
    }

    /*
    * @brief 获取PQ量化特征的偏移量
    * @@return int : PQ量化特征的偏移量
    **/
    virtual int get_fea_offset() const {
        return _fea_offset;
    }

    virtual float get_fea_offset(uint32_t idx) const {
        return 0.0;
    }
    /*
    * @brief 获取某个样本的量化特征的总长度（偏移值+PQ量化特征）
    * @@return int : 每个样本量化特征的长度
    **/
    virtual int get_per_fea_len() const {
        return _per_fea_len;
    }
    /*
    * @brief 更新某个样本的偏移值
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int set_static_value_of_formula(uint64_t idx, float*);
    /*
    * @brief 加载量化索引
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int load(const std::string& codebook_file, const std::string& quantized_feafile,
            std::vector<std::vector<uint32_t>>& datapoints_by_token,
            const uint32_t* local_2memory_idx = nullptr, float min_offset = 0.0f) {return 0;}

    virtual int load(const std::string& codebook_file, const std::string& quantized_feafile,
            const uint32_t* local_2memory_idx = nullptr, float min_offset = 0.0f, 
            unsigned char* bias = nullptr, size_t bias_step = 0);
    /*
    * @brief 加载量化码本
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int load_codebooks(const std::string& codebook_file);

    /*
    * @brief 直接加载文件数据使用
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int load_codebooks(char * fileBaseAddress, uint64_t fileLength);
    /*
    * @brief 获取配置信息
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual const QuantizationParams& get_quantization_params() const {
        return _params;
    }
    /*
     * @brief 写码本文件
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    virtual int save_codebooks(const std::string& file_name) const;
    /*
     * @brief 写索引文件(建库的产出，与建库样本相关)
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    virtual int save_index(const std::string& file_name) const;
    virtual int save_index_bak(const std::string& file_name) const {return 0;}

    virtual void reset_quantized_feature() {
        _quantized_feature.reset(nullptr);
    }

    virtual void reset() {
        if(_isUniquePtr) {
            _quantized_feature.reset(nullptr);
            _codebooks.reset(nullptr);
        }else{
            _quantized_feature.release();
            _codebooks.release();
        }
    }

    virtual std::unique_ptr<float[]> decode_packed(uint64_t idx) {return nullptr;}

    virtual int get_norm(uint64_t idx) const {
        auto vec = decode(idx);
        return cblas_sdot(_params.dim, vec.get(), 1, vec.get(), 1);
    }

    /*
    * 直接加载文件数据
    */ 
    int load_quantized_feature_address(char *fileBaseAddress, uint64_t fileLength);
    virtual int load_quantized_feature_address(char *indexFileBaseAddress, uint64_t indexFileLength, 
                char *offsetFileBaseAddress, uint64_t offsetFileLength)
    {
        return -1;
    }

private:
    /*
    * @brief 计算码本长度
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual size_t get_codebook_length() {
        //LOG(INFO) << "get_codebook_length = " << _params.nsq* _params.ks* _params.lsq* sizeof(float);
        return _params.nsq * _params.ks * _params.lsq * sizeof(float);
    }
    /*
    * @brief 计算量化特征长度
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual size_t get_quantized_feature_length() {
        return _per_fea_len * _total_point_count * sizeof(unsigned char);
    }
    /*
    * @brief decode
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual std::unique_ptr<float[]> decode(uint64_t idx) const;

    /*
    * @brief 加载索引
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int load_quantized_feature(
                                const std::string& quantized_feafile,
                                const uint32_t* cnts_index,
                                unsigned char* bias = nullptr,
                                size_t bias_step = 0);

    virtual int load_quantized_feature(const std::string& quantized_feafile,
                                std::vector<std::vector<uint32_t>>& datapoints_by_token,
                                const uint32_t* cnts_index) {return 0;}

public:
    /*
    * @brief 初始化码本所需内存
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int init_codebooks_memory();

protected:
    const int _per_subspace_len;
    const int _fea_offset;
    uint32_t _total_point_count;
    int _per_fea_len;
    float _min_offset;
    QuantizationParams _params;
    std::unique_ptr<float[]> _codebooks;
    std::unique_ptr<unsigned char[]> _quantized_feature;

protected:
    bool _isUniquePtr = true;   //如果为true，则通过智能指针释放内存，否则，通过加载的文件释放，这里不再需要释放。
};

}//namespace puck