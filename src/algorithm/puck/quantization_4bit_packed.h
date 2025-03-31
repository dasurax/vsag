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
 * @file quantization_4bit_packed.h
 * @author suguan.dx
 * @date 2024/08/06 14:39
 * @brief
 *
 **/
#pragma once
#include <string>
#include <memory>
#include <functional>
#include <numeric>
#include <math.h>
#include <vector>

#include "algorithm/puck/index_conf.h"
#include "algorithm/puck/quantization.h"
namespace puck {

///仅支持KS=16 && 分层索引量化后的残差
class Quantization4BitPacked: public Quantization {
public:
    Quantization4BitPacked(const QuantizationParams& params, uint32_t point_count);
    ~Quantization4BitPacked() {
        reset();
    }
    /*
    * @brief 初始化量化特征所需的内存
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    int init_quantized_feature_memory() override;
    /*
    * @brief 计算pq table
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    int get_dist_table(const float*  query, float* dist_table) const override;
    int get_ip_dist_table(const float*  query, float* dist_table) const override;
    int get_dist_table_int8(float* dist_table, uint8_t* dist_table_int8, float* fixed_point_multiplier) const override;
    /*
    * @brief 获取量化码本
    * @@return float* : 码本指针
    **/
    inline float* get_codebooks() const override {
        return _codebooks.get();
    }
    /*
    * @brief 获取指定空间的量化码本
    * @@return float* : 子空间码本指针
    **/
    inline float* get_sub_codebooks(uint64_t n = 0, uint64_t k = 0) const override {
        return _codebooks.get() + (u_int64_t)n * _params.ks * _params.lsq + k * _params.lsq;
    }
    /*
    * @brief 获取某个样本的量化特征（偏移值+PQ量化特征）
    * @@return float* : 量化特征
    **/
    inline unsigned char* get_quantized_feature(uint64_t idx = 0) const override {
        return _quantized_feature.get() + (u_int64_t)idx * _per_fea_len;
        // return _quantized_feature.get() + (u_int64_t)idx * _per_fea_len;
    }

    void get_quantized_feature(uint64_t idx, uint8_t* pq_feature) override {
        uint64_t k = idx / 32;
        uint64_t r = idx % 32;
        auto pq_feature_start = _quantized_feature_packed.get() + k * _params.nsq * 16;

        int hi = r / 16;
        int rr = r % 16;
        for (size_t i = 0; i < _params.nsq; i++) {
            pq_feature[i] = pq_feature_start[i * 16 + rr];
            if (hi) {
                pq_feature[i] >>= 4;
            } else {
                pq_feature[i] &= 0x0F;
            }
        }
    }

    inline unsigned char* get_quantized_feature_packed(uint64_t k = 0) const override {
        // LOG(INFO) << "---------------- " << idx;
        return _quantized_feature_packed.get() + k * _params.nsq * 16;
        // return _quantized_feature.get() + (u_int64_t)idx * _per_fea_len;
    }

    inline float* get_fea_offset_packed(uint64_t k = 0) const override {
        // LOG(INFO) << "---------------- " << idx;
        return _fea_offset_packed.get() + k * 32;
        // return _quantized_feature.get() + (u_int64_t)idx * _per_fea_len;
    }

    inline unsigned char* get_quantized_feature_packed(size_t token, uint64_t k) const override {
        //LOG(INFO) << "---------------- get_quantized_feature_packed " << _quantized_feature_packed_vec.size() << " " << _quantized_feature_packed_vec[token].size();
        return (unsigned char*)_quantized_feature_packed_vec[token].data() + k * _params.nsq * 16;
        // return _quantized_feature.get() + (u_int64_t)idx * _per_fea_len;
    }

    inline float* get_fea_offset_packed(size_t token, uint64_t k) const override {
        //LOG(INFO) << "---------------- get_fea_offset_packed " << _fea_offset_packed_vec.size() << " " << _fea_offset_packed_vec[token].size();
        return (float* )_fea_offset_packed_vec[token].data() + k * 32;
        // return _quantized_feature.get() + (u_int64_t)idx * _per_fea_len;
    }
    /*
    * @brief 获取PQ量化特征的偏移量
    * @@return int : PQ量化特征的偏移量
    **/
    int get_fea_offset() const override {
        return _fea_offset;
    }

    float get_fea_offset(uint32_t idx) const override {
        return _fea_offset_packed[idx];
    }

    /*
    * @brief 获取某个样本的量化特征的总长度（偏移值+PQ量化特征）
    * @@return int : 每个样本量化特征的长度
    **/
    int get_per_fea_len() const override {
        return _per_fea_len;
    }
    /*
    * @brief 更新某个样本的偏移值
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    int set_static_value_of_formula(uint64_t idx, float*) override;
    /*
    * @brief 加载量化索引
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    int load(const std::string& codebook_file, const std::string& quantized_feafile,
            std::vector<std::vector<uint32_t>>& datapoints_by_token,
            const uint32_t* local_2memory_idx = nullptr, float min_offset = 0.0f) override;

    int load(const std::string& codebook_file, const std::string& quantized_feafile,
            const uint32_t* local_2memory_idx = nullptr, float min_offset = 0.0f, 
            unsigned char* bias = nullptr, size_t bias_step = 0) override;
    /*
    * @brief 加载量化码本
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    int load_codebooks(const std::string& codebook_file) override;
    /*
    * @brief 直接加载文件数据使用
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    //int load_codebooks(char * fileBaseAddress, uint64_t fileLength);
    /*
    * @brief 获取配置信息
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    const QuantizationParams& get_quantization_params() const override {
        return _params;
    }
    /*
     * @brief 写码本文件
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    int save_codebooks(const std::string& file_name) const override;
    /*
     * @brief 写索引文件(建库的产出，与建库样本相关)
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    int save_index(const std::string& file_name) const override;
    int save_index_bak(const std::string& file_name) const override;

    void reset_quantized_feature() override {
        _quantized_feature.reset(nullptr);
    }

    void reset() override {
        if(_isUniquePtr) {
            _quantized_feature.reset(nullptr);
            _quantized_feature_packed.reset(nullptr);
            _fea_offset_packed.reset(nullptr);
            _codebooks.reset(nullptr);
        }else{
            _quantized_feature.release();
            _quantized_feature_packed.release();
            _fea_offset_packed.release();
            _codebooks.release();
        }
    }

    std::unique_ptr<float[]> decode_packed(uint64_t idx) override;
    
    int load_quantized_feature_address(char * indexFileBaseAddress, uint64_t indexFileLength, 
            char * offsetFileBaseAddress, uint64_t offsetFileLength) override;
    int load_codebooks(char * fileBaseAddress, uint64_t fileLength) override;
private:
    /*
    * @brief 计算码本长度
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    size_t get_codebook_length() override {
        //LOG(INFO) << "get_codebook_length = " << _params.nsq* _params.ks* _params.lsq* sizeof(float);
        return _params.nsq * _params.ks * _params.lsq * sizeof(float);
    }
    /*
    * @brief 计算量化特征长度
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    size_t get_quantized_feature_length() override {
        return _per_fea_len * _total_point_count * sizeof(unsigned char);
    }
    /*
    * @brief decode
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    std::unique_ptr<float[]> decode(uint64_t idx) const override;

    
    /*
    * @brief 加载索引
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    int load_quantized_feature(const std::string& quantized_feafile,
                            const uint32_t* cnts_index,
                            unsigned char* bias = nullptr,
                            size_t bias_step = 0) override;

    int load_quantized_feature(const std::string& quantized_feafile,
                            std::vector<std::vector<uint32_t>>& datapoints_by_token,
                            const uint32_t* cnts_index) override;
public:
    /*
    * @brief 初始化码本所需内存
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    int init_codebooks_memory() override;
private:
    const int _per_subspace_len;
    const int _fea_offset;
    uint32_t _total_point_count;
    uint32_t _total_copy_num;
    int _per_fea_len;
    float _min_offset;
    QuantizationParams _params;
    std::unique_ptr<float[]> _codebooks;
    std::vector<std::vector<uint8_t>> _quantized_feature_packed_vec;
    std::vector<std::vector<float>> _fea_offset_packed_vec;
    std::unique_ptr<unsigned char[]> _quantized_feature;
    std::unique_ptr<unsigned char[]> _quantized_feature_packed;
    std::unique_ptr<float[]> _fea_offset_packed;
};

}//namespace puck