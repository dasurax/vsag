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
 * @file quantization_4bit_packed.cpp
 * @author suguan.dx
 * @date 2024/08/06 14:39
 * @brief
 *
 **/
#include<thread>
#include <numeric>
#include <fstream>
#include <functional>
#include <unistd.h>
#include <cblas.h>
#include <limits>
#include "algorithm/puck/quantization_4bit_packed.h"
#include "algorithm/puck/hierarchical_cluster_index.h"
#include <glog/logging.h>
namespace faiss {
void fvec_L2sqr_ny(float* dis, const float* x,
                   const float* y, size_t d, size_t ny);
};

namespace puck {

Quantization4BitPacked::Quantization4BitPacked(const QuantizationParams& params,
                           uint32_t point_count) : //Quantization(params, point_count),
    
    _per_subspace_len(sizeof(char)),
    _fea_offset(std::ceil(1.0 * sizeof(float) / _per_subspace_len)) {
    LOG(INFO) << "Quantization4BitPacked " << params.ks;
    _params = params;
    reset();

    LOG(INFO) << "###init in 4bit: lsq: " << _params.lsq << " nsq: " << _params.nsq << " ks: " << _params.ks;

    _quantized_feature_packed_vec.resize(FLAGS_coarse_cluster_count * FLAGS_fine_cluster_count);
    _fea_offset_packed_vec.resize(FLAGS_coarse_cluster_count * FLAGS_fine_cluster_count);
    if (_params.ks != 256 && _params.ks != 16) {
        LOG(ERROR) << "_params.ks != 256 and 16";
        return;
    }

    _total_point_count = point_count;
    _per_fea_len = _params.nsq * _per_subspace_len;
    _per_fea_len += _fea_offset;
    LOG(INFO) << "_per_fea_len=" << _per_fea_len;
    LOG(INFO) << "_per_subspace_len=" << _per_subspace_len;
    LOG(INFO) << "_params.nsq=" << _params.nsq;
    //init_codebooks_memory();
}

struct ThreadReaderParam {
    std::string file_name;
    uint32_t thread_idx;
    uint32_t start_id;
    uint32_t count;
    uint32_t read_len;
    uint64_t file_offset;
    uint32_t read_offset;
};

void load_bytes_array_thread(size_t dim,
                            float* codebooks,
                            const ThreadReaderParam reader, 
                            unsigned char* code_bytes,
                            unsigned char* code_bytes_packed,
                            uint8_t* code_bytes_packed_hi,
                            float* fea_offset_packed,
                            const uint32_t* local_2memory_idx,
                            const uint32_t nsq,
                            const int fea_offset,
                            float min_offset,
                            unsigned char* bias = nullptr,
                            size_t bias_step = 0) {
    std::ifstream input_file(reader.file_name.c_str(), std::ios::binary | std::ios::in);
    u_int64_t curr_file_offset = (u_int64_t)reader.start_id * reader.read_len + reader.file_offset;
    input_file.seekg(curr_file_offset);
    //LOG(ERROR) << "reader.start_id: " << reader.start_id << " " << reader.start_id + reader.count;
    for (uint32_t i = 0; i < reader.count; ++i) {
        uint32_t memory_idx = reader.start_id + i;
        //LOG(ERROR) << "memory_idx: " << memory_idx;
        if (local_2memory_idx != nullptr) {
            memory_idx = local_2memory_idx[reader.start_id + i];
        }

        u_int64_t pq_curr_offset = (u_int64_t)memory_idx * (reader.read_len + reader.read_offset) +
                                   reader.read_offset;
        input_file.read((char*)(code_bytes + pq_curr_offset), reader.read_len);
        //if (memory_idx % 10000 == 0)
        //LOG(INFO) << "memory_idx: " << memory_idx << " " << fea_offset_packed[1252951] << " " << (fea_offset_packed == nullptr) << " " << reader.count;
        if (FLAGS_use_pq_method_0 && !FLAGS_convert_local_to_memory_while_build) {
        //if (FLAGS_use_pq_method_0) {
            unsigned char* pq_feature = code_bytes + pq_curr_offset + fea_offset;
            auto lsq = dim / nsq;
            std::unique_ptr<float[]> residual(new float[dim]);

            for (uint32_t m = 0; m < nsq; ++m) {
                const float* codeword = codebooks + m * 16 * lsq + pq_feature[m] * lsq;

                for (uint32_t d = 0; d < lsq; ++d) {
                    residual.get()[m * lsq + d] = codeword[d];
                }
            }
            auto norm = cblas_sdot(dim, residual.get(), 1, residual.get(), 1);
            auto ori = ((float*)(code_bytes + pq_curr_offset))[0];
            if (bias != nullptr) {
                auto cur_bias = ((float*)(bias + (u_int64_t)memory_idx * bias_step))[0];
                ((float*)(code_bytes + pq_curr_offset))[0] += cur_bias;
                //LOG(INFO) << ori << " " << cur_bias << " " << " " << bias_step << " " << ((float*)(code_bytes + pq_curr_offset))[0]  << " " << ((float*)(code_bytes + pq_curr_offset))[0]  + norm;
            }
            
            ((float*)(code_bytes + pq_curr_offset))[0] += norm;
         
        }
        
        fea_offset_packed[memory_idx] = ((float*)(code_bytes + pq_curr_offset))[0] - min_offset;

        //if (min_offset < 0)
          //  LOG(INFO) << "offset: " << ((float*)(code_bytes + pq_curr_offset))[0] << " " << min_offset << " " << fea_offset_packed[memory_idx];
        // create packed dataset
        for (auto j = 0; j < nsq; j++) {
            
            auto ptr = code_bytes + pq_curr_offset + fea_offset + j;
            int hi = (memory_idx % 32) / 16; 
            uint8_t v = *((uint8_t*)ptr);
          
            u_int64_t k = memory_idx / 32;
            u_int64_t start = k * 16 * nsq;

            if (hi) {
                v *= 16;
                //if (193536 <= start + j * 16 + memory_idx % 16)
                  //  LOG(ERROR) << "start + j * 16 + memory_idx % 16: " << start + j * 16 + memory_idx % 16;
                code_bytes_packed_hi[start + j * 16 + memory_idx % 16] = v;
            } else {
                ((uint8_t*)code_bytes_packed)[start + j * 16 + memory_idx % 16] = v;
            }
             
        }

        /*
        auto idx = memory_idx;
        auto ptr = code_bytes + pq_curr_offset + fea_offset;
        for (size_t j = 0; j < nsq; j++) {
            u_int64_t k = idx / 32;
            u_int64_t start = k * 16 * nsq;
            int hi = (idx % 32) / 16; 
            uint8_t v = code_bytes_packed[start + j * 16 + idx % 16];
            auto ori = ((uint8_t*)ptr)[j];
            uint8_t newv = 0;
            if (hi) {
                newv = v >> 4;
            } else {
                newv = v & 0x0F;
            }
            //if (idx == 16) {
                if ((int)ori != (int)newv)
                    LOG(INFO) << idx << " " << j << " " << (int)ori << " " << (int)newv << " " << start + j * 16 + idx % 16 << " " << (int)v;
            //}
            // ptr[i] = 0;
        }
        */

        if ((i + 1) % 1000000 == 0) {
            LOG(INFO) << "loading index file " << reader.file_name << " thread " << reader.thread_idx << " processed "
                      << 1.0 * i / reader.count;
        }
    }

    input_file.close();
}

void load_bytes_array_thread2(const ThreadReaderParam reader, 
                            unsigned char* code_bytes,
                            unsigned char* code_bytes_packed,
                            uint8_t* code_bytes_packed_hi,
                            float* fea_offset_packed,
                            const uint32_t* local_2memory_idx,
                            const uint32_t nsq,
                            const int fea_offset,
                            float min_offset,
                            std::vector<std::vector<uint32_t>>& datapoints_by_token,
                            std::vector<std::vector<uint8_t>>& quantized_feature_packed_vec,
                            std::vector<std::vector<float>>& fea_offset_packed_vec
                        ) {
    std::ifstream input_file(reader.file_name.c_str(), std::ios::binary | std::ios::in);
    u_int64_t curr_file_offset = (u_int64_t)reader.start_id * reader.read_len + reader.file_offset;
    input_file.seekg(curr_file_offset);
    // LOG(INFO) << reader.start_id << " " << reader.start_id + reader.count;
    
    
    for (uint32_t i = 0; i < reader.count; ++i) {
        uint32_t memory_idx = reader.start_id + i;

        if (local_2memory_idx != nullptr) {
            memory_idx = local_2memory_idx[reader.start_id + i];
        }

        u_int64_t pq_curr_offset = (u_int64_t)memory_idx * (reader.read_len + reader.read_offset) +
                                   reader.read_offset;
        input_file.read((char*)(code_bytes + pq_curr_offset), reader.read_len);
        //if (memory_idx % 10000 == 0)
        //LOG(INFO) << "memory_idx: " << memory_idx << " " << fea_offset_packed[1252951] << " " << (fea_offset_packed == nullptr) << " " << reader.count;
        fea_offset_packed[memory_idx] = ((float*)(code_bytes + pq_curr_offset))[0] - min_offset;
        //if (min_offset < 0)
          //  LOG(INFO) << "offset: " << ((float*)(code_bytes + pq_curr_offset))[0] << " " << min_offset << " " << fea_offset_packed[memory_idx];
        // create packed dataset
        for (auto j = 0; j < nsq; j++) {
            
            auto ptr = code_bytes + pq_curr_offset + fea_offset + j;
            int hi = (memory_idx % 32) / 16; 
            uint8_t v = *((uint8_t*)ptr);
          
            u_int64_t k = memory_idx / 32;
            u_int64_t start = k * 16 * nsq;

            if (hi) {
                v *= 16;
                code_bytes_packed_hi[start + j * 16 + memory_idx % 16] = v;
            } else {
                ((uint8_t*)code_bytes_packed)[start + j * 16 + memory_idx % 16] = v;
            }
             
        }

        /*
        auto idx = memory_idx;
        auto ptr = code_bytes + pq_curr_offset + fea_offset;
        for (size_t j = 0; j < nsq; j++) {
            u_int64_t k = idx / 32;
            u_int64_t start = k * 16 * nsq;
            int hi = (idx % 32) / 16; 
            uint8_t v = code_bytes_packed[start + j * 16 + idx % 16];
            auto ori = ((uint8_t*)ptr)[j];
            uint8_t newv = 0;
            if (hi) {
                newv = v >> 4;
            } else {
                newv = v & 0x0F;
            }
            //if (idx == 16) {
                if ((int)ori != (int)newv)
                    LOG(INFO) << idx << " " << j << " " << (int)ori << " " << (int)newv << " " << start + j * 16 + idx % 16 << " " << (int)v;
            //}
            // ptr[i] = 0;
        }
        */

        if ((i + 1) % 1000000 == 0) {
            LOG(INFO) << "loading index file " << reader.file_name << " thread " << reader.thread_idx << " processed "
                      << 1.0 * i / reader.count;
        }
    }

    input_file.close();

    
    std::vector<char> bytes(reader.read_len * reader.count, 0);
    std::ifstream input_file2(reader.file_name.c_str(), std::ios::binary | std::ios::in);
    input_file2.read(bytes.data(), reader.read_len * reader.count);
 
 
    // _quantized_feature_packed_vec
    // _fea_offset_packed_vec
    
    for (size_t i = 0; i < datapoints_by_token.size(); i++) {
        if (i == 1025) LOG(INFO) << i << " " << datapoints_by_token[i].size();
        auto& quantized_feature_packed = quantized_feature_packed_vec[i];
        auto& fea_offset_packed = fea_offset_packed_vec[i];
        size_t s = nsq * (((u_int64_t)datapoints_by_token[i].size() + 31) & (~31)) / 2;
        size_t s2 = (((u_int64_t)datapoints_by_token[i].size() + 31) & (~31));
        quantized_feature_packed.resize(s, 0);
        fea_offset_packed.resize(s2, 0);
        for (size_t memory_idx = 0; memory_idx < datapoints_by_token[i].size(); memory_idx++) {
            
            size_t local_idx = datapoints_by_token[i][memory_idx];
            auto offset_ptr = bytes.data() + reader.read_len * local_idx;
            fea_offset_packed[memory_idx] = (((float*)offset_ptr)[0]);
            
            for (auto j = 0; j < nsq; j++) {
                auto ptr = offset_ptr + fea_offset + j;
                int hi = (memory_idx % 32) / 16; 
                uint8_t v = *((uint8_t*)ptr);
            
                u_int64_t k = memory_idx / 32;
                u_int64_t start = k * 16 * nsq;

                if (hi) {
                    v *= 16;
                } 
                quantized_feature_packed[start + j * 16 + memory_idx % 16] += v;
            }
            
        }
        
    }
    LOG(INFO) << 1025 << " " << datapoints_by_token[1025].size() << " " << quantized_feature_packed_vec[1025].size();
    LOG(INFO) << 1025 << " " << datapoints_by_token[1025].size();
    
}

bool check_file_length_info(const std::string& file_name,
                            const uint64_t file_length);

int Quantization4BitPacked::init_codebooks_memory() {
    u_int64_t pq_codebook_length = (u_int64_t)_params.nsq * _params.ks * _params.lsq;
    _codebooks.reset(new float[pq_codebook_length]);
    LOG(ERROR) << "init_codebooks_memory: " << pq_codebook_length;
    if (_codebooks.get() == nullptr) {
        LOG(FATAL) << "malloc memory Quantization4BitPacked codebooks " << pq_codebook_length << " error.";
        return -1;
    }

    return 0;
}

int Quantization4BitPacked::load_codebooks(const std::string& codebook_file) {
    if(init_codebooks_memory() != 0) {
        return -1;
    }
    LOG(INFO) << "--------------- === ===== load_codebook: " << codebook_file;
    //int ret = fvecs_read(codebook_file.c_str(), _params.lsq,
    //                     _params.nsq * _params.ks, _codebooks.get());
    int ret = read_fvec_format(codebook_file.c_str(), _params.lsq,
                               _params.nsq * _params.ks, _codebooks.get());
    if (ret != int(_params.nsq * _params.ks)) {
        LOG(FATAL) << "load file error, file : " << codebook_file << " feature_dim : " <<
                   _params.lsq
                   << " number : " << _params.nsq* _params.ks << " return code : " << ret;
        return -1;
    }

    return 0;
}

int Quantization4BitPacked::load_codebooks(char * fileBaseAddress, uint64_t fileLength) {
   if (fileLength != int(_params.nsq * _params.ks * _params.lsq) * sizeof(float)) {
        LOG(ERROR) << "file length in 4bit error: fileLength: " << fileLength << " lsq: " <<
                   _params.lsq << " nsq: " << _params.nsq << " ks: " << _params.ks;
        return -1;
    }
    _codebooks.reset((float*) fileBaseAddress);

    _isUniquePtr = false;
    return 0;
}

int Quantization4BitPacked::init_quantized_feature_memory() {
    uint64_t pq_feature_length = (u_int64_t)_total_point_count * _per_fea_len;
    uint64_t pq_feature_packed_length = 
            _params.nsq * (((u_int64_t)_total_point_count + 31) & (~31)) / 2;
    uint64_t offset_length = (u_int64_t)_total_point_count * _fea_offset;
    
    LOG(INFO) << "init quantized feature memory, length = " << pq_feature_length << " " << _total_point_count;
    int32_t pagesize = getpagesize();
    void* memb = nullptr;
    pq_feature_length =  pq_feature_length + (pagesize - pq_feature_length % pagesize);
    int err = posix_memalign(&memb, pagesize, pq_feature_length);
    if (err != 0) {
        std::runtime_error("alloc_aligned_mem_failed errno=" + errno);
        return -1;
    }

    void* memb_packed = nullptr;
    pq_feature_packed_length =  pq_feature_packed_length + (pagesize - pq_feature_packed_length % pagesize);
    err = posix_memalign(&memb_packed, pagesize, pq_feature_packed_length);
    if (err != 0) {
        std::runtime_error("alloc_aligned_mem_failed errno=" + errno);
        return -1;
    }


    void* offset_memb_packed = nullptr;
    offset_length =  offset_length + (pagesize - offset_length % pagesize);
    err = posix_memalign(&offset_memb_packed, pagesize, offset_length);
    if (err != 0) {
        std::runtime_error("alloc_aligned_mem_failed errno=" + errno);
        return -1;
    }
    LOG(ERROR) << "init_quantized_feature_memory: " << pq_feature_length + pq_feature_packed_length + offset_length;
    LOG(INFO) << "init_quantized_feature_memory: " << (pq_feature_length + pq_feature_packed_length + offset_length) / (1024.0);
    LOG(INFO) << "init_quantized_feature_memory: " << (pq_feature_length) / (1024.0);
    LOG(INFO) << "init_quantized_feature_memory: " << (pq_feature_packed_length) / (1024.0);
    LOG(INFO) << "init_quantized_feature_memory: " << (offset_length) / (1024.0);

    LOG(INFO) << "pq_feature_packed_length: " << pq_feature_packed_length 
            << " " << _total_point_count << " " << _params.nsq
            << " " << (((u_int64_t)_total_point_count + 31) & (~31))
            << " " << ((_total_point_count + 31) & (~31))
            << " " << offset_length;
    _quantized_feature.reset(reinterpret_cast<unsigned char*>(memb));
    _quantized_feature_packed.reset(reinterpret_cast<unsigned char*>(memb_packed));
    _fea_offset_packed.reset(reinterpret_cast<float*>(offset_memb_packed));

    if (_quantized_feature.get() == nullptr) {
        LOG(FATAL) << "malloc memory quantized feature vector " << pq_feature_length << " error.";
        return -1;
    }
    if (_quantized_feature_packed.get() == nullptr) {
        LOG(FATAL) << "malloc memory quantized feature vector " << pq_feature_packed_length << " error.";
        return -1;
    }
    if (_fea_offset_packed.get() == nullptr) {
        LOG(FATAL) << "malloc memory quantized feature vector " << pq_feature_packed_length << " error.";
        return -1;
    }
    memset(_quantized_feature.get(), 0, pq_feature_length);
    memset(_quantized_feature_packed.get(), 0, pq_feature_packed_length);
    memset(_fea_offset_packed.get(), 0, offset_length);
    return 0;
}

/*
vector<uint8_t> CreatePackedDataset(
    const DenseDataset<uint8_t>& hashed_database) {
  vector<uint8_t> packed_dataset;
  if (hashed_database.empty()) {
    return packed_dataset;
  }

  DimensionIndex num_blocks = hashed_database[0].nonzero_entries();
  packed_dataset.resize(num_blocks * ((hashed_database.size() + 31) & (~31)) /
                        2);
  DatapointIndex k = 0;
  for (; k < hashed_database.size() / 32; ++k) {
    size_t start = k * 16 * num_blocks;
    for (size_t j = 0; j < num_blocks; ++j) {
      for (size_t m = 0; m < 16; m++) {
        uint8_t u0 = hashed_database[k * 32 + m].values()[j];
        uint8_t u1 = hashed_database[k * 32 + m + 16].values()[j];
        packed_dataset[start + j * 16 + m] = u1 * 16 + u0;
      }
    }
  }

  if (k * 32 < hashed_database.size()) {
    size_t start = k * 16 * num_blocks;
    for (size_t j = 0; j < num_blocks; ++j) {
      for (size_t m = 0; m < 16; m++) {
        DatapointIndex dp_idx = k * 32 + m;
        dp_idx = dp_idx >= hashed_database.size() ? (hashed_database.size() - 1)
                                                  : dp_idx;
        uint8_t u0 = hashed_database[dp_idx].values()[j];

        dp_idx = k * 32 + m + 16;
        dp_idx = dp_idx >= hashed_database.size() ? (hashed_database.size() - 1)
                                                  : dp_idx;
        uint8_t u1 = hashed_database[dp_idx].values()[j];
        packed_dataset[start + j * 16 + m] = u1 * 16 + u0;
      }
    }
  }

  return packed_dataset;
}
*/

int Quantization4BitPacked::load_quantized_feature_address(char * indexFileBaseAddress, uint64_t indexFileLength, 
                    char * offsetFileBaseAddress, uint64_t offsetFileLength) {
    if (_params.ks != 256 && _params.ks != 16) {
        LOG(FATAL) << "only for ks = 256 or ks = 16";
        return -1;
    }

    uint64_t pq_feature_packed_length = _params.nsq * (((u_int64_t)_total_point_count + 31) & (~31)) / 2;
    if(pq_feature_packed_length != indexFileLength) {
        LOG(ERROR) << "quantization4bit load feature error: fileLength: " << indexFileLength << " expected: " << pq_feature_packed_length;
        return -1;
    }
    _quantized_feature_packed.reset(reinterpret_cast<unsigned char*>(indexFileBaseAddress));

    uint64_t offset_length = (u_int64_t)_total_point_count * _fea_offset;
    if(offset_length != offsetFileLength) {
        LOG(ERROR) << "quantization4bit load offset error: fileLength: " << offsetFileLength << " expected: " << offset_length;
        return -1;
    }
    _fea_offset_packed.reset(reinterpret_cast<float*>(offsetFileBaseAddress));
    
    _isUniquePtr = false;
    return 0;
}

int Quantization4BitPacked::load_quantized_feature(const std::string& quantization_file,
        const uint32_t* local_2memory_idx,
        unsigned char* bias,
        size_t bias_step) {

    
    if (_params.ks != 256 && _params.ks != 16) {
        LOG(FATAL) << "only for ks = 256 or ks = 16";
        return -1;
    }

    LOG(INFO) << "load quantized feature " << quantization_file << "; _total_point_count = " <<
              _total_point_count << " " << _params.ks;
    constexpr size_t kMinMin =
        std::numeric_limits<int16_t>::min() / std::numeric_limits<int8_t>::min();
    constexpr size_t kMaxMax =
        std::numeric_limits<int16_t>::max() / std::numeric_limits<int8_t>::max();
    constexpr size_t kGuaranteedToWork = (kMaxMax < kMinMin) ? kMaxMax : kMinMin;
    LOG(INFO) << "limit: " << kMinMin << " " << kMaxMax << " " << kGuaranteedToWork;
    //文件长度检查
    u_int64_t pq_feature_length = (u_int64_t)_total_point_count * _per_fea_len;
    bool is_ok = check_file_length_info(quantization_file, pq_feature_length);

    if (is_ok == false) {
        return -1;
    }

    if (true) {
        //return 0;
    }

    LOG(INFO) << "_quantized_feature == nullptr: " << (_quantized_feature == nullptr);
    if (_quantized_feature == nullptr && init_quantized_feature_memory() != 0) {
        return -1;
    }

    if (false) {
        auto quantization_packed_file = quantization_file + "_packed";
        auto quantization_packed_offset_file = quantization_file + "_packed_fea_offset";
        uint64_t pq_feature_packed_length = 
            _params.nsq * (((u_int64_t)_total_point_count + 31) & (~31)) / 2;
        uint64_t offset_length = (u_int64_t)_total_point_count * _fea_offset;
        std::ifstream input_file(quantization_packed_file.c_str(), std::ios::binary | std::ios::in);
        input_file.read((char*)(_quantized_feature_packed.get()), pq_feature_packed_length);

        LOG(ERROR) << "load_quantized_feature direct: " << input_file.gcount() << " " << input_file.fail();
        std::ifstream input_file2(quantization_packed_offset_file.c_str(), std::ios::binary | std::ios::in);
        input_file2.read((char*)(_fea_offset_packed.get()), offset_length);

        return 0;
    }

    //多线程加载文件
    uint32_t threads_count = std::thread::hardware_concurrency();
    // threads_count = 85;
    uint32_t per_thread_count = ceil(1.0 * _total_point_count / threads_count);
    std::vector<std::thread> threads;
    ThreadReaderParam params;
    params.file_name = quantization_file;
    params.read_len = _per_fea_len;
    params.file_offset = 0;
    params.read_offset = 0;
    LOG(INFO) << "params.read_len = " << params.read_len << " " << params.read_offset << " " <<
              (local_2memory_idx == nullptr);

    // avoid parallel writes
    uint64_t pq_feature_packed_length = 
            _params.nsq * (((u_int64_t)_total_point_count + 31) & (~31)) / 2;
    std::vector<uint8_t> quantized_feature_packed_hi(pq_feature_packed_length, 0);
    LOG(INFO) << "pq_feature_packed_length: " << pq_feature_packed_length << " _params.nsq: " << _params.nsq;
    for (uint32_t thread_id = 0; thread_id < threads_count; ++thread_id) {
        params.start_id = thread_id * per_thread_count;
        if (params.start_id >= _total_point_count) {
            params.count = 0;
        } else {
            params.count = std::min(per_thread_count, _total_point_count - params.start_id);
        }
        params.thread_idx = thread_id;
        float* codebooks = get_codebooks();
        //LOG(ERROR) << "params.start_id: " << params.start_id << " " << params.count << " " << _total_point_count;
        threads.push_back(std::thread(std::bind(load_bytes_array_thread,
                                                _params.dim,
                                                codebooks,
                                                params, 
                                                _quantized_feature.get(),
                                                _quantized_feature_packed.get(),
                                                quantized_feature_packed_hi.data(),
                                                _fea_offset_packed.get(),
                                                local_2memory_idx,
                                                _params.nsq,
                                                _fea_offset,
                                                _min_offset,
                                                bias,
                                                bias_step)));
    }

    for (uint32_t thread_id = 0; thread_id < threads_count; ++thread_id) {
        threads[thread_id].join();
    }

     LOG(INFO) << "finish load_bytes_array_thread";

    /*
    LOG(INFO) << "load_index " << quantization_file;
    for (size_t i = 0; i < _total_point_count; i++) {
        unsigned char* pq_feature = get_quantized_feature(i);
        pq_feature += _fea_offset;
        //LOG(INFO) << i << " " << _total_point_count;
        for (uint32_t m = 0; m < _params.nsq; ++m) {
            if (pq_feature[m] >= 16) {
                LOG(INFO) << "============ pq_feature[m] >= 16 " << pq_feature[m] << " ";
            }
        }
    }
    */

    for (size_t i = 0; i < pq_feature_packed_length; i++) {
        ((uint8_t*)_quantized_feature_packed.get())[i] += quantized_feature_packed_hi[i];
    }
    
    float* offset = (float*)get_quantized_feature(0);
    LOG(INFO) << "offset: " << *offset;
    LOG(INFO) << "load_index " << quantization_file;
    LOG(INFO) << "load quantized feature " << quantization_file << " suc.";
    return 0;
}

int Quantization4BitPacked::load_quantized_feature(const std::string& quantization_file,
        std::vector<std::vector<uint32_t>>& datapoints_by_token,
        const uint32_t* local_2memory_idx) {
    if (_params.ks != 256 && _params.ks != 16) {
        LOG(FATAL) << "only for ks = 256 or ks = 16";
        return -1;
    }

    LOG(INFO) << "load quantized feature " << quantization_file << "; _total_point_count = " <<
              _total_point_count;
    constexpr size_t kMinMin =
        std::numeric_limits<int16_t>::min() / std::numeric_limits<int8_t>::min();
    constexpr size_t kMaxMax =
        std::numeric_limits<int16_t>::max() / std::numeric_limits<int8_t>::max();
    constexpr size_t kGuaranteedToWork = (kMaxMax < kMinMin) ? kMaxMax : kMinMin;
    LOG(INFO) << "limit: " << kMinMin << " " << kMaxMax << " " << kGuaranteedToWork;
    //文件长度检查
    u_int64_t pq_feature_length = (u_int64_t)_total_point_count * _per_fea_len;
    bool is_ok = check_file_length_info(quantization_file, pq_feature_length);

    if (is_ok == false) {
        return -1;
    }

    if (true) {
        //return 0;
    }

    if (init_quantized_feature_memory() != 0) {
        return -1;
    }

    //多线程加载文件
    uint32_t threads_count = std::thread::hardware_concurrency();
    threads_count = 1;
    uint32_t per_thread_count = ceil(1.0 * _total_point_count / threads_count);
    std::vector<std::thread> threads;
    ThreadReaderParam params;
    params.file_name = quantization_file;
    params.read_len = _per_fea_len;
    params.file_offset = 0;
    params.read_offset = 0;
    LOG(INFO) << "params.read_len = " << params.read_len << " " << params.read_offset << " " <<
              (local_2memory_idx == nullptr);

    // avoid parallel writes
    uint64_t pq_feature_packed_length = 
            _params.nsq * (((u_int64_t)_total_point_count + 31) & (~31)) / 2;
    std::vector<uint8_t> quantized_feature_packed_hi(pq_feature_packed_length, 0);

    for (uint32_t thread_id = 0; thread_id < threads_count; ++thread_id) {
        params.start_id = thread_id * per_thread_count;
        params.count = std::min(per_thread_count, _total_point_count - params.start_id);
        params.thread_idx = thread_id;
        threads.push_back(std::thread(std::bind(load_bytes_array_thread2,
                                                params, 
                                                _quantized_feature.get(),
                                                _quantized_feature_packed.get(),
                                                quantized_feature_packed_hi.data(),
                                                _fea_offset_packed.get(),
                                                local_2memory_idx,
                                                _params.nsq,
                                                _fea_offset,
                                                _min_offset,
                                                datapoints_by_token,
                                                std::ref(_quantized_feature_packed_vec),
                                                std::ref(_fea_offset_packed_vec))));
                                                
    }



    for (uint32_t thread_id = 0; thread_id < threads_count; ++thread_id) {
        threads[thread_id].join();
    }
    LOG(INFO) << "_quantized_feature_packed_vec: " << _quantized_feature_packed_vec[0].size();
    LOG(INFO) << "_quantized_feature_packed_vec: " << _fea_offset_packed_vec[1025].size();
    /*
    LOG(INFO) << "load_index " << quantization_file;
    for (size_t i = 0; i < _total_point_count; i++) {
        unsigned char* pq_feature = get_quantized_feature(i);
        pq_feature += _fea_offset;
        //LOG(INFO) << i << " " << _total_point_count;
        for (uint32_t m = 0; m < _params.nsq; ++m) {
            if (pq_feature[m] >= 16) {
                LOG(INFO) << "============ pq_feature[m] >= 16 " << pq_feature[m] << " ";
            }
        }
    }
    */
    for (size_t i = 0; i < pq_feature_packed_length; i++) {
        ((uint8_t*)_quantized_feature_packed.get())[i] += quantized_feature_packed_hi[i];
    }

    float* offset = (float*)get_quantized_feature(0);
    LOG(INFO) << "offset: " << *offset;
    LOG(INFO) << "load_index " << quantization_file;
    LOG(INFO) << "load quantized feature " << quantization_file << " suc.";
    return 0;
}

int Quantization4BitPacked::load(const std::string& codebook_file, const std::string& quantized_feafile,
                        const uint32_t* local_2memory_idx, float min_offset, unsigned char* bias, size_t bias_step) {
    _min_offset = min_offset;
    LOG(INFO) << "_min_offset: " << _min_offset << " " << min_offset;
    if (_params.ks != 256 && _params.ks != 16) {
        LOG(FATAL) << "only for ks = 256 or ks = 16";
        return -1;
    }

    if (load_codebooks(codebook_file) != 0) {
        return -1;
    }

    if (local_2memory_idx != nullptr) {
        if (load_quantized_feature(quantized_feafile, local_2memory_idx, bias, bias_step) != 0) {
            return -1;
        }
    } else {
        std::vector<uint32_t> temp_cnts_index(_total_point_count);
        std::iota(temp_cnts_index.begin(), temp_cnts_index.end(), 0);

        if (load_quantized_feature(quantized_feafile, temp_cnts_index.data(), bias, bias_step) != 0) {
            return -1;
        }
    }
    return 0;
}

int Quantization4BitPacked::load(const std::string& codebook_file, const std::string& quantized_feafile,
                        std::vector<std::vector<uint32_t>>& datapoints_by_token,
                        const uint32_t* local_2memory_idx, float min_offset) {
    _min_offset = min_offset;
    LOG(INFO) << "_min_offset: " << _min_offset << " " << min_offset;
    if (_params.ks != 256 && _params.ks != 16) {
        LOG(FATAL) << "only for ks = 256 or ks = 16";
        return -1;
    }

    if (load_codebooks(codebook_file) != 0) {
        return -1;
    }

    if (local_2memory_idx != nullptr) {
        if (load_quantized_feature(quantized_feafile, datapoints_by_token, local_2memory_idx) != 0) {
            return -1;
        }
    } else {
        std::vector<uint32_t> temp_cnts_index(_total_point_count);
        std::iota(temp_cnts_index.begin(), temp_cnts_index.end(), 0);

        if (load_quantized_feature(quantized_feafile, datapoints_by_token, temp_cnts_index.data()) != 0) {
            return -1;
        }
    }

    return 0;
}

std::unique_ptr<float[]> Quantization4BitPacked::decode(uint64_t idx) const {
    unsigned char* pq_feature = get_quantized_feature(idx);
    pq_feature += _fea_offset;
    std::unique_ptr<float[]> residual(new float[_params.nsq * _params.lsq]);
    for (uint32_t m = 0; m < _params.nsq; ++m) {
        const float* codeword = get_codebooks() + m * _params.ks * _params.lsq + pq_feature[m] * _params.lsq;

        for (uint32_t d = 0; d < _params.lsq; ++d) {
            residual.get()[m * _params.lsq + d] = codeword[d];
        }
    }

    return residual;
}

std::unique_ptr<float[]> Quantization4BitPacked::decode_packed(uint64_t idx) {
    std::vector<uint8_t> pq_feature_vec(_params.nsq);
    get_quantized_feature(idx, pq_feature_vec.data());
 
    std::unique_ptr<float[]> residual(new float[_params.nsq * _params.lsq]);
    const unsigned char* pq_feature = (unsigned char*)pq_feature_vec.data();

    for (uint32_t m = 0; m < _params.nsq; ++m) {
        const float* codeword = get_codebooks() + m * _params.ks * _params.lsq + pq_feature[m] * _params.lsq;

        for (uint32_t d = 0; d < _params.lsq; ++d) {
            residual.get()[m * _params.lsq + d] = codeword[d];
        }
    }

    return residual;
}


int Quantization4BitPacked::set_static_value_of_formula(uint64_t idx, float* vocab) {
    //LOG(ERROR) << "set_static_value_of_formula";
    if (idx >= _total_point_count || !vocab) {
        return -1;
    }

    float* static_dist = (float*)get_quantized_feature(idx);
    std::unique_ptr<float[]> residual = decode(idx);
    static_dist[0] = 2 * cblas_sdot(_params.dim, residual.get(), 1, vocab, 1);

    if (FLAGS_use_pq_method_0 && FLAGS_convert_local_to_memory_while_build) {
        auto lsq = _params.dim / _params.nsq;
      
        auto norm = cblas_sdot(_params.dim, residual.get(), 1, residual.get(), 1);
        static_dist[0] += norm;

    }

    for (uint32_t idx = 0; idx < _params.dim; ++idx) {
        vocab[idx] += residual[idx];
    }

    return 0;
}

int Quantization4BitPacked::get_ip_dist_table(const float* feature, float* dist_table) const {
    float totalf = 0.0;
    //LOG(INFO) << "get_ip_dist_table _params.nsq: " << _params.nsq;
    for (uint32_t m = 0; m < _params.nsq; ++m) {
        matrix_multiplication(_codebooks.get() + (u_int64_t)m * _params.ks * _params.lsq, 
                            feature + m * _params.lsq,
                            _params.ks, 1, _params.lsq,
                            "TN", 
                            dist_table + m * _params.ks);
        for (size_t i = 0; i < _params.ks; i++) {
            
            dist_table[m * _params.ks + i] *= -2;
        }
        /*
        LOG(INFO) << "get_ip_dist_table m: " << m 
                << " _params.nsq: " << _params.nsq
                << " _params.lsq: " << _params.lsq
                << " _params.ks: " << _params.ks
                << " dist_table: " << dist_table[m * _params.ks + 0] << " " << dist_table[m * _params.ks + 1]
                << " feature: " << feature[m * _params.lsq + 0] << " " << feature[m * _params.lsq + 1]
                << " _codebooks: " << _codebooks.get()[m * _params.ks * _params.lsq + 0] << " " << _codebooks.get()[m * _params.ks * _params.lsq + 1];
        */
    }
    //LOG(INFO) << "totalf: " << totalf << " " << -2 * totalf;
    return 0;
}

int Quantization4BitPacked::get_dist_table(const float* feature, float* dist_table) const {
    //LOG(INFO) << "get_dist_table _params.nsq: " << _params.nsq;
    for (uint32_t m = 0; m < _params.nsq; ++m) {
        faiss::fvec_L2sqr_ny(dist_table + m * _params.ks,
                             feature + m * _params.lsq,
                             _codebooks.get() + (u_int64_t)m * _params.ks * _params.lsq,
                             _params.lsq,
                             _params.ks);
        /*
        LOG(INFO) << "get_dist_table m: " << m 
                << " _params.nsq: " << _params.nsq
                << " _params.lsq: " << _params.lsq
                << " _params.ks: " << _params.ks
                << " dist_table: " << dist_table[m * _params.ks + 0] << " " << dist_table[m * _params.ks + 1]
                << " feature: " << feature[m * _params.lsq + 0] << " " << feature[m * _params.lsq + 1]
                << " _codebooks: " << _codebooks.get()[m * _params.ks * _params.lsq + 0] << " " << _codebooks.get()[m * _params.ks * _params.lsq + 1];
        */
    }
    //LOG(INFO) << "get_dist_table: " <<  dist_table[0] << " " << dist_table[1] << " " << dist_table[_params.ks * _params.lsq - 1];
    return 0;
}

float max_abs_value(const float* arr, size_t num_simd_iter) {
    //LOG(INFO) << "max_abs_value " << num_simd_iter;
    const float* ptr = arr;
    __m128 accumulators = _mm_setzero_ps();
    constexpr int32_t abs_mask_scalar = 0x7FFFFFFF;
    static const __m128 abs_mask_vector =
        _mm_castsi128_ps(_mm_set1_epi32(abs_mask_scalar));

    float result = 0.0f;
    if (num_simd_iter != 0) {
        for (; num_simd_iter != 0; --num_simd_iter, ptr += 4) {
            __m128 vals = _mm_loadu_ps(ptr);
            __m128 abs_vals = _mm_and_ps(vals, abs_mask_vector);
            accumulators = _mm_max_ps(accumulators, abs_vals);
        }

        accumulators = _mm_max_ps(
            accumulators, _mm_shuffle_ps(accumulators, accumulators, 0b1110));
        result = std::max(accumulators[0], accumulators[1]);
        //LOG(INFO) << "result: " << result;
    }

    const float* end = arr + num_simd_iter;
    for (; ptr < end; ++ptr) {
        result = std::max(result, std::abs(*ptr));
    }

    return result;
}

int Quantization4BitPacked::get_dist_table_int8(float* dist_table, uint8_t* dist_table_int8, float* fixed_point_multiplier) const {
    //if (true) return 0;
    size_t dist_talbe_size = _params.nsq * _params.ks;
    const float max_abs_lookup_element = std::max(
            std::sqrt(std::numeric_limits<float>::epsilon()), max_abs_value(dist_table, dist_talbe_size / 4));
    /*
    float max_abs_lookup_element_raw = 0.0;
    
    for (size_t i = 0; i < dist_talbe_size; i++) {
        if (dist_table[i] > max_abs_lookup_element_raw) {
            max_abs_lookup_element_raw = dist_table[i];
        }
    }
    */
    // return max_integer_value / max_abs_lookup_element;
    auto max_integer_value = std::numeric_limits<int8_t>::max();
    auto multiplier = max_integer_value / max_abs_lookup_element;

    constexpr uint8_t bias = static_cast<uint8_t>(1) << ((sizeof(uint8_t) * 8) - 1);
    //LOG(INFO) << "dist_talbe_size: " << dist_talbe_size << " " << multiplier << " " << max_abs_lookup_element << " " << max_abs_lookup_element_raw;
    for (size_t i = 0; i < dist_talbe_size; ++i) {
        //LOG(WARNING) << "dist_table_int8 + i: " << i << " " << (float*)(dist_table_int8 + i);
        dist_table_int8[i] = std::round(dist_table[i] * multiplier) + bias;
        //LOG(INFO) << i << " " << dist_table[i] << " " << (dist_table_int8[i] - bias) / multiplier;
        // dist_table[i] = (dist_table_int8[i] - bias) / multiplier;
    }
    *fixed_point_multiplier = multiplier;
    //LOG(INFO) << "multiplier: " << multiplier;
    return 0;
}

/* T = uint8_t
template <typename T>  
vector<T> ConvertLookupToFixedPoint(
    ConstSpan<float> raw_lookup,
    const AsymmetricHasherConfig::FixedPointLUTConversionOptions&
        conversion_options,
    float* multiplier) {
  DCHECK_GT(conversion_options.multiplier_quantile(), 0.0f);
  DCHECK_LE(conversion_options.multiplier_quantile(), 1.0f);
  using SignedT = make_signed_t<T>;
  *multiplier = ComputeMultiplierByQuantile(
      raw_lookup, conversion_options.multiplier_quantile(),
      numeric_limits<SignedT>::max());
  constexpr int kRound =
      AsymmetricHasherConfig::FixedPointLUTConversionOptions::ROUND;
  if (conversion_options.multiplier_quantile() == 1.0f) {
    if (conversion_options.float_to_int_conversion_method() == kRound) {
      return ConvertLookupToFixedPointImpl<T>(
          raw_lookup, [](float f) { return std::round(f); }, *multiplier);
    } else {
      return ConvertLookupToFixedPointImpl<T>(
          raw_lookup, [](float f) { return static_cast<SignedT>(f); },
          *multiplier);
    }
  }

float ComputeMultiplierByQuantile(ConstSpan<float> raw_lookup, float quantile,
                                  int32_t max_integer_value) {
  const size_t k = raw_lookup.size() * (1.0 - quantile) + 1;
  if (k == 1) {
    // 如果 MaxAbsValue(raw_lookup) 为 0 ，那么 max_abs_lookup_element 会是 epsilon
    // 以此来避免除 0
    const float max_abs_lookup_element = std::max(
        std::sqrt(numeric_limits<float>::epsilon()), MaxAbsValue(raw_lookup));
    return max_integer_value / max_abs_lookup_element;
  } else {
    DCHECK_LT(quantile, 1.0f);
    TopNAmortizedConstant<float> tn(k);
    for (auto& elem : raw_lookup) {
      tn.push(std::abs(elem));
    }
    return max_integer_value / tn.exact_bottom();
  }
}

template <typename T, typename Lambda>
inline vector<T> ConvertLookupToFixedPointImpl(ConstSpan<float> raw_lookup,
                                               Lambda convert_to_int_lambda,
                                               float multiplier) {

  constexpr uint8_t kBias = static_cast<uint8_t>(1) << ((sizeof(uint8_t) * 8) - 1);
  vector<T> result(raw_lookup.size());
  for (size_t i = 0; i < raw_lookup.size(); ++i) {
    result[i] = convert_to_int_lambda(raw_lookup[i] * multiplier) + kBias;
  }
  return result;
}
  */

int write_fvec_format(const char* file_name, const uint32_t dim, const uint64_t n, const float* fea_vocab);
int Quantization4BitPacked::save_codebooks(const std::string& file_name) const {
    return write_fvec_format(file_name.c_str(), _params.lsq, _params.nsq * _params.ks, get_codebooks());
}

int Quantization4BitPacked::save_index(const std::string& file_name) const {
    FILE* f = fopen(file_name.c_str(), "wb");

    if (f == nullptr) {
        LOG(ERROR) << "cannot open " << file_name << " for writing";
        return -1;
    }
    LOG(INFO) << "save_index " << file_name;
    /*
    for (size_t i = 0; i < _total_point_count; i++) {
        unsigned char* pq_feature = get_quantized_feature(i);
        pq_feature += _fea_offset;
        for (uint32_t m = 0; m < _params.nsq; ++m) {
            if (pq_feature[m] >= 16) {
                LOG(INFO) << "============ pq_feature[m] >= 16 " << pq_feature[m] << " " << file_name;
            }
        }
    }
    */

    auto real_total_point_count = _total_point_count;

    unsigned char* feature_data = get_quantized_feature(0);
    long ret = fwrite(feature_data, get_per_fea_len(), real_total_point_count, f);

    if (ferror(f)) {
        LOG(INFO) << feature_data[0];
        fprintf(stderr, "Write error: %s\n", strerror(errno));
    }
    fclose(f);
    LOG(INFO) << "ret: " << ret << " " << real_total_point_count;
    if (ret != real_total_point_count) {
        LOG(ERROR) << "writint to " << file_name << " has error";
        return -1;
    }
    //LOG(INFO) << "FLAGS_convert_local_to_memory_while_build: " << FLAGS_convert_local_to_memory_while_build << " " << !_quantized_feature_packed << " " << !_fea_offset_packed;
    if (FLAGS_convert_local_to_memory_while_build && _quantized_feature_packed && _fea_offset_packed) {
        auto file_name_packed = file_name + "_packed";
        FILE* f = fopen(file_name_packed.c_str(), "wb");
        long ret = fwrite(_quantized_feature_packed.get(), 16 * _params.nsq, (((u_int64_t)_total_point_count + 31) & (~31)) / 32, f);
        //LOG(INFO) << "_quantized_feature_packed.get(): " << (int)_quantized_feature_packed.get()[0] << " " << (int)_quantized_feature_packed.get()[1];
        auto file_name_packed_offset = file_name + "_packed_fea_offset";
        FILE* f2 = fopen(file_name_packed_offset.c_str(), "wb");

        ret = fwrite(_fea_offset_packed.get(), _fea_offset, (u_int64_t)_total_point_count, f2);
        fclose(f);
        fclose(f2); 
    }

    return 0;
}

int Quantization4BitPacked::save_index_bak(const std::string& file_name) const {
    FILE* f = fopen(file_name.c_str(), "wb");

    if (f == nullptr) {
        LOG(ERROR) << "cannot open " << file_name << " for writing";
        return -1;
    }
    LOG(INFO) << "save_index_bak " << file_name;
    
    auto real_total_point_count = _total_point_count;
    unsigned char* feature_data = get_quantized_feature(0);
    long ret = fwrite(feature_data, get_per_fea_len(), real_total_point_count, f);
    fclose(f);

    if (ret != real_total_point_count) {
        LOG(ERROR) << "writint to " << file_name << " has error";
        return -1;
    }

    return 0;
}

}//namespace puck