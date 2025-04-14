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
 * @file    kmeans.cpp
 * @author  yinjie06(yinjie06@baidu.com)
 * @date    2023/07/25 11:11
 * @brief
 *
 **/

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
//#include <mkl.h>
#include <cblas.h>
#include <glog/logging.h>
#include <immintrin.h>
#include <omp.h>
#include <random>
#include "algorithm/puck/kmeans.h"
#include "algorithm/puck/thread_pool.h"

namespace puck {

#if (defined(__STD_C_VERSION__) && (__STD_C_VERSION__ >= 201112L)) || (__cplusplus >= 201703L) || \
    defined(_ISOC11_SOURCE)
void*
aligned_malloc(size_t size, size_t minimum_alignment) {
    size = (size + minimum_alignment - 1) / minimum_alignment * minimum_alignment;
    return aligned_alloc(minimum_alignment, size);
}

void
aligned_free(void* aligned_memory) {
    free(aligned_memory);
}

#elif defined(_MSC_VER)
#include <malloc.h>

void*
aligned_malloc(size_t size, size_t minimum_alignment) {
    return _aligned_alloc(size, minimum_alignment);
}

void
aligned_free(void* aligned_memory) {
    _aligned_free(aligned_memory);
}

#else
#endif

void
Kmeans::random_init_center(const size_t total_cnt,
                           const size_t sample_cnt,
                           const uint32_t dim,
                           const float* train_dataset,
                           std::vector<size_t>& sample_ids) {
    sample_ids.clear();
    std::uniform_int_distribution<> dis(0, total_cnt - 1);
    std::vector<bool> filter(total_cnt, false);

    while (sample_ids.size() < sample_cnt) {
        size_t sample_id = dis(_rnd);

        //去重
        if (filter[sample_id]) {
            continue;
        }

        sample_ids.push_back(sample_id);
        filter[sample_id] = true;
    }
}

int
Kmeans::roulette_selection(std::vector<float>& wheel) {
    float total_val = 0;

    for (auto& val : wheel) {
        total_val += val;
    }

    cblas_sscal(wheel.size(), 1.0 / total_val, wheel.data(), 1);
    std::uniform_real_distribution<double> dis(0, 1.0);
    double rd = dis(_rnd);

    for (auto id = 0; id < wheel.size(); ++id) {
        rd -= wheel[id];

        if (rd < 0) {
            return id;
        }
    }

    return wheel.size() - 1;
}

void
Kmeans::kmeanspp_init_center(const size_t total_cnt,
                             const size_t sample_cnt,
                             const uint32_t dim,
                             const float* train_dataset,
                             std::vector<size_t>& sample_ids) {
    std::vector<float> disbest(total_cnt, std::numeric_limits<float>::max());
    std::vector<float> distmp(total_cnt);
    sample_ids.resize(sample_cnt, 0);
    sample_ids[0] = _rnd() % total_cnt;

    std::vector<float> points_norm(total_cnt, 0);
    int max_threads = omp_get_max_threads();

#pragma omp parallel for schedule(dynamic) num_threads(_params.nt)
    for (size_t j = 0; j < total_cnt; j++) {
        int num_threads = omp_get_num_threads();
        if (j == 0)
            LOG(INFO) << "kmeanspp_init_center: " << _params.nt << " " << max_threads << " "
                      << num_threads;
        points_norm[j] = cblas_sdot(dim, train_dataset + j * dim, 1, train_dataset + j * dim, 1);
    }

    for (size_t i = 1; i < sample_cnt; i++) {
        size_t newsel = sample_ids[i - 1];
        const float* last_center = train_dataset + newsel * dim;

#pragma omp parallel for schedule(dynamic) num_threads(_params.nt)

        for (size_t j = 0; j < total_cnt; j++) {
            float temp = points_norm[j] + points_norm[newsel] -
                         2.0 * cblas_sdot(dim, train_dataset + j * dim, 1, last_center, 1);

            if (temp < disbest[j]) {
                disbest[j] = temp;
            }
        }

        memcpy(distmp.data(), disbest.data(), total_cnt * sizeof(distmp[0]));
        sample_ids[i] = roulette_selection(distmp);
    }
}

int
Kmeans::kmeans_reassign_empty(
    uint32_t dim, size_t total_cnt, size_t k, float* centroids, int* assign, int* nassign) {
    std::vector<float> proba_split(k);
    std::vector<float> vepsilon(dim);
    std::normal_distribution<> d_normal(0, _rnd() / ((double)RAND_MAX + 1.0));
#pragma omp parallel for schedule(dynamic) num_threads(_params.nt)

    for (auto c = 0; c < k; c++) {
        proba_split[c] = (nassign[c] < 2 ? 0 : nassign[c] * nassign[c] - 1);
    }

    int nreassign = 0;

    for (auto c = 0; c < k; c++) {
        if (nassign[c] == 0) {
            nreassign++;

            auto j = roulette_selection(proba_split);

            memcpy(centroids + c * dim, centroids + j * dim, dim * sizeof(centroids[0]));
            double s = cblas_snrm2(dim, centroids + j * dim, 1) * 0.0000001;

            for (auto& v : vepsilon) {
                v = d_normal(_rnd);
            }

            cblas_sscal(dim, s, vepsilon.data(), 1);
            cblas_saxpy(dim, 1.0, vepsilon.data(), 1, centroids + j * dim, 1);
            cblas_saxpy(dim, -1.0, vepsilon.data(), 1, centroids + c * dim, 1);

            proba_split[j] = 0;
        }
    }

    return nreassign;
}

float
Kmeans::kmeans(uint32_t dim,
               size_t total_cnt,
               size_t k,
               const float* train_dataset,
               float* centroids_out,
               float* dis_out,
               int* assign_out) {
    if (k >= total_cnt) {
        LOG(ERROR) << "better to have fewer clusters than points";
        return -1;
    }

    int nt = std::max(_params.nt, 1);

    std::unique_ptr<float[]> centroids(new float[k * dim]);

    std::unique_ptr<float[]> dis(new float[total_cnt]);
    std::unique_ptr<int[]> assign(new int[total_cnt]);
    std::unique_ptr<int[]> nassign(new int[k]);

    double qerr = std::numeric_limits<double>::max();
    double qerr_best = std::numeric_limits<double>::max();

    std::vector<size_t> selected(k);

    int core_ret = 0;

    for (auto run = 0; run < _params.redo; run++) {
#ifdef DEBUG
        LOG(INFO) << "\nkmeans / run " << run;
#endif
        //_params.init_type = KMeansCenterInitType::RANDOM;
        float norm = 0.0;
        for (size_t i = 0; i < dim; i++) {
            norm += train_dataset[i] * train_dataset[i];
        }
        LOG(INFO) << "kmeans 0 " << train_dataset[0] << " " << train_dataset[1] << " "
                  << train_dataset[2] << " " << norm << " " << dim;
        if (_params.init_type == KMeansCenterInitType::KMEANS_PLUS_PLUS) {
            //数据集太大时候，缩小范围，待开发
            uint32_t nsubset = (total_cnt > k * 8 && total_cnt > 8 * 1024) ? k * 8 : total_cnt;
            LOG(INFO) << "true nsubset for  KMEANS_PLUS_PLUS = " << nsubset;
            kmeanspp_init_center(nsubset, k, dim, train_dataset, selected);
        } else {
            random_init_center(total_cnt, k, dim, train_dataset, selected);
        }
        LOG(INFO) << "kmeans 1";
        for (auto i = 0; i < k; i++) {
            memcpy(centroids.get() + i * dim,
                   train_dataset + selected[i] * dim,
                   dim * sizeof(centroids[0]));
        }
        LOG(INFO) << "kmeans 2 " << dim << " " << total_cnt << " " << k << " " << _params.niter
                  << " " << nt;
        core_ret = kmeans_core(dim,
                               total_cnt,
                               k,
                               _params.niter,
                               nt,
                               centroids.get(),
                               train_dataset,
                               assign.get(),
                               nassign.get(),
                               dis.get(),
                               &qerr);
        LOG(INFO) << "kmeans 3";
        if (core_ret < 0) {
            return -1;
            break;
        }

        if (qerr < qerr_best) {
            qerr_best = qerr;

            if (centroids_out != nullptr) {
                memcpy(centroids_out, centroids.get(), k * dim * sizeof(*centroids.get()));
            }

            if (dis_out != nullptr) {
                memcpy(dis_out, dis.get(), total_cnt * sizeof(*dis.get()));
            }

            if (assign_out != nullptr) {
                memcpy(assign_out, assign.get(), total_cnt * sizeof(*assign.get()));
            }
        }
    }

    return qerr_best / total_cnt;
}

float
Kmeans::assign(uint32_t dim,
               size_t total_cnt,
               size_t k,
               const float* train_dataset,
               float* centroids,
               float* dis_out,
               int* assign_out) {
    std::unique_ptr<int[]> nassign(new int[k]);

    double qerr = std::numeric_limits<double>::max();

    int core_ret = 0;

    core_ret = kmeans_assign(dim,
                             total_cnt,
                             k,
                             _params.niter,
                             1,
                             centroids,
                             train_dataset,
                             assign_out,
                             nassign.get(),
                             dis_out,
                             &qerr);

    return qerr / total_cnt;
}

/*
template <typename Int, typename DenomInt>
constexpr Int DivRoundUp(Int num, DenomInt denom) {
  return (num + static_cast<Int>(denom) - static_cast<Int>(1)) /
         static_cast<Int>(denom);
}

template <typename Int, typename DenomInt>
constexpr Int NextMultipleOf(Int num, DenomInt denom) {
  return DivRoundUp(num, denom) * denom;
}

constexpr bool ShouldTranspose(size_t n_queries, size_t dimensionality) {
    const size_t n_dims = dimensionality;

    constexpr size_t kDoubleDivisor = sizeof(float) / sizeof(float);

    constexpr size_t kMaxMidLevelQueryBatchSize = 256 / kDoubleDivisor;

    const size_t n_mid_level_batches =
        DivRoundUp(n_queries, kMaxMidLevelQueryBatchSize);

    constexpr float kUntransposedCostPerQuery = 2.0;

    constexpr float kUntransposedCostPerDimPerQuery = 0.025;

    constexpr float kRemainderCostPerDimPerQuery = 0.025;

    constexpr float kRemainderCostPerQuery = 2.0;

    const float transposed_cost_per_dp = dimensionality * n_mid_level_batches;

    const bool has_remainder = 0;
    const float untransposed_postprocess_cost_per_dp =
        n_queries * kUntransposedCostPerQuery +
        n_queries * n_dims * kUntransposedCostPerDimPerQuery +
        n_queries * n_dims * kRemainderCostPerDimPerQuery * has_remainder +
        n_queries * kRemainderCostPerQuery * has_remainder;

    return transposed_cost_per_dp < untransposed_postprocess_cost_per_dp;
}

template <bool kIsSquaredL2>
class M2MTransposer {
 public:
  static constexpr size_t kElementsPerRegister = 8;

  static std::unique_ptr<M2MTransposer> New(const size_t dimensionality,
                                       const size_t result_entries) {
    const size_t transposed_entries =
        (dimensionality + kIsSquaredL2) * kElementsPerRegister;
    const size_t total_entries = 2 * transposed_entries + result_entries;

    constexpr size_t kCacheLineBytes = 64;
    float* storage = static_cast<float*>(
        aligned_malloc(total_entries * sizeof(float), kCacheLineBytes));
    std::fill(storage, storage + total_entries,
              std::numeric_limits<float>::quiet_NaN());
    std::unique_ptr<M2MTransposer> t_ptr;
    t_ptr.reset(reinterpret_cast<M2MTransposer*>(storage));
    //return std::make_unique<M2MTransposer>(reinterpret_cast<M2MTransposer*>(storage));
    return t_ptr;
  }

  static void operator delete(void* ptr) { aligned_free(ptr); }

  float* __restrict__ GetTransposedPtr0(const size_t dimensionality) {
    const size_t transposed_sz =
        (dimensionality + kIsSquaredL2) * kElementsPerRegister;
    float* storage = &first_buffer_entry_;
    return storage + 0 * transposed_sz +
           (kIsSquaredL2 ? kElementsPerRegister : 0);
  }

  float* __restrict__ GetTransposedPtr1(const size_t dimensionality) {
    const size_t transposed_sz =
        (dimensionality + kIsSquaredL2) * kElementsPerRegister;
    float* storage = &first_buffer_entry_;
    return storage + 1 * transposed_sz +
           (kIsSquaredL2 ? kElementsPerRegister : 0);
  }

  float* __restrict__ GetResultsPtr(const size_t dimensionality) {
    const size_t transposed_sz =
        (dimensionality + kIsSquaredL2) * kElementsPerRegister;
    float* storage = &first_buffer_entry_;
    return storage + 2 * transposed_sz;
  }

   void TransposeDatabaseBlock(const size_t dimensionality,
                                                const float* database,
                                                size_t first_dp_idx,
                                                size_t n_to_transpose) {
    float* __restrict__ transposed_ptr0 = GetTransposedPtr0(dimensionality);
    float* __restrict__ transposed_ptr1 = GetTransposedPtr1(dimensionality);

    if (n_to_transpose <= kElementsPerRegister) {
      TransposeDatabaseBlockImpl(dimensionality, database, first_dp_idx,
                                 n_to_transpose, transposed_ptr0);
    } else {
      TransposeDatabaseBlockImpl(dimensionality, database, first_dp_idx,
                                 kElementsPerRegister, transposed_ptr0);
      TransposeDatabaseBlockImpl(
          dimensionality, database, first_dp_idx + kElementsPerRegister,
          n_to_transpose - kElementsPerRegister, transposed_ptr1);
    }
    if constexpr (kIsSquaredL2) {
      AugmentWithL2Norms(transposed_ptr0, transposed_ptr1, dimensionality);
    }
  }

   static void TransposeDatabaseBlockImpl(
      const size_t dimensionality, const float* database, size_t first_dp_idx,
      size_t n_to_transpose, float* __restrict__ transposed_storage) {
    size_t j = 0;
    for (; j + 4 <= n_to_transpose; j += 4) {
      const float* database_ptr =
          database + (first_dp_idx + j) * dimensionality;
      float* __restrict__ dest = transposed_storage + j;

      constexpr size_t kCacheLineElements = 64 / sizeof(float);
      
      const float* prefetch = database_ptr + 0 * dimensionality;
      const float* prefetch_end = database_ptr + 4 * dimensionality;
      
      for (; prefetch < prefetch_end; prefetch += kCacheLineElements) {
        //::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
          //  prefetch);
      }
      
      const float* ptr_begin = database_ptr + 1 * dimensionality;
      const float* ptr_end = database_ptr + 2 * dimensionality;
      for (const float* ptr = ptr_begin; ptr != ptr_end;
           ++ptr, dest += kElementsPerRegister) {
        dest[0] = *(ptr - dimensionality);
        dest[1] = ptr[0 * dimensionality];
        dest[2] = ptr[1 * dimensionality];
        dest[3] = ptr[2 * dimensionality];
      }
    }
    for (; j < n_to_transpose; ++j) {
      const float* untransposed0 =
          database + (first_dp_idx + j) * dimensionality;
      float* dest = transposed_storage + j;
      for (size_t dim_idx = 0; dim_idx < dimensionality;
           ++dim_idx, dest += kElementsPerRegister) {
        dest[0] = untransposed0[dim_idx];
      }
    }
  }

 private:
    static void AugmentWithL2Norms(
      float* __restrict__ transposed_ptr0,
      float* __restrict__ transposed_ptr1, size_t dimensionality) {
    
    __m256 norm0 = _mm256_setzero_ps();
    __m256 norm1 = _mm256_setzero_ps();
    __m256 two = _mm256_set1_ps(2.0);

    for (size_t dim = 0; dim < dimensionality; dim++) {
      __m256 transposed_simd0 = _mm256_load_ps(transposed_ptr0 + dim * kElementsPerRegister);
      __m256 transposed_simd1 = _mm256_load_ps(transposed_ptr1 + dim * kElementsPerRegister);
      
      norm0 = _mm256_fnmadd_ps(transposed_simd0, transposed_simd0, norm0);
      norm1 = _mm256_fnmadd_ps(transposed_simd1, transposed_simd1, norm1);
      auto tmp0 = _mm256_mul_ps(transposed_simd0, two);
      _mm256_storeu_ps(transposed_ptr0 + dim * kElementsPerRegister, tmp0);
      auto tmp1 = _mm256_mul_ps(transposed_simd1, two);
      _mm256_storeu_ps(transposed_ptr1 + dim * kElementsPerRegister, tmp1);
    }
    auto neg1 = _mm256_set1_ps(-1.0);
    auto tmp0 = _mm256_mul_ps(norm0, neg1);
    _mm256_storeu_ps(transposed_ptr0 - kElementsPerRegister, tmp0);
    auto tmp1 = _mm256_mul_ps(norm1, neg1);
    _mm256_storeu_ps(transposed_ptr1 - kElementsPerRegister, tmp1);
    
  }

  struct alignas(kElementsPerRegister * sizeof(float)) {
    float first_buffer_entry_;
  };
};

using Transposer = M2MTransposer<true>;
struct BottomLevelBatchArgs {
    size_t dimensionality;
    const float* queries;
    const float* query_norms;
    size_t first_q_idx;
    size_t first_dp_idx;
    size_t num_datapoints;
    Transposer* transposer;
    //CallbackT* callback;
};

template <size_t kNumQueries>
 void DoAccumulationTransposedTemplate(const float* transposed_block0,
                                   const float* transposed_block1,
                                   const float** query_ptrs,
                                   const float* query_norms,
                                   size_t dimensionality,
                                   __m256 (&accumulators)[6][2]) {
    size_t kElementsPerRegister = 8;
    for (size_t j = 0; j < kNumQueries; j++) {
        auto query_norm = _mm256_set1_ps(query_norms[j]);
        auto db_norms0 = _mm256_load_ps(transposed_block0 - kElementsPerRegister);
        auto db_norms1 = _mm256_load_ps(transposed_block1 - kElementsPerRegister);
        accumulators[j][0] = _mm256_add_ps(db_norms0, query_norm);

    }

    for (size_t dim = 0; dim < dimensionality; dim++) {
      auto transposed_simd0 = _mm256_load_ps(transposed_block0 + dim * kElementsPerRegister);
      auto transposed_simd1 = _mm256_load_ps(transposed_block1 + dim * kElementsPerRegister);

      for (size_t j = 0; j < kNumQueries; j++) {
        auto query_simd = _mm256_set1_ps(query_ptrs[j][dim]);
        accumulators[j][0] = _mm256_fnmadd_ps(query_simd, transposed_simd0, accumulators[j][0]);
        accumulators[j][1] = _mm256_fnmadd_ps(query_simd, transposed_simd0, accumulators[j][1]);
      }
    }
}

template <size_t kNumQueries>
   static void BottomLevelBatch(BottomLevelBatchArgs args) {
    const size_t dimensionality = args.dimensionality;
    const float* queries = args.queries;
    const float* query_norms = args.query_norms;
    const size_t first_q_idx = args.first_q_idx;
    // LOG(INFO) << "BottomLevelBatch: " << first_q_idx;
    const size_t first_dp_idx = args.first_dp_idx;
    const size_t num_datapoints = args.num_datapoints;
    const float* transposed_ptr0 =
        args.transposer->GetTransposedPtr0(dimensionality);
    const float* transposed_ptr1 =
        args.transposer->GetTransposedPtr1(dimensionality);

    const float* volatile query_ptrs_vol[kNumQueries];
    for (size_t j = 0; j < kNumQueries; j++) {
      query_ptrs_vol[j] = queries + j * dimensionality;
    }
    const float* query_ptrs[kNumQueries];
    for (size_t j = 0; j < kNumQueries; j++) {
      query_ptrs[j] = query_ptrs_vol[j];
    }
    __m256 accumulators[kNumQueries][2];
    DoAccumulationTransposedTemplate<kNumQueries>(
        transposed_ptr0, transposed_ptr1, query_ptrs, query_norms,
        dimensionality, accumulators);

    std::array<std::array<float, 16>, kNumQueries> re;
    for (size_t j = 0; j < kNumQueries; j++) {
        _mm256_storeu_ps(re[j].data(), accumulators[j][0]);
        _mm256_storeu_ps(re[j].data() + 8, accumulators[j][1]);
    }
    if (first_q_idx == 0 && first_dp_idx == 0) {
        LOG(INFO) << "dist: " << re[0][0];
    }
  }

void MidLevelBatch(const size_t dimensionality, size_t first_q_idx, size_t num_queries,
                            size_t first_dp_idx, size_t num_datapoints, 
                            const float* queries, const float* database, const float* query_norms) {
    // LOG(INFO) << "MidLevelBatch: " << first_q_idx << " " << num_queries << " " << first_dp_idx << " " << num_datapoints;
    constexpr static size_t kResultsSize =
        2 * 6 * 8;
    constexpr static size_t kSmallQueryStride = 6;
    const size_t q_idx_end = first_q_idx + num_queries;

    thread_local size_t allocated_dimensionality = 0;
    thread_local std::unique_ptr<Transposer> transposer_storage;
    if (allocated_dimensionality < dimensionality) {
      transposer_storage = Transposer::New(dimensionality, kResultsSize);
      allocated_dimensionality = dimensionality;
    }

    Transposer* transposer = transposer_storage.get();

    transposer->TransposeDatabaseBlock(dimensionality, database,
                                         first_dp_idx, num_datapoints);


    BottomLevelBatchArgs args;
    args.dimensionality = dimensionality;
    args.queries = queries + first_q_idx * dimensionality;

    args.query_norms = query_norms + first_q_idx;

    args.first_q_idx = first_q_idx;
    args.first_dp_idx = first_dp_idx;
    args.num_datapoints = num_datapoints;
    args.transposer = transposer;
    //args.callback = &this->callback_;

    while (args.first_q_idx + kSmallQueryStride <= q_idx_end) {
        BottomLevelBatch<kSmallQueryStride>(args);
        args.first_q_idx += kSmallQueryStride;

        args.query_norms += kSmallQueryStride;
        
        args.queries += kSmallQueryStride * dimensionality;
    }

    const size_t final_batch_size = q_idx_end - args.first_q_idx;
    //LOG(INFO) << "final_batch_size: " << final_batch_size;
    //SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE_6(final_batch_size, BottomLevelBatch,
      //                                  args);
}

void TopLevelBatch(size_t dimensionality, const float* queries, const size_t num_queries, 
        const float* database, const size_t num_datapoints, const float* query_norms) {
    const size_t q_stride_est1 =
            std::max<size_t>(1, (1 << 19) / (dimensionality * sizeof(float)));
    static constexpr size_t kElementsPerRegister = 8;
    constexpr static size_t kSmallQueryStride = 6;
    const size_t q_stride_est2 =
            num_queries / DivRoundUp(num_queries, q_stride_est1);

    const size_t q_stride = NextMultipleOf(q_stride_est2, kSmallQueryStride);

    for (size_t q_idx = 0; q_idx < num_queries; q_idx += q_stride) {
        const size_t q_batch_size = std::min(q_stride, num_queries - q_idx);

        constexpr size_t kDatabaseStride = 2 * kElementsPerRegister;
        const size_t num_db_blocks = DivRoundUp(num_datapoints, kDatabaseStride);

        similarity::ParallelFor(0, num_db_blocks, 0, [&](int block_idx, int threadId) {
            const size_t first_dp_idx = block_idx * kDatabaseStride;
            const size_t dp_batch_size =
                std::min(kDatabaseStride, num_datapoints - first_dp_idx);

            MidLevelBatch(dimensionality, q_idx, q_batch_size, first_dp_idx, dp_batch_size, queries, database, query_norms);
        });
       
    }
}


int nearest_center2(uint32_t dim, const float* centroids, const size_t centroid_cnt,
                   const float* train_dataset, const size_t point_cnt, int* assign, float* dis) {
    
    LOG(INFO) << "ShouldTranspose: " << ShouldTranspose(128, 1024);


            //return DenseManyToManyTransposedImpl<true>(queries, database, pool,
              //                                     std::move(callback));
    std::vector<float> points_norm(point_cnt);
    int nt = std::thread::hardware_concurrency();

    #pragma omp parallel for schedule(dynamic) num_threads(nt)

    for (size_t j = 0; j < point_cnt; j++) {
        points_norm[j] = cblas_sdot(dim, train_dataset + j * dim, 1, train_dataset + j * dim, 1);
    }

    TopLevelBatch(dim, train_dataset, 128, centroids, centroid_cnt, points_norm.data());
    
    exit(0);

    return 0;
}

*/
inline int
Kmeans::nearest_center(uint32_t dim,
                       const float* centroids,
                       const size_t centroid_cnt,
                       const float* train_dataset,
                       const size_t point_cnt,
                       int* assign,
                       float* dis) {
    std::vector<float> points_norm(point_cnt);
    std::vector<float> centroids_norm(centroid_cnt);
    int nt = _params.nt;  //std::thread::hardware_concurrency();

#pragma omp parallel for schedule(dynamic) num_threads(nt)

    for (size_t j = 0; j < point_cnt; j++) {
        points_norm[j] = cblas_sdot(dim, train_dataset + j * dim, 1, train_dataset + j * dim, 1);
    }

#pragma omp parallel for schedule(dynamic) num_threads(nt)

    for (size_t j = 0; j < centroid_cnt; j++) {
        centroids_norm[j] = cblas_sdot(dim, centroids + j * dim, 1, centroids + j * dim, 1);
    }

#pragma omp parallel for schedule(dynamic) num_threads(nt)

    for (size_t j = 0; j < point_cnt; j++) {
        std::pair<float, uint32_t> min_centroid = {std::numeric_limits<float>::max(), 0};

        for (size_t c = 0; c < centroid_cnt; c++) {
            float cur_dist =
                points_norm[j] + centroids_norm[c] -
                2.0 * cblas_sdot(dim, train_dataset + j * dim, 1, centroids + c * dim, 1);
            /*
            if (j == 0 && c < 30) {
                LOG(INFO) << "c: " << c << " " << cur_dist << " " << points_norm[j] << " "
                          << centroids_norm[c] << " "
                          << 2.0 * cblas_sdot(
                                       dim, train_dataset + j * dim, 1, centroids + c * dim, 1);
            }
            */
            if (cur_dist < min_centroid.first) {
                min_centroid = {cur_dist, c};
            }
        }

        assign[j] = min_centroid.second;
        dis[j] = min_centroid.first;
    }

    return 0;
}

int
Kmeans::kmeans_core(uint32_t d,
                    size_t n,
                    size_t k,
                    int niter,
                    int nt,
                    float* centroids,
                    const float* v,
                    int* assign,
                    int* nassign,
                    float* dis,
                    double* qerr_out) {
    double qerr = std::numeric_limits<double>::max();
    double qerr_old = std::numeric_limits<double>::max();

    int tot_nreassign = 0;

    for (auto iter = 1; iter <= niter; iter++) {
        nearest_center(d, centroids, k, v, n, assign, dis);
        // nearest_center2(d, centroids, k, v, n, assign, dis);

        memset(nassign, 0, k * sizeof(int));
        {
            memset(centroids, 0, sizeof(centroids[0] * k * d));

            for (auto i = 0; i < n; i++) {
                if (assign[i] < 0 || assign[i] >= k) {
                    LOG(ERROR) << "assign to invalid center, something wrong in input. Maybe there "
                                  "are NaNs?";
                    return -1;
                }

                nassign[assign[i]]++;
                cblas_saxpy(d, 1.0, v + i * d, 1, centroids + assign[i] * d, 1);
            }

            for (auto i = 0; i < k; i++) {
                cblas_sscal(d, 1.0 / nassign[i], centroids + i * d, 1);
            }
        }

        auto nreassign = kmeans_reassign_empty(d, n, k, centroids, assign, nassign);
#ifdef DEBUG

        if (nreassign > 0) {
            LOG(INFO) << nreassign << " empty clusters are splited";
        }

#endif
        tot_nreassign += nreassign;

        if (tot_nreassign > n / 100 && tot_nreassign > 1000) {
            LOG(ERROR) << "kmeans: reassigned " << tot_nreassign << " times, abandoning";
            return -1;
        }

        qerr_old = qerr;
        qerr = 0;

        for (auto i = 0; i < n; i++) {
            qerr += dis[i];
        }

#ifdef DEBUG
        LOG(INFO) << qerr_old << " " << qerr;
#endif

        if (std::fabs(qerr_old - qerr) < 1e-6 && nreassign == 0) {
            break;
        }

        //#ifdef DEBUG
        if (k != 16 && k != 256)
            LOG(INFO) << "ite " << iter << "th, err = " << qerr << ", nreassign = " << nreassign;
        //#endif
    }
    size_t maxn = 0;
    for (size_t kk = 0; kk < k; kk++) {
        if (maxn < nassign[kk]) {
            maxn = nassign[kk];
        }
    }
    LOG(INFO) << "maxn: "
              << " " << maxn << " k: " << k;
    if (k == 50 || k == 500) {
        for (size_t kk = 0; kk < k; kk++) {
            LOG(INFO) << kk << " " << nassign[kk];
        }
    }

    *qerr_out = qerr;
    return 0;
}

int
Kmeans::kmeans_assign(uint32_t d,
                      size_t n,
                      size_t k,
                      int niter,
                      int nt,
                      float* centroids,
                      const float* v,
                      int* assign,
                      int* nassign,
                      float* dis,
                      double* qerr_out) {
    double qerr = std::numeric_limits<double>::max();
    double qerr_old = std::numeric_limits<double>::max();

    int tot_nreassign = 0;

    nearest_center(d, centroids, k, v, n, assign, dis);
    // nearest_center2(d, centroids, k, v, n, assign, dis);

    memset(nassign, 0, k * sizeof(int));

    for (auto i = 0; i < n; i++) {
        if (assign[i] < 0 || assign[i] >= k) {
            LOG(ERROR)
                << "assign to invalid center, something wrong in input. Maybe there are NaNs?";
            return -1;
        }

        nassign[assign[i]]++;
    }

    qerr = 0;

    for (auto i = 0; i < n; i++) {
        qerr += dis[i];
    }

    LOG(INFO) << "err = " << qerr;

    size_t maxn = 0;
    for (size_t kk = 0; kk < k; kk++) {
        if (maxn < nassign[kk]) {
            maxn = nassign[kk];
        }
    }
    LOG(INFO) << "maxn: "
              << " " << maxn << " k: " << k;
    if (k == 50) {
        for (size_t kk = 0; kk < k; kk++) {
            LOG(INFO) << kk << " " << nassign[kk];
        }
    }

    *qerr_out = qerr;
    return 0;
}

};  //namespace puck
