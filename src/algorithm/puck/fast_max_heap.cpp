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
 * @file max_heap.cpp
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2019/8/20 10:43
 * @brief
 *
 **/
#include <memory>
#include <iostream>
#include <glog/logging.h>
#include "algorithm/puck/fast_max_heap.h"
#include "algorithm/puck/zip_sort.h"
#include <immintrin.h>
#include <math.h>
namespace puck {

template <typename Int, typename DenomInt>
constexpr Int DivRoundUp(Int num, DenomInt denom) {
  return (num + static_cast<Int>(denom) - static_cast<Int>(1)) /
         static_cast<Int>(denom);
}

template <typename Int, typename DenomInt>
constexpr Int NextMultipleOf(Int num, DenomInt denom) {
  return DivRoundUp(num, denom) * denom;
}

uint32_t GetFinalMask32(size_t num_datapoints) {
  const size_t remainder_bits = num_datapoints % 32;
  return remainder_bits ? (1u << remainder_bits) - 1 : 0xFFFFFFFF;
}

inline uint32_t GetComparisonMask(__m256 v0, __m256 v1, __m256 v2, __m256 v3) {
  const uint32_t m00 = _mm256_movemask_ps(v0);
  const uint32_t m08 = _mm256_movemask_ps(v1);
  const uint32_t m16 = _mm256_movemask_ps(v2);
  const uint32_t m24 = _mm256_movemask_ps(v3);
  return m00 + (m08 << 8) + (m16 << 16) + (m24 << 24);
}

template <bool kIsEquality>
inline size_t CalculateSwapMasks(const float* values,
                                            uint32_t* masks, size_t n_masks,
                                            uint32_t final_mask,
                                            float threshold) {
    size_t n_kept = 0;
    const auto simd_threshold = _mm256_set1_ps(threshold);
    for (size_t j = 0; j < n_masks; j++) {
        
        auto v0 = _mm256_loadu_ps(values + 32 * j);
        
        auto v1 = _mm256_loadu_ps(values + 32 * j + 8);

        auto v2 = _mm256_loadu_ps(values + 32 * j + 16);

        auto v3 = _mm256_loadu_ps(values + 32 * j + 24);

        const uint32_t mask = kIsEquality
                                ? GetComparisonMask(
                                    _mm256_cmp_ps(v0, simd_threshold, _CMP_EQ_OS),
                                    _mm256_cmp_ps(v1, simd_threshold, _CMP_EQ_OS),
                                    _mm256_cmp_ps(v2, simd_threshold, _CMP_EQ_OS),
                                    _mm256_cmp_ps(v3, simd_threshold, _CMP_EQ_OS))
                                : GetComparisonMask(
                                    _mm256_cmp_ps(v0, simd_threshold, _CMP_LT_OS),
                                    _mm256_cmp_ps(v1, simd_threshold, _CMP_LT_OS),
                                    _mm256_cmp_ps(v2, simd_threshold, _CMP_LT_OS),
                                    _mm256_cmp_ps(v3, simd_threshold, _CMP_LT_OS));
        n_kept += __builtin_popcount(mask);

        masks[j] = mask;

    }

    uint32_t& last_mask = masks[n_masks - 1];
    n_kept -= __builtin_popcount(last_mask);
    last_mask &= final_mask;
    n_kept += __builtin_popcount(last_mask);

    return n_kept;
}

inline size_t CalculateLtSwapMasks(const float* values,
                                            uint32_t* masks, size_t n_masks,
                                            uint32_t final_mask,
                                            float threshold) {
  return CalculateSwapMasks<false>(values, masks, n_masks, final_mask,
                                   threshold);
}


inline size_t CalculateEqSwapMasks(const float* values,
                                            uint32_t* masks, size_t n_masks,
                                            uint32_t final_mask,
                                            float threshold) {
    return CalculateSwapMasks<true>(values, masks, n_masks, final_mask,
                                  threshold);
}

float FastMedianOf3(float v0, float v1, float v2) {
  float big = std::max(v0, v1);
  float sml = std::min(v0, v1);
  return std::max(sml, std::min(big, v2));
}

inline int FindLSBSetNonZero(uint32_t n) { return __builtin_ctz(n); }

inline float UseMasksToFindNewMedian(float* values,
                                                uint32_t* lt_masks,
                                                uint32_t* eq_masks,
                                                size_t n_masks) {
    size_t n_idx = 0;
    float vals[3];
    for (size_t j = 0; j < n_masks; j++) {
        uint32_t mask = ~(lt_masks[j] + eq_masks[j]);
        while (mask) {
            const int offset = FindLSBSetNonZero(mask);
            mask &= (mask - 1);
            const size_t idx = j * 32 + offset;
            vals[n_idx++] = values[idx];
            if (n_idx == 3) {
                return FastMedianOf3(vals[0], vals[1], vals[2]);
            }
        }
    }

  return vals[0];
}

void ZipSwap(size_t a, size_t b, uint32_t* indices,
                          float* values) {
  std::swap(indices[a], indices[b]);
  std::swap(values[a], values[b]);
}

void ZipSwap(size_t a, size_t b, uint32_t* indices,
                          float* values, float* values_aux) {
  std::swap(indices[a], indices[b]);
  std::swap(values[a], values[b]);
  std::swap(values_aux[a], values_aux[b]);
}

inline size_t UseMasksToPartition(uint32_t* indices,
                                              float* values, uint32_t* masks,
                                              size_t n_masks) {
  size_t mask_idx1 = 0;
  size_t mask_idx2 = n_masks - 1;

  uint32_t mask1 = ~masks[mask_idx1];

  uint32_t mask2 = masks[mask_idx2];

  if (n_masks > 1) {
    for (;;) {
      while (mask1 && mask2) {
        const int offset1 = FindLSBSetNonZero(mask1);
        const int offset2 = FindLSBSetNonZero(mask2);
        mask1 &= mask1 - 1;
        mask2 &= mask2 - 1;
        const size_t idx1 = mask_idx1 * 32 + offset1;
        const size_t idx2 = mask_idx2 * 32 + offset2;
        ZipSwap(idx1, idx2, indices, values);
      }
      if (!mask1) {
        ++mask_idx1;
        if (mask_idx1 == mask_idx2) {
          break;
        }
        mask1 = ~masks[mask_idx1];
      }
      if (!mask2) {
        --mask_idx2;
        if (mask_idx1 == mask_idx2) {
          mask2 = ~mask1;
          break;
        }
        mask2 = masks[mask_idx2];
      }
    }
  }

  size_t write_offset = mask_idx2 * 32;
  while (mask2) {
    const int offset = FindLSBSetNonZero(mask2);
    const size_t idx = mask_idx2 * 32 + offset;
    mask2 &= mask2 - 1;
    ZipSwap(write_offset++, idx, indices, values);
  }
  return write_offset;
}

inline size_t UseMasksToPartition(uint32_t* indices,
                                              float* values, float* values_aux, uint32_t* masks,
                                              size_t n_masks) {
  size_t mask_idx1 = 0;
  size_t mask_idx2 = n_masks - 1;

  uint32_t mask1 = ~masks[mask_idx1];

  uint32_t mask2 = masks[mask_idx2];

  if (n_masks > 1) {
    for (;;) {
      while (mask1 && mask2) {
        const int offset1 = FindLSBSetNonZero(mask1);
        const int offset2 = FindLSBSetNonZero(mask2);
        mask1 &= mask1 - 1;
        mask2 &= mask2 - 1;
        const size_t idx1 = mask_idx1 * 32 + offset1;
        const size_t idx2 = mask_idx2 * 32 + offset2;
        ZipSwap(idx1, idx2, indices, values, values_aux);
      }
      if (!mask1) {
        ++mask_idx1;
        if (mask_idx1 == mask_idx2) {
          break;
        }
        mask1 = ~masks[mask_idx1];
      }
      if (!mask2) {
        --mask_idx2;
        if (mask_idx1 == mask_idx2) {
          mask2 = ~mask1;
          break;
        }
        mask2 = masks[mask_idx2];
      }
    }
  }

  size_t write_offset = mask_idx2 * 32;
  while (mask2) {
    const int offset = FindLSBSetNonZero(mask2);
    const size_t idx = mask_idx2 * 32 + offset;
    mask2 &= mask2 - 1;
    ZipSwap(write_offset++, idx, indices, values, values_aux);
  }
  return write_offset;
}

size_t UseMasksToSelect(uint32_t* to, uint32_t* from,
                                           uint32_t* masks, size_t n_masks) {
  size_t write_idx = 0;
  for (size_t j = 0; j < n_masks; j++) {
    uint32_t mask = masks[j];
    while (mask) {
      const int offset = FindLSBSetNonZero(mask);
      mask &= (mask - 1);
      to[write_idx++] = from[32 * j + offset];
    }
  }
  return write_idx;
}

inline size_t UseMaskToCompact(uint32_t* indices,
                                        float* values, uint32_t mask) {
  size_t write_idx = 0;
  while (mask) {
    const int offset = FindLSBSetNonZero(mask);
    mask &= (mask - 1);
    indices[write_idx] = indices[offset];
    values[write_idx] = values[offset];
    ++write_idx;
  }
  return write_idx;
}

inline size_t UseMaskToCompact(uint32_t* indices,
                                        float* values, float* values_aux, uint32_t mask) {
  size_t write_idx = 0;
  while (mask) {
    const int offset = FindLSBSetNonZero(mask);
    mask &= (mask - 1);
    indices[write_idx] = indices[offset];
    values[write_idx] = values[offset];
    values_aux[write_idx] = values_aux[offset];
    ++write_idx;
  }
  return write_idx;
}

size_t UseMasksToCompactDoublePorted(
    uint32_t* indices, float* values, uint32_t* masks, size_t n_masks) {

  std::copy(values, values + 64, values + n_masks * 32);
  std::copy(indices, indices + 64, indices + n_masks * 32);
  std::copy(masks, masks + 2, masks + n_masks);
  n_masks += 2;

  uint32_t mask1 = masks[2];
  uint32_t* indices1 = indices + 2 * 32;
  float* values1 = values + 2 * 32;

  uint32_t mask2 = masks[3];
  uint32_t* indices2 = indices + 3 * 32;
  float* values2 = values + 3 * 32;

  uint32_t* masks_ptr = masks + 3;
  uint32_t* masks_end = masks + n_masks;

  uint32_t* indices_write_ptr = indices;
  float* values_write_ptr = values;

  for (;;) {
    if (!mask1 || !mask2) {
      bool proceed_to_cooldown = false;

      do {
        if (!mask1) {
          mask1 = mask2;
          indices1 = indices2;
          values1 = values2;
        }

        if (++masks_ptr >= masks_end) {
          proceed_to_cooldown = true;
          break;
        }

        mask2 = *masks_ptr;
        indices2 += 32;
        values2 += 32;

      } while (!mask1 || !mask2);

      if (proceed_to_cooldown) break;
    }
    const int offset2 = FindLSBSetNonZero(mask2);
    const int offset1 = FindLSBSetNonZero(mask1);

    *indices_write_ptr++ = indices2[offset2];
    *values_write_ptr++ = values2[offset2];

    *indices_write_ptr++ = indices1[offset1];
    *values_write_ptr++ = values1[offset1];

    mask2 &= (mask2 - 1);
    mask1 &= (mask1 - 1);
  }

  while (mask1) {
    const int offset1 = FindLSBSetNonZero(mask1);
    mask1 &= (mask1 - 1);
    *indices_write_ptr++ = indices1[offset1];
    *values_write_ptr++ = values1[offset1];
  }

  return indices_write_ptr - indices;
}

size_t UseMasksToCompactDoublePorted(
    uint32_t* indices, float* values, float* values_aux, uint32_t* masks, size_t n_masks) {

  std::copy(values, values + 64, values + n_masks * 32);
  std::copy(values_aux, values_aux + 64, values_aux + n_masks * 32);
  std::copy(indices, indices + 64, indices + n_masks * 32);
  std::copy(masks, masks + 2, masks + n_masks);
  n_masks += 2;

  uint32_t mask1 = masks[2];
  uint32_t* indices1 = indices + 2 * 32;
  float* values1 = values + 2 * 32;
  float* values1_aux = values_aux + 2 * 32;

  uint32_t mask2 = masks[3];
  uint32_t* indices2 = indices + 3 * 32;
  float* values2 = values + 3 * 32;
  float* values2_aux = values_aux + 3 * 32;

  uint32_t* masks_ptr = masks + 3;
  uint32_t* masks_end = masks + n_masks;

  uint32_t* indices_write_ptr = indices;
  float* values_write_ptr = values;
  float* values_aux_write_ptr = values_aux;

  for (;;) {
    if (!mask1 || !mask2) {
      bool proceed_to_cooldown = false;

      do {
        if (!mask1) {
          mask1 = mask2;
          indices1 = indices2;
          values1 = values2;
          values1_aux = values2_aux;
        }

        if (++masks_ptr >= masks_end) {
          proceed_to_cooldown = true;
          break;
        }

        mask2 = *masks_ptr;
        indices2 += 32;
        values2 += 32;
        values2_aux += 32;

      } while (!mask1 || !mask2);

      if (proceed_to_cooldown) break;
    }
    const int offset2 = FindLSBSetNonZero(mask2);
    const int offset1 = FindLSBSetNonZero(mask1);

    *indices_write_ptr++ = indices2[offset2];
    *values_write_ptr++ = values2[offset2];
    *values_aux_write_ptr++ = values2_aux[offset2];

    *indices_write_ptr++ = indices1[offset1];
    *values_write_ptr++ = values1[offset1];
    *values_aux_write_ptr++ = values1_aux[offset1];

    mask2 &= (mask2 - 1);
    mask1 &= (mask1 - 1);
  }

  while (mask1) {
    const int offset1 = FindLSBSetNonZero(mask1);
    mask1 &= (mask1 - 1);
    *indices_write_ptr++ = indices1[offset1];
    *values_write_ptr++ = values1[offset1];
    *values_aux_write_ptr++ = values1_aux[offset1];
  }

  return indices_write_ptr - indices;
}

inline size_t UseMasksToCompact(uint32_t* indices,
                                        float* values, uint32_t* masks,
                                        size_t n_masks) {
  if (n_masks == 1) {
    return UseMaskToCompact(indices, values, masks[0]);
  }
  return UseMasksToCompactDoublePorted(indices, values, masks, n_masks);
}

inline size_t UseMasksToCompact(uint32_t* indices,
                                        float* values, float* values_aux, uint32_t* masks,
                                        size_t n_masks) {
  if (n_masks == 1) {
    return UseMaskToCompact(indices, values, values_aux, masks[0]);
  }
  return UseMasksToCompactDoublePorted(indices, values, values_aux, masks, n_masks);
}

inline float DecrementThreshold(float threshold) {
    constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
    return std::nextafter(threshold, kNegativeInfinity);
}

inline bool CompIV(uint32_t idx_a, uint32_t idx_b,
                         float value_a, float value_b) {
  const bool is_eq_or_nan =
      value_a == value_b || std::isunordered(value_a, value_b);
  if (is_eq_or_nan) {
    return idx_a < idx_b;
  }
  return value_a < value_b;
}

inline void CompOrSwap(size_t a, size_t b, uint32_t* indices,
                             float* values) {
  if (!CompIV(indices[a], indices[b], values[a], values[b])) {
    ZipSwap(a, b, indices, values);
  }
}

inline void CompOrSwap(size_t a, size_t b, uint32_t* indices,
                             float* values, float* values_aux) {
  if (!CompIV(indices[a], indices[b], values[a], values[b])) {
    ZipSwap(a, b, indices, values, values_aux);
  }
}

inline void SelectionSort(uint32_t* indices, float* values,
                                size_t sz) {
  switch (sz) {
    case 3:

      CompOrSwap(0, 1, indices, values);
      CompOrSwap(1, 2, indices, values);
      //ABSL_FALLTHROUGH_INTENDED;
    case 2:

      CompOrSwap(0, 1, indices, values);
      //ABSL_FALLTHROUGH_INTENDED;
    case 1:

      break;
  }
}

inline void SelectionSort(uint32_t* indices, float* values, float* values_aux,
                                size_t sz) {
  switch (sz) {
    case 3:

      CompOrSwap(0, 1, indices, values, values_aux);
      CompOrSwap(1, 2, indices, values, values_aux);
      //ABSL_FALLTHROUGH_INTENDED;
    case 2:

      CompOrSwap(0, 1, indices, values, values_aux);
      //ABSL_FALLTHROUGH_INTENDED;
    case 1:

      break;
  }
}

FastMaxHeap::FastMaxHeap(uint32_t size, float* val, float* val_aux, uint32_t* tag, uint32_t* masks) :
    distances_(val), distances_aux_(val_aux), indices_(tag), masks_(masks), max_results_(size), epsilon_(std::numeric_limits<float>::infinity()) {
    capacity_ = max_capacity_ = NextMultipleOf(2 * max_results_, 32);
    sz_ = 0;
    epsilon_.store(std::numeric_limits<float>::infinity(), std::memory_order_relaxed);
    indices_end_ = indices_ + capacity_;
    distances_end_ = distances_ + capacity_;
    distances_aux_end_ = distances_aux_ + capacity_;
    pushes_remaining_negated_ = sz_ - capacity_;
    //LOG(INFO) << "pushes_remaining_negated_: " << pushes_remaining_negated_ << " " << sz_ << " " << capacity_;
}


bool FastMaxHeap::max_heap_update(const float new_val, const float new_val_aux, const uint32_t new_tag) {
    //LOG(INFO) << "indices_: " << indices_[0];
    //LOG(INFO) << "indices_: " << indices_[capacity_ - 1];
    indices_end_[pushes_remaining_negated_] = new_tag;
    //LOG(INFO) << "indices_: " << indices_[0];
    distances_end_[pushes_remaining_negated_] = new_val;
    //LOG(INFO) << "indices_: " << indices_[0];
    distances_aux_end_[pushes_remaining_negated_] = new_val_aux;
    //LOG(INFO) << "indices_: " << indices_[0];
    ++pushes_remaining_negated_;
    
    return pushes_remaining_negated_ == 0;
}

void FastMaxHeap::garbage_collect(size_t keep_min, size_t keep_max) {

    if (keep_min == 0) {
        sz_ = 0;
        return;
    }
    if (sz_ <= keep_max) return;

    { // ApproxNthElementImpl
        auto sz = sz_;
        auto values = distances_;
        auto indices = indices_;
        auto masks = masks_;
        float threshold_value;
        size_t n_already_kept = 0;
        bool skip_threshold_selection = false;
        //LOG(INFO) << "123";
        for (;;) {
            // LOG(INFO) << "123";
            if (!skip_threshold_selection) {
                if (sz <= 3) {
                    const size_t final_size = n_already_kept + keep_min;

                    SelectionSort(indices, values, sz);

                    values[keep_min] = values[keep_min - 1];
                    indices[keep_min] = indices[keep_min - 1];

                    sz_ = final_size;
                    break;
                }
                
                float big = std::max(values[0], values[sz / 2]);
                float sml = std::min(values[0], values[sz / 2]);
                threshold_value = std::max(sml, std::min(big, values[sz - 1]));

            }
            skip_threshold_selection = false;
        
            const uint32_t final_mask = GetFinalMask32(sz);
            const size_t n_masks = DivRoundUp(sz, 32);
            
            size_t n_kept = CalculateLtSwapMasks(values, masks, n_masks, final_mask,
                                                threshold_value);

   
            auto handle_overly_picky_pivot = [&]() {
                if (n_kept < sz * 3 / 4) {
                    uint32_t* eq_masks = masks + n_masks;
                    threshold_value =
                        UseMasksToFindNewMedian(values, masks, eq_masks, n_masks);
                    skip_threshold_selection = true;
                } else {
                    const size_t pivot_idx =
                        UseMasksToPartition(indices, values, masks, n_masks);
                    n_already_kept += n_kept;
                    keep_min -= n_kept;
                    keep_max -= n_kept;
                    sz -= n_kept;
                    indices += n_kept;
                    values += n_kept;
                }
            };

            const bool compute_eq_masks = n_kept < keep_min;
            
            if (compute_eq_masks) {
                const size_t n_needed = keep_min - n_kept;
                size_t n_found;
                uint32_t* eq_masks = masks + n_masks;
                n_found = CalculateEqSwapMasks(values, eq_masks, n_masks, final_mask,
                                                threshold_value);
        
                if (n_found < n_needed) {
                    handle_overly_picky_pivot();
                    continue;
                }

                uint32_t* scratch = indices + 32 * n_masks + 64;
                const size_t sz = UseMasksToSelect(scratch, indices, eq_masks, n_masks);

                if (n_found > n_needed) {
                    ZipNthElementBranchOptimized(std::less<uint32_t>(), n_needed - 1,
                                                scratch, scratch + n_found);
                }
            }
           
            
            sz = UseMasksToCompact(indices, values, masks, n_masks);

            if (n_kept > keep_max) {
                continue;
            }

            uint32_t tiebreaker_idx = std::numeric_limits<uint32_t>::max();
            if (compute_eq_masks) {
                const size_t n_needed = keep_min - n_kept;
                uint32_t* scratch = indices + 32 * n_masks + 64;
                std::copy(scratch, scratch + n_needed, indices + n_kept);
                std::fill(values + n_kept, values + n_kept + n_needed, threshold_value);
                n_kept = keep_min;
                tiebreaker_idx = scratch[n_needed - 1];
            } else {
                threshold_value = DecrementThreshold(threshold_value);
            }

            values[n_kept] = threshold_value;
            indices[n_kept] = tiebreaker_idx;

            const size_t final_size = n_already_kept + n_kept;
            
            sz_ = final_size;
            break;

        }
  
    } // end ApproxNthElementImpl

    epsilon_ = distances_[sz_];

    // init
    indices_end_ = indices_ + capacity_;
    distances_end_ = distances_ + capacity_;
    pushes_remaining_negated_ = sz_ - capacity_;

}

void FastMaxHeap::garbage_collect_aux(size_t keep_min, size_t keep_max) {

    if (keep_min == 0) {
        sz_ = 0;
        return;
    }
    if (sz_ <= keep_max) return;

    { // ApproxNthElementImpl
        auto sz = sz_;
        auto values = distances_;
        auto values_aux = distances_aux_;
        auto indices = indices_;
        auto masks = masks_;
        float threshold_value;
        size_t n_already_kept = 0;
        bool skip_threshold_selection = false;
        //LOG(INFO) << "123";
        for (;;) {
            //LOG(INFO) << "1";
            if (!skip_threshold_selection) {
                if (sz <= 3) {
                    const size_t final_size = n_already_kept + keep_min;

                    SelectionSort(indices, values, values_aux, sz);

                    values[keep_min] = values[keep_min - 1];
                    values_aux[keep_min] = values_aux[keep_min - 1];
                    indices[keep_min] = indices[keep_min - 1];

                    sz_ = final_size;
                    break;
                }
                
                float big = std::max(values[0], values[sz / 2]);
                float sml = std::min(values[0], values[sz / 2]);
                threshold_value = std::max(sml, std::min(big, values[sz - 1]));

            }
            skip_threshold_selection = false;
        
            const uint32_t final_mask = GetFinalMask32(sz);
            const size_t n_masks = DivRoundUp(sz, 32);
            
            size_t n_kept = CalculateLtSwapMasks(values, masks, n_masks, final_mask,
                                                threshold_value);

            //LOG(INFO) << "2";
            auto handle_overly_picky_pivot = [&]() {
                if (n_kept < sz * 3 / 4) {
                    uint32_t* eq_masks = masks + n_masks;
                    threshold_value =
                        UseMasksToFindNewMedian(values, masks, eq_masks, n_masks);
                    skip_threshold_selection = true;
                } else {
                    const size_t pivot_idx =
                        UseMasksToPartition(indices, values, values_aux, masks, n_masks);
                    n_already_kept += n_kept;
                    keep_min -= n_kept;
                    keep_max -= n_kept;
                    sz -= n_kept;
                    indices += n_kept;
                    values += n_kept;
                    values_aux += n_kept;
                }
            };

            const bool compute_eq_masks = n_kept < keep_min;
            
            if (compute_eq_masks) {
                const size_t n_needed = keep_min - n_kept;
                size_t n_found;
                uint32_t* eq_masks = masks + n_masks;
                n_found = CalculateEqSwapMasks(values, eq_masks, n_masks, final_mask,
                                                threshold_value);
        
                if (n_found < n_needed) {
                    handle_overly_picky_pivot();
                    continue;
                }

                uint32_t* scratch = indices + 32 * n_masks + 64;
                const size_t sz = UseMasksToSelect(scratch, indices, eq_masks, n_masks);

                if (n_found > n_needed) {
                    ZipNthElementBranchOptimized(std::less<uint32_t>(), n_needed - 1,
                                                scratch, scratch + n_found);
                }
            }
            //LOG(INFO) << "3";
            
            sz = UseMasksToCompact(indices, values, values_aux, masks, n_masks);

            if (n_kept > keep_max) {
                continue;
            }

            uint32_t tiebreaker_idx = std::numeric_limits<uint32_t>::max();
            float v_aux = 0.0f;
            if (compute_eq_masks) {
                const size_t n_needed = keep_min - n_kept;
                uint32_t* scratch = indices + 32 * n_masks + 64;
                float* scratch_aux = values_aux + 32 * n_masks + 64;
                std::copy(scratch, scratch + n_needed, indices + n_kept);
                std::fill(values + n_kept, values + n_kept + n_needed, threshold_value);
                std::copy(scratch_aux, scratch_aux + n_needed, values_aux + n_kept);
                n_kept = keep_min;
                tiebreaker_idx = scratch[n_needed - 1];
                v_aux = scratch_aux[n_needed - 1];
            } else {
                threshold_value = DecrementThreshold(threshold_value);
            }
            //LOG(INFO) << "4 " << tiebreaker_idx << " " << n_kept << " " << keep_max;
            values[n_kept] = threshold_value;
            //LOG(INFO) << "4 " << tiebreaker_idx << " " << n_kept << " " << values - distances_;
            indices[n_kept] = tiebreaker_idx;
            //LOG(INFO) << "4 " << tiebreaker_idx << " " << n_kept << " " << values_aux - distances_aux_ << " " << distances_aux_[0];
            if (compute_eq_masks) {
              //LOG(INFO) << "40 " << values_aux - distances_aux_ + n_kept << " " << values_aux - distances_aux_ + tiebreaker_idx;
              //LOG(INFO) << "40 " << " n_kept: " << n_kept << " " << values_aux[n_kept];
              //LOG(INFO) << "40 " << " tiebreaker_idx: " << tiebreaker_idx << " " << values_aux[tiebreaker_idx];
              values_aux[n_kept] = v_aux;
              //LOG(INFO) << "41";
            }

            const size_t final_size = n_already_kept + n_kept;
            
            sz_ = final_size;
            //LOG(INFO) << "5";
            break;

        }
  
    } // end ApproxNthElementImpl

    epsilon_ = distances_[sz_];

    // init
    indices_end_ = indices_ + capacity_;
    distances_end_ = distances_ + capacity_;
    distances_aux_end_ = distances_aux_ + capacity_;
    pushes_remaining_negated_ = sz_ - capacity_;

}

void FastMaxHeap::reorder() {
    sz_ = capacity_ + pushes_remaining_negated_;
    //LOG(INFO) << "be sz_: " << sz_;
    garbage_collect(max_results_, max_results_);
    //LOG(INFO) << "af sz_: " << sz_;
}

void FastMaxHeap::garbage_collect() {
    sz_ = capacity_ + pushes_remaining_negated_;

    // 
    size_t keep_max = (max_results_ + capacity_) / 2 - 1;
    size_t keep_min = max_results_;
    //LOG(INFO) << "be sz_: " << sz_;
    garbage_collect(keep_min, keep_max);
    //LOG(INFO) << "af sz_: " << sz_;
}

void FastMaxHeap::reorder_aux() {
    sz_ = capacity_ + pushes_remaining_negated_;
    //LOG(INFO) << "be sz_: " << sz_;
    garbage_collect_aux(max_results_, max_results_);
    //LOG(INFO) << "af sz_: " << sz_;
}

void FastMaxHeap::garbage_collect_aux() {
    sz_ = capacity_ + pushes_remaining_negated_;

    // 
    size_t keep_max = (max_results_ + capacity_) / 2 - 1;
    size_t keep_min = max_results_;
    //LOG(INFO) << "be sz_: " << sz_;
    garbage_collect_aux(keep_min, keep_max);
    //LOG(INFO) << "af sz_: " << sz_;
}

}