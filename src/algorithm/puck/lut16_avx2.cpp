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
#include <immintrin.h>
#include "algorithm/puck/lut16_avx2.h"

namespace puck {

__m256i CombineAvxLanes(__m256i a, __m256i b) {
  constexpr uint8_t kDestLoEqALo = 0x00;
  constexpr uint8_t kDestLoEqAHi = 0x01;
  constexpr uint8_t kDestHiEqBLo = 0x20;
  constexpr uint8_t kDestHiEqBHi = 0x30;
  constexpr uint8_t t1spec = (kDestLoEqALo + kDestHiEqBHi);
  constexpr uint8_t t2spec = (kDestLoEqAHi + kDestHiEqBLo);
  __m256i term0 = _mm256_permute2x128_si256(a, b, t1spec);
  __m256i term1 = _mm256_permute2x128_si256(a, b, t2spec);
  return _mm256_add_epi16(term0, term1);
}

__m256i PostprocessAccumulatorPair(
    const __m256i even_plus_tag_along_bits, const __m256i odd) {
  __m256i even = _mm256_sub_epi16(even_plus_tag_along_bits, _mm256_slli_epi16(odd, 8));

  __m256i lo_per_lane = _mm256_unpacklo_epi16(even, odd);
  __m256i hi_per_lane = _mm256_unpackhi_epi16(even, odd);

  return CombineAvxLanes(lo_per_lane, hi_per_lane);
}

uint32_t get_comparison_mask(__m256i a, __m256i b) {
  constexpr uint8_t kDestLoEqALo = 0x00;
  constexpr uint8_t kDestLoEqAHi = 0x01;
  constexpr uint8_t kDestHiEqBLo = 0x20;
  constexpr uint8_t kDestHiEqBHi = 0x30;
  constexpr uint8_t lo_spec = (kDestLoEqALo + kDestHiEqBLo);
  constexpr uint8_t hi_spec = (kDestLoEqAHi + kDestHiEqBHi);
  __m256i alo_blo = _mm256_permute2x128_si256(a, b, lo_spec);
  __m256i ahi_bhi = _mm256_permute2x128_si256(a, b, hi_spec);
  __m256i aa_bb = _mm256_packs_epi16(alo_blo, ahi_bhi);
  return _mm256_movemask_epi8(aa_bb);
}

void print_m256i_as_16bits(__m256i var) {
    // 提取出 16 个 16-bit 元素
    uint16_t elements[16];
    _mm256_storeu_si256((__m256i*)elements, var);

    // 打印每个 16-bit 元素
    for (int i = 0; i < 16; ++i) {
      //  std::cout << (int)elements[i] << " ";
    }
    //std::cout << std::endl; // 恢复正常的输出格式
}

int16_t get_int16_threshold(float float_threshold) {
  constexpr float kMaxValue = std::numeric_limits<int16_t>::max();

  return std::min(float_threshold, kMaxValue);
}

int find_lsb_set_non_zero(uint32_t n) {
  return __builtin_ctz(n);
}


uint32_t lookup_dist_table_16(std::vector<float>& dist, const uint8_t* data_start,
                        const uint8_t* lookups, size_t num_blocks, 
                        float multiplier, __m256& inv_mults, __m256i& simd_thresholds, 
                        __m256& simd_biases, float* fea_offset, __m256i* results, bool debug) {

    constexpr uint8_t bias = static_cast<uint8_t>(1) << ((sizeof(uint8_t) * 8) - 1);
    __m256i int16_accums[4];
    for (size_t i = 0; i < 4; i++) {
        int16_accums[i] = _mm256_setzero_si256();
    }
    __m256i sign7 = _mm256_set1_epi8(0x0F);

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


}
