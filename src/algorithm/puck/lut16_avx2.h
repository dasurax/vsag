#pragma once

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

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#elif defined(__ARM_FEATURE_SIMD32) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace puck {

__m256i CombineAvxLanes(__m256i a, __m256i b);

__m256i PostprocessAccumulatorPair(
    const __m256i even_plus_tag_along_bits, const __m256i odd);

uint32_t get_comparison_mask(__m256i a, __m256i b);

void print_m256i_as_16bits(__m256i var);

int16_t get_int16_threshold(float float_threshold);
int find_lsb_set_non_zero(uint32_t n);


uint32_t lookup_dist_table_16(std::vector<float>& dist, const uint8_t* data_start,
                        const uint8_t* lookups, size_t num_blocks, 
                        float multiplier, __m256& inv_mults, __m256i& simd_thresholds, 
                        __m256& simd_biases, float* fea_offset, __m256i* results, bool debug);

}
