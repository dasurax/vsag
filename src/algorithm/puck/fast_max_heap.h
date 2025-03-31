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
 * @file fast_max_heap.h
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2019/8/20 10:43
 * @brief
 *
 **/
#pragma once
#include <cstdio>
#include <limits>
#include <atomic>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
namespace puck {
class FastMaxHeap {
public:
    FastMaxHeap(uint32_t size, float* val, float* val_aux, uint32_t* tag, uint32_t* masks);
    ~FastMaxHeap() {}
    /*
     * @brief 小于堆顶元素时，需要更新堆
     * @@param [in] new_val : 距离
     * @@param [in] new_tag : 标记
     **/
    bool max_heap_update(const float new_val, const float new_val_aux, const uint32_t new_tag);
    /*
     * @brief 排序
     **/
    void reorder();

    void garbage_collect();
    void garbage_collect(size_t keep_min, size_t keep_max);

    void reorder_aux();

    void garbage_collect_aux();
    void garbage_collect_aux(size_t keep_min, size_t keep_max);

    /*
     * @brief 获取堆内元素个数
     **/
    uint32_t get_heap_size() {
        return sz_;
    }
    /*
     * @brief 获取堆顶val的指针
     **/
    float* get_top_addr() const {
        return 0;
    }

    float epsilon() const {
        return epsilon_.load(std::memory_order_relaxed);
    }

private:
    FastMaxHeap();
private:
    float* distances_;
    float* distances_aux_;
    uint32_t* indices_;
    uint32_t* masks_;
    size_t sz_ = 0;
    size_t max_results_;

    size_t capacity_ = 0;
    
    size_t max_capacity_ = 0;
    std::atomic<float> epsilon_;

    uint32_t* indices_end_;

    float* distances_end_;
    float* distances_aux_end_;

    ssize_t pushes_remaining_negated_;
};

}