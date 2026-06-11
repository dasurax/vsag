
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <optional>

#include "algorithm/sindi/sindi_parameter.h"
#include "hash_types.h"
#include "impl/searcher/basic_searcher.h"
#include "quantization/sparse_quantization//sparse_term_computer.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/pointer_define.h"
#include "vsag/allocator.h"
#include "vsag/dataset.h"

namespace vsag {

DEFINE_POINTER(SparseTermDataCell);
class SparseTermDataCell {
public:
    using MappedQueryTerms = Vector<std::pair<uint32_t, uint32_t>>;

    SparseTermDataCell() = default;

    SparseTermDataCell(float doc_retain_ratio,
                       uint32_t term_id_limit,
                       Allocator* allocator,
                       bool use_quantization,
                       std::shared_ptr<QuantizationParams> quantization_params,
                       bool use_local_term_map = false)
        : doc_retain_ratio_(doc_retain_ratio),
          term_id_limit_(term_id_limit),
          allocator_(allocator),
          term_ids_(allocator),
          term_datas_(allocator),
          term_sizes_(allocator),
          flat_term_offsets_(allocator),
          flat_term_ids_(allocator),
          flat_term_datas_(allocator),
          use_quantization_(use_quantization),
          quantization_params_(std::move(quantization_params)),
          local_to_global_terms_(allocator),
          sorted_global_to_local_terms_(allocator),
          use_local_term_map_(use_local_term_map) {
        if (use_local_term_map_) {
            global_to_local_terms_ = std::make_unique<UnorderedMap<uint32_t, uint32_t>>(allocator_);
        }
    }

    void
    Query(float* global_dists, const SparseTermComputerPtr& computer) const;

    void
    MapQueryTerms(const SparseTermComputerPtr& computer, MappedQueryTerms& mapped_terms) const;

    void
    QueryByMappedTerms(float* global_dists,
                       const SparseTermComputerPtr& computer,
                       const MappedQueryTerms& mapped_terms) const;

    /**
     * @brief Insert candidates into heap by iterating through term lists
     * 
     * @param dists Pre-allocated distance array (will be modified during processing)
     * @param computer SparseTermComputer for iterating through terms
     * @param heap MaxHeap to store candidate results
     * @param param Inner search parameters
     * @param offset_id Offset to add to inner IDs when inserting into heap
     */
    template <InnerSearchMode mode = InnerSearchMode::KNN_SEARCH,
              InnerSearchType type = InnerSearchType::PURE>
    void
    InsertHeapByTermLists(float* dists,
                          const SparseTermComputerPtr& computer,
                          MaxHeap& heap,
                          const InnerSearchParam& param,
                          uint32_t offset_id) const;

    template <InnerSearchMode mode = InnerSearchMode::KNN_SEARCH,
              InnerSearchType type = InnerSearchType::PURE>
    void
    InsertHeapByMappedTerms(float* dists,
                            const SparseTermComputerPtr& computer,
                            const MappedQueryTerms& mapped_terms,
                            MaxHeap& heap,
                            const InnerSearchParam& param,
                            uint32_t offset_id) const;

    /**
     * @brief Insert candidates into heap directly from precomputed distance array
     * 
     * @param dists Precomputed distance array (will be modified during processing)
     * @param dists_size Size of the distance array
     * @param heap MaxHeap to store candidate results
     * @param param Inner search parameters
     * @param offset_id Offset to add to inner IDs when inserting into heap
     */
    template <InnerSearchMode mode = InnerSearchMode::KNN_SEARCH,
              InnerSearchType type = InnerSearchType::PURE>
    void
    InsertHeapByDists(float* dists,
                      uint32_t dists_size,
                      MaxHeap& heap,
                      const InnerSearchParam& param,
                      uint32_t offset_id) const;

    void
    DocPrune(Vector<std::pair<uint32_t, float>>& sorted_base) const;

    void
    InsertVector(const SparseVector& sparse_base, uint16_t base_id);

    void
    InsertSortedVector(const Vector<std::pair<uint32_t, float>>& sorted_base, uint16_t base_id);

    void
    BuildFlatFromSortedVectors(Vector<Vector<std::pair<uint32_t, float>>>& sorted_bases);

    void
    ResizeTermList(InnerIdType new_term_capacity);

    uint32_t
    MapTermToLocalForBuild(uint32_t global_term);

    uint32_t
    MapTermToLocalForFlatBuild(uint32_t global_term);

    [[nodiscard]] std::optional<uint32_t>
    TryMapTermToLocal(uint32_t global_term) const;

    [[nodiscard]] uint32_t
    GetGlobalTerm(uint32_t local_term) const;

    void
    BuildSortedTermLookup();

    void
    Serialize(StreamWriter& writer) const;

    void
    ShrinkTermList();

    void
    FlattenTermLists();

    [[nodiscard]] bool
    IsFlatStorage() const {
        return use_flat_storage_;
    }

    [[nodiscard]] const uint16_t*
    GetTermIdsData(uint32_t term) const;

    [[nodiscard]] const uint8_t*
    GetTermDataBytes(uint32_t term) const;

    [[nodiscard]] size_t
    GetTermDataByteSize(uint32_t term) const;

    void
    Deserialize(StreamReader& reader);

    float
    CalcDistanceByInnerId(const SparseTermComputerPtr& computer, uint16_t base_id);

    void
    Encode(float val, uint8_t* dst) const;

    void
    Decode(const uint8_t* src, size_t size, float* dst) const;

    void
    GetSparseVector(uint32_t base_id, SparseVector* data, Allocator* specified_allocator);

    [[nodiscard]] int64_t
    GetMemoryUsage() const;

private:
    template <InnerSearchMode mode, InnerSearchType type>
    void
    insert_candidate_into_heap(uint32_t id,
                               float& dist,
                               float& cur_heap_top,
                               MaxHeap& heap,
                               uint32_t offset_id,
                               float radius,
                               const FilterPtr& filter) const;

    template <InnerSearchType type>
    bool
    fill_heap_initial(uint32_t id,
                      float& dist,
                      float& cur_heap_top,
                      MaxHeap& heap,
                      uint32_t offset_id,
                      uint32_t n_candidate,
                      const FilterPtr& filter) const;

public:
    uint32_t term_id_limit_{0};

    float doc_retain_ratio_{0};

    uint32_t term_capacity_{0};

    Vector<std::unique_ptr<Vector<uint16_t>>> term_ids_;

    Vector<std::unique_ptr<Vector<uint8_t>>> term_datas_;

    Vector<uint32_t> term_sizes_;

    Vector<uint32_t> flat_term_offsets_;

    Vector<uint16_t> flat_term_ids_;

    Vector<uint8_t> flat_term_datas_;

    bool use_flat_storage_{false};

    std::unique_ptr<UnorderedMap<uint32_t, uint32_t>> global_to_local_terms_{nullptr};

    Vector<uint32_t> local_to_global_terms_;

    Vector<std::pair<uint32_t, uint32_t>> sorted_global_to_local_terms_;

    bool use_local_term_map_{false};

    Allocator* const allocator_{nullptr};

    bool use_quantization_{false};

    int64_t total_count_{0};

    std::shared_ptr<QuantizationParams> quantization_params_;
};
}  // namespace vsag
