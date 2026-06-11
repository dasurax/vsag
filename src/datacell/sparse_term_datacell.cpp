
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

#include "sparse_term_datacell.h"

#include <algorithm>

#include "utils/util_functions.h"
#include "vsag/allocator.h"
#include "vsag_exception.h"
namespace vsag {
namespace {

size_t
TermDataUnitSize(bool use_quantization) {
    return use_quantization ? sizeof(uint8_t) : sizeof(float);
}

}  // namespace

void
SparseTermDataCell::Query(float* global_dists, const SparseTermComputerPtr& computer) const {
    MappedQueryTerms mapped_terms(allocator_);
    MapQueryTerms(computer, mapped_terms);
    QueryByMappedTerms(global_dists, computer, mapped_terms);
}

void
SparseTermDataCell::MapQueryTerms(const SparseTermComputerPtr& computer,
                                  SparseTermDataCell::MappedQueryTerms& mapped_terms) const {
    mapped_terms.clear();
    mapped_terms.reserve(computer->pruned_len_);
    for (uint32_t it = 0; it < computer->pruned_len_; ++it) {
        auto local_term = TryMapTermToLocal(computer->GetTerm(it));
        if (not local_term.has_value()) {
            continue;
        }
        auto term = local_term.value();
        if (term >= term_sizes_.size() || term_sizes_[term] == 0) {
            continue;
        }
        mapped_terms.emplace_back(term, it);
    }
}

void
SparseTermDataCell::QueryByMappedTerms(
    float* global_dists,
    const SparseTermComputerPtr& computer,
    const SparseTermDataCell::MappedQueryTerms& mapped_terms) const {
    for (uint32_t pos = 0; pos < mapped_terms.size(); ++pos) {
        auto term = mapped_terms[pos].first;
        auto it = mapped_terms[pos].second;
        if (pos + 1 < mapped_terms.size()) {
            auto next_term = mapped_terms[pos + 1].first;
            __builtin_prefetch(GetTermIdsData(next_term), 0, 3);
            __builtin_prefetch(GetTermDataBytes(next_term), 0, 3);
        }

        auto term_size = static_cast<uint32_t>(static_cast<float>(term_sizes_[term]) *
                                               computer->term_retain_ratio_);
        auto* term_ids = GetTermIdsData(term);
        auto* term_data = GetTermDataBytes(term);
        if (term_ids == nullptr || term_data == nullptr) {
            continue;
        }

        if (use_quantization_) {
            computer->ScanForAccumulate(it, term_ids, term_data, term_size, global_dists);
        } else {
            computer->ScanForAccumulate(it,
                                        term_ids,
                                        reinterpret_cast<const float*>(term_data),
                                        term_size,
                                        global_dists);
        }
    }
    computer->ResetTerm();
}

template <InnerSearchMode mode, InnerSearchType type>
void
SparseTermDataCell::insert_candidate_into_heap(uint32_t id,
                                               float& dist,
                                               float& cur_heap_top,
                                               MaxHeap& heap,
                                               uint32_t offset_id,
                                               float radius,
                                               const FilterPtr& filter) const {
    if constexpr (type == InnerSearchType::WITH_FILTER) {
#if __cplusplus >= 202002L
        if (dist > cur_heap_top or not filter->CheckValid(id + offset_id)) [[likely]] {
#else
        if (__builtin_expect(dist > cur_heap_top or not filter->CheckValid(id + offset_id), 1)) {
#endif
            dist = 0;
            return;
        }
    } else {
#if __cplusplus >= 202002L
        if (dist > cur_heap_top) [[likely]] {
#else
        if (__builtin_expect(dist > cur_heap_top, 1)) {
#endif
            dist = 0;
            return;
        }
    }
    heap.emplace(dist, id + offset_id);
    if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
        heap.pop();
        cur_heap_top = heap.top().first;
    }
    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        cur_heap_top = radius - 1;
    }
    dist = 0;
}

template <InnerSearchType type>
bool
SparseTermDataCell::fill_heap_initial(uint32_t id,
                                      float& dist,
                                      float& cur_heap_top,
                                      MaxHeap& heap,
                                      uint32_t offset_id,
                                      uint32_t n_candidate,
                                      const FilterPtr& filter) const {
    if (dist < 0) {
        if constexpr (type == InnerSearchType::WITH_FILTER) {
            if (not filter->CheckValid(id + offset_id)) {
                dist = 0;
                return false;
            }
        }
        heap.emplace(dist, id + offset_id);
        cur_heap_top = heap.top().first;
        dist = 0;
        return heap.size() == n_candidate;
    }
    return false;
}

template <InnerSearchMode mode, InnerSearchType type>
void
SparseTermDataCell::InsertHeapByTermLists(float* dists,
                                          const SparseTermComputerPtr& computer,
                                          MaxHeap& heap,
                                          const InnerSearchParam& param,
                                          uint32_t offset_id) const {
    MappedQueryTerms mapped_terms(allocator_);
    MapQueryTerms(computer, mapped_terms);
    InsertHeapByMappedTerms<mode, type>(dists, computer, mapped_terms, heap, param, offset_id);
}

template <InnerSearchMode mode, InnerSearchType type>
void
SparseTermDataCell::InsertHeapByMappedTerms(
    float* dists,
    const SparseTermComputerPtr& computer,
    const SparseTermDataCell::MappedQueryTerms& mapped_terms,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const {
    uint32_t id = 0;
    float cur_heap_top = std::numeric_limits<float>::max();
    auto n_candidate = param.ef;
    auto radius = param.radius;
    auto filter = param.is_inner_id_allowed;

    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        // note that radius = 1 - ip -> radius - 1 = 0 - ip
        // the dist in heap is equal to 0 - ip
        // thus, we need to compare dist with radius - 1
        cur_heap_top = radius - 1;
    }

    for (const auto& [term, term_iterator] : mapped_terms) {
        uint32_t i = 0;
        auto term_size = static_cast<uint32_t>(static_cast<float>(term_sizes_[term]) *
                                               computer->term_retain_ratio_);
        auto* one_term_ids = GetTermIdsData(term);
        if (one_term_ids == nullptr) {
            continue;
        }
        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if (heap.size() < n_candidate) {
                for (; i < term_size; i++) {
                    id = one_term_ids[i];
                    if (fill_heap_initial<type>(
                            id, dists[id], cur_heap_top, heap, offset_id, n_candidate, filter)) {
                        i++;
                        break;
                    }
                }
            }
        }

        for (; i < term_size; i++) {
            id = one_term_ids[i];
            insert_candidate_into_heap<mode, type>(
                id, dists[id], cur_heap_top, heap, offset_id, radius, filter);
        }
    }
    computer->ResetTerm();
}

template <InnerSearchMode mode, InnerSearchType type>
void
SparseTermDataCell::InsertHeapByDists(float* dists,
                                      uint32_t dists_size,
                                      MaxHeap& heap,
                                      const InnerSearchParam& param,
                                      uint32_t offset_id) const {
    float cur_heap_top = std::numeric_limits<float>::max();
    auto n_candidate = param.ef;
    auto radius = param.radius;
    auto filter = param.is_inner_id_allowed;

    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        cur_heap_top = radius - 1;
    }

    uint32_t id = 0;
    if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
        if (heap.size() < n_candidate) {
            for (; id < total_count_; id++) {
                if (fill_heap_initial<type>(
                        id, dists[id], cur_heap_top, heap, offset_id, n_candidate, filter)) {
                    id++;
                    break;
                }
            }
        }
    }

    for (; id < total_count_; id++) {
        insert_candidate_into_heap<mode, type>(
            id, dists[id], cur_heap_top, heap, offset_id, radius, filter);
    }
}

void
SparseTermDataCell::DocPrune(Vector<std::pair<uint32_t, float>>& sorted_base) const {
    // use this function when inserting
    if (sorted_base.size() <= 1 || doc_retain_ratio_ == 1) {
        return;
    }
    float total_mass = 0.0F;
    for (const auto& pair : sorted_base) {
        total_mass += pair.second;
    }

    float part_mass = total_mass * doc_retain_ratio_;
    float temp_mass = 0.0F;
    int pruned_doc_len = 0;

    while (temp_mass < part_mass) {
        temp_mass += sorted_base[pruned_doc_len++].second;
    }

    sorted_base.resize(pruned_doc_len);
}

uint32_t
SparseTermDataCell::MapTermToLocalForBuild(uint32_t global_term) {
    if (not use_local_term_map_) {
        ResizeTermList(global_term + 1);
        return global_term;
    }
    if (global_to_local_terms_ == nullptr) {
        global_to_local_terms_ = std::make_unique<UnorderedMap<uint32_t, uint32_t>>(allocator_);
    }
    auto it = global_to_local_terms_->find(global_term);
    if (it != global_to_local_terms_->end()) {
        return it->second;
    }
    auto local_term = static_cast<uint32_t>(local_to_global_terms_.size());
    (*global_to_local_terms_)[global_term] = local_term;
    local_to_global_terms_.push_back(global_term);
    sorted_global_to_local_terms_.clear();
    ResizeTermList(local_term + 1);
    return local_term;
}

uint32_t
SparseTermDataCell::MapTermToLocalForFlatBuild(uint32_t global_term) {
    if (global_term > term_id_limit_) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("term id {} is greater than term id limit {}", global_term, term_id_limit_));
    }
    if (not use_local_term_map_) {
        if (global_term >= term_sizes_.size()) {
            term_sizes_.resize(static_cast<size_t>(global_term) + 1, 0);
            term_capacity_ = static_cast<uint32_t>(term_sizes_.size());
        }
        return global_term;
    }
    if (global_to_local_terms_ == nullptr) {
        global_to_local_terms_ = std::make_unique<UnorderedMap<uint32_t, uint32_t>>(allocator_);
    }
    auto it = global_to_local_terms_->find(global_term);
    if (it != global_to_local_terms_->end()) {
        return it->second;
    }
    auto local_term = static_cast<uint32_t>(local_to_global_terms_.size());
    (*global_to_local_terms_)[global_term] = local_term;
    local_to_global_terms_.push_back(global_term);
    sorted_global_to_local_terms_.clear();
    term_sizes_.push_back(0);
    term_capacity_ = static_cast<uint32_t>(term_sizes_.size());
    return local_term;
}

std::optional<uint32_t>
SparseTermDataCell::TryMapTermToLocal(uint32_t global_term) const {
    if (not use_local_term_map_) {
        return global_term;
    }
    if (not sorted_global_to_local_terms_.empty()) {
        auto it = std::lower_bound(
            sorted_global_to_local_terms_.begin(),
            sorted_global_to_local_terms_.end(),
            global_term,
            [](const std::pair<uint32_t, uint32_t>& item, uint32_t term) {
                return item.first < term;
            });
        if (it != sorted_global_to_local_terms_.end() && it->first == global_term) {
            return it->second;
        }
        return std::nullopt;
    }
    if (global_to_local_terms_ == nullptr) {
        return std::nullopt;
    }
    auto it = global_to_local_terms_->find(global_term);
    if (it == global_to_local_terms_->end()) {
        return std::nullopt;
    }
    return it->second;
}

uint32_t
SparseTermDataCell::GetGlobalTerm(uint32_t local_term) const {
    return use_local_term_map_ ? local_to_global_terms_[local_term] : local_term;
}

void
SparseTermDataCell::BuildSortedTermLookup() {
    sorted_global_to_local_terms_.clear();
    if (not use_local_term_map_) {
        return;
    }
    sorted_global_to_local_terms_.reserve(local_to_global_terms_.size());
    for (uint32_t local_term = 0; local_term < local_to_global_terms_.size(); ++local_term) {
        sorted_global_to_local_terms_.emplace_back(local_to_global_terms_[local_term], local_term);
    }
    std::sort(sorted_global_to_local_terms_.begin(),
              sorted_global_to_local_terms_.end(),
              [](const std::pair<uint32_t, uint32_t>& lhs,
                 const std::pair<uint32_t, uint32_t>& rhs) { return lhs.first < rhs.first; });
}

const uint16_t*
SparseTermDataCell::GetTermIdsData(uint32_t term) const {
    if (term >= term_sizes_.size() || term_sizes_[term] == 0) {
        return nullptr;
    }
    if (use_flat_storage_) {
        return flat_term_ids_.data() + flat_term_offsets_[term];
    }
    if (term >= term_ids_.size() || term_ids_[term] == nullptr) {
        return nullptr;
    }
    return term_ids_[term]->data();
}

const uint8_t*
SparseTermDataCell::GetTermDataBytes(uint32_t term) const {
    if (term >= term_sizes_.size() || term_sizes_[term] == 0) {
        return nullptr;
    }
    if (use_flat_storage_) {
        return flat_term_datas_.data() +
               flat_term_offsets_[term] * TermDataUnitSize(use_quantization_);
    }
    if (term >= term_datas_.size() || term_datas_[term] == nullptr) {
        return nullptr;
    }
    return term_datas_[term]->data();
}

size_t
SparseTermDataCell::GetTermDataByteSize(uint32_t term) const {
    if (term >= term_sizes_.size() || term_sizes_[term] == 0) {
        return 0;
    }
    return static_cast<size_t>(term_sizes_[term]) * TermDataUnitSize(use_quantization_);
}

void
SparseTermDataCell::InsertVector(const SparseVector& sparse_base, uint16_t base_id) {
    if (use_flat_storage_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "cannot insert sparse vector into flattened term data cell");
    }
    // resize term
    uint32_t max_term_id = 0;
    for (auto i = 0; i < sparse_base.len_; i++) {
        auto term_id = sparse_base.ids_[i];
        max_term_id = std::max(max_term_id, term_id);
    }
    if (max_term_id > term_id_limit_) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("max term id of sparse vector {} is greater than term id limit {}",
                        max_term_id,
                        term_id_limit_));
    }

    Vector<std::pair<uint32_t, float>> sorted_base(allocator_);
    sort_sparse_vector(sparse_base, sorted_base);

    // doc prune
    DocPrune(sorted_base);

    InsertSortedVector(sorted_base, base_id);
}

void
SparseTermDataCell::InsertSortedVector(const Vector<std::pair<uint32_t, float>>& sorted_base,
                                       uint16_t base_id) {
    if (use_flat_storage_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "cannot insert sparse vector into flattened term data cell");
    }

    // insert vector
    for (const auto& item : sorted_base) {
        auto term = MapTermToLocalForBuild(item.first);
        auto val = item.second;

        if (term_sizes_[term] == 0) {  // create term until needed
            term_ids_[term] = std::make_unique<Vector<uint16_t>>(allocator_);
            term_datas_[term] = std::make_unique<Vector<uint8_t>>(allocator_);
        }

        term_ids_[term]->push_back(base_id);

        auto& data_vec = *term_datas_[term];
        if (use_quantization_) {
            uint8_t buffer;
            Encode(val, &buffer);
            data_vec.push_back(buffer);
        } else {
            auto old_size = data_vec.size();
            data_vec.resize(old_size + sizeof(float));
            *reinterpret_cast<float*>(data_vec.data() + old_size) = val;
        }

        term_sizes_[term] += 1;
    }
    total_count_++;
}

void
SparseTermDataCell::ResizeTermList(InnerIdType new_term_capacity) {
    if (use_flat_storage_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "cannot resize flattened sparse term data cell");
    }
    if (new_term_capacity <= term_capacity_) {
        return;
    }
    InnerIdType new_capacity = term_capacity_ == 0 ? new_term_capacity : term_capacity_;
    while (new_capacity < new_term_capacity) {
        if (new_capacity > std::numeric_limits<InnerIdType>::max() / 2) {
            new_capacity = new_term_capacity;
            break;
        }
        new_capacity *= 2;
    }
    Vector<std::unique_ptr<Vector<uint16_t>>> new_ids(new_capacity, allocator_);
    Vector<std::unique_ptr<Vector<uint8_t>>> new_datas(new_capacity, allocator_);
    Vector<uint32_t> new_sizes(new_capacity, 0, allocator_);

    std::move(term_ids_.begin(), term_ids_.end(), new_ids.begin());
    std::move(term_datas_.begin(), term_datas_.end(), new_datas.begin());
    std::copy(term_sizes_.begin(), term_sizes_.end(), new_sizes.begin());

    term_ids_.swap(new_ids);
    term_datas_.swap(new_datas);
    term_sizes_.swap(new_sizes);
    term_capacity_ = new_capacity;
}

void
SparseTermDataCell::ShrinkTermList() {
    uint32_t real_capacity = 0;
    for (auto i = term_sizes_.size(); i > 0; --i) {
        if (term_sizes_[i - 1] != 0) {
            real_capacity = static_cast<uint32_t>(i);
            break;
        }
    }

    if (real_capacity >= term_capacity_) {
        return;
    }

    if (use_flat_storage_) {
        flat_term_offsets_.resize(static_cast<size_t>(real_capacity) + 1);
    } else {
        term_ids_.resize(real_capacity);
        term_datas_.resize(real_capacity);
    }
    term_sizes_.resize(real_capacity);
    term_capacity_ = real_capacity;
}

void
SparseTermDataCell::FlattenTermLists() {
    if (use_flat_storage_) {
        return;
    }
    ShrinkTermList();

    uint64_t posting_count = 0;
    for (auto size : term_sizes_) {
        posting_count += size;
    }
    if (posting_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "flattened sparse posting list is too large");
    }
    if (posting_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "flattened sparse posting list offset exceeds uint32 range");
    }

    Vector<uint32_t> flat_offsets(term_sizes_.size() + 1, 0, allocator_);
    Vector<uint16_t> flat_ids(allocator_);
    Vector<uint8_t> flat_datas(allocator_);
    flat_ids.resize(static_cast<size_t>(posting_count));
    flat_datas.resize(static_cast<size_t>(posting_count) * TermDataUnitSize(use_quantization_));

    size_t posting_offset = 0;
    size_t data_offset = 0;
    const auto data_unit_size = TermDataUnitSize(use_quantization_);
    for (uint32_t term = 0; term < term_sizes_.size(); ++term) {
        flat_offsets[term] = static_cast<uint32_t>(posting_offset);
        const auto term_size = term_sizes_[term];
        if (term_size == 0) {
            flat_offsets[term + 1] = static_cast<uint32_t>(posting_offset);
            continue;
        }
        if (term_ids_[term] == nullptr || term_datas_[term] == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "non-empty sparse term has empty posting storage");
        }
        if (term_ids_[term]->size() != term_size ||
            term_datas_[term]->size() != static_cast<size_t>(term_size) * data_unit_size) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "sparse term posting size does not match term size");
        }
        std::memcpy(flat_ids.data() + posting_offset,
                    term_ids_[term]->data(),
                    static_cast<size_t>(term_size) * sizeof(uint16_t));
        std::memcpy(flat_datas.data() + data_offset,
                    term_datas_[term]->data(),
                    static_cast<size_t>(term_size) * data_unit_size);
        posting_offset += term_size;
        data_offset += static_cast<size_t>(term_size) * data_unit_size;
        flat_offsets[term + 1] = static_cast<uint32_t>(posting_offset);
    }

    flat_term_offsets_.swap(flat_offsets);
    flat_term_ids_.swap(flat_ids);
    flat_term_datas_.swap(flat_datas);

    Vector<std::unique_ptr<Vector<uint16_t>>> empty_ids(allocator_);
    Vector<std::unique_ptr<Vector<uint8_t>>> empty_datas(allocator_);
    term_ids_.swap(empty_ids);
    term_datas_.swap(empty_datas);
    use_flat_storage_ = true;
    if (use_local_term_map_) {
        BuildSortedTermLookup();
        global_to_local_terms_.reset();
    }
}

void
SparseTermDataCell::BuildFlatFromSortedVectors(
    Vector<Vector<std::pair<uint32_t, float>>>& sorted_bases) {
    if (use_flat_storage_ || total_count_ != 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "cannot bulk build a non-empty sparse term data cell");
    }
    if (sorted_bases.size() >
        static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "sparse term data cell window exceeds uint16 id range");
    }

    Vector<std::unique_ptr<Vector<uint16_t>>> empty_ids(allocator_);
    Vector<std::unique_ptr<Vector<uint8_t>>> empty_datas(allocator_);
    term_ids_.swap(empty_ids);
    term_datas_.swap(empty_datas);
    term_sizes_.clear();
    flat_term_offsets_.clear();
    flat_term_ids_.clear();
    flat_term_datas_.clear();
    local_to_global_terms_.clear();
    sorted_global_to_local_terms_.clear();
    if (global_to_local_terms_ != nullptr) {
        global_to_local_terms_->clear();
    } else if (use_local_term_map_) {
        global_to_local_terms_ = std::make_unique<UnorderedMap<uint32_t, uint32_t>>(allocator_);
    }
    term_capacity_ = 0;

    uint64_t posting_count = 0;
    for (auto& sorted_base : sorted_bases) {
        for (auto& item : sorted_base) {
            auto term = MapTermToLocalForFlatBuild(item.first);
            item.first = term;
            term_sizes_[term] += 1;
            posting_count += 1;
        }
    }
    if (posting_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "flattened sparse posting list offset exceeds uint32 range");
    }

    flat_term_offsets_.resize(static_cast<size_t>(term_capacity_) + 1);
    uint32_t offset = 0;
    for (uint32_t term = 0; term < term_capacity_; ++term) {
        flat_term_offsets_[term] = offset;
        offset += term_sizes_[term];
    }
    flat_term_offsets_[term_capacity_] = offset;

    flat_term_ids_.resize(static_cast<size_t>(posting_count));
    flat_term_datas_.resize(static_cast<size_t>(posting_count) *
                            TermDataUnitSize(use_quantization_));
    Vector<uint32_t> write_offsets(allocator_);
    write_offsets.assign(flat_term_offsets_.begin(), flat_term_offsets_.end());

    for (size_t doc_idx = 0; doc_idx < sorted_bases.size(); ++doc_idx) {
        auto base_id = static_cast<uint16_t>(doc_idx);
        for (const auto& item : sorted_bases[doc_idx]) {
            auto term = item.first;
            auto write_offset = write_offsets[term]++;
            flat_term_ids_[write_offset] = base_id;
            auto data_offset = static_cast<size_t>(write_offset) *
                               TermDataUnitSize(use_quantization_);
            if (use_quantization_) {
                Encode(item.second, flat_term_datas_.data() + data_offset);
            } else {
                std::memcpy(flat_term_datas_.data() + data_offset, &item.second, sizeof(float));
            }
        }
    }

    for (uint32_t term = 0; term < term_capacity_; ++term) {
        if (write_offsets[term] != flat_term_offsets_[term + 1]) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "flat sparse build posting count mismatch");
        }
    }

    use_flat_storage_ = true;
    total_count_ = static_cast<int64_t>(sorted_bases.size());
    if (use_local_term_map_) {
        BuildSortedTermLookup();
        global_to_local_terms_.reset();
    }
}

float
SparseTermDataCell::CalcDistanceByInnerId(const SparseTermComputerPtr& computer, uint16_t base_id) {
    float ip = 0;
    Vector<float> temp_data(allocator_);
    while (computer->HasNextTerm()) {
        auto it = computer->NextTermIter();
        auto local_term = TryMapTermToLocal(computer->GetTerm(it));
        if (not local_term.has_value()) {
            continue;
        }
        auto term = local_term.value();
        if (computer->HasNextTerm()) {
            auto next_it = it + 1;
            auto next_term = TryMapTermToLocal(computer->GetTerm(next_it));
            if (next_term.has_value() && next_term.value() < term_sizes_.size() &&
                term_sizes_[next_term.value()] != 0) {
                __builtin_prefetch(GetTermIdsData(next_term.value()), 0, 3);
                __builtin_prefetch(GetTermDataBytes(next_term.value()), 0, 3);
            }
        }
        // Fix: Check term_sizes_[term] == 0 to avoid null pointer dereference
        if (term >= term_sizes_.size() || term_sizes_[term] == 0) {
            continue;
        }

        // Dequantize
        const float* vals = nullptr;
        auto size = term_sizes_[term];
        auto* term_ids = GetTermIdsData(term);
        auto* term_data = GetTermDataBytes(term);
        if (term_ids == nullptr || term_data == nullptr) {
            continue;
        }
        if (use_quantization_) {
            temp_data.resize(size);
            Decode(term_data, size, temp_data.data());
            vals = temp_data.data();
        } else {
            vals = reinterpret_cast<const float*>(term_data);
        }

        computer->ScanForCalculateDist(it, term_ids, vals, term_sizes_[term], base_id, &ip);
    }
    computer->ResetTerm();
    return 1 + ip;
}

int64_t
SparseTermDataCell::GetMemoryUsage() const {
    auto memory = sizeof(SparseTermDataCell);
    if (use_flat_storage_) {
        memory += flat_term_offsets_.size() * sizeof(uint32_t);
        memory += flat_term_ids_.size() * sizeof(uint16_t);
        memory += flat_term_datas_.size() * sizeof(uint8_t);
    } else {
        memory += term_ids_.size() * sizeof(std::unique_ptr<Vector<uint16_t>>);
        memory += term_datas_.size() * sizeof(std::unique_ptr<Vector<uint8_t>>);
        for (const auto& ptr : term_ids_) {
            if (ptr != nullptr) {
                memory += ptr->size() * sizeof(uint16_t);
            }
        }
        for (const auto& ptr : term_datas_) {
            if (ptr != nullptr) {
                memory += ptr->size() * sizeof(uint8_t);
            }
        }
    }
    memory += sizeof(QuantizationParams);
    memory += term_sizes_.size() * sizeof(uint32_t);
    memory += local_to_global_terms_.size() * sizeof(uint32_t);
    memory += sorted_global_to_local_terms_.size() * sizeof(std::pair<uint32_t, uint32_t>);
    if (global_to_local_terms_ != nullptr) {
        memory += sizeof(UnorderedMap<uint32_t, uint32_t>);
        memory += global_to_local_terms_->size() *
                  (sizeof(std::pair<uint32_t, uint32_t>) + sizeof(void*) * 2);
    }
    return static_cast<int64_t>(memory);
}

void
SparseTermDataCell::GetSparseVector(uint32_t base_id,
                                    SparseVector* data,
                                    Allocator* specified_allocator) {
    Allocator* allocator = specified_allocator != nullptr ? specified_allocator : allocator_;

    Vector<uint32_t> ids(allocator);
    Vector<float> vals(allocator);

    for (auto term = 0; term < term_sizes_.size(); term++) {
        if (term_sizes_[term] == 0) {
            continue;
        }
        auto* one_term_ids = GetTermIdsData(static_cast<uint32_t>(term));
        auto* term_data = GetTermDataBytes(static_cast<uint32_t>(term));
        if (one_term_ids == nullptr || term_data == nullptr) {
            continue;
        }
        for (auto i = 0; i < term_sizes_[term]; i++) {
            if (one_term_ids[i] == base_id) {
                ids.push_back(GetGlobalTerm(static_cast<uint32_t>(term)));
                float v;
                if (use_quantization_) {
                    Decode(term_data + i, 1, &v);
                } else {
                    v = reinterpret_cast<const float*>(term_data)[i];
                }
                vals.push_back(v);
            }
        }
    }

    data->len_ = ids.size();
    data->ids_ = static_cast<uint32_t*>(allocator->Allocate(sizeof(uint32_t) * data->len_));
    data->vals_ = static_cast<float*>(allocator->Allocate(sizeof(float) * data->len_));

    memcpy(data->ids_, ids.data(), data->len_ * sizeof(uint32_t));
    memcpy(data->vals_, vals.data(), data->len_ * sizeof(float));
}

template <typename T, typename U>
void
convert(const Vector<T>& input, Vector<U>& output) {
    output.clear();
    output.reserve(input.size());
    for (const auto& value : input) {
        output.push_back(static_cast<U>(value));
    }
}

void
SparseTermDataCell::Serialize(StreamWriter& writer) const {
    auto serialize_term_capacity = use_local_term_map_
                                       ? static_cast<uint32_t>(local_to_global_terms_.size())
                                       : term_capacity_;
    StreamWriter::WriteObj(writer, serialize_term_capacity);
    Vector<float> empty_data(allocator_);
    Vector<uint32_t> empty_ids(allocator_);
    Vector<float> buffer_data(allocator_);
    Vector<uint32_t> buffer_ids(allocator_);
    for (auto i = 0; i < serialize_term_capacity; i++) {
        if (term_sizes_[i] != 0) {
            auto* term_ids = GetTermIdsData(i);
            auto* term_data = GetTermDataBytes(i);
            if (term_ids == nullptr || term_data == nullptr) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "non-empty sparse term has empty posting storage");
            }
            buffer_ids.clear();
            buffer_ids.reserve(term_sizes_[i]);
            for (uint32_t id_idx = 0; id_idx < term_sizes_[i]; ++id_idx) {
                buffer_ids.push_back(term_ids[id_idx]);
            }
            StreamWriter::WriteVector(writer, buffer_ids);
            auto term_data_size = GetTermDataByteSize(i);
            auto buffer_size =
                align_up(static_cast<int64_t>(term_data_size), sizeof(float)) / sizeof(float);
            buffer_data.resize(buffer_size);
            std::memset(buffer_data.data(), 0, sizeof(float) * buffer_data.size());
            std::memcpy(buffer_data.data(), term_data, term_data_size);
            StreamWriter::WriteVector(writer, buffer_data);
        } else {
            StreamWriter::WriteVector(writer, empty_ids);
            StreamWriter::WriteVector(writer, empty_data);
        }
    }
    Vector<uint32_t> serialize_term_sizes(allocator_);
    serialize_term_sizes.reserve(serialize_term_capacity);
    for (uint32_t i = 0; i < serialize_term_capacity; ++i) {
        serialize_term_sizes.push_back(term_sizes_[i]);
    }
    StreamWriter::WriteVector(writer, serialize_term_sizes);
    if (use_local_term_map_) {
        StreamWriter::WriteVector(writer, local_to_global_terms_);
    }
}

void
SparseTermDataCell::Deserialize(StreamReader& reader) {
    uint32_t term_capacity;
    StreamReader::ReadObj(reader, term_capacity);

    Vector<std::unique_ptr<Vector<uint16_t>>> empty_ids(allocator_);
    Vector<std::unique_ptr<Vector<uint8_t>>> empty_datas(allocator_);
    term_ids_.swap(empty_ids);
    term_datas_.swap(empty_datas);

    term_capacity_ = term_capacity;
    use_flat_storage_ = true;
    Vector<uint32_t> new_sizes(term_capacity, 0, allocator_);
    term_sizes_.swap(new_sizes);
    Vector<uint32_t> new_offsets(static_cast<size_t>(term_capacity) + 1, 0, allocator_);
    flat_term_offsets_.swap(new_offsets);
    flat_term_ids_.clear();
    flat_term_datas_.clear();

    Vector<uint32_t> ids_buffer(allocator_);
    Vector<float> data_buffer(allocator_);
    for (auto i = 0; i < term_capacity; i++) {
        StreamReader::ReadVector(reader, ids_buffer);
        StreamReader::ReadVector(reader, data_buffer);
        if (flat_term_ids_.size() > std::numeric_limits<uint32_t>::max()) {
            throw VsagException(ErrorType::READ_ERROR,
                                "sparse posting list offset exceeds uint32 range");
        }
        flat_term_offsets_[i] = static_cast<uint32_t>(flat_term_ids_.size());
        if (not ids_buffer.empty()) {
            term_sizes_[i] = static_cast<uint32_t>(ids_buffer.size());
            auto old_id_size = flat_term_ids_.size();
            flat_term_ids_.resize(old_id_size + ids_buffer.size());
            for (uint32_t id_idx = 0; id_idx < ids_buffer.size(); ++id_idx) {
                if (ids_buffer[id_idx] > std::numeric_limits<uint16_t>::max()) {
                    throw VsagException(ErrorType::READ_ERROR,
                                        "sparse posting inner id exceeds uint16 range");
                }
                flat_term_ids_[old_id_size + id_idx] = static_cast<uint16_t>(ids_buffer[id_idx]);
            }

            auto data_byte_size = ids_buffer.size() * TermDataUnitSize(use_quantization_);
            auto available_byte_size = data_buffer.size() * sizeof(float);
            if (available_byte_size < data_byte_size) {
                throw VsagException(ErrorType::READ_ERROR,
                                    "sparse term data payload is shorter than posting ids");
            }
            auto old_data_size = flat_term_datas_.size();
            flat_term_datas_.resize(old_data_size + data_byte_size);
            std::memcpy(flat_term_datas_.data() + old_data_size,
                        data_buffer.data(),
                        data_byte_size);
        }
        if (flat_term_ids_.size() > std::numeric_limits<uint32_t>::max()) {
            throw VsagException(ErrorType::READ_ERROR,
                                "sparse posting list offset exceeds uint32 range");
        }
        flat_term_offsets_[i + 1] = static_cast<uint32_t>(flat_term_ids_.size());
    }
    Vector<uint32_t> serialized_term_sizes(allocator_);
    StreamReader::ReadVector(reader, serialized_term_sizes);
    if (serialized_term_sizes.size() != term_capacity) {
        throw VsagException(
            ErrorType::READ_ERROR,
            fmt::format("term size vector size ({}) does not match term capacity ({})",
                        serialized_term_sizes.size(),
                        term_capacity));
    }
    for (uint32_t i = 0; i < term_capacity; ++i) {
        if (serialized_term_sizes[i] != term_sizes_[i]) {
            throw VsagException(
                ErrorType::READ_ERROR,
                fmt::format("term size ({}) does not match posting list size ({}) for term {}",
                            serialized_term_sizes[i],
                            term_sizes_[i],
                            i));
        }
    }
    term_sizes_.swap(serialized_term_sizes);
    if (use_local_term_map_) {
        StreamReader::ReadVector(reader, local_to_global_terms_);
        if (local_to_global_terms_.size() != term_sizes_.size()) {
            throw VsagException(
                ErrorType::READ_ERROR,
                fmt::format("local term map size ({}) does not match term size vector ({})",
                            local_to_global_terms_.size(),
                            term_sizes_.size()));
        }
        BuildSortedTermLookup();
        global_to_local_terms_.reset();
    } else {
        ShrinkTermList();
    }

    // Restore total_count_ from deserialized data (not serialized to maintain compatibility)
    total_count_ = 0;
    for (uint32_t term = 0; term < term_sizes_.size(); ++term) {
        auto* term_ids = GetTermIdsData(term);
        if (term_ids == nullptr) {
            continue;
        }
        for (uint32_t i = 0; i < term_sizes_[term]; ++i) {
            total_count_ = std::max(total_count_, static_cast<int64_t>(term_ids[i]) + 1);
        }
    }
}

void
SparseTermDataCell::Encode(float val, uint8_t* dst) const {
    float x = (val - quantization_params_->min_val) / quantization_params_->diff * 255.0F;
    *dst = static_cast<uint8_t>(std::clamp(x, 0.0F, 255.0F));
}

void
SparseTermDataCell::Decode(const uint8_t* src, size_t size, float* dst) const {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = static_cast<float>(src[i]) / 255.0F * quantization_params_->diff +
                 quantization_params_->min_val;
    }
}

template void
SparseTermDataCell::InsertHeapByTermLists<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByTermLists<InnerSearchMode::KNN_SEARCH,
                                          InnerSearchType::WITH_FILTER>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByTermLists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::PURE>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByTermLists<InnerSearchMode::RANGE_SEARCH,
                                          InnerSearchType::WITH_FILTER>(
    float* dists,
    const SparseTermComputerPtr& computer,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByMappedTerms<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
    float* dists,
    const SparseTermComputerPtr& computer,
    const SparseTermDataCell::MappedQueryTerms& mapped_terms,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByMappedTerms<InnerSearchMode::KNN_SEARCH,
                                            InnerSearchType::WITH_FILTER>(
    float* dists,
    const SparseTermComputerPtr& computer,
    const SparseTermDataCell::MappedQueryTerms& mapped_terms,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByMappedTerms<InnerSearchMode::RANGE_SEARCH,
                                            InnerSearchType::PURE>(
    float* dists,
    const SparseTermComputerPtr& computer,
    const SparseTermDataCell::MappedQueryTerms& mapped_terms,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByMappedTerms<InnerSearchMode::RANGE_SEARCH,
                                            InnerSearchType::WITH_FILTER>(
    float* dists,
    const SparseTermComputerPtr& computer,
    const SparseTermDataCell::MappedQueryTerms& mapped_terms,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByDists<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByDists<InnerSearchMode::KNN_SEARCH, InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::PURE>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

template void
SparseTermDataCell::InsertHeapByDists<InnerSearchMode::RANGE_SEARCH, InnerSearchType::WITH_FILTER>(
    float* dists,
    uint32_t dists_size,
    MaxHeap& heap,
    const InnerSearchParam& param,
    uint32_t offset_id) const;

}  // namespace vsag
