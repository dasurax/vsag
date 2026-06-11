
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

#include <sstream>

#include "impl/allocator/safe_allocator.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("SparseTermDatacell Basic Test", "[ut][SparseTermDatacell]") {
    // prepare data
    auto count_base = 10;
    auto len_base = 10;
    std::vector<SparseVector> sparse_vectors(count_base);
    for (int i = 0; i < count_base; i++) {
        sparse_vectors[i].len_ = len_base;
        sparse_vectors[i].ids_ = new uint32_t[sparse_vectors[i].len_];
        sparse_vectors[i].vals_ = new float[sparse_vectors[i].len_];
        // base[0] = [0:0, 1:1, 2:2, ..., 9:9] = after_prune = [7:7, 8:8, 9:9]
        // base[1] = [1:1, 2:2, 3:3, ..., 10:10] = after_prune = [7:7, 8:8, 9:9, 10:10]
        // base[2] = [2:2, 3:3, 4:4, ..., 11:11] = after_prune = [8:8, 9:9, 10:10, 11:11]
        // base[3] = [3:3, 4:4, 5:5, ..., 12:12] = after_prune = [9:9, 10:10, 11:11, 12:12]
        // base[4] = [4:4, 5:5, 6:6, ..., 13:13] = after_prune = [10:10, 11:11, 12:12, 13:13]
        // base[5] = [5:5, 6:6, 7:7, ..., 14:14] = after_prune = [11:11, 12:12, 13:13, 14:14]
        // base[6] = [6:6, 7:7, 8:8, ..., 15:15] = after_prune = [12:12, 13:13, 14:14, 15:15]
        // base[7] = [7:7, 8:8, 9:9, ..., 16:16] = after_prune = [13:13, 14:14, 15:15, 16:16]
        // base[8] = [8:8, 9:9, 10:10, ..., 17:17]
        // after_prune = [13:13, 14:14, 15:15, 16:16, 17:17]
        // base[9] = [9:9, 10:10, 11:11, ..., 18:18]
        // after_prune = [14:14, 15:15, 16:16, 17:17, 18:18]
        for (int d = 0; d < sparse_vectors[i].len_; d++) {
            sparse_vectors[i].ids_[d] = i + d;
            sparse_vectors[i].vals_[d] = i + d;
        }
    }

    // query: [0:1, 1:1, 2:1 .... 18:1]
    // dis(q, b0) = 9 + 8 + 7 = 24
    // dis(q, b1) = 10 + 9 + 8 + 7 = 34
    // ...
    // dis(q, b9) = 80
    SparseVector query_sv;
    query_sv.len_ = 19;
    query_sv.ids_ = new uint32_t[query_sv.len_];
    query_sv.vals_ = new float[query_sv.len_];
    for (int d = 0; d < query_sv.len_; d++) {
        query_sv.ids_[d] = d;
        query_sv.vals_[d] = 1;
    }

    // prepare data_cell
    float query_prune_ratio = 0.0;
    float doc_retain_ratio = 0.5;
    float term_prune_ratio = 0.0;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    // disable quantization for this basic test
    std::shared_ptr<QuantizationParams> q_params = nullptr;
    auto data_cell = std::make_shared<SparseTermDataCell>(
        doc_retain_ratio, DEFAULT_TERM_ID_LIMIT, allocator.get(), false, q_params);
    REQUIRE(std::abs(data_cell->doc_retain_ratio_ - doc_retain_ratio) < 1e-3);

    // test factory computer
    SINDISearchParameter search_params;
    search_params.term_prune_ratio = term_prune_ratio;
    search_params.query_prune_ratio = query_prune_ratio;
    auto computer = std::make_shared<SparseTermComputer>(query_sv, search_params, allocator.get());
    REQUIRE(computer->pruned_len_ == (1.0F - query_prune_ratio) * query_sv.len_);

    // test insert
    auto exp_id_size = 19;
    std::vector<uint32_t> exp_size = {0, 0, 0, 0, 0, 0, 0, 2, 3, 4, 4, 4, 4, 5, 5, 4, 3, 2, 1};
    for (auto i = 0; i < count_base; i++) {
        data_cell->InsertVector(sparse_vectors[i], i);
    }
    REQUIRE(data_cell->term_capacity_ >= exp_id_size);
    REQUIRE(data_cell->term_ids_.size() == data_cell->term_capacity_);
    REQUIRE(data_cell->term_datas_.size() == data_cell->term_capacity_);
    for (auto i = 0; i < exp_id_size; i++) {
        if (exp_size[i] == 0) {
            REQUIRE(data_cell->term_ids_[i] == nullptr);
            REQUIRE(data_cell->term_datas_[i] == nullptr);
        } else {
            REQUIRE(data_cell->term_ids_[i]->size() == data_cell->term_sizes_[i]);
            REQUIRE(data_cell->term_ids_[i]->size() == exp_size[i]);
            REQUIRE(data_cell->term_datas_[i]->size() == exp_size[i] * sizeof(float));
        }
    }
    for (auto i = exp_id_size; i < data_cell->term_capacity_; i++) {
        REQUIRE(data_cell->term_ids_[i] == nullptr);
        REQUIRE(data_cell->term_datas_[i] == nullptr);
    }

    // Calculate expected distances programmatically to match the test logic
    std::vector<float> exp_dists(count_base, 0.0f);
    for (int i = 0; i < count_base; ++i) {
        // 1. Get the original vector and sort it
        const auto& vec = sparse_vectors[i];
        Vector<std::pair<uint32_t, float>> sorted_base(allocator.get());
        sort_sparse_vector(vec, sorted_base);

        // 2. Call the actual DocPrune function
        data_cell->DocPrune(sorted_base);

        // 3. Simulate quantization and inner product calculation
        float total_dist = 0.0f;
        for (const auto& pair : sorted_base) {
            float val = pair.second;
            float query_val = -1.0f;  // The computer uses -1.0 as query value
            total_dist += query_val * val;
        }
        exp_dists[i] = total_dist;
    }

    SECTION("test query") {
        std::vector<float> dists(count_base, 0);
        data_cell->Query(dists.data(), computer);
        for (auto i = 0; i < dists.size(); i++) {
            REQUIRE(std::abs(dists[i] - exp_dists[i]) < 1e-3);
        }
    }

    SECTION("test insert heap in knn search") {
        auto topk = 5;
        auto pos = count_base - topk;
        InnerSearchParam inner_param;
        inner_param.ef = topk;
        MaxHeap heap(allocator.get());
        std::vector<float> dists(count_base, 0);
        data_cell->Query(dists.data(), computer);

        data_cell->InsertHeapByTermLists<KNN_SEARCH, PURE>(
            dists.data(), computer, heap, inner_param, 0);
        REQUIRE(heap.size() == topk);

        // Extract results from InsertHeapByTermLists
        std::vector<std::pair<float, int64_t>> results_by_term_lists;
        while (!heap.empty()) {
            results_by_term_lists.push_back(heap.top());
            heap.pop();
        }

        for (auto i = 0; i < topk; i++) {
            auto exp_id = pos + i;
            REQUIRE(results_by_term_lists[i].second == exp_id);
            REQUIRE(std::abs(results_by_term_lists[i].first - exp_dists[exp_id]) < 1e-3);
        }

        std::vector<float> dists2(count_base, 0);
        data_cell->Query(dists2.data(), computer);
        MaxHeap heap2(allocator.get());
        data_cell->InsertHeapByDists<KNN_SEARCH, PURE>(
            dists2.data(), dists2.size(), heap2, inner_param, 0);

        // Extract results from InsertHeapByDists
        std::vector<std::pair<float, int64_t>> results_by_dists;
        while (!heap2.empty()) {
            results_by_dists.push_back(heap2.top());
            heap2.pop();
        }
        // Compare results from both methods
        REQUIRE(results_by_term_lists.size() == results_by_dists.size());
        for (size_t i = 0; i < results_by_term_lists.size(); i++) {
            REQUIRE(results_by_term_lists[i].second == results_by_dists[i].second);
            REQUIRE(std::abs(results_by_term_lists[i].first - results_by_dists[i].first) < 1e-3);
        }
        for (auto i = 0; i < dists.size(); i++) {
            REQUIRE(std::abs(dists[i] - 0) < 1e-3);
        }
    }

    SECTION("test insert heap in range search") {
        auto range_topk = 3;
        auto pos = count_base - range_topk - 1;  // note that we retrieval dist < dists[pos]
        InnerSearchParam inner_param;
        std::vector<float> dists(count_base, 0);
        data_cell->Query(dists.data(), computer);
        inner_param.radius = dists[pos];
        MaxHeap heap(allocator.get());

        data_cell->InsertHeapByTermLists<RANGE_SEARCH, PURE>(
            dists.data(), computer, heap, inner_param, 0);
        REQUIRE(heap.size() == range_topk);

        // Extract results from InsertHeapByTermLists
        std::vector<std::pair<float, int64_t>> results_by_term_lists;
        while (!heap.empty()) {
            results_by_term_lists.push_back(heap.top());
            heap.pop();
        }
        for (auto i = 0; i < range_topk; i++) {
            auto exp_id = pos + i + 1;
            REQUIRE(results_by_term_lists[i].second == exp_id);
            REQUIRE(std::abs(results_by_term_lists[i].first - exp_dists[exp_id]) < 1e-3);
        }

        std::vector<float> dists2(count_base, 0);
        data_cell->Query(dists2.data(), computer);
        MaxHeap heap2(allocator.get());
        data_cell->InsertHeapByDists<RANGE_SEARCH, PURE>(
            dists2.data(), dists2.size(), heap2, inner_param, 0);

        // Extract results from InsertHeapByDists
        std::vector<std::pair<float, int64_t>> results_by_dists;
        while (!heap2.empty()) {
            results_by_dists.push_back(heap2.top());
            heap2.pop();
        }

        // Compare results from both methods
        REQUIRE(results_by_term_lists.size() == results_by_dists.size());
        for (size_t i = 0; i < results_by_term_lists.size(); i++) {
            REQUIRE(results_by_term_lists[i].second == results_by_dists[i].second);
            REQUIRE(std::abs(results_by_term_lists[i].first - results_by_dists[i].first) < 1e-3);
        }
        for (auto i = 0; i < range_topk; i++) {
            REQUIRE(std::abs(dists[i] - 0) < 1e-3);
        }
    }
    // clean
    for (auto& item : sparse_vectors) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
    delete[] query_sv.ids_;
    delete[] query_sv.vals_;
}

TEST_CASE("SparseTermDatacell Deserialize Shrinks Empty Tail", "[ut][SparseTermDatacell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    std::shared_ptr<QuantizationParams> q_params = nullptr;
    SparseTermDataCell data_cell(1.0F, DEFAULT_TERM_ID_LIMIT, allocator.get(), false, q_params);

    std::vector<uint32_t> ids = {2, 9};
    std::vector<float> vals = {0.2F, 0.9F};
    SparseVector sv;
    sv.len_ = static_cast<uint32_t>(ids.size());
    sv.ids_ = ids.data();
    sv.vals_ = vals.data();
    data_cell.InsertVector(sv, 0);

    auto capacity_after_insert = data_cell.term_capacity_;
    data_cell.ResizeTermList(32);
    REQUIRE(data_cell.term_capacity_ >= 32);
    REQUIRE(capacity_after_insert < data_cell.term_capacity_);

    std::stringstream ss;
    IOStreamWriter writer(ss);
    data_cell.Serialize(writer);

    SparseTermDataCell restored(1.0F, DEFAULT_TERM_ID_LIMIT, allocator.get(), false, q_params);
    IOStreamReader reader(ss);
    restored.Deserialize(reader);

    REQUIRE(restored.term_capacity_ == 10);
    REQUIRE(restored.term_sizes_.size() == 10);
}

TEST_CASE("SparseTermDatacell Local Term Remap Round Trip", "[ut][SparseTermDatacell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    std::shared_ptr<QuantizationParams> q_params = nullptr;
    SparseTermDataCell data_cell(
        1.0F, DEFAULT_TERM_ID_LIMIT, allocator.get(), false, q_params, true);

    std::vector<uint32_t> ids0 = {100, 999999};
    std::vector<float> vals0 = {1.0F, 2.0F};
    SparseVector sv0;
    sv0.len_ = static_cast<uint32_t>(ids0.size());
    sv0.ids_ = ids0.data();
    sv0.vals_ = vals0.data();
    data_cell.InsertVector(sv0, 0);

    std::vector<uint32_t> ids1 = {999999};
    std::vector<float> vals1 = {3.0F};
    SparseVector sv1;
    sv1.len_ = static_cast<uint32_t>(ids1.size());
    sv1.ids_ = ids1.data();
    sv1.vals_ = vals1.data();
    data_cell.InsertVector(sv1, 1);

    REQUIRE(data_cell.use_local_term_map_);
    REQUIRE(data_cell.local_to_global_terms_.size() == 2);
    REQUIRE(data_cell.term_capacity_ == 2);
    REQUIRE(data_cell.term_sizes_.size() == 2);

    std::vector<uint32_t> q_ids = {999999};
    std::vector<float> q_vals = {1.0F};
    SparseVector query;
    query.len_ = static_cast<uint32_t>(q_ids.size());
    query.ids_ = q_ids.data();
    query.vals_ = q_vals.data();
    SINDISearchParameter search_params;
    search_params.term_prune_ratio = 0;
    search_params.query_prune_ratio = 0;
    auto computer = std::make_shared<SparseTermComputer>(query, search_params, allocator.get());
    std::vector<float> dists(2, 0);
    data_cell.Query(dists.data(), computer);
    REQUIRE(std::abs(dists[0] - (-2.0F)) < 1e-3);
    REQUIRE(std::abs(dists[1] - (-3.0F)) < 1e-3);

    std::stringstream ss;
    IOStreamWriter writer(ss);
    data_cell.Serialize(writer);

    SparseTermDataCell restored(
        1.0F, DEFAULT_TERM_ID_LIMIT, allocator.get(), false, q_params, true);
    IOStreamReader reader(ss);
    restored.Deserialize(reader);
    REQUIRE(restored.IsFlatStorage());
    REQUIRE(restored.local_to_global_terms_.size() == 2);
    REQUIRE(restored.term_capacity_ == 2);

    SparseVector restored_sv;
    restored.GetSparseVector(0, &restored_sv, allocator.get());
    REQUIRE(restored_sv.len_ == 2);
    allocator->Deallocate(restored_sv.ids_);
    allocator->Deallocate(restored_sv.vals_);
}

TEST_CASE("SparseTermDatacell Local Term Remap Equivalent To Direct Terms",
          "[ut][SparseTermDatacell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    std::shared_ptr<QuantizationParams> q_params = nullptr;
    SparseTermDataCell direct_cell(
        1.0F, DEFAULT_TERM_ID_LIMIT, allocator.get(), false, q_params, false);
    SparseTermDataCell local_cell(
        1.0F, DEFAULT_TERM_ID_LIMIT, allocator.get(), false, q_params, true);
    SparseTermDataCell flat_cell(
        1.0F, DEFAULT_TERM_ID_LIMIT, allocator.get(), false, q_params, true);

    std::vector<std::vector<uint32_t>> doc_ids = {
        {999999, 42, 7},
        {42, 500000, 123456},
        {8, 999999},
        {7, 123456, 500000, 999999},
    };
    std::vector<std::vector<float>> doc_vals = {
        {2.0F, 1.5F, 0.5F},
        {0.7F, 1.1F, 2.3F},
        {3.1F, 0.4F},
        {0.9F, 0.6F, 1.8F, 2.2F},
    };

    auto make_sv = [](std::vector<uint32_t>& ids, std::vector<float>& vals) {
        SparseVector sv;
        sv.len_ = static_cast<uint32_t>(ids.size());
        sv.ids_ = ids.data();
        sv.vals_ = vals.data();
        return sv;
    };

    for (uint16_t i = 0; i < doc_ids.size(); ++i) {
        auto sv = make_sv(doc_ids[i], doc_vals[i]);
        direct_cell.InsertVector(sv, i);
        local_cell.InsertVector(sv, i);
    }

    Vector<Vector<std::pair<uint32_t, float>>> sorted_docs(allocator.get());
    for (uint32_t i = 0; i < doc_ids.size(); ++i) {
        auto sv = make_sv(doc_ids[i], doc_vals[i]);
        Vector<std::pair<uint32_t, float>> sorted_doc(allocator.get());
        sort_sparse_vector(sv, sorted_doc);
        sorted_docs.emplace_back(allocator.get());
        sorted_docs.back().swap(sorted_doc);
    }
    flat_cell.BuildFlatFromSortedVectors(sorted_docs);
    REQUIRE(flat_cell.IsFlatStorage());

    REQUIRE(local_cell.use_local_term_map_);
    REQUIRE(local_cell.local_to_global_terms_.size() == 6);
    REQUIRE(local_cell.local_to_global_terms_.size() < direct_cell.term_capacity_);
    REQUIRE(flat_cell.local_to_global_terms_.size() == 6);

    local_cell.FlattenTermLists();
    REQUIRE(local_cell.IsFlatStorage());

    std::stringstream ss;
    IOStreamWriter writer(ss);
    local_cell.Serialize(writer);
    SparseTermDataCell restored_local(
        1.0F, DEFAULT_TERM_ID_LIMIT, allocator.get(), false, q_params, true);
    IOStreamReader reader(ss);
    restored_local.Deserialize(reader);
    REQUIRE(restored_local.IsFlatStorage());

    auto sparse_vector_to_map = [&](SparseTermDataCell& cell, uint32_t base_id) {
        SparseVector sv;
        cell.GetSparseVector(base_id, &sv, allocator.get());
        std::map<uint32_t, float> result;
        for (uint32_t i = 0; i < sv.len_; ++i) {
            result[sv.ids_[i]] = sv.vals_[i];
        }
        allocator->Deallocate(sv.ids_);
        allocator->Deallocate(sv.vals_);
        return result;
    };

    for (uint32_t base_id = 0; base_id < doc_ids.size(); ++base_id) {
        auto direct_map = sparse_vector_to_map(direct_cell, base_id);
        auto local_map = sparse_vector_to_map(local_cell, base_id);
        auto flat_map = sparse_vector_to_map(flat_cell, base_id);
        auto restored_map = sparse_vector_to_map(restored_local, base_id);
        REQUIRE(local_map == direct_map);
        REQUIRE(flat_map == direct_map);
        REQUIRE(restored_map == direct_map);
    }

    std::vector<std::vector<uint32_t>> query_ids = {
        {999999, 42, 777777},
        {7, 8, 123456, 500000},
        {777777, 888888},
    };
    std::vector<std::vector<float>> query_vals = {
        {1.0F, 0.25F, 4.0F},
        {0.5F, 1.2F, 0.75F, 0.3F},
        {2.0F, 3.0F},
    };

    SINDISearchParameter search_params;
    search_params.term_prune_ratio = 0;
    search_params.query_prune_ratio = 0;
    InnerSearchParam inner_param;
    inner_param.ef = 3;

    auto extract_heap = [](MaxHeap& heap) {
        std::vector<std::pair<float, InnerIdType>> result;
        while (not heap.empty()) {
            result.push_back(heap.top());
            heap.pop();
        }
        return result;
    };

    auto compare_cell_with_direct = [&](SparseTermDataCell& candidate_cell,
                                        const SparseVector& query) {
        std::vector<float> direct_dists(doc_ids.size(), 0.0F);
        std::vector<float> candidate_dists(doc_ids.size(), 0.0F);
        auto direct_computer =
            std::make_shared<SparseTermComputer>(query, search_params, allocator.get());
        auto candidate_computer =
            std::make_shared<SparseTermComputer>(query, search_params, allocator.get());
        direct_cell.Query(direct_dists.data(), direct_computer);
        candidate_cell.Query(candidate_dists.data(), candidate_computer);
        for (uint32_t i = 0; i < doc_ids.size(); ++i) {
            REQUIRE(std::abs(candidate_dists[i] - direct_dists[i]) < 1e-6F);
        }

        MaxHeap direct_heap(allocator.get());
        MaxHeap candidate_heap(allocator.get());
        direct_cell.InsertHeapByTermLists<KNN_SEARCH, PURE>(
            direct_dists.data(), direct_computer, direct_heap, inner_param, 0);
        candidate_cell.InsertHeapByTermLists<KNN_SEARCH, PURE>(
            candidate_dists.data(), candidate_computer, candidate_heap, inner_param, 0);
        REQUIRE(extract_heap(candidate_heap) == extract_heap(direct_heap));

        for (uint16_t base_id = 0; base_id < doc_ids.size(); ++base_id) {
            auto direct_dist = direct_cell.CalcDistanceByInnerId(direct_computer, base_id);
            auto candidate_dist = candidate_cell.CalcDistanceByInnerId(candidate_computer, base_id);
            REQUIRE(std::abs(candidate_dist - direct_dist) < 1e-6F);
        }
    };

    for (uint32_t i = 0; i < query_ids.size(); ++i) {
        auto query = make_sv(query_ids[i], query_vals[i]);
        compare_cell_with_direct(local_cell, query);
        compare_cell_with_direct(flat_cell, query);
        compare_cell_with_direct(restored_local, query);
    }
}

TEST_CASE("SparseTermDatacell Encode/Decode Test", "[ut][SparseTermDatacell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    // Prepare data
    std::vector<uint32_t> ids = {10, 20, 30};
    std::vector<float> vals = {1.1f, 2.2f, 3.3f};
    SparseVector sv;
    sv.len_ = ids.size();
    sv.ids_ = ids.data();
    sv.vals_ = vals.data();

    float min_val = 1.1f;
    float max_val = 3.3f;

    // Prepare datacell
    auto q_params = std::make_shared<QuantizationParams>();
    q_params->min_val = min_val;
    q_params->max_val = max_val;
    q_params->diff = max_val - min_val;
    auto data_cell = std::make_shared<SparseTermDataCell>(
        1.0f, DEFAULT_TERM_ID_LIMIT, allocator.get(), true, q_params);

    // Insert vector (tests Encode)
    uint16_t base_id = 5;
    data_cell->InsertVector(sv, base_id);
    data_cell->FlattenTermLists();
    REQUIRE(data_cell->IsFlatStorage());

    // Get vector (tests Decode)
    SparseVector retrieved_sv;
    data_cell->GetSparseVector(base_id, &retrieved_sv, allocator.get());

    REQUIRE(retrieved_sv.len_ == sv.len_);

    // Verify results
    std::map<uint32_t, float> retrieved_map;
    for (size_t i = 0; i < retrieved_sv.len_; ++i) {
        retrieved_map[retrieved_sv.ids_[i]] = retrieved_sv.vals_[i];
    }

    float tolerance = 0.1f;

    for (size_t i = 0; i < sv.len_; ++i) {
        REQUIRE(retrieved_map.count(sv.ids_[i]));
        REQUIRE(std::abs(retrieved_map[sv.ids_[i]] - sv.vals_[i]) < tolerance);
    }

    allocator->Deallocate(retrieved_sv.ids_);
    allocator->Deallocate(retrieved_sv.vals_);
}

TEST_CASE("SparseTermDatacell Last Term Test", "[ut][SparseTermDatacell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    auto make_sv = [](const std::vector<uint32_t>& ids, const std::vector<float>& vals) {
        vsag::SparseVector sv;
        sv.len_ = static_cast<uint32_t>(ids.size());
        sv.ids_ = const_cast<uint32_t*>(ids.data());
        sv.vals_ = const_cast<float*>(vals.data());
        return sv;
    };

    std::vector<int64_t> ids = {0, 1};

    {
        std::vector<uint32_t> ids0 = {1, 2};
        std::vector<float> vals0 = {0.1f, 0.0f};
        std::vector<uint32_t> ids1 = {1};
        std::vector<float> vals1 = {0.1f};

        auto sv0 = make_sv(ids0, vals0);
        auto sv1 = make_sv(ids1, vals1);

        auto q_params = std::make_shared<QuantizationParams>();
        q_params->min_val = 0.0f;
        q_params->max_val = 0.1f;
        q_params->diff = q_params->max_val - q_params->min_val;
        auto data_cell = std::make_shared<SparseTermDataCell>(
            1, DEFAULT_TERM_ID_LIMIT, allocator.get(), false, q_params);
        data_cell->InsertVector(sv0, ids[0]);
        data_cell->InsertVector(sv1, ids[1]);

        std::vector<uint32_t> q_ids = {1, 4};
        std::vector<float> q_vals = {1.0f, 1.0f};
        auto sv_query = make_sv(q_ids, q_vals);

        SINDISearchParameter search_params;
        search_params.term_prune_ratio = 0;
        search_params.query_prune_ratio = 0;
        auto computer =
            std::make_shared<SparseTermComputer>(sv_query, search_params, allocator.get());

        std::vector<float> dists(2, 0);
        data_cell->Query(dists.data(), computer);
        REQUIRE(std::abs(dists[0] - (-0.1f)) < 1e-2f);
        REQUIRE(std::abs(dists[1] - (-0.1f)) < 1e-2f);
    }
}
