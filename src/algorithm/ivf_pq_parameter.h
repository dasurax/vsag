
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
#include "data_cell/bucket_datacell_parameter.h"
#include "fmt/format-inl.h"
#include "inner_string_params.h"
#include "parameter.h"
#include "typing.h"

namespace vsag {
class IVFPQParameter : public Parameter {
public:
    explicit IVFPQParameter();

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() override;

public:
    BucketDataCellParamPtr bucket_param{nullptr};
    bool use_residual{false};
    uint64_t coarse_cluster_count{100};
    uint64_t fine_cluster_count{100};
    uint64_t filter_nsq{4};
    uint64_t train_points_count{1000000};
    uint64_t pq_train_points_count{200000};
};

using IVFPQParameterPtr = std::shared_ptr<IVFPQParameter>;

class IVFPQSearchParameters {
public:
    static IVFPQSearchParameters
    FromJson(const std::string& json_string) {
        JsonType params = JsonType::parse(json_string);

        IVFPQSearchParameters obj;

        // set obj.scan_buckets_count
        CHECK_ARGUMENT(params.contains(INDEX_TYPE_IVF_PQ),
                       fmt::format("parameters must contains {}", INDEX_TYPE_IVF_PQ));

        CHECK_ARGUMENT(params[INDEX_TYPE_IVF_PQ].contains(IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT),
                       fmt::format("parameters[{}] must contains {}",
                                   INDEX_TYPE_IVF_PQ,
                                   IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT));
        obj.scan_buckets_count = params[INDEX_TYPE_IVF_PQ][IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT];

        
        CHECK_ARGUMENT(params[INDEX_TYPE_IVF_PQ].contains(IVF_SEARCH_PARAM_SEARCH_COARSE_COUNT),
                       fmt::format("parameters[{}] must contains {}",
                                   INDEX_TYPE_IVF_PQ,
                                   IVF_SEARCH_PARAM_SEARCH_COARSE_COUNT));
        obj.search_coarse_count = params[INDEX_TYPE_IVF_PQ][IVF_SEARCH_PARAM_SEARCH_COARSE_COUNT];

        CHECK_ARGUMENT(params[INDEX_TYPE_IVF_PQ].contains(IVF_SEARCH_PARAM_SEARCH_FINE_COUNT),
                       fmt::format("parameters[{}] must contains {}",
                                   INDEX_TYPE_IVF_PQ,
                                   IVF_SEARCH_PARAM_SEARCH_FINE_COUNT));
        obj.search_fine_count = params[INDEX_TYPE_IVF_PQ][IVF_SEARCH_PARAM_SEARCH_FINE_COUNT];

        CHECK_ARGUMENT(params[INDEX_TYPE_IVF_PQ].contains(IVF_SEARCH_PARAM_FILTER_TOPK),
                       fmt::format("parameters[{}] must contains {}",
                                   INDEX_TYPE_IVF_PQ,
                                   IVF_SEARCH_PARAM_FILTER_TOPK));
        obj.filter_topk = params[INDEX_TYPE_IVF_PQ][IVF_SEARCH_PARAM_FILTER_TOPK];

        CHECK_ARGUMENT(params[INDEX_TYPE_IVF_PQ].contains(IVF_SEARCH_PARAM_WINDOW_SIZE),
                       fmt::format("parameters[{}] must contains {}",
                                   INDEX_TYPE_IVF_PQ,
                                   IVF_SEARCH_PARAM_WINDOW_SIZE));
        obj.window_size = params[INDEX_TYPE_IVF_PQ][IVF_SEARCH_PARAM_WINDOW_SIZE];
        
        return obj;
    }

public:
    int64_t scan_buckets_count{30};
    int64_t search_coarse_count{10};
    int64_t search_fine_count{100};
    int64_t filter_topk{200};
    int64_t window_size{5};

private:
    IVFPQSearchParameters() = default;
};

}  // namespace vsag
