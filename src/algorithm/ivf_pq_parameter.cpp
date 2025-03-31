
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

#include "ivf_pq_parameter.h"

#include <fmt/format-inl.h>

#include "inner_string_params.h"
#include "vsag/constants.h"
#include <iostream>
namespace vsag {

IVFPQParameter::IVFPQParameter() = default;

void
IVFPQParameter::FromJson(const JsonType& json) {
    std::cout << "dump: " << json.dump() << std::endl;
    this->bucket_param = std::make_shared<BucketDataCellParameter>();
    CHECK_ARGUMENT(json.contains(BUCKET_PARAMS_KEY),
                   fmt::format("ivf parameters must contains {}", BUCKET_PARAMS_KEY));
    this->bucket_param->FromJson(json[BUCKET_PARAMS_KEY]);

    CHECK_ARGUMENT(json.contains(COARSE_CLUSTER_COUNT_KEY),
                   fmt::format("ivf parameters must contains {}", COARSE_CLUSTER_COUNT_KEY));
    this->coarse_cluster_count = json[COARSE_CLUSTER_COUNT_KEY];
    
    if (json.contains(FINE_CLUSTER_COUNT_KEY)) {
        this->fine_cluster_count = json[FINE_CLUSTER_COUNT_KEY];
    } else {
        this->fine_cluster_count = this->coarse_cluster_count;
    }

    CHECK_ARGUMENT(json.contains(FILTER_NSQ_KEY),
                   fmt::format("ivf parameters must contains {}", FILTER_NSQ_KEY));
    this->filter_nsq = this->filter_nsq;
    
    if (json.contains(TRAIN_POINTS_COUNT_KEY)) {
        this->train_points_count = json[TRAIN_POINTS_COUNT_KEY];
    }
    
    if (json.contains(FINE_CLUSTER_COUNT_KEY)) {
        this->pq_train_points_count = json[PQ_TRAIN_POINTS_COUNT_KEY];
    }
}

JsonType
IVFPQParameter::ToJson() {
    JsonType json;
    json["type"] = INDEX_IVF_PQ;
    json[BUCKET_PARAMS_KEY] = this->bucket_param->ToJson();
    json["coarse_cluster_count"] = this->coarse_cluster_count;
    json["fine_cluster_count"] = this->fine_cluster_count;
    json["filter_nsq"] = this->filter_nsq;
    json["train_points_count"] = this->train_points_count;
    json["pq_train_points_count"] = this->pq_train_points_count;
    return json;
}
}  // namespace vsag
