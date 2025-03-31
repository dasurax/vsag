
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

#include "data_cell/bucket_datacell.h"
#include "index/index_common_param.h"
#include "inner_index_interface.h"
#include "ivf_pq_parameter.h"
#include "ivf_partition/ivf_partition_strategy.h"
#include "stream_reader.h"
#include "stream_writer.h"
#include "typing.h"
#include "vsag/index.h"
#include "puck/puck_index.h"
#include <glog/logging.h>

namespace vsag {
class IVFPQ : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    explicit IVFPQ(const IVFPQParameterPtr& param, const IndexCommonParam& common_param);

    explicit IVFPQ(const ParamPtr& param, const IndexCommonParam& common_param)
        : IVFPQ(std::dynamic_pointer_cast<IVFPQParameter>(param), common_param){};

    ~IVFPQ() override = default;

    [[nodiscard]] std::string
    GetName() const override {
        return INDEX_IVF_PQ;
    }

    void
    InitFeatures() override;

    std::vector<int64_t>
    Build(const DatasetPtr& base) override;

    std::vector<int64_t>
    Add(const DatasetPtr& base) override;

    DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    int64_t
    GetNumElements() const override;

    inline int64_t
    GetMemoryUsage() const override {
        return 0;
    }

private:
    BucketInterfacePtr bucket_{nullptr};

    IVFPartitionStrategyPtr partition_strategy_{nullptr};
    std::shared_ptr<puck::PuckIndex> puck_index_{nullptr};
    int64_t total_elements_{0};
};
}  // namespace vsag
