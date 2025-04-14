
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

#include <filesystem>  // C++17 引入的文件系统库
#include <fstream>
#include <iostream>

#include "gno_imi.h"
#include "impl/basic_searcher.h"
#include "inner_string_params.h"
#include "ivf_partition/gno_imi_partition.h"
#include "utils/util_functions.h"

namespace vsag {
namespace fs = std::filesystem;
static const std::unordered_map<std::string, std::vector<std::string> > EXTERNAL_MAPPING = {
    {
        IVF_BASE_QUANTIZATION_TYPE,
        {BUCKET_PARAMS_KEY, QUANTIZATION_PARAMS_KEY, QUANTIZATION_TYPE_KEY},
    },
    {
        IVF_BASE_IO_TYPE,
        {BUCKET_PARAMS_KEY, IO_PARAMS_KEY, IO_TYPE_KEY},
    },
    {
        IVF_BUCKETS_COUNT,
        {BUCKET_PARAMS_KEY, BUCKETS_COUNT_KEY},
    },
    {
        IVF_PQ_COARSE_CLUSTER_COUNT,
        {COARSE_CLUSTER_COUNT_KEY},
    },
    {
        IVF_PQ_FINE_CLUSTER_COUNT,
        {FINE_CLUSTER_COUNT_KEY},
    },
    {
        IVF_PQ_FILTER_NSQ,
        {FILTER_NSQ_KEY},
    },
    {
        IVF_PQ_TRAIN_POINTS_COUNT,
        {TRAIN_POINTS_COUNT_KEY},
    },
    {
        IVF_PQ_PQ_TRAIN_POINTS_COUNT,
        {PQ_TRAIN_POINTS_COUNT_KEY},
    }};

static constexpr const char* IVF_PQ_PARAMS_TEMPLATE =
    R"(
    {
        "type": "{INDEX_TYPE_IVF_PQ}",
        "{BUCKET_PARAMS_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}"
            },
            "{QUANTIZATION_PARAMS_KEY}": {
                "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}"
            },
            "{BUCKETS_COUNT_KEY}": 10
        },
        "{COARSE_CLUSTER_COUNT_KEY}": 10,
        "{FINE_CLUSTER_COUNT_KEY}": 10,
        "{FILTER_NSQ_KEY}": 10,
        "{TRAIN_POINTS_COUNT_KEY}": 1000000,
        "{PQ_TRAIN_POINTS_COUNT_KEY}": 200000
    })";

ParamPtr
GNOIMI::CheckAndMappingExternalParam(const JsonType& external_param,
                                     const IndexCommonParam& common_param) {
    if (common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
        throw std::invalid_argument(fmt::format("IVF PQ not support {} datatype", DATATYPE_INT8));
    }

    std::string str = format_map(IVF_PQ_PARAMS_TEMPLATE, DEFAULT_MAP);
    //std::cout << "cout " << str << std::endl;
    auto inner_json = JsonType::parse(str);
    mapping_external_param_to_inner(external_param, EXTERNAL_MAPPING, inner_json);

    auto ivf_pq_parameter = std::make_shared<IVFPQParameter>();
    ivf_pq_parameter->FromJson(inner_json);

    return ivf_pq_parameter;
}

GNOIMI::GNOIMI(const IVFPQParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param) {
    this->bucket_ = BucketInterface::MakeInstance(param->bucket_param, common_param);
    this->partition_strategy_ = std::make_shared<GNOIMIPartition>(
        /*bucket_->bucket_count_*/ 100,
        common_param,
        IVFNearestPartitionTrainerType::KMeansTrainer);
    /*
    puck_index_.reset(new puck::PuckIndex());
    auto& conf = puck_index_->get_conf_file();
    conf.spilled_copy_num = 1;
    conf.feature_dim = common_param.dim_;
    conf.nsq = common_param.dim_;
    conf.whether_norm = false;

    conf.filter_nsq = param->filter_nsq;
    conf.coarse_cluster_count = param->coarse_cluster_count;
    conf.fine_cluster_count = param->fine_cluster_count;
    conf.train_points_count = param->train_points_count;
    conf.pq_train_points_count = param->pq_train_points_count;
    conf.show();
    */
}

void
GNOIMI::InitFeatures() {
    // Common Init
    // Build & Add
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
    });

    // search
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
    });
    // concurrency
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_SEARCH_CONCURRENT);
    // serialize
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
        IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_SERIALIZE_FILE,
    });

    auto name = this->bucket_->GetQuantizerName();
    if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16) {
        this->index_feature_list_->SetFeature(IndexFeature::NEED_TRAIN);
    } else {
        this->index_feature_list_->SetFeatures({
            IndexFeature::SUPPORT_RANGE_SEARCH,
            IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
        });
    }
}

std::vector<int64_t>
GNOIMI::Build(const DatasetPtr& base) {
    /*
    std::vector<int64_t> failed_ids;
    
    std::string folder_path = "./puck_index";
    if (!fs::exists(folder_path)) {  // 检查文件夹是否存在
        fs::create_directory(folder_path);
    }

    std::string filename = folder_path + "/all_data.feat.bin";
    std::ofstream file(filename, std::ios::binary);

    const float* vectors = base->GetFloat32Vectors();
    size_t num_vectors = base->GetNumElements();
    size_t dim = base->GetDim();

    for (size_t i = 0; i < num_vectors; ++i) {
        uint32_t dim_uint32 = static_cast<uint32_t>(dim);
        file.write(reinterpret_cast<const char*>(&dim_uint32), sizeof(uint32_t));

        const float* vector_data = vectors + i * dim;
        file.write(reinterpret_cast<const char*>(vector_data), dim * sizeof(float));
    }

    file.close();
    */

    partition_strategy_->Train(base);
    this->bucket_->Train(base->GetFloat32Vectors(), base->GetNumElements());
    //puck_index_->train();
    //puck_index_->build();

    return this->Add(base);
}

std::vector<int64_t>
GNOIMI::Add(const DatasetPtr& base) {
    // TODO(LHT): duplicate
    if (not partition_strategy_->is_trained_) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "ivf index add without train error");
    }
    auto dim = partition_strategy_->dim_;
    auto num_element = base->GetNumElements();
    const auto* ids = base->GetIds();
    const auto* vectors = base->GetFloat32Vectors();
    auto labels = partition_strategy_->ClassifyDatas(vectors, num_element, 1);
    std::cout << "labels1: " << labels[0] << std::endl;
    for (int64_t i = 0; i < num_element; ++i) {
        bucket_->InsertVector(vectors + i * dim, labels[i], ids[i]);
    }
    this->total_elements_ += num_element;
    return {};
}

DatasetPtr
GNOIMI::KnnSearch(const vsag::DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const vsag::FilterPtr& filter) const {
    auto* allocator = allocator_;
    MaxHeap heap(allocator);
    auto param = IVFPQSearchParameters::FromJson(parameters);
    int scan_buckets_count =
        std::min(static_cast<BucketIdType>(param.scan_buckets_count), bucket_->bucket_count_);
    auto candidate_buckets = partition_strategy_->ClassifyDatasForSearch(
        query->GetFloat32Vectors(), 1, scan_buckets_count);
    auto computer = bucket_->FactoryComputer(query->GetFloat32Vectors());
    Vector<float> dist(allocator);
    std::set<BucketIdType> ss;
    std::set<LabelType> ss2;
    BucketIdType min = 9999999, max = 0;
    LabelType minid = 9999999, maxid = 0;
    size_t total = 0;
    auto cur_heap_top = std::numeric_limits<float>::max();
    for (auto& bucket_id : candidate_buckets) {
        //ss.insert(bucket_id);
        min = std::min(min, bucket_id);
        max = std::max(max, bucket_id);
        auto bucket_size = bucket_->GetBucketSize(bucket_id);
        const auto* labels = bucket_->GetLabel(bucket_id);
        if (bucket_size > dist.size()) {
            dist.resize(bucket_size);
        }
        total += bucket_size;
        bucket_->ScanBucketById(dist.data(), computer, bucket_id);
        for (int j = 0; j < bucket_size; ++j) {
            //ss2.insert(labels[j]);
            minid = std::min(minid, labels[j]);
            maxid = std::max(maxid, labels[j]);
            if (filter == nullptr or filter->CheckValid(labels[j])) {
                if (heap.size() < k or dist[j] < cur_heap_top) {
                    heap.emplace(dist[j], labels[j]);
                }
                if (heap.size() > k) {
                    heap.pop();
                }
                cur_heap_top = heap.top().first;
            }
        }
    }
    //std::cout << "total: " << total << std::endl;
    //std::cout << "ss: " << ss.size() << " max: " << max << " min: " << min << " total: " << total
    //        << " maxid: " << maxid << " minid: " << minid << std::endl;
    auto dataset_results = Dataset::Make();
    dataset_results->Dim(static_cast<int64_t>(heap.size()))->NumElements(1)->Owner(true, allocator);

    auto* ids = (int64_t*)allocator->Allocate(sizeof(int64_t) * heap.size());
    dataset_results->Ids(ids);
    auto* dists = (float*)allocator->Allocate(sizeof(float) * heap.size());
    dataset_results->Distances(dists);
    for (auto j = static_cast<int64_t>(heap.size() - 1); j >= 0; --j) {
        dists[j] = heap.top().first;
        ids[j] = heap.top().second;
        heap.pop();
    }
    return std::move(dataset_results);
}

DatasetPtr
GNOIMI::RangeSearch(const vsag::DatasetPtr& query,
                    float radius,
                    const std::string& parameters,
                    const vsag::FilterPtr& filter,
                    int64_t limited_size) const {
    auto* allocator = allocator_;
    MaxHeap heap(allocator);
    auto param = IVFPQSearchParameters::FromJson(parameters);
    int scan_buckets_count =
        std::min(static_cast<BucketIdType>(param.scan_buckets_count), bucket_->bucket_count_);
    auto candidate_buckets = partition_strategy_->ClassifyDatasForSearch(
        query->GetFloat32Vectors(), 1, scan_buckets_count);
    auto computer = bucket_->FactoryComputer(query->GetFloat32Vectors());
    Vector<float> dist(allocator);
    auto cur_heap_top = std::numeric_limits<float>::max();
    if (limited_size < 0) {
        limited_size = std::numeric_limits<int64_t>::max();
    }
    for (auto& bucket_id : candidate_buckets) {
        auto bucket_size = bucket_->GetBucketSize(bucket_id);
        const auto* labels = bucket_->GetLabel(bucket_id);
        if (bucket_size > dist.size()) {
            dist.resize(bucket_size);
        }
        bucket_->ScanBucketById(dist.data(), computer, bucket_id);
        for (int j = 0; j < bucket_size; ++j) {
            if (filter == nullptr or filter->CheckValid(labels[j])) {
                if (dist[j] <= radius + THRESHOLD_ERROR and dist[j] < cur_heap_top) {
                    heap.emplace(dist[j], labels[j]);
                }
                if (heap.size() > limited_size) {
                    heap.pop();
                }
                if (not heap.empty() and heap.size() == limited_size) {
                    cur_heap_top = heap.top().first;
                }
            }
        }
    }
    auto dataset_results = Dataset::Make();
    dataset_results->Dim(static_cast<int64_t>(heap.size()))->NumElements(1)->Owner(true, allocator);

    auto* ids = (int64_t*)allocator->Allocate(sizeof(int64_t) * heap.size());
    dataset_results->Ids(ids);
    auto* dists = (float*)allocator->Allocate(sizeof(float) * heap.size());
    dataset_results->Distances(dists);
    for (auto j = static_cast<int64_t>(heap.size() - 1); j >= 0; --j) {
        dists[j] = heap.top().first;
        ids[j] = heap.top().second;
        heap.pop();
    }
    return std::move(dataset_results);
}

int64_t
GNOIMI::GetNumElements() const {
    return this->total_elements_;
}

void
GNOIMI::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, this->total_elements_);
    this->bucket_->Serialize(writer);
    this->partition_strategy_->Serialize(writer);
    this->label_table_->Serialize(writer);
}

void
GNOIMI::Deserialize(StreamReader& reader) {
    StreamReader::ReadObj(reader, this->total_elements_);
    this->bucket_->Deserialize(reader);
    this->partition_strategy_->Deserialize(reader);
    this->label_table_->Deserialize(reader);
}

}  // namespace vsag
