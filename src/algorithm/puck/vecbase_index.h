#pragma once
#include <string>
#include <vector>
#include <map>
#include "algorithm/puck/factory.h"

namespace puck {

class VecbaseIndex {
public:
    VecbaseIndex() : _index(nullptr) {}

    bool create_index(const std::string& index_type) {
        _index = VecbaseIndexFactory::instance().create_index(index_type);
        return _index != nullptr;
    }

    virtual int init() {
        if (_index) {
            return _index->init();
        }
        return -1; // 未初始化
    }

    virtual int init(std::map<std::string, std::pair<void*, size_t>>& data_ptr) {
        if (_index) {
            return _index->init(data_ptr);
        }
        return -1; // 未初始化
    }

    virtual int search(const Request* request, Response* response) {
        if (_index) {
            return _index->search(request, response);
        }
        return -1;
    }

    virtual int train() {
        if (_index) {
            return _index->train();
        }
        return -1;
    }

    virtual int build() {
        if (_index) {
            return _index->build();
        }
        return -1;
    }

    virtual IndexConf& get_conf_file() {
        if (_index) {
            return _index->get_conf_file();
        }
        throw std::runtime_error("Index not initialized!");
    }

    virtual void set_conf_file(IndexConf& conf) {
        if (_index) {
            _index->set_conf_file(conf);
        }
    }

    virtual void info() {
        if (_index) {
            _index->info();
        }
    }

    /*
    std::vector<Triple> index_file_to_save() {
        return std::move(_index->index_file_to_save());
    }
    */
    virtual std::vector<std::string> index_file_to_save() {
        return std::move(_index->index_file_to_save());
    }

    virtual size_t get_total_point_count() {
        return _index->get_total_point_count();
    }

    virtual bool update_conf(const std::string& puck_params) {
        _index->update_conf(puck_params);
        return true;
    }

protected:
    std::shared_ptr<Index> _index;
};

} //namesapce puck
