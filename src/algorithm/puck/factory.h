#pragma once
#include "algorithm/puck/index.h"
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include <mutex>

namespace puck {

class VecbaseIndexFactory {
public:
    using CreateFunc = std::function<std::shared_ptr<Index>()>;

    static VecbaseIndexFactory& instance() {
        static VecbaseIndexFactory factory;
        return factory;
    }

    void register_index(const std::string& index_type, CreateFunc create_func) {
        //std::lock_guard<std::mutex> lock(_mutex);
        std::cout << "Registering index type: " << index_type << " " << &_creators << std::endl;
        _creators[index_type] = create_func;
    }

    std::shared_ptr<Index> create_index(const std::string& index_type) {
        // 检查是否已经初始化，如果没有就自动初始化
        ensure_initialized();

        std::cout << "create_index index type: " << index_type << " " << &_creators << std::endl;
        // 尝试在注册的索引类型中查找
        auto it = _creators.find(index_type);
        if (it != _creators.end()) {
            return it->second();
        }
        std::cerr << "Error: Index type '" << index_type << "' not registered!" << std::endl;
        return nullptr;
    }

    // 显式调用注册的方法
    static void initialize() {
        auto& instance_ref = instance();
        std::lock_guard<std::mutex> lock(instance_ref._mutex);

        if (instance_ref._initialized) {
            return; // 已经初始化过，直接返回
        }

        std::cout << "Executing all initializers..." << std::endl;
        for (const auto& initializer : instance_ref._initializers) {
            initializer();
        }

        instance_ref._initialized = true; // 标记为已初始化
    }

    void add_initializer(std::function<void()> initializer) {
        std::lock_guard<std::mutex> lock(_mutex);
        _initializers.push_back(initializer);
    }

private:
    VecbaseIndexFactory() : _initialized(false) {}

    // 确保初始化被调用
    void ensure_initialized() {
        if (!_initialized) {
            initialize();
        }
    }

    std::unordered_map<std::string, CreateFunc> _creators;
    std::vector<std::function<void()>> _initializers; // 存储静态注册的初始化器
    std::mutex _mutex;
    bool _initialized; // 是否已经初始化
};

// 注册宏
#define REGISTER_INDEX(index_type, class_type)                            \
    namespace {                                                           \
        void register_##class_type() {                                    \
            puck::VecbaseIndexFactory::instance().register_index(         \
                index_type, []() { return std::make_shared<class_type>(); }); \
        }                                                                 \
        struct Register##class_type {                                     \
            Register##class_type() {                                      \
                puck::VecbaseIndexFactory::instance().add_initializer(register_##class_type); \
            }                                                             \
        };                                                                \
        static Register##class_type g_register_##class_type;              \
    }

}// namespace puck