#pragma once
#include <iostream>
#include <string>
#include <filesystem> 
#include <glog/logging.h>
#include "algorithm/puck/time.h"

namespace puck {
namespace base {

class FunctionTimer {
public:
    explicit FunctionTimer(const char* file, const char* func, int line, const char* info = "")
        : file_(extract_filename(file)), func_(func), line_(line), info_(info) {
        timer.start();
        LOG(INFO) << "========================="
                << "[" << file_ << ":" << line_ << "][" << func_ << "] "
                << "Start. "; //  <<  "[" << info_ << "]";
    }

    ~FunctionTimer() {
        timer.stop();
        LOG(INFO) << "=========================" 
                << "[" << file_ << ":" <<  line_ << "][" << func_ << "] "
                << "Finish. " // <<  "[" << info_ << "]"
                << "(" << timer.s_elapsed() << " seconds)";
    }

private:
    std::string file_;  
    const char* func_;  
    const char* info_; 
    int line_;         
    Timer timer;

    static std::string extract_filename(const char* full_path) {
        return std::filesystem::path(full_path).filename().string();
    }
};

#define TIME_FUNCTION() \
    FunctionTimer timer_instance(__FILE__, __func__, __LINE__)

}  // namespace base
}  // namespace puck