#pragma once
#include "mcts.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace gkt {

class ScriptedNet {
public:
    ScriptedNet(const ScriptedNet&) = delete;
    ScriptedNet& operator=(const ScriptedNet&) = delete;
    ScriptedNet(ScriptedNet&&) = delete;
    ScriptedNet& operator=(ScriptedNet&&) = delete;
    ~ScriptedNet();
#ifdef GKT_WITH_TORCH
    explicit ScriptedNet(const std::string& path, const std::string& device = "cpu");
    void set_graph(const Graph& graph);
    NetBatch predict_batch(const float* X, int B, int n, int F, const float* masks);
    PredictBatchFn as_fn();
    std::string device() const { return device_; }
private:
    void* module_ = nullptr;
    std::string device_;
    std::vector<float> adj_in_, adj_out_;
    int n_ = 0;
#else
    explicit ScriptedNet(const std::string&, const std::string& = "cpu") {
        throw std::runtime_error("gkt_native was built without LibTorch");
    }
    void set_graph(const Graph&) {}
    NetBatch predict_batch(const float*, int, int, int, const float*) { return {}; }
    PredictBatchFn as_fn() { return {}; }
#endif
};

bool torch_available();

}  // namespace gkt
