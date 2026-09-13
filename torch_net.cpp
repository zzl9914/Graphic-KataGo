#include "gkt/torch_net.hpp"

#ifdef GKT_WITH_TORCH
#include <torch/script.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace gkt {

static torch::Device parse_device(const std::string& d) {
    if (d.rfind("cuda", 0) == 0) {
        if (!torch::cuda::is_available())
            throw std::runtime_error(
                "CUDA requested (" + d + ") but torch::cuda is not available");
        return torch::Device(d);
    }
    return torch::Device(torch::kCPU);
}

ScriptedNet::ScriptedNet(const std::string& path, const std::string& device) {
    device_ = device;
    std::unique_ptr<torch::jit::Module> mod(
        new torch::jit::Module(torch::jit::load(path, parse_device(device))));
    mod->eval();
    module_ = mod.release();
}

ScriptedNet::~ScriptedNet() {
    delete static_cast<torch::jit::Module*>(module_);
    module_ = nullptr;
}

void ScriptedNet::set_graph(const Graph& graph) {
    n_ = graph.n;
    adj_in_ = graph.adj_in;
    adj_out_ = graph.adj_out;
}

NetBatch ScriptedNet::predict_batch(const float* X, int B, int n, int F, const float* masks) {
    if (!module_) throw std::runtime_error("no module");
    if (n_ <= 0)
        throw std::runtime_error("ScriptedNet: set_graph not called");
    if (n != n_)
        throw std::runtime_error("ScriptedNet: n != set_graph n");
    if (static_cast<int>(adj_in_.size()) != n * n
        || static_cast<int>(adj_out_.size()) != n * n)
        throw std::runtime_error("ScriptedNet: adj size mismatch");
    torch::NoGradGuard ng;
    auto* mod = static_cast<torch::jit::Module*>(module_);
    auto device = parse_device(device_);
    auto x = torch::from_blob(const_cast<float*>(X), {B, n, F}, torch::kFloat32).to(device).clone();
    auto m = torch::from_blob(const_cast<float*>(masks), {B, n + 1}, torch::kFloat32).to(device).clone();
    auto ain = torch::from_blob(adj_in_.data(), {n, n}, torch::kFloat32).to(device).clone();
    auto aout = torch::from_blob(adj_out_.data(), {n, n}, torch::kFloat32).to(device).clone();
    auto out = mod->forward(std::vector<torch::jit::IValue>{x, m > 0, ain, aout});
    auto tup = out.toTuple();
    auto pol = tup->elements()[0].toTensor().contiguous().cpu();
    auto val = tup->elements()[1].toTensor().contiguous().cpu();
    const size_t npol = static_cast<size_t>(B) * static_cast<size_t>(n + 1);
    if (static_cast<size_t>(pol.numel()) != npol)
        throw std::runtime_error(
            "predict_batch policy size != B*(n+1) (pass dim included)");
    NetBatch nb;
    nb.policy.resize(npol);
    nb.value.resize(static_cast<size_t>(std::max(B, 0)));
    if (npol)
        std::memcpy(nb.policy.data(), pol.data_ptr<float>(), npol * sizeof(float));
    auto v = val.contiguous().reshape({-1});
    if (v.dim() == 0) {
        if (B != 1)
            throw std::runtime_error("predict_batch scalar value only valid for B=1");
        if (!nb.value.empty()) nb.value[0] = v.item<float>();
    } else if (static_cast<size_t>(v.numel()) != nb.value.size()) {
        throw std::runtime_error("predict_batch value size != B");
    } else if (!nb.value.empty()) {
        std::memcpy(nb.value.data(), v.data_ptr<float>(),
                    nb.value.size() * sizeof(float));
    }
    if (tup->elements().size() >= 3) {
        auto st = tup->elements()[2].toTensor().contiguous().cpu().reshape({-1});
        nb.stdev.resize(nb.value.size());
        if (st.dim() == 0) {
            if (B != 1)
                throw std::runtime_error("predict_batch scalar stdev only valid for B=1");
            if (!nb.stdev.empty()) nb.stdev[0] = st.item<float>();
        } else if (static_cast<size_t>(st.numel()) != nb.stdev.size()) {
            throw std::runtime_error("predict_batch stdev size != B");
        } else if (!nb.stdev.empty()) {
            std::memcpy(nb.stdev.data(), st.data_ptr<float>(),
                        nb.stdev.size() * sizeof(float));
        }
    }
    return nb;
}

PredictBatchFn ScriptedNet::as_fn() {
    return [this](const float* X, int B, int n, int F, const float* masks) {
        return this->predict_batch(X, B, n, F, masks);
    };
}

bool torch_available() { return true; }

}  // namespace gkt

#else

namespace gkt {
ScriptedNet::~ScriptedNet() {}
bool torch_available() { return false; }
}

#endif
