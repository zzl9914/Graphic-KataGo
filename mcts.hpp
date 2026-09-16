#pragma once
#include "engine.hpp"
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace gkt {

struct NetBatch {
    // X: (B, n, F) row-major; masks: (B, n+1)
    std::vector<float> policy;  // B * (n+1)
    std::vector<float> value;   // B
    std::vector<float> stdev;   // B; empty → treated as ~ln(2)
};

using PredictBatchFn = std::function<NetBatch(const float* X, int B, int n, int F,
                                              const float* masks)>;
// After each MCTS GPU/CPU eval batch: (batches_done, n_batches, sims_remaining).
using MctsBatchFn = std::function<void(int batch_done, int n_batches, int remaining)>;

struct MCTSConfig {
    bool forbid_pass = false;  // training: drop pass if a stone move exists
    float c_puct = 1.5f;
    int n_simulations = 50;
    float dirichlet_alpha = 0.3f;
    float dirichlet_frac = 0.25f;
    int batch_size = 32;
    float virtual_loss = 3.0f;
    float fpu_reduction = 0.2f;
    float var_floor = 0.04f;
};

struct SearchResult {
    std::vector<int> legal_actions;
    std::vector<float> visit_counts;
    std::vector<float> nn_policy;  // pre-Dirichlet prior, aligned with legal_actions
    float q = 0.f;
};

class MCTS {
public:
    // Non-owning: the function may hold a Python object and must outlive MCTS,
    // and be destroyed while the GIL is held.
    PredictBatchFn* predict = nullptr;
    MCTSConfig cfg;
    std::mt19937 rng;

    MCTS(PredictBatchFn& pred, MCTSConfig c, uint32_t seed = 0)
        : predict(&pred), cfg(c), rng(seed) {}

    SearchResult search(const Position& pos, bool return_q = false,
                        std::optional<int> n_simulations = std::nullopt,
                        MctsBatchFn on_batch = {});

private:
    struct Node {
        // Heap board: presence of the pointer means this node has a Position.
        // Tree uses indices, so vector realloc does not need stable Node addresses.
        std::unique_ptr<Position> pos;
        int parent = -1;
        int action = -1;
        float prior = 0.f;
        float nn_prior = 0.f;
        std::vector<int> children;
        int visits = 0;
        double total_value = 0.0;
        double weight_sum = 0.0;  // Σ 1/(1+σ²); Q = total_value / weight_sum
        int n_backup = 0;
        double raw_sum = 0.0;
        double sq_sum = 0.0;
        float stdev = 0.f;
        std::vector<int> legal_actions;
        bool has_legal = false;
        bool expanded = false;
        float value = 0.f;
        std::vector<float> value_by_side;
        bool has_vside = false;
    };

    int n_ = 0;
    std::vector<Node> nodes_;

    int new_node(int parent, int action, float prior);
    bool ensure_pos(int node);
    std::pair<int, std::vector<int>> select_leaf(int root);
    int select_child(int node);
    std::vector<std::vector<float>> evaluate_batch(const std::vector<int>& leaves);
    void evaluate_leaf(int node, bool add_noise);
    void expand(int node, const float* policy);
    Position apply_action(const Position& pos, int action);
    std::vector<int> legal_moves(const Position& pos);
    std::vector<float> terminal_value_by_side(const Position& pos);
    void mark_terminal(int node);
    void backprop(const std::vector<int>& path, const std::vector<float>& vside,
                  float stdev);
    static double node_q(const Node& nd, double unvisited);
};

float score_lead(float my, float total, int n, int num_players);
float score_lead_abs(float my, float total, int num_players);

}  // namespace gkt
