#include "gkt/mcts.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace gkt {

static void require_net_batch(const NetBatch& out, int B, int n, const char* where) {
    const size_t npol = static_cast<size_t>(std::max(B, 0)) * static_cast<size_t>(n + 1);
    if (out.policy.size() != npol)
        throw std::runtime_error(std::string(where) + ": policy size != B*(n+1)");
    if (out.value.size() != static_cast<size_t>(std::max(B, 0)))
        throw std::runtime_error(std::string(where) + ": value size != B");
    if (!out.stdev.empty() && out.stdev.size() != out.value.size())
        throw std::runtime_error(std::string(where) + ": stdev size != B");
}

static float batch_stdev(const NetBatch& out, int i) {
    if (out.stdev.empty()) return 0.693147f;  // softplus(0)
    float s = out.stdev[static_cast<size_t>(i)];
    if (!(s >= 0.f)) s = 0.f;
    return s;
}

static double leaf_weight(float stdev) {
    double s = static_cast<double>(stdev);
    return 1.0 / (1.0 + s * s);
}

double MCTS::node_q(const Node& nd, double unvisited) {
    if (nd.visits <= 0)
        return unvisited;
    if (nd.weight_sum > 1e-12)
        return nd.total_value / nd.weight_sum;
    return nd.total_value / static_cast<double>(nd.visits);
}

float score_lead(float my, float total, int n, int num_players) {
    int k = std::max(num_players, 2);
    float denom = static_cast<float>((k - 1) * std::max(n, 1));
    float x = (static_cast<float>(k) * my - total) / denom;
    if (x < -1.f) x = -1.f;
    if (x > 1.f) x = 1.f;
    return x;
}

float score_lead_abs(float my, float total, int num_players) {
    int k = std::max(num_players, 2);
    return (static_cast<float>(k) * my - total) / static_cast<float>(k - 1);
}

int MCTS::new_node(int parent, int action, float prior) {
    Node nd;
    nd.parent = parent;
    nd.action = action;
    nd.prior = prior;
    nd.nn_prior = prior;
    nodes_.push_back(std::move(nd));
    return static_cast<int>(nodes_.size()) - 1;
}

bool MCTS::ensure_pos(int node) {
    if (node < 0 || node >= static_cast<int>(nodes_.size())) return false;
    if (nodes_[node].pos)
        return nodes_[node].pos->graph != nullptr;
    std::vector<int> chain;
    int cur = node;
    while (cur >= 0 && cur < static_cast<int>(nodes_.size()) && !nodes_[cur].pos) {
        chain.push_back(cur);
        if (chain.size() > nodes_.size()) return false;  // parent cycle
        cur = nodes_[cur].parent;
    }
    if (cur < 0 || cur >= static_cast<int>(nodes_.size()) || !nodes_[cur].pos
        || nodes_[cur].pos->graph == nullptr)
        return false;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        int ni = *it;
        int par = nodes_[ni].parent;
        nodes_[ni].pos = std::make_unique<Position>(
            apply_action(*nodes_[par].pos, nodes_[ni].action));
        if (nodes_[ni].pos->graph == nullptr) return false;
    }
    return true;
}

std::vector<int> MCTS::legal_moves(const Position& pos) {
    auto moves = legal_moves_for(pos);
    if (cfg.forbid_pass && !pos.is_gomoku()) {
        int n = pos.n();
        bool stone = false;
        for (int a : moves) {
            if (a != n) {
                stone = true;
                break;
            }
        }
        if (stone) {
            moves.erase(std::remove(moves.begin(), moves.end(), n),
                        moves.end());
        }
    }
    std::shuffle(moves.begin(), moves.end(), rng);
    return moves;
}

Position MCTS::apply_action(const Position& pos, int action) {
    if (action == n_) {
        if (pos.is_gomoku() || position_game_over(pos))
            throw std::runtime_error("MCTS apply_action: pass is illegal");
        return apply_pass(pos);
    }
    auto r = try_move(pos, action, pos.to_move);
    if (!r.legal)
        throw std::runtime_error(
            std::string("MCTS apply_action: illegal move: ") + r.reason);
    return r.new_position;
}

void MCTS::mark_terminal(int node) {
    nodes_[node].value_by_side = terminal_value_by_side(*nodes_[node].pos);
    nodes_[node].has_vside = true;
    int t = nodes_[node].pos->to_move - 1;
    if (t < 0 || t >= static_cast<int>(nodes_[node].value_by_side.size())) t = 0;
    nodes_[node].value = nodes_[node].value_by_side[t];
    nodes_[node].stdev = 0.f;
    nodes_[node].expanded = true;
}

std::vector<float> MCTS::terminal_value_by_side(const Position& pos) {
    if (pos.is_gomoku()) {
        std::vector<float> out(pos.num_players, 0.f);
        if (pos.winner > 0) {
            for (int p = 1; p <= pos.num_players; p++)
                out[p - 1] = (p == pos.winner) ? 1.f : -1.f;
        }
        return out;
    }
    auto fr = finalize(pos);
    int n = std::max(pos.n(), 1);
    float total = 0.f;
    for (int p = 1; p <= pos.num_players; p++) total += fr.scores[p];
    std::vector<float> out(pos.num_players);
    for (int p = 1; p <= pos.num_players; p++)
        out[p - 1] = score_lead(fr.scores[p], total, n, pos.num_players);
    return out;
}

int MCTS::select_child(int node) {
    const auto& kids = nodes_[node].children;
    int best = kids[0];
    double best_score = -1e18;
    int total_visits = 1;
    for (int c : kids) total_visits += nodes_[c].visits;
    const Node& parent = nodes_[node];
    double parent_q = node_q(parent, static_cast<double>(parent.value));
    double explored = 0.0;
    for (int c : kids)
        if (nodes_[c].visits > 0) explored += nodes_[c].nn_prior;
    if (explored < 0.0) explored = 0.0;
    if (explored > 1.0) explored = 1.0;
    double fpu = parent_q - static_cast<double>(cfg.fpu_reduction) * std::sqrt(explored);
    double var_floor = std::max(static_cast<double>(cfg.var_floor), 1e-8);
    double var = var_floor;
    if (parent.n_backup > 1) {
        double mean = parent.raw_sum / parent.n_backup;
        double v = parent.sq_sum / parent.n_backup - mean * mean;
        if (v > var) var = v;
    }
    double c = static_cast<double>(cfg.c_puct)
        * std::sqrt(var + var_floor) / std::sqrt(var_floor);
    for (int cidx : kids) {
        const Node& ch = nodes_[cidx];
        double u = c * ch.prior * std::sqrt(static_cast<double>(total_visits))
                   / (1.0 + ch.visits);
        double q = node_q(ch, fpu);
        double score = q + u;
        if (score > best_score) {
            best_score = score;
            best = cidx;
        }
    }
    return best;
}

std::pair<int, std::vector<int>> MCTS::select_leaf(int root) {
    int node = root;
    std::vector<int> path = {node};
    nodes_[node].visits += 1;
    nodes_[node].total_value -= cfg.virtual_loss;
    while (nodes_[node].expanded && !nodes_[node].children.empty()) {
        node = select_child(node);
        path.push_back(node);
        if (path.size() > nodes_.size() + 1)
            throw std::runtime_error("MCTS select_leaf: tree cycle");
        nodes_[node].visits += 1;
        nodes_[node].total_value -= cfg.virtual_loss;
        if (!ensure_pos(node))
            throw std::runtime_error("MCTS select_leaf: ensure_pos failed");
    }
    if (!ensure_pos(node))
        throw std::runtime_error("MCTS select_leaf: ensure_pos failed");
    return {node, path};
}

void MCTS::expand(int node, const float* policy) {
    if (!nodes_[node].children.empty()) return;
    std::vector<int> legal = nodes_[node].legal_actions;
    nodes_.reserve(nodes_.size() + legal.size());
    nodes_[node].children.reserve(legal.size());
    for (int a : legal) {
        float pr = 0.f;
        if (policy && a >= 0 && a <= n_) pr = policy[a];
        int ch = new_node(node, a, pr);
        nodes_[node].children.push_back(ch);
    }
    nodes_[node].expanded = true;
}

void MCTS::evaluate_leaf(int node, bool add_noise) {
    if (!ensure_pos(node))
        throw std::runtime_error("MCTS evaluate_leaf: ensure_pos failed");
    if (!nodes_[node].has_legal) {
        nodes_[node].legal_actions = legal_moves(*nodes_[node].pos);
        nodes_[node].has_legal = true;
    }
    if (nodes_[node].legal_actions.empty()) {
        mark_terminal(node);
        return;
    }
    int me = nodes_[node].pos->to_move;
    int k = nodes_[node].pos->num_players;
    int n = n_;
    int F = feature_dim(k);
    std::vector<float> Xs(static_cast<size_t>(k) * n * F);
    std::vector<float> masks(static_cast<size_t>(k) * (n + 1), 1.f);
    std::vector<float> mask_me(n + 1, 0.f);
    for (int a : nodes_[node].legal_actions)
        if (a >= 0 && a <= n) mask_me[a] = 1.f;
    for (int p = 1; p <= k; p++) {
        extract_features_into(*nodes_[node].pos, p, Xs.data() + (p - 1) * n * F);
        if (p == me)
            std::copy(mask_me.begin(), mask_me.end(), masks.begin() + (p - 1) * (n + 1));
    }
    NetBatch out = (*predict)(Xs.data(), k, n, F, masks.data());
    require_net_batch(out, k, n, "MCTS evaluate_leaf");
    if (me < 1 || me > k)
        throw std::runtime_error("MCTS evaluate_leaf: to_move out of range");
    const float* pol = out.policy.data() + static_cast<size_t>(me - 1) * (n + 1);
    expand(node, pol);
    if (add_noise && cfg.dirichlet_frac > 0.f && cfg.dirichlet_alpha > 0.f
        && !nodes_[node].children.empty()) {
        std::gamma_distribution<float> gamma(cfg.dirichlet_alpha, 1.f);
        int nc = static_cast<int>(nodes_[node].children.size());
        std::vector<float> noise(nc);
        float s = 0.f;
        for (int i = 0; i < nc; i++) { noise[i] = gamma(rng); s += noise[i]; }
        if (s <= 0.f) s = 1.f;
        for (int i = 0; i < nc; i++) {
            int ch = nodes_[node].children[i];
            float nz = noise[i] / s;
            nodes_[ch].prior = (1.f - cfg.dirichlet_frac) * nodes_[ch].prior
                               + cfg.dirichlet_frac * nz;
        }
    }
    nodes_[node].value_by_side.assign(k, 0.f);
    for (int i = 0; i < k; i++)
        nodes_[node].value_by_side[i] = out.value[i];
    nodes_[node].has_vside = true;
    int vidx = me - 1;
    if (vidx < 0 || vidx >= k) vidx = 0;
    nodes_[node].value = nodes_[node].value_by_side[vidx];
    nodes_[node].stdev = batch_stdev(out, vidx);
}

std::vector<std::vector<float>> MCTS::evaluate_batch(const std::vector<int>& leaves) {
    std::vector<std::vector<float>> values(leaves.size());
    std::vector<int> to_eval_idx;
    for (size_t i = 0; i < leaves.size(); i++) {
        int leaf = leaves[i];
        if (!ensure_pos(leaf))
            throw std::runtime_error("MCTS evaluate_batch: ensure_pos failed");
        if (nodes_[leaf].expanded && nodes_[leaf].has_vside) {
            values[i] = nodes_[leaf].value_by_side;
            continue;
        }
        if (!nodes_[leaf].has_legal) {
            nodes_[leaf].legal_actions = legal_moves(*nodes_[leaf].pos);
            nodes_[leaf].has_legal = true;
        }
        if (nodes_[leaf].legal_actions.empty()) {
            mark_terminal(leaf);
            values[i] = nodes_[leaf].value_by_side;
        } else {
            to_eval_idx.push_back(static_cast<int>(i));
        }
    }
    if (to_eval_idx.empty()) return values;
    int n = n_;
    int rows = 0;
    for (int idx : to_eval_idx) rows += nodes_[leaves[idx]].pos->num_players;
    int F = feature_dim(nodes_[leaves[to_eval_idx[0]]].pos->num_players);
    std::vector<float> Xs(static_cast<size_t>(rows) * n * F);
    std::vector<float> masks(static_cast<size_t>(rows) * (n + 1));
    int off = 0;
    for (int idx : to_eval_idx) {
        int leaf = leaves[idx];
        int k = nodes_[leaf].pos->num_players;
        int me = nodes_[leaf].pos->to_move;
        std::vector<float> mask_me(n + 1, 0.f);
        for (int a : nodes_[leaf].legal_actions)
            if (a >= 0 && a <= n) mask_me[a] = 1.f;
        for (int p = 1; p <= k; p++) {
            extract_features_into(*nodes_[leaf].pos, p, Xs.data() + (off + p - 1) * n * F);
            float* m = masks.data() + (off + p - 1) * (n + 1);
            if (p == me) std::copy(mask_me.begin(), mask_me.end(), m);
            else std::fill(m, m + n + 1, 1.f);
        }
        off += k;
    }
    NetBatch bout = (*predict)(Xs.data(), rows, n, F, masks.data());
    require_net_batch(bout, rows, n, "MCTS evaluate_batch");
    off = 0;
    for (int idx : to_eval_idx) {
        int leaf = leaves[idx];
        int k = nodes_[leaf].pos->num_players;
        int me = nodes_[leaf].pos->to_move;
        if (me < 1 || me > k)
            throw std::runtime_error("MCTS evaluate_batch: to_move out of range");
        const float* pol = bout.policy.data()
            + static_cast<size_t>(off + me - 1) * (n + 1);
        expand(leaf, pol);
        nodes_[leaf].value_by_side.resize(k);
        for (int i = 0; i < k; i++)
            nodes_[leaf].value_by_side[i] = bout.value[static_cast<size_t>(off + i)];
        nodes_[leaf].has_vside = true;
        int vidx = me - 1;
        if (vidx < 0 || vidx >= k) vidx = 0;
        nodes_[leaf].value = nodes_[leaf].value_by_side[vidx];
        nodes_[leaf].stdev = batch_stdev(bout, off + vidx);
        values[idx] = nodes_[leaf].value_by_side;
        off += k;
    }
    return values;
}

void MCTS::backprop(const std::vector<int>& path, const std::vector<float>& vside,
                    float stdev) {
    double w = leaf_weight(stdev);
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        int ni = *it;
        nodes_[ni].total_value += cfg.virtual_loss;
        if (!ensure_pos(ni))
            throw std::runtime_error("MCTS backprop: ensure_pos failed");
        int me = nodes_[ni].pos->to_move;
        float v = 0.f;
        if (me >= 1 && me <= static_cast<int>(vside.size()))
            v = vside[me - 1];
        nodes_[ni].total_value += w * static_cast<double>(v);
        nodes_[ni].weight_sum += w;
        nodes_[ni].n_backup += 1;
        nodes_[ni].raw_sum += static_cast<double>(v);
        nodes_[ni].sq_sum += static_cast<double>(v) * static_cast<double>(v);
    }
}

SearchResult MCTS::search(const Position& pos, bool return_q,
                          std::optional<int> n_simulations, MctsBatchFn on_batch) {
    if (!predict)
        throw std::runtime_error("MCTS: predict is null");
    nodes_.clear();
    n_ = pos.n();
    int remaining0 = std::max(1, n_simulations ? *n_simulations : cfg.n_simulations);
    // Modest hint only. Tree uses indices, not Node pointers. expand() already
    // reserves before children; realloc is cheap because Node holds unique_ptr
    // rather than an inlined Position.
    nodes_.reserve(static_cast<size_t>(n_) + static_cast<size_t>(remaining0) + 16);
    int root = new_node(-1, -1, 0.f);
    nodes_[root].pos = std::make_unique<Position>(pos);
    nodes_[root].legal_actions = legal_moves(pos);
    nodes_[root].has_legal = true;
    SearchResult sr;
    if (nodes_[root].legal_actions.empty()) {
        if (return_q) sr.q = 0.f;
        nodes_.clear();
        return sr;
    }
    evaluate_leaf(root, true);
    int remaining = remaining0;
    int bs = std::max(1, cfg.batch_size);
    int n_batches = (remaining0 + bs - 1) / bs;
    if (on_batch) on_batch(0, n_batches, remaining);
    int batch_i = 0;
    while (remaining > 0) {
        int b = std::min(bs, remaining);
        std::vector<int> leaves;
        std::vector<std::vector<int>> paths;
        leaves.reserve(b);
        for (int i = 0; i < b; i++) {
            auto [leaf, path] = select_leaf(root);
            leaves.push_back(leaf);
            paths.push_back(std::move(path));
        }
        auto values = evaluate_batch(leaves);
        for (int i = 0; i < b; i++)
            backprop(paths[i], values[i], nodes_[leaves[i]].stdev);
        remaining -= b;
        batch_i++;
        if (on_batch) on_batch(batch_i, n_batches, remaining);
    }
    sr.legal_actions = nodes_[root].legal_actions;
    if (nodes_[root].children.size() != sr.legal_actions.size())
        throw std::runtime_error("MCTS: children/legal size mismatch");
    sr.visit_counts.assign(sr.legal_actions.size(), 0.f);
    sr.nn_policy.assign(sr.legal_actions.size(), 0.f);
    for (size_t i = 0; i < nodes_[root].children.size(); i++) {
        const Node& ch = nodes_[nodes_[root].children[i]];
        sr.visit_counts[i] = static_cast<float>(ch.visits);
        sr.nn_policy[i] = ch.nn_prior;
    }
    if (return_q)
        sr.q = static_cast<float>(node_q(nodes_[root], nodes_[root].value));
    nodes_.clear();
    return sr;
}

}  // namespace gkt
