#include "gkt/graph.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace gkt {

static void build_csr(int n, const std::vector<std::vector<int>>& adj,
                      std::vector<int64_t>& indptr, std::vector<int64_t>& indices) {
    indptr.assign(n + 1, 0);
    for (int i = 0; i < n; i++) indptr[i + 1] = indptr[i] + static_cast<int64_t>(adj[i].size());
    indices.resize(static_cast<size_t>(indptr[n]));
    size_t p = 0;
    for (int i = 0; i < n; i++) {
        for (int j : adj[i]) indices[p++] = j;
    }
}

Graph::Graph(int n_, const std::vector<std::pair<int, int>>& directed_edges,
             std::optional<GridMeta> grid_)
    : n(n_), grid(std::move(grid_)) {
    if (n_ < 0) throw std::invalid_argument("n < 0");
    std::vector<std::vector<int>> in_adj(n_), out_adj(n_), und_adj(n_);
    std::vector<std::set<int>> und_set(n_);
    for (auto [u, v] : directed_edges) {
        if (u < 0 || v < 0 || u >= n_ || v >= n_)
            throw std::invalid_argument("edge out of range");
        out_adj[u].push_back(v);
        in_adj[v].push_back(u);
        und_set[u].insert(v);
        und_set[v].insert(u);
    }
    for (int i = 0; i < n_; i++) {
        std::sort(in_adj[i].begin(), in_adj[i].end());
        in_adj[i].erase(std::unique(in_adj[i].begin(), in_adj[i].end()), in_adj[i].end());
        std::sort(out_adj[i].begin(), out_adj[i].end());
        out_adj[i].erase(std::unique(out_adj[i].begin(), out_adj[i].end()), out_adj[i].end());
        und_adj[i].assign(und_set[i].begin(), und_set[i].end());
    }
    build_csr(n_, in_adj, in_indptr, in_indices);
    build_csr(n_, out_adj, out_indptr, out_indices);
    build_csr(n_, und_adj, und_indptr, und_indices);

    adj_in.assign(static_cast<size_t>(n_) * n_, 0.f);
    adj_out.assign(static_cast<size_t>(n_) * n_, 0.f);
    for (int u = 0; u < n_; u++) {
        for (int64_t e = out_indptr[u]; e < out_indptr[u + 1]; e++) {
            int v = static_cast<int>(out_indices[e]);
            adj_out[static_cast<size_t>(u) * n_ + v] = 1.f;
            adj_in[static_cast<size_t>(v) * n_ + u] = 1.f;
        }
    }
}

void Graph::ensure_zobrist(int num_players) const {
    if (zobrist_.count(num_players)) return;
    PythonRandom rng(ZOBRIST_SEED);
    Zobrist z;
    z.table.resize(n);
    for (int v = 0; v < n; v++) {
        z.table[v].assign(num_players + 1, 0);
        for (int c = 1; c <= num_players; c++)
            z.table[v][c] = rng.getrandbits64();
    }
    z.turn_salt.resize(num_players + 1);
    for (int i = 0; i <= num_players; i++)
        z.turn_salt[i] = rng.getrandbits64();
    zobrist_[num_players] = std::move(z);
}

const std::vector<std::vector<uint64_t>>& Graph::table(int num_players) const {
    ensure_zobrist(num_players);
    return zobrist_.at(num_players).table;
}

const std::vector<uint64_t>& Graph::turn_salt(int num_players) const {
    ensure_zobrist(num_players);
    return zobrist_.at(num_players).turn_salt;
}

std::shared_ptr<Graph> make_graph(
    int n, const std::vector<std::pair<int, int>>& edges,
    std::optional<GridMeta> grid) {
    return std::make_shared<Graph>(n, edges, grid);
}

}  // namespace gkt
