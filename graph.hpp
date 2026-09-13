#pragma once
#include "rng.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gkt {

struct GridMeta {
    int rows = 0;
    int cols = 0;
    bool toroidal = false;
};

class Graph {
public:
    int n = 0;
    std::vector<int64_t> in_indptr, in_indices;
    std::vector<int64_t> out_indptr, out_indices;
    std::vector<int64_t> und_indptr, und_indices;
    std::optional<GridMeta> grid;
    std::vector<float> adj_in;    // n*n 0/1 A_in
    std::vector<float> adj_out;   // n*n 0/1 A_out

    Graph() = default;
    Graph(int n_, const std::vector<std::pair<int, int>>& directed_edges,
          std::optional<GridMeta> grid_ = std::nullopt);

    void ensure_zobrist(int num_players) const;
    const std::vector<std::vector<uint64_t>>& table(int num_players) const;
    const std::vector<uint64_t>& turn_salt(int num_players) const;

    static constexpr uint32_t ZOBRIST_SEED = 0x5EEDC0DEu;

private:
    struct Zobrist {
        std::vector<std::vector<uint64_t>> table;  // [v][color] color 0 unused
        std::vector<uint64_t> turn_salt;
    };
    mutable std::unordered_map<int, Zobrist> zobrist_;
};

std::shared_ptr<Graph> make_graph(
    int n, const std::vector<std::pair<int, int>>& edges,
    std::optional<GridMeta> grid = std::nullopt);

}  // namespace gkt
