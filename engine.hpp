#pragma once
#include "graph.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gkt {

constexpr int8_t EMPTY = 0;
constexpr int8_t BLACK = 1;
constexpr int8_t WHITE = 2;

enum class Rules : int { GraphGo = 0, Gomoku = 1, AntiGomoku = 2 };

inline bool is_k_line_rules(Rules r) {
    return r == Rules::Gomoku || r == Rules::AntiGomoku;
}

struct Position {
    std::shared_ptr<const Graph> graph;
    std::vector<int8_t> occupancy;
    int to_move = 1;
    int num_players = 2;
    std::vector<int> pass_streak;
    std::shared_ptr<const std::unordered_set<uint64_t>> history;
    uint64_t zobrist = 0;
    int move_no = 0;
    Rules rules = Rules::GraphGo;
    int win_length = 5;
    /// k-in-a-row: 0 in progress, -1 draw, 1..k winner. Unused for GraphGo.
    int winner = 0;
    /// Last action: -1 none, `0..n-1` vertex, `n` pass.
    int last_action = -1;
    /// Vertices captured by the last stone play (now empty). Pass clears this.
    std::vector<int> last_captured;

    int n() const { return graph ? graph->n : 0; }
    /// Gomoku and Anti-Gomoku: empty-point moves, no pass, binary {+1,0,-1}.
    bool is_gomoku() const { return is_k_line_rules(rules); }
};

/// Extra channels after the `num_players` occupancy one-hot:
/// 1-liberty, 2-liberty, 3+ liberty, log1p(group size), last-move, just-captured.
constexpr int N_EXTRA_FEATURES = 6;
inline int feature_dim(int num_players) {
    return num_players + N_EXTRA_FEATURES;
}
void extract_features_into(const Position& pos, int me, float* out);

struct MoveResult {
    bool legal = false;
    std::string reason;
    Position new_position;
    std::vector<int> captured;
};

struct FinalizeResult {
    std::vector<float> scores;                 // index 0 unused, 1..k
    std::vector<std::vector<float>> shares;    // [vertex][side] side 0 unused
    /// If true, scores are already {0, 0.5, 1} outcomes (k-in-a-row), not territory.
    bool binary = false;
};

Position make_initial(std::shared_ptr<const Graph> g, int num_players = 2,
                      const std::vector<int8_t>* occupancy = nullptr,
                      int starting_player = 1,
                      Rules rules = Rules::GraphGo, int win_length = 5);

bool is_eliminated(const Position& pos, int side);
std::vector<int> active_sides(const Position& pos);
int next_active(const Position& pos, int current);
bool position_game_over(const Position& pos);

MoveResult try_move(const Position& pos, int vertex_idx, int player);
/// GraphGo: empty vertices plus pass (`n`). k-in-a-row: empty vertices only.
std::vector<int> legal_moves_for(const Position& pos);
/// GraphGo pass primitive. Identity for k-in-a-row and after game-over.
/// Game::pass_move() does not call this blindly: it goes through play(n) and
/// reports illegal for k-in-a-row / game-over instead of looking like a no-op success.
Position apply_pass(const Position& pos);
std::vector<int8_t> legalize_occupancy(const std::vector<int8_t>& occ,
                                       const Graph& graph);

FinalizeResult finalize(const Position& pos);

class Game {
public:
    std::shared_ptr<const Graph> graph;
    int num_players = 2;
    std::vector<float> komi_schedule;
    std::vector<int> sides;
    Position position;

    Game(std::shared_ptr<const Graph> g, int num_players = 2,
         std::vector<float> komi = {}, int starting_player = 1,
         const std::vector<int8_t>* occupancy = nullptr,
         Rules rules = Rules::GraphGo, int win_length = 5);

    std::vector<int> legal_moves() const { return legal_moves_for(position); }
    bool game_over() const { return position_game_over(position); }
    MoveResult play(int vertex_idx);  // vertex; GraphGo also allows n = pass (illegal after game over)
    MoveResult pass_move();           // same as play(n): illegal for k-in-a-row / game-over
    FinalizeResult finalize_game() const { return finalize(position); }
};

}  // namespace gkt
