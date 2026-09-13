#include "gkt/engine.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gkt {

namespace {

int side_index(int side) { return side - 1; }

struct BlocksInfo {
    std::vector<int64_t> block_id, lib_count, lib_only, block_flat, block_start;
    int64_t nblocks = 0;
};

BlocksInfo blocks_info(const int8_t* occ, int n,
                       const int64_t* und_indptr, const int64_t* und_indices,
                       const int64_t* in_indptr, const int64_t* in_indices,
                       int side) {
    BlocksInfo info;
    info.block_id.assign(n, -1);
    std::vector<int8_t> visited(n, 0);
    std::vector<int64_t> stack(n, 0);
    int64_t nblocks = 0;
    for (int i = 0; i < n; i++) {
        if (visited[i] || occ[i] != side) continue;
        int64_t top = 0;
        stack[0] = i;
        visited[i] = 1;
        info.block_id[i] = nblocks;
        while (top >= 0) {
            int64_t u = stack[top];
            top--;
            for (int64_t e = und_indptr[u]; e < und_indptr[u + 1]; e++) {
                int w = static_cast<int>(und_indices[e]);
                if (!visited[w] && occ[w] == side) {
                    visited[w] = 1;
                    info.block_id[w] = nblocks;
                    if (top + 1 >= n)
                        throw std::runtime_error("blocks_info: stack overflow");
                    top++;
                    stack[top] = w;
                }
            }
        }
        nblocks++;
    }
    info.nblocks = nblocks;
    info.block_start.assign(static_cast<size_t>(nblocks) + 1, 0);
    for (int i = 0; i < n; i++) {
        if (occ[i] == side) info.block_start[info.block_id[i] + 1]++;
    }
    for (int64_t b = 0; b < nblocks; b++)
        info.block_start[b + 1] += info.block_start[b];
    info.block_flat.assign(static_cast<size_t>(info.block_start[nblocks]), 0);
    std::vector<int64_t> fill(static_cast<size_t>(nblocks), 0);
    for (int i = 0; i < n; i++) {
        if (occ[i] == side) {
            int64_t b = info.block_id[i];
            info.block_flat[info.block_start[b] + fill[b]] = i;
            fill[b]++;
        }
    }
    info.lib_count.assign(static_cast<size_t>(nblocks), 0);
    info.lib_only.assign(static_cast<size_t>(nblocks), -1);
    std::vector<int64_t> lib_mark(n, 0);
    for (int64_t b = 0; b < nblocks; b++) {
        for (int64_t p = info.block_start[b]; p < info.block_start[b + 1]; p++) {
            int u = static_cast<int>(info.block_flat[p]);
            for (int64_t e = in_indptr[u]; e < in_indptr[u + 1]; e++) {
                int w = static_cast<int>(in_indices[e]);
                if (occ[w] == 0 && lib_mark[w] != b + 1) {
                    lib_mark[w] = b + 1;
                    info.lib_count[b]++;
                    info.lib_only[b] = w;
                }
            }
        }
    }
    return info;
}

struct TryCore {
    std::vector<int8_t> new_occ;
    std::vector<int> captured;
    bool legal = false;
};

TryCore try_move_core(const int8_t* occ, int n,
                      const int64_t* und_indptr, const int64_t* und_indices,
                      const int64_t* in_indptr, const int64_t* in_indices,
                      int vertex, int player, int num_players) {
    TryCore r;
    r.new_occ.assign(occ, occ + n);
    r.new_occ[vertex] = static_cast<int8_t>(player);
    std::vector<int> captured;
    captured.reserve(n);
    for (int opp = 1; opp <= num_players; opp++) {
        if (opp == player) continue;
        auto bid = blocks_info(r.new_occ.data(), n, und_indptr, und_indices,
                               in_indptr, in_indices, opp);
        for (int64_t b = 0; b < bid.nblocks; b++) {
            if (bid.lib_count[b] == 0) {
                for (int64_t p = bid.block_start[b]; p < bid.block_start[b + 1]; p++)
                    captured.push_back(static_cast<int>(bid.block_flat[p]));
            }
        }
    }
    for (int c : captured) r.new_occ[c] = 0;
    auto own = blocks_info(r.new_occ.data(), n, und_indptr, und_indices,
                           in_indptr, in_indices, player);
    bool suicide = false;
    for (int64_t b = 0; b < own.nblocks; b++) {
        if (own.lib_count[b] == 0) { suicide = true; break; }
    }
    // After opponent captures, every own block must still have a liberty.
    // A capture does not excuse remaining 0-liberty own stones: on a digraph
    // the captured vertices need not be empty in-neighbours of the mover.
    if (suicide) {
        r.new_occ.assign(occ, occ + n);
        r.legal = false;
        return r;
    }
    r.captured = std::move(captured);
    r.legal = true;
    return r;
}

void require_gomoku_grid(const Graph& g) {
    if (!g.grid)
        throw std::invalid_argument("k-in-a-row requires rectangular .grid metadata");
    const GridMeta& gm = *g.grid;
    if (gm.rows < 2 || gm.cols < 2
        || static_cast<long long>(gm.rows) * gm.cols != g.n)
        throw std::invalid_argument(
            "k-in-a-row .grid needs rows>=2, cols>=2, and rows*cols==n");
}

bool grid_rc(int r, int c, int R, int C, bool torus, int* out) {
    if (torus) {
        if (R <= 0 || C <= 0) return false;
        r = ((r % R) + R) % R;
        c = ((c % C) + C) % C;
        *out = r * C + c;
        return true;
    }
    if (r < 0 || r >= R || c < 0 || c >= C) return false;
    *out = r * C + c;
    return true;
}

int gomoku_run_len(const int8_t* occ, int n, int v, int player,
                   int dr, int dc, int R, int C, bool torus) {
    if (C <= 0 || v < 0 || v >= n || occ == nullptr) return 0;
    int r0 = v / C, c0 = v % C;
    if (r0 < 0 || r0 >= R) return 0;
    int cnt = 1;
    for (int s = 1; s >= -1; s -= 2) {
        for (int t = 1; t < n; t++) {
            int w = 0;
            if (!grid_rc(r0 + s * t * dr, c0 + s * t * dc, R, C, torus, &w)) break;
            if (w < 0 || w >= n) break;
            if (w == v) {
                // Closed wrap: the whole cycle is the same colour. `t` is the
                // period; do not walk the opposite ray (that would double-count).
                return t;
            }
            if (occ[w] != player) break;
            cnt++;
        }
    }
    return cnt;
}

int gomoku_winner_from(const Position& pos, int v, int player) {
    if (!pos.graph || !pos.graph->grid) return 0;
    const GridMeta& gm = *pos.graph->grid;
    int need = std::max(2, pos.win_length);
    static const int D[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (const auto& d : D) {
        if (gomoku_run_len(pos.occupancy.data(), pos.n(), v, player,
                           d[0], d[1], gm.rows, gm.cols, gm.toroidal) >= need)
            return player;
    }
    return 0;
}

/// Lined player from geometry; AntiGomoku flips the winner.
int kline_outcome(Rules rules, int lined) {
    if (lined <= 0) return lined;
    if (rules == Rules::AntiGomoku) return 3 - lined;
    return lined;
}

bool occupancy_has_empty(const std::vector<int8_t>& occ) {
    return std::find(occ.begin(), occ.end(), EMPTY) != occ.end();
}

int scan_gomoku_winner(const Position& pos) {
    const int n = pos.n();
    if (n <= 0 || static_cast<int>(pos.occupancy.size()) != n)
        throw std::invalid_argument("occupancy length != n");
    bool empty = false;
    for (int v = 0; v < n; v++) {
        int8_t p = pos.occupancy[v];
        if (p == EMPTY) { empty = true; continue; }
        if (gomoku_winner_from(pos, v, p))
            return kline_outcome(pos.rules, static_cast<int>(p));
    }
    return empty ? 0 : -1;
}

void xor_place(uint64_t& z, const Graph& g, int np, int vertex, int from, int to) {
    const auto& table = g.table(np);
    const auto& salt = g.turn_salt(np);
    if (from >= 0 && from < static_cast<int>(salt.size())) z ^= salt[from];
    if (to >= 0 && to < static_cast<int>(salt.size())) z ^= salt[to];
    if (vertex >= 0 && vertex < static_cast<int>(table.size())
        && from >= 0 && from < static_cast<int>(table[vertex].size()))
        z ^= table[vertex][from];
}

std::vector<int> legal_moves_fast(const int8_t* occ, int n,
                                  const Graph& g, int player,
                                  uint64_t base_delta, uint64_t cur_zobrist,
                                  const std::unordered_set<uint64_t>& history) {
    const int other = 3 - player;
    const auto* und_indptr = g.und_indptr.data();
    const auto* und_indices = g.und_indices.data();
    const auto* in_indptr = g.in_indptr.data();
    const auto* in_indices = g.in_indices.data();
    const auto& table = g.table(2);

    auto opp = blocks_info(occ, n, und_indptr, und_indices, in_indptr, in_indices, other);
    auto own = blocks_info(occ, n, und_indptr, und_indices, in_indptr, in_indices, player);

    std::vector<std::vector<int8_t>> dead_revive(
        static_cast<size_t>(std::max<int64_t>(own.nblocks, 0)), std::vector<int8_t>(n, 0));
    std::vector<int8_t> is_dead(static_cast<size_t>(own.nblocks), 0);
    for (int64_t b = 0; b < own.nblocks; b++) {
        if (own.lib_count[b] == 0) {
            is_dead[b] = 1;
            for (int64_t p = own.block_start[b]; p < own.block_start[b + 1]; p++) {
                int u = static_cast<int>(own.block_flat[p]);
                for (int64_t e = in_indptr[u]; e < in_indptr[u + 1]; e++) {
                    int w = static_cast<int>(in_indices[e]);
                    if (occ[w] == other) dead_revive[b][w] = 1;
                }
            }
        }
    }

    std::vector<int> out;
    std::vector<int8_t> captured_mark(n), merged_mark(n), lib_mark(n);
    std::vector<int8_t> absorbed(static_cast<size_t>(own.nblocks));
    std::vector<int64_t> stack(n);

    for (int v = 0; v < n; v++) {
        if (occ[v] != 0) continue;
        std::fill(captured_mark.begin(), captured_mark.end(), 0);
        uint64_t cap_delta = 0;
        for (int64_t b = 0; b < opp.nblocks; b++) {
            if (opp.lib_count[b] == 0 || (opp.lib_count[b] == 1 && opp.lib_only[b] == v)) {
                for (int64_t p = opp.block_start[b]; p < opp.block_start[b + 1]; p++) {
                    int c = static_cast<int>(opp.block_flat[p]);
                    captured_mark[c] = 1;
                    cap_delta ^= table[c][occ[c]];
                }
            }
        }
        std::fill(merged_mark.begin(), merged_mark.end(), 0);
        merged_mark[v] = 1;
        int64_t top = 0;
        for (int64_t e = und_indptr[v]; e < und_indptr[v + 1]; e++) {
            int w = static_cast<int>(und_indices[e]);
            if (occ[w] == player && !merged_mark[w]) {
                merged_mark[w] = 1;
                if (top >= n)
                    throw std::runtime_error("legal_moves_fast: merge stack overflow");
                stack[top++] = w;
            }
        }
        std::fill(absorbed.begin(), absorbed.end(), 0);
        while (top > 0) {
            int u = static_cast<int>(stack[--top]);
            int64_t b = own.block_id[u];
            if (b >= 0) absorbed[b] = 1;
            for (int64_t e = und_indptr[u]; e < und_indptr[u + 1]; e++) {
                int w = static_cast<int>(und_indices[e]);
                if (occ[w] == player && !merged_mark[w]) {
                    merged_mark[w] = 1;
                    if (top >= n)
                        throw std::runtime_error("legal_moves_fast: merge stack overflow");
                    stack[top++] = w;
                }
            }
        }
        std::fill(lib_mark.begin(), lib_mark.end(), 0);
        int lib_total = 0;
        for (int i = 0; i < n; i++) {
            if (!merged_mark[i]) continue;
            for (int64_t e = in_indptr[i]; e < in_indptr[i + 1]; e++) {
                int w = static_cast<int>(in_indices[e]);
                if (merged_mark[w] || lib_mark[w]) continue;
                // Captured opponent stones become empty; count them as
                // liberties only if they actually feed this merged block.
                const bool empty_after = (occ[w] == 0) || captured_mark[w];
                if (!empty_after) continue;
                lib_mark[w] = 1;
                lib_total++;
            }
        }
        if (lib_total == 0) continue;
        bool dead = false;
        for (int64_t b = 0; b < own.nblocks; b++) {
            if (!is_dead[b] || absorbed[b]) continue;
            bool revived = false;
            for (int w = 0; w < n; w++) {
                if (dead_revive[b][w] && captured_mark[w]) { revived = true; break; }
            }
            if (!revived) { dead = true; break; }
        }
        if (dead) continue;
        uint64_t delta = base_delta ^ table[v][player] ^ cap_delta;
        uint64_t new_z = cur_zobrist ^ delta;
        if (history.count(new_z)) continue;
        out.push_back(v);
    }
    return out;
}

void require_occupancy(const std::vector<int8_t>& occ, int n, int num_players) {
    if (static_cast<int>(occ.size()) != n)
        throw std::invalid_argument("occupancy length != n");
    if (num_players < 2)
        throw std::invalid_argument("num_players < 2");
    for (int8_t c : occ) {
        if (c == EMPTY) continue;
        if (c < 1 || c > num_players)
            throw std::invalid_argument("occupancy color out of range");
    }
}

uint64_t hash_occupancy(const Graph& g, const std::vector<int8_t>& occ,
                        int to_move, int num_players) {
    require_occupancy(occ, g.n, num_players);
    const auto& table = g.table(num_players);
    const auto& salt = g.turn_salt(num_players);
    if (to_move < 1 || to_move > num_players
        || to_move >= static_cast<int>(salt.size()))
        throw std::invalid_argument("to_move out of range");
    uint64_t z = salt[to_move];
    for (int i = 0; i < g.n; i++) {
        int c = occ[i];
        if (c == EMPTY) continue;
        if (c < 0 || c >= static_cast<int>(table[i].size()))
            throw std::invalid_argument("occupancy color out of range");
        z ^= table[i][c];
    }
    return z;
}

}  // namespace

bool is_eliminated(const Position& pos, int side) {
    int i = side_index(side);
    if (i < 0 || i >= static_cast<int>(pos.pass_streak.size())) return false;
    return pos.pass_streak[i] >= 2;
}

std::vector<int> active_sides(const Position& pos) {
    std::vector<int> a;
    for (int s = 1; s <= pos.num_players; s++)
        if (!is_eliminated(pos, s)) a.push_back(s);
    return a;
}

int next_active(const Position& pos, int current) {
    int np = pos.num_players;
    for (int i = 0; i < np; i++) {
        current = current % np + 1;
        if (!is_eliminated(pos, current)) return current;
    }
    return current;
}

bool position_game_over(const Position& pos) {
    if (pos.is_gomoku()) return pos.winner != 0;
    auto active = active_sides(pos);
    if (active.size() <= 1) return true;
    for (int s : active)
        if (pos.pass_streak[side_index(s)] < 1) return false;
    return true;
}

Position make_initial(std::shared_ptr<const Graph> g, int num_players,
                      const std::vector<int8_t>* occupancy, int starting_player,
                      Rules rules, int win_length) {
    Position p;
    p.graph = std::move(g);
    if (!p.graph)
        throw std::invalid_argument("empty graph");
    if (num_players < 2)
        throw std::invalid_argument("num_players < 2");
    if (starting_player < 1 || starting_player > num_players)
        throw std::invalid_argument("starting_player out of range");
    p.num_players = num_players;
    p.to_move = starting_player;
    p.pass_streak.assign(num_players, 0);
    p.occupancy.assign(p.graph->n, EMPTY);
    p.rules = rules;
    p.win_length = std::max(2, win_length);
    p.winner = 0;
    if (p.is_gomoku()) require_gomoku_grid(*p.graph);
    if (occupancy) {
        if (static_cast<int>(occupancy->size()) != p.graph->n)
            throw std::invalid_argument("occupancy length != n");
        p.occupancy = *occupancy;
    }
    require_occupancy(p.occupancy, p.graph->n, num_players);
    p.history = std::make_shared<std::unordered_set<uint64_t>>();
    p.zobrist = hash_occupancy(*p.graph, p.occupancy, p.to_move, num_players);
    p.move_no = 0;
    p.last_action = -1;
    p.last_captured.clear();
    if (p.is_gomoku()) p.winner = scan_gomoku_winner(p);
    return p;
}

MoveResult try_move(const Position& pos, int vertex_idx, int player) {
    MoveResult r;
    int n = pos.n();
    if (!pos.graph || n <= 0) {
        r.reason = "empty position";
        return r;
    }
    if (vertex_idx < 0 || vertex_idx >= n) {
        r.reason = "vertex out of range";
        return r;
    }
    if (static_cast<int>(pos.occupancy.size()) != n)
        throw std::invalid_argument("occupancy length != n");
    if (pos.occupancy[vertex_idx] != EMPTY) {
        r.reason = "vertex occupied";
        return r;
    }
    if (player != pos.to_move) {
        r.reason = "not this player's turn";
        return r;
    }
    if (position_game_over(pos)) {
        r.reason = "game over";
        return r;
    }
    const Graph& g = *pos.graph;

    if (pos.is_gomoku()) {
        if (!g.grid) {
            r.reason = "k-in-a-row requires grid metadata";
            return r;
        }
        Position nxt = pos;
        nxt.occupancy[vertex_idx] = static_cast<int8_t>(player);
        nxt.winner = kline_outcome(pos.rules,
            gomoku_winner_from(nxt, vertex_idx, player));
        if (nxt.winner == 0 && !occupancy_has_empty(nxt.occupancy))
            nxt.winner = -1;
        nxt.to_move = 3 - player;
        xor_place(nxt.zobrist, g, pos.num_players, vertex_idx, player, nxt.to_move);
        nxt.move_no = pos.move_no + 1;
        nxt.last_action = vertex_idx;
        nxt.last_captured.clear();
        r.legal = true;
        r.new_position = std::move(nxt);
        return r;
    }

    auto core = try_move_core(pos.occupancy.data(), n,
                              g.und_indptr.data(), g.und_indices.data(),
                              g.in_indptr.data(), g.in_indices.data(),
                              vertex_idx, player, pos.num_players);
    if (!core.legal) {
        r.reason = "suicide/occupied forbidden";
        return r;
    }
    std::vector<int> new_streak = pos.pass_streak;
    new_streak[side_index(player)] = 0;
    Position tmp = pos;
    tmp.pass_streak = new_streak;
    int next_player = next_active(tmp, player);
    const auto& table = g.table(pos.num_players);
    const auto& salt = g.turn_salt(pos.num_players);
    uint64_t delta = salt[player] ^ salt[next_player] ^ table[vertex_idx][player];
    for (int c : core.captured) delta ^= table[c][pos.occupancy[c]];
    uint64_t new_z = pos.zobrist ^ delta;
    if (pos.history && pos.history->count(new_z)) {
        r.reason = "superko (situational)";
        return r;
    }
    auto nh = std::make_shared<std::unordered_set<uint64_t>>();
    if (pos.history) *nh = *pos.history;
    nh->insert(pos.zobrist);
    r.legal = true;
    r.captured = core.captured;
    r.new_position = pos;
    r.new_position.occupancy = std::move(core.new_occ);
    r.new_position.to_move = next_player;
    r.new_position.pass_streak = std::move(new_streak);
    r.new_position.history = std::move(nh);
    r.new_position.zobrist = new_z;
    r.new_position.move_no = pos.move_no + 1;
    r.new_position.winner = 0;
    r.new_position.last_action = vertex_idx;
    r.new_position.last_captured = core.captured;
    return r;
}

std::vector<int> legal_moves_for(const Position& pos) {
    if (!pos.graph) return {};
    if (position_game_over(pos)) return {};
    int me = pos.to_move;
    int n = pos.n();
    if (n <= 0) return {};
    if (static_cast<int>(pos.occupancy.size()) != n)
        throw std::invalid_argument("occupancy length != n");
    if (pos.is_gomoku()) {
        std::vector<int> out;
        for (int v = 0; v < n; v++)
            if (pos.occupancy[v] == EMPTY) out.push_back(v);
        return out;
    }
    const Graph& g = *pos.graph;
    int nxt = next_active(pos, me);
    const auto& salt = g.turn_salt(pos.num_players);
    uint64_t base_delta = salt[me] ^ salt[nxt];
    static const std::unordered_set<uint64_t> kEmptyHist;
    const auto& hist = pos.history ? *pos.history : kEmptyHist;
    std::vector<int> out;
    if (pos.num_players == 2) {
        out = legal_moves_fast(pos.occupancy.data(), n, g, me, base_delta,
                               pos.zobrist, hist);
    } else {
        for (int v = 0; v < n; v++) {
            if (pos.occupancy[v] != EMPTY) continue;
            auto r = try_move(pos, v, me);
            if (r.legal) out.push_back(v);
        }
    }
    out.push_back(n);  // pass is always legal
    return out;
}

Position apply_pass(const Position& pos) {
    if (!pos.graph) return pos;
    if (pos.is_gomoku()) return pos;
    if (position_game_over(pos)) return pos;
    int me = pos.to_move;
    std::vector<int> new_streak = pos.pass_streak;
    int si = side_index(me);
    if (si >= 0 && si < static_cast<int>(new_streak.size()))
        new_streak[si] += 1;
    Position tmp = pos;
    tmp.pass_streak = new_streak;
    int nxt = next_active(tmp, me);
    const auto& salt = pos.graph->turn_salt(pos.num_players);
    uint64_t new_z = pos.zobrist;
    if (me >= 0 && me < static_cast<int>(salt.size())) new_z ^= salt[me];
    if (nxt >= 0 && nxt < static_cast<int>(salt.size())) new_z ^= salt[nxt];
    auto nh = std::make_shared<std::unordered_set<uint64_t>>();
    if (pos.history) *nh = *pos.history;
    nh->insert(pos.zobrist);
    Position np = pos;
    np.to_move = nxt;
    np.pass_streak = std::move(new_streak);
    np.history = std::move(nh);
    np.zobrist = new_z;
    np.move_no = pos.move_no + 1;
    np.last_action = pos.n();
    np.last_captured.clear();
    return np;
}

std::vector<int8_t> legalize_occupancy(const std::vector<int8_t>& occupancy,
                                       const Graph& graph) {
    int n = graph.n;
    if (static_cast<int>(occupancy.size()) != n)
        throw std::invalid_argument("occupancy length != n");
    for (int8_t c : occupancy) {
        if (c != EMPTY && (c < 1 || c > 16))
            throw std::invalid_argument("occupancy color out of range");
    }
    std::vector<int8_t> occ = occupancy;
    int k = 0;
    for (int x : occ) if (x > k) k = x;
    for (int side = std::max(k, 2); side >= 1; --side) {
        auto info = blocks_info(occ.data(), n, graph.und_indptr.data(),
                                graph.und_indices.data(), graph.in_indptr.data(),
                                graph.in_indices.data(), side);
        for (int64_t b = 0; b < info.nblocks; b++) {
            if (info.lib_count[b] == 0) {
                for (int64_t p = info.block_start[b]; p < info.block_start[b + 1]; p++)
                    occ[info.block_flat[p]] = EMPTY;
            }
        }
    }
    return occ;
}

void extract_features_into(const Position& pos, int me, float* out) {
    int n = pos.n();
    int np = pos.num_players;
    int F = feature_dim(np);
    if (n <= 0 || np <= 0 || !out) return;
    if (static_cast<int>(pos.occupancy.size()) != n)
        throw std::invalid_argument("occupancy length != n");
    if (me < 1 || me > np)
        throw std::invalid_argument("extract_features: me out of range");
    std::fill(out, out + static_cast<size_t>(n) * F, 0.f);
    for (int i = 0; i < n; i++) {
        int8_t occ = pos.occupancy[i];
        if (occ == EMPTY) continue;
        if (occ < 1 || occ > np)
            throw std::invalid_argument("occupancy color out of range");
        int ch = (static_cast<int>(occ) - me) % np;
        if (ch < 0) ch += np;
        out[static_cast<size_t>(i) * F + ch] = 1.f;
    }
    if (!pos.graph) return;
    const Graph& g = *pos.graph;
    const int extra = np;
    for (int side = 1; side <= np; side++) {
        auto info = blocks_info(pos.occupancy.data(), n,
                                g.und_indptr.data(), g.und_indices.data(),
                                g.in_indptr.data(), g.in_indices.data(), side);
        for (int64_t b = 0; b < info.nblocks; b++) {
            int64_t sz = info.block_start[b + 1] - info.block_start[b];
            float logsz = std::log1p(static_cast<float>(std::max<int64_t>(sz, 0)));
            int64_t libs = info.lib_count[b];
            for (int64_t p = info.block_start[b]; p < info.block_start[b + 1]; p++) {
                int i = static_cast<int>(info.block_flat[p]);
                float* row = out + static_cast<size_t>(i) * F + extra;
                if (libs == 1) row[0] = 1.f;
                else if (libs == 2) row[1] = 1.f;
                else if (libs >= 3) row[2] = 1.f;
                row[3] = logsz;
            }
        }
    }
    if (pos.last_action >= 0 && pos.last_action < n)
        out[static_cast<size_t>(pos.last_action) * F + extra + 4] = 1.f;
    for (int v : pos.last_captured) {
        if (v >= 0 && v < n && pos.occupancy[v] == EMPTY)
            out[static_cast<size_t>(v) * F + extra + 5] = 1.f;
    }
}

FinalizeResult finalize(const Position& pos) {
    int n = pos.n();
    FinalizeResult fr;
    fr.scores.assign(pos.num_players + 1, 0.f);
    fr.shares.assign(n, std::vector<float>(pos.num_players + 1, 0.f));
    if (pos.is_gomoku()) {
        fr.binary = true;
        for (int i = 0; i < n; i++) {
            int8_t c = pos.occupancy[i];
            if (c != EMPTY) fr.shares[i][c] = 1.f;
        }
        if (pos.winner > 0 && pos.winner <= pos.num_players) {
            fr.scores[pos.winner] = 1.f;
        } else {
            for (int p = 1; p <= pos.num_players; p++)
                fr.scores[p] = 0.5f;
        }
        return fr;
    }
    const Graph& g = *pos.graph;
    std::vector<int8_t> occupancy = pos.occupancy;
    std::vector<std::vector<int>> in_neighs(n);
    for (int i = 0; i < n; i++) {
        for (int64_t e = g.in_indptr[i]; e < g.in_indptr[i + 1]; e++)
            in_neighs[i].push_back(static_cast<int>(g.in_indices[e]));
    }
    bool changed = true;
    int guard = 0;
    const int cap = n * std::max(pos.num_players, 1) + 2;
    while (changed) {
        if (++guard > cap)
            throw std::runtime_error("finalize: focus fill did not converge");
        changed = false;
        for (int side = 1; side <= pos.num_players; side++) {
            for (int i = 0; i < n; i++) {
                if (occupancy[i] != EMPTY) continue;
                const auto& in_neigh = in_neighs[i];
                if (in_neigh.empty()) continue;  // no in-edges: not a focus (rules §3)
                bool all = true;
                for (int w : in_neigh) if (occupancy[w] != side) { all = false; break; }
                if (all) { occupancy[i] = static_cast<int8_t>(side); changed = true; }
            }
        }
    }
    changed = true;
    guard = 0;
    while (changed) {
        if (++guard > n + 2)
            throw std::runtime_error("finalize: dame fill did not converge");
        std::vector<std::pair<int, int>> to_fill;
        for (int i = 0; i < n; i++) {
            if (occupancy[i] != EMPTY) continue;
            int party = 0;
            int nparties = 0;
            std::vector<int8_t> seen(pos.num_players + 1, 0);
            for (int w : in_neighs[i]) {
                int8_t c = occupancy[w];
                if (c != EMPTY && !seen[c]) {
                    seen[c] = 1;
                    party = c;
                    nparties++;
                }
            }
            if (nparties == 1) to_fill.emplace_back(i, party);
        }
        changed = !to_fill.empty();
        for (auto [i, side] : to_fill) occupancy[i] = static_cast<int8_t>(side);
    }
    for (int i = 0; i < n; i++) {
        if (occupancy[i] != EMPTY) {
            fr.scores[occupancy[i]] += 1.f;
            fr.shares[i][occupancy[i]] = 1.f;
            continue;
        }
        std::vector<int> parties;
        std::vector<int8_t> seen(pos.num_players + 1, 0);
        for (int w : in_neighs[i]) {
            int8_t c = occupancy[w];
            if (c != EMPTY && !seen[c]) {
                seen[c] = 1;
                parties.push_back(c);
            }
        }
        if (parties.size() >= 2) {
            float share = 1.f / static_cast<float>(parties.size());
            for (int p : parties) {
                fr.scores[p] += share;
                fr.shares[i][p] = share;
            }
        }
    }
    return fr;
}

Game::Game(std::shared_ptr<const Graph> g, int np, std::vector<float> komi,
           int starting_player, const std::vector<int8_t>* occupancy,
           Rules rules, int win_length)
    : graph(std::move(g)), num_players(np) {
    if (is_k_line_rules(rules)) {
        if (!graph)
            throw std::invalid_argument("Gomoku / Anti-Gomoku requires a graph");
        if (np != 2)
            throw std::invalid_argument("Gomoku / Anti-Gomoku is two-player");
        require_gomoku_grid(*graph);
    }
    komi_schedule = std::move(komi);
    komi_schedule.resize(np, 0.f);
    for (int s = 1; s <= np; s++) sides.push_back(s);
    position = make_initial(graph, np, occupancy, starting_player, rules, win_length);
}

MoveResult Game::play(int vertex_idx) {
    if (vertex_idx == position.n()) {
        MoveResult r;
        if (position.is_gomoku()) {
            r.reason = "pass is not a k-in-a-row move";
            return r;
        }
        if (position_game_over(position)) {
            r.reason = "game over";
            return r;
        }
        position = apply_pass(position);
        r.legal = true;
        r.new_position = position;
        return r;
    }
    auto r = try_move(position, vertex_idx, position.to_move);
    if (r.legal) position = r.new_position;
    return r;
}

MoveResult Game::pass_move() {
    return play(position.n());
}

}  // namespace gkt
