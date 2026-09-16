#include "gkt/selfplay.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace gkt {

int randomized_log_count(int nominal, double lo_mult, double hi_mult,
                         std::mt19937& rng) {
    int n = std::max(nominal, 1);
    double lo = std::max(lo_mult * n, 1.0);
    double hi = std::max(hi_mult * n, lo);
    std::uniform_real_distribution<double> u(std::log(lo), std::log(hi));
    return std::max(1, static_cast<int>(std::round(std::exp(u(rng)))));
}

int randomized_sim_count(int nominal, std::mt19937& rng) {
    return randomized_log_count(nominal, 0.25, 4.0, rng);
}

Game make_training_game(std::shared_ptr<const Graph> graph, int num_players,
                        std::mt19937& rng, Rules rules, int win_length) {
    int k = std::max(2, num_players);
    int n = graph->n;
    std::uniform_real_distribution<float> uf(0.f, 1.f);
    std::uniform_int_distribution<int> col(1, k);
    float pmax = is_k_line_rules(rules) ? 0.12f : INIT_OCCUPY_P_MAX;
    for (int attempt = 0; attempt < 32; attempt++) {
        float p = uf(rng) * pmax;
        std::vector<int8_t> occ(n, EMPTY);
        for (int i = 0; i < n; i++) {
            if (uf(rng) < p) occ[i] = static_cast<int8_t>(col(rng));
        }
        if (!is_k_line_rules(rules))
            occ = legalize_occupancy(occ, *graph);
        std::vector<int> counts(k, 0);
        for (int8_t c : occ) if (c >= 1 && c <= k) counts[c - 1]++;
        int fewest = *std::min_element(counts.begin(), counts.end());
        int to_move = 1;
        for (int i = 0; i < k; i++) if (counts[i] == fewest) { to_move = i + 1; break; }
        Game game(graph, k, {}, to_move, &occ, rules, win_length);
        if (!is_k_line_rules(rules) || !game.game_over()) return game;
    }
    return Game(graph, k, {}, 1, nullptr, rules, win_length);
}

std::pair<float, std::vector<float>> score_lead_and_ownership(
    const FinalizeResult& fr, int me, int n, int num_players) {
    float total = 0.f;
    for (int p = 1; p <= num_players; p++) total += fr.scores[p];
    float my = (me < static_cast<int>(fr.scores.size())) ? fr.scores[me] : 0.f;
    float lead = 0.f;
    if (fr.binary) {
        lead = 2.f * my - 1.f;
        if (lead < -1.f) lead = -1.f;
        if (lead > 1.f) lead = 1.f;
    } else {
        lead = score_lead(my, total, n, num_players);
    }
    std::vector<float> own(n, 0.f);
    for (int i = 0; i < n; i++) {
        float mine = (me < static_cast<int>(fr.shares[i].size())) ? fr.shares[i][me] : 0.f;
        float rest = 0.f;
        for (int p = 1; p <= num_players; p++) rest += fr.shares[i][p];
        own[i] = mine - (rest - mine);
    }
    return {lead, own};
}

static void occupancy_view(const std::vector<int8_t>& occ, int me, float* out) {
    int n = static_cast<int>(occ.size());
    for (int i = 0; i < n; i++) {
        if (occ[i] == me) out[i] = 1.f;
        else if (occ[i] != EMPTY) out[i] = -1.f;
        else out[i] = 0.f;
    }
}

std::vector<Sample> play_one_game(std::shared_ptr<const Graph> graph,
                                  PredictBatchFn& predict,
                                  int num_players,
                                  const SelfPlayConfig& cfg,
                                  const SelfPlayHeartbeat& heartbeat) {
    std::mt19937 rng(cfg.rng_seed);
    Game game = make_training_game(graph, num_players, rng, cfg.rules, cfg.win_length);
    MCTSConfig mc;
    mc.n_simulations = cfg.n_simulations;
    mc.batch_size = cfg.batch_size;
    MCTS mcts(predict, mc, cfg.rng_seed + 1);
    int max_moves;
    if (is_k_line_rules(cfg.rules)) {
        max_moves = graph->n;
    } else {
        max_moves = cfg.max_moves > 0 ? cfg.max_moves : (graph->n * 2 + 40);
        if (cfg.randomize_sim)
            max_moves = randomized_log_count(max_moves, 0.5, 2.0, rng);
    }
    int n = graph->n;
    int pass_after = max_moves;
    if (!is_k_line_rules(cfg.rules)) {
        pass_after = std::min(max_moves, std::max(n / 8, 16));
    }
    int F = feature_dim(num_players);

    struct Partial {
        std::vector<float> X, mask, pol;
        int me;
        float q;
        float weight;
        std::vector<int8_t> occ;
    };
    std::vector<Partial> samples;
    int move_count = 0;
    if (heartbeat)
        heartbeat("game_start", 0, max_moves, cfg.n_simulations, 0, 0);
    while (move_count < max_moves && !game.game_over()) {
        int me = game.position.to_move;
        std::optional<int> n_this;
        if (cfg.randomize_sim)
            n_this = randomized_sim_count(mcts.cfg.n_simulations, rng);
        int n_search = n_this ? *n_this : cfg.n_simulations;
        if (heartbeat)
            heartbeat("move", move_count, max_moves, n_search, 0, 0);
        MctsBatchFn on_batch;
        if (heartbeat) {
            int ply = move_count;
            int mm = max_moves;
            on_batch = [&heartbeat, ply, mm](int done, int total, int rem) {
                heartbeat("batch", ply, mm, rem, done, total);
            };
        }
        mcts.cfg.forbid_pass = (!is_k_line_rules(cfg.rules))
            && (move_count < pass_after);
        auto sr = mcts.search(game.position, true, n_this, on_batch);
        if (sr.legal_actions.empty())
            throw std::runtime_error("self-play: no legal moves but game not over");
        if (sr.visit_counts.size() != sr.legal_actions.size())
            throw std::runtime_error("self-play: visit_counts/legal size mismatch");
        if (sr.nn_policy.size() != sr.legal_actions.size())
            throw std::runtime_error("self-play: nn_policy/legal size mismatch");

        std::vector<float> raw = sr.visit_counts;
        float vis_sum = 0.f;
        for (float c : raw) vis_sum += c;
        if (vis_sum <= 0.f)
            throw std::runtime_error("self-play: visit counts sum to 0");
        float kl = 0.f;
        for (size_t i = 0; i < raw.size(); i++) {
            float p = raw[i] / vis_sum;
            float q = std::max(sr.nn_policy[i], 1e-8f);
            if (p > 0.f) kl += p * std::log(p / q);
        }
        float surprise_w = 1.f + 2.f * kl;
        if (surprise_w < 0.5f) surprise_w = 0.5f;
        if (surprise_w > 8.f) surprise_w = 8.f;

        float maxv = 0.f;
        for (float c : raw) if (c > maxv) maxv = c;
        float thresh = std::max(2.f, 0.02f * maxv);
        std::vector<float> pruned = raw;
        float prune_sum = 0.f;
        for (size_t i = 0; i < pruned.size(); i++) {
            if (raw[i] < thresh && sr.nn_policy[i] < 0.03f) pruned[i] = 0.f;
            prune_sum += pruned[i];
        }
        if (prune_sum <= 0.f) {
            pruned = raw;
            prune_sum = vis_sum;
        }
        for (float& c : pruned) c /= prune_sum;

        std::vector<float> sample_counts = raw;
        if (cfg.temperature > 0.f) {
            float inv = 1.f / cfg.temperature;
            for (float& c : sample_counts) c = std::pow(c, inv);
        }
        float sum = 0.f;
        for (float c : sample_counts) sum += c;
        if (sum <= 0.f)
            throw std::runtime_error("self-play: visit counts sum to 0");
        for (float& c : sample_counts) c /= sum;
        std::discrete_distribution<int> dist(sample_counts.begin(), sample_counts.end());
        int action_idx = dist(rng);
        int action = sr.legal_actions[action_idx];
        Partial p;
        p.X.assign(static_cast<size_t>(n) * F, 0.f);
        extract_features_into(game.position, me, p.X.data());
        p.mask.assign(n + 1, 0.f);
        p.pol.assign(n + 1, 0.f);
        for (size_t i = 0; i < sr.legal_actions.size(); i++) {
            p.mask[sr.legal_actions[i]] = 1.f;
            p.pol[sr.legal_actions[i]] = pruned[i];
        }
        p.me = me;
        p.q = sr.q;
        p.weight = surprise_w;
        p.occ = game.position.occupancy;
        samples.push_back(std::move(p));
        auto played = game.play(action);
        if (!played.legal)
            throw std::runtime_error(
                std::string("self-play illegal move: ") + played.reason);
        move_count++;
    }
    if (heartbeat)
        heartbeat("game_end", move_count, max_moves, cfg.n_simulations, 0, 0);
    auto fr = game.finalize_game();
    std::vector<Sample> out;
    int n_s = static_cast<int>(samples.size());
    for (int i = 0; i < n_s; i++) {
        auto [lead, own] = score_lead_and_ownership(fr, samples[i].me, n, num_players);
        float total = 0.f;
        for (int p = 1; p <= num_players; p++) total += fr.scores[p];
        float my = (samples[i].me < static_cast<int>(fr.scores.size()))
            ? fr.scores[samples[i].me] : 0.f;
        float lead_abs = fr.binary ? lead : score_lead_abs(my, total, num_players);
        // rto (search): mix z/n with root Q. abs (supervision): MC stones only
        // on Graph-Go so the two heads can disagree; k-in-a-row shares the mix
        // (already in [-1, 1], no board-size scale).
        float rto_t = cfg.q_lambda * lead + (1.f - cfg.q_lambda) * samples[i].q;
        float abs_t = fr.binary ? rto_t : lead_abs;
        Sample s;
        s.features = samples[i].X;
        s.legal_mask = samples[i].mask;
        s.policy = samples[i].pol;
        s.me = samples[i].me;
        s.value = abs_t;
        s.value_rto = rto_t;
        s.ownership = std::move(own);
        s.q = samples[i].q;
        if (i + 1 < n_s) {
            s.opp_policy = samples[i + 1].pol;
            s.opp_weight = 1.f;
        } else {
            s.opp_policy.assign(n + 1, 0.f);
            s.opp_weight = 0.f;
        }
        int j = std::min(i + FUTURE_PLIES, std::max(n_s - 1, 0));
        s.future.assign(n, 0.f);
        if (n_s > 0) occupancy_view(samples[j].occ, samples[i].me, s.future.data());
        s.lead = lead;
        s.weight = samples[i].weight;
        out.push_back(std::move(s));
    }
    return out;
}

}  // namespace gkt
