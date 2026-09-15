#pragma once
#include "mcts.hpp"
#include <functional>
#include <string>
#include <vector>

namespace gkt {

constexpr int FUTURE_PLIES = 4;
constexpr float INIT_OCCUPY_P_MAX = 0.25f;

struct Sample {
    std::vector<float> features;     // n * F
    std::vector<float> legal_mask;   // n+1
    std::vector<float> policy;       // n+1
    int me = 1;
    float value = 0.f;
    std::vector<float> ownership;    // n
    float q = 0.f;
    std::vector<float> opp_policy;   // n+1
    float opp_weight = 0.f;
    std::vector<float> future;       // n
    float lead = 0.f;
    float weight = 1.f;              // policy-surprise sample weight
};

struct SelfPlayConfig {
    int n_simulations = 50;
    float temperature = 1.0f;
    int max_moves = 0;
    int batch_size = 32;
    float q_lambda = 0.5f;  // mix: weight of MC z; 1-q_lambda weights root Q
    bool randomize_sim = true;
    uint32_t rng_seed = 0;
    Rules rules = Rules::GraphGo;
    int win_length = 5;
};

int randomized_log_count(int nominal, double lo_mult, double hi_mult,
                         std::mt19937& rng);
int randomized_sim_count(int nominal, std::mt19937& rng);

Game make_training_game(std::shared_ptr<const Graph> graph, int num_players,
                        std::mt19937& rng, Rules rules = Rules::GraphGo,
                        int win_length = 5);

std::pair<float, std::vector<float>> score_lead_and_ownership(
    const FinalizeResult& fr, int me, int n, int num_players);

// event: game_start | move | batch | game_end
// batch: ply=`move`, this-search sim budget=`n_sim` (remaining after the batch),
//        `batch_done`/`batch_total` count finished eval batches (0 = root expand).
using SelfPlayHeartbeat = std::function<void(const char* event, int move,
                                             int max_moves, int n_sim,
                                             int batch_done, int batch_total)>;

std::vector<Sample> play_one_game(std::shared_ptr<const Graph> graph,
                                  PredictBatchFn& predict,
                                  int num_players,
                                  const SelfPlayConfig& cfg,
                                  const SelfPlayHeartbeat& heartbeat = {});

}  // namespace gkt
