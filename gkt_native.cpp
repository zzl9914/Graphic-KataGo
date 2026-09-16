#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>

#include "gkt/engine.hpp"
#include "gkt/mcts.hpp"
#include "gkt/selfplay.hpp"
#include "gkt/torch_net.hpp"

#include <memory>
#include <stdexcept>

namespace py = pybind11;

namespace {

std::shared_ptr<gkt::Graph> graph_from_py(int n, const std::vector<std::pair<int, int>>& edges,
                                          py::object grid) {
    std::optional<gkt::GridMeta> gm;
    if (!grid.is_none()) {
        auto t = grid.cast<py::tuple>();
        gm = gkt::GridMeta{t[0].cast<int>(), t[1].cast<int>(), t[2].cast<bool>()};
    }
    return std::make_shared<gkt::Graph>(n, edges, gm);
}

gkt::PredictBatchFn wrap_python_net(py::object net) {
    py::object predict = net.attr("predict_batch");
    return [predict](const float* X, int B, int n, int F, const float* masks) {
        py::gil_scoped_acquire gil;
        py::array_t<float> xa({B, n, F});
        py::array_t<float> ma({B, n + 1});
        size_t nx = static_cast<size_t>(B) * n * F;
        size_t nm = static_cast<size_t>(B) * (n + 1);
        if (nx) std::memcpy(xa.mutable_data(), X, sizeof(float) * nx);
        if (nm) std::memcpy(ma.mutable_data(), masks, sizeof(float) * nm);
        py::object res = predict(xa, ma);
        py::tuple tup = py::cast<py::tuple>(res);
        if (tup.size() < 2)
            throw std::runtime_error("predict_batch must return (policy, value[, stdev])");
        auto pol = py::cast<py::array_t<float>>(tup[0]);
        auto val = py::cast<py::array_t<float>>(tup[1]);
        auto polc = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(pol);
        auto valc = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(val);
        if (!polc)
            throw std::runtime_error("predict_batch policy is not a float array");
        if (static_cast<size_t>(polc.size()) != nm)
            throw std::runtime_error("predict_batch policy size != B*(n+1) (pass dim included)");
        gkt::NetBatch nb;
        nb.policy.resize(nm);
        nb.value.resize(static_cast<size_t>(std::max(B, 0)));
        if (nm) std::memcpy(nb.policy.data(), polc.data(), sizeof(float) * nm);
        if (!valc)
            throw std::runtime_error("predict_batch value is not a float array");
        if (valc.ndim() == 0) {
            if (B != 1)
                throw std::runtime_error("predict_batch scalar value only valid for B=1");
            if (!nb.value.empty()) nb.value[0] = valc.data()[0];
        } else if (static_cast<size_t>(valc.size()) != nb.value.size()) {
            throw std::runtime_error("predict_batch value size != B");
        } else if (!nb.value.empty()) {
            std::memcpy(nb.value.data(), valc.data(), sizeof(float) * nb.value.size());
        }
        if (tup.size() >= 3) {
            auto st = py::cast<py::array_t<float>>(tup[2]);
            auto stc = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(st);
            if (!stc)
                throw std::runtime_error("predict_batch stdev is not a float array");
            if (stc.ndim() == 0) {
                if (B != 1)
                    throw std::runtime_error("predict_batch scalar stdev only valid for B=1");
                nb.stdev.assign(1, stc.data()[0]);
            } else if (static_cast<size_t>(stc.size()) != nb.value.size()) {
                throw std::runtime_error("predict_batch stdev size != B");
            } else {
                nb.stdev.resize(nb.value.size());
                if (!nb.stdev.empty())
                    std::memcpy(nb.stdev.data(), stc.data(), sizeof(float) * nb.stdev.size());
            }
        }
        return nb;
    };
}

gkt::SelfPlayHeartbeat wrap_heartbeat(py::object hb) {
    if (hb.is_none()) return {};
    return [hb](const char* event, int move, int max_moves, int n_sim,
                int batch_done, int batch_total) {
        py::gil_scoped_acquire gil;
        hb(event, move, max_moves, n_sim, batch_done, batch_total);
    };
}

gkt::Rules parse_rules(const std::string& s) {
    std::string t;
    t.reserve(s.size());
    for (unsigned char c : s) {
        if (c == '-' || c == '_') continue;
        t.push_back(static_cast<char>(std::tolower(c)));
    }
    if (t == "antigomoku")
        return gkt::Rules::AntiGomoku;
    if (t == "gomoku")
        return gkt::Rules::Gomoku;
    if (t == "go" || t == "graphgo")
        return gkt::Rules::GraphGo;
    throw std::invalid_argument(
        "unknown rules '" + s + "'; expected 'go', 'gomoku', or 'antigomoku'");
}

gkt::SelfPlayConfig selfplay_cfg(int n_sim, float temperature, int max_moves,
                                 int batch_size, float q_lambda, bool randomize_sim,
                                 uint32_t seed,
                                 const std::string& rules = "go", int win_length = 5) {
    gkt::SelfPlayConfig cfg;
    cfg.n_simulations = n_sim;
    cfg.temperature = temperature;
    cfg.max_moves = max_moves;
    cfg.batch_size = batch_size;
    cfg.q_lambda = q_lambda;
    cfg.randomize_sim = randomize_sim;
    cfg.rng_seed = seed;
    cfg.rules = parse_rules(rules);
    cfg.win_length = win_length;
    return cfg;
}

py::list samples_to_py(const std::vector<gkt::Sample>& samples, int n, int F) {
    py::list out;
    for (const auto& s : samples) {
        py::array_t<float> X({n, F}), mask({n + 1}), pol({n + 1}), own({n}),
            opp({n + 1}), fut({n});
        auto copy_f = [](py::array_t<float>& a, const std::vector<float>& v,
                         const char* name) {
            if (static_cast<size_t>(a.size()) != v.size())
                throw std::runtime_error(std::string("sample ") + name + " size mismatch");
            if (!v.empty())
                std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(float));
        };
        copy_f(X, s.features, "features");
        copy_f(mask, s.legal_mask, "legal_mask");
        copy_f(pol, s.policy, "policy");
        copy_f(own, s.ownership, "ownership");
        copy_f(opp, s.opp_policy, "opp_policy");
        copy_f(fut, s.future, "future");
        out.append(py::make_tuple(X, mask, pol, s.me, s.value, own, s.q, opp,
                                  s.opp_weight, fut, s.lead, s.weight,
                                  s.value_rto));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(gkt_native, m) {
    m.doc() = "GKT C++ engine, MCTS, and self-play";
    m.def("torch_available", &gkt::torch_available);

    py::class_<gkt::Graph, std::shared_ptr<gkt::Graph>>(m, "Graph")
        .def(py::init(&graph_from_py), py::arg("n"), py::arg("edges"),
             py::arg("grid") = py::none())
        .def_readonly("n", &gkt::Graph::n)
        .def("adj_in", [](const gkt::Graph& g) {
            py::array_t<float> a({g.n, g.n});
            std::memcpy(a.mutable_data(), g.adj_in.data(),
                        sizeof(float) * static_cast<size_t>(g.n) * g.n);
            return a;
        })
        .def("adj_out", [](const gkt::Graph& g) {
            py::array_t<float> a({g.n, g.n});
            std::memcpy(a.mutable_data(), g.adj_out.data(),
                        sizeof(float) * static_cast<size_t>(g.n) * g.n);
            return a;
        })
        .def("zobrist_table", [](const gkt::Graph& g, int k) {
            const auto& t = g.table(k);
            py::list rows;
            for (int v = 0; v < g.n; v++) {
                py::list row;
                for (auto x : t[v]) row.append(py::int_(x));
                rows.append(row);
            }
            return rows;
        })
        .def("turn_salt", [](const gkt::Graph& g, int k) {
            py::list s;
            for (auto x : g.turn_salt(k)) s.append(py::int_(x));
            return s;
        });

    py::class_<gkt::Position>(m, "Position")
        .def_property_readonly("n", &gkt::Position::n)
        .def_readonly("to_move", &gkt::Position::to_move)
        .def_readonly("num_players", &gkt::Position::num_players)
        .def_readonly("move_no", &gkt::Position::move_no)
        .def_readonly("zobrist", &gkt::Position::zobrist)
        .def_property_readonly("rules", [](const gkt::Position& p) {
            if (p.rules == gkt::Rules::AntiGomoku) return "antigomoku";
            if (p.rules == gkt::Rules::Gomoku) return "gomoku";
            return "go";
        })
        .def_readonly("win_length", &gkt::Position::win_length)
        .def_readonly("winner", &gkt::Position::winner)
        .def_property_readonly("occupancy", [](const gkt::Position& p) {
            int n = p.n();
            if (n < 0)
                throw std::runtime_error("occupancy: n < 0");
            if (static_cast<int>(p.occupancy.size()) != n)
                throw std::runtime_error("occupancy length != n");
            py::array_t<int8_t> a({n});
            if (n > 0)
                std::memcpy(a.mutable_data(), p.occupancy.data(),
                            static_cast<size_t>(n));
            return a;
        })
        .def_property_readonly("pass_streak", [](const gkt::Position& p) {
            return p.pass_streak;
        });

    py::class_<gkt::MoveResult>(m, "MoveResult")
        .def_readonly("legal", &gkt::MoveResult::legal)
        .def_readonly("reason", &gkt::MoveResult::reason)
        .def_readonly("new_position", &gkt::MoveResult::new_position)
        .def_readonly("captured", &gkt::MoveResult::captured);

    py::class_<gkt::Game>(m, "Game")
        .def(py::init([](std::shared_ptr<gkt::Graph> g, int np, std::vector<float> komi,
                         int start, py::object occ, std::string rules, int win_length) {
                 std::vector<int8_t> ov;
                 const std::vector<int8_t>* ptr = nullptr;
                 if (!occ.is_none()) {
                     auto a = py::cast<py::array_t<int8_t>>(occ);
                     ov.assign(a.data(), a.data() + a.size());
                     ptr = &ov;
                 }
                 return gkt::Game(std::move(g), np, std::move(komi), start, ptr,
                                  parse_rules(rules), win_length);
             }),
             py::arg("graph"), py::arg("num_players") = 2,
             py::arg("komi_schedule") = std::vector<float>{},
             py::arg("starting_player") = 1,
             py::arg("occupancy") = py::none(),
             py::arg("rules") = "go",
             py::arg("win_length") = 5)
        .def("legal_moves", &gkt::Game::legal_moves)
        .def("game_over", &gkt::Game::game_over)
        .def("play", &gkt::Game::play)
        .def("pass_move", &gkt::Game::pass_move)
        .def("finalize", [](const gkt::Game& g) {
            auto fr = g.finalize_game();
            py::dict scores;
            for (int s : g.sides) scores[py::int_(s)] = fr.scores[s];
            py::list shares;
            int n = g.position.n();
            for (int i = 0; i < n; i++) {
                py::dict d;
                for (int s : g.sides)
                    if (fr.shares[i][s] != 0.f) d[py::int_(s)] = fr.shares[i][s];
                shares.append(d);
            }
            return py::make_tuple(scores, shares);
        })
        .def_property_readonly("position", [](const gkt::Game& g) { return g.position; })
        .def_property_readonly("sides", [](const gkt::Game& g) { return g.sides; })
        .def_property_readonly("komi_schedule", [](const gkt::Game& g) { return g.komi_schedule; });

    m.def("legal_moves_for", &gkt::legal_moves_for);
    m.def("try_move", &gkt::try_move);
    m.def("apply_pass", &gkt::apply_pass);
    m.def("position_game_over", &gkt::position_game_over);
    m.def("legalize_occupancy", [](py::array_t<int8_t> occ, std::shared_ptr<gkt::Graph> g) {
        if (!g) throw std::invalid_argument("empty graph");
        std::vector<int8_t> v(occ.data(), occ.data() + occ.size());
        auto out = gkt::legalize_occupancy(v, *g);
        py::array_t<int8_t> a({static_cast<int>(out.size())});
        if (!out.empty())
            std::memcpy(a.mutable_data(), out.data(), out.size());
        return a;
    });
    m.def("extract_features", [](const gkt::Position& pos, int me) {
        int n = pos.n(), F = gkt::feature_dim(pos.num_players);
        py::array_t<float> a({n, F});
        gkt::extract_features_into(pos, me, a.mutable_data());
        return a;
    });

    m.def("search", [](const gkt::Position& pos, py::object net, int n_sim,
                       int batch_size, float dirichlet_frac, float c_puct,
                       uint32_t seed, bool return_q, py::object on_batch) {
        gkt::MCTSConfig cfg;
        cfg.n_simulations = n_sim;
        cfg.batch_size = batch_size;
        cfg.dirichlet_frac = dirichlet_frac;
        cfg.c_puct = c_puct;
        auto pred = wrap_python_net(net);
        gkt::MCTS mcts(pred, cfg, seed);
        gkt::MctsBatchFn ob;
        if (!on_batch.is_none()) {
            ob = [on_batch](int done, int total, int rem) {
                py::gil_scoped_acquire gil;
                on_batch(done, total, rem);
            };
        }
        gkt::SearchResult sr = mcts.search(pos, return_q, n_sim, ob);
        py::array_t<float> counts({static_cast<int>(sr.visit_counts.size())});
        if (!sr.visit_counts.empty())
            std::memcpy(counts.mutable_data(), sr.visit_counts.data(),
                        sr.visit_counts.size() * sizeof(float));
        if (return_q)
            return py::object(py::make_tuple(sr.legal_actions, counts, sr.q));
        return py::object(py::make_tuple(sr.legal_actions, counts, py::none()));
    }, py::arg("pos"), py::arg("net"), py::arg("n_simulations") = 50,
       py::arg("batch_size") = 32, py::arg("dirichlet_frac") = 0.25f,
       py::arg("c_puct") = 1.5f, py::arg("seed") = 0, py::arg("return_q") = false,
       py::arg("on_batch") = py::none());

    m.def("play_one_game", [](std::shared_ptr<gkt::Graph> graph, py::object net,
                              int num_players, int n_sim, float temperature,
                              int max_moves, int batch_size, float q_lambda,
                              bool randomize_sim, uint32_t seed,
                              py::object heartbeat, std::string rules, int win_length) {
        auto cfg = selfplay_cfg(n_sim, temperature, max_moves, batch_size,
                                q_lambda, randomize_sim, seed,
                                rules, win_length);
        auto pred = wrap_python_net(net);
        auto hb = wrap_heartbeat(heartbeat);
        std::vector<gkt::Sample> samples;
        {
            py::gil_scoped_release rel;
            samples = gkt::play_one_game(graph, pred, num_players, cfg, hb);
        }
        return samples_to_py(samples, graph->n, gkt::feature_dim(num_players));
    }, py::arg("graph"), py::arg("net"), py::arg("num_players") = 2,
       py::arg("n_simulations") = 50, py::arg("temperature") = 1.f,
       py::arg("max_moves") = 0, py::arg("batch_size") = 32,
       py::arg("q_lambda") = 0.5f,
       py::arg("randomize_sim") = true, py::arg("seed") = 0,
       py::arg("heartbeat") = py::none(),
       py::arg("rules") = "go", py::arg("win_length") = 5);

    py::class_<gkt::ScriptedNet>(m, "ScriptedNet")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("path"), py::arg("device") = "cpu")
        .def("set_graph", &gkt::ScriptedNet::set_graph);

    m.def("play_one_game_script", [](std::shared_ptr<gkt::Graph> graph,
                                     gkt::ScriptedNet& net, int num_players,
                                     int n_sim, float temperature, int max_moves,
                                     int batch_size, float q_lambda, bool randomize_sim,
                                     uint32_t seed, std::string rules, int win_length) {
        net.set_graph(*graph);
        auto cfg = selfplay_cfg(n_sim, temperature, max_moves, batch_size,
                                q_lambda, randomize_sim, seed,
                                rules, win_length);
        auto pred = net.as_fn();
        std::vector<gkt::Sample> samples;
        {
            py::gil_scoped_release rel;
            samples = gkt::play_one_game(graph, pred, num_players, cfg);
        }
        return samples_to_py(samples, graph->n, gkt::feature_dim(num_players));
    }, py::arg("graph"), py::arg("net"), py::arg("num_players") = 2,
       py::arg("n_simulations") = 50, py::arg("temperature") = 1.f,
       py::arg("max_moves") = 0, py::arg("batch_size") = 32,
       py::arg("q_lambda") = 0.5f,
       py::arg("randomize_sim") = true, py::arg("seed") = 0,
       py::arg("rules") = "go", py::arg("win_length") = 5);
}
