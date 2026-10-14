// The first C++ version of the PSSA annealing loop, for the speed comparison
// (Section 2.3.4 of the report). Like the Python loop, it recounts Eemb in full at every step.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Random numbers, as Python's random(), choice() and choices(weights=)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
struct PyRandom {
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> unif{0.0, 1.0};

    explicit PyRandom(long long seed) {
        if (seed < 0) {
            std::random_device rd;
            rng.seed(((uint64_t)rd() << 32) ^ (uint64_t)rd());
        } else {
            rng.seed((uint64_t)seed);
        }
    }

    double random() { return unif(rng); }

    // returns an index in [0, n)
    int randint(int n) {
        std::uniform_int_distribution<int> d(0, n - 1);
        return d(rng);
    }

    // weighted choice, as random.choices(population, weights=weights, k=1)[0]
    int weighted_choice(const std::vector<double>& weights) {
        double total = 0.0;
        for (double w : weights) total += w;
        if (total <= 0.0) return randint((int)weights.size());
        double r = random() * total;
        double cum = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) {
            cum += weights[i];
            if (r <= cum) return (int)i;
        }
        return (int)weights.size() - 1;
    }
};

static double now_epoch_seconds() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// State: I and H as contiguous integers
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
struct State {
    int n_hw = 0;                          // hardware nodes are 1..n_hw
    std::vector<std::vector<int>> H_adj;   // size n_hw+1
    std::vector<int> gp_lookup;            // size n_hw+1, -1 if absent
    std::vector<std::pair<int, int>> H_edges;

    int N = 0;                             // number of logical nodes
    std::vector<std::vector<int>> I_adj;   // size N (logical-index adjacency)
    std::vector<int> deg;                  // size N
    std::unordered_set<int64_t> I_edges_set; // key = a*N+b, a<b
    std::vector<std::pair<int, int>> edges_I; // logical-index edge list

    // scratch marks, reset by bumping a counter
    std::vector<int> markA; int ctrA = 0;   // used by chain_linked()
    std::vector<int> markB; int ctrB = 0;  // used by leaves() and as a scratch set
    std::vector<int> markC; int ctrC = 0;   // used by leaves() BFS-visited

    std::unordered_set<int64_t> covered;    // scratch for compute_eemb()
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Primitives
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static bool chain_linked(State& st, const std::vector<int>& a, const std::vector<int>& b) {
    const std::vector<int>* small;
    const std::vector<int>* big;
    if (a.size() <= b.size()) { small = &a; big = &b; }
    else { small = &b; big = &a; }

    st.ctrA++;
    int stamp = st.ctrA;
    for (int u : *big) st.markA[u] = stamp;
    for (int u : *small) {
        for (int w : st.H_adj[u]) {
            if (st.markA[w] == stamp) return true;
        }
    }
    return false;
}

static std::vector<int> remove_leaf(const std::vector<int>& chain, int u) {
    std::vector<int> out;
    out.reserve(chain.size() > 0 ? chain.size() - 1 : 0);
    for (int w : chain) if (w != u) out.push_back(w);
    return out;
}

// _attach_leaf(chain, u, v) == chain + [u], as in Python
static void attach_leaf(std::vector<int>& chain, int u) { chain.push_back(u); }

static std::vector<int> leaves(State& st, const std::vector<int>& chain) {
    if (chain.size() <= 1) return chain;

    st.ctrB++;
    int chainStamp = st.ctrB;
    for (int u : chain) st.markB[u] = chainStamp;

    std::vector<int> result;
    for (int u : chain) {
        int d = 0;
        for (int w : st.H_adj[u]) if (st.markB[w] == chainStamp) d++;
        if (d <= 1) result.push_back(u);
    }
    if (!result.empty()) return result;

    std::vector<int> out;
    for (int u : chain) {
        int start = -1;
        for (int w : chain) if (w != u) { start = w; break; }
        if (start == -1) continue;

        st.ctrC++;
        int visitStamp = st.ctrC;
        st.markC[start] = visitStamp;
        std::vector<int> stack = {start};
        int seen = 1;
        while (!stack.empty()) {
            int x = stack.back();
            stack.pop_back();
            for (int w : st.H_adj[x]) {
                if (w == u) continue;
                if (st.markB[w] != chainStamp) continue;
                if (st.markC[w] == visitStamp) continue;
                st.markC[w] = visitStamp;
                seen++;
                stack.push_back(w);
            }
        }
        if (seen == (int)chain.size() - 1) out.push_back(u);
    }
    return out;
}

static double shift_direction_prob(int i, int j, const std::vector<std::vector<int>>& phi,
                                    const std::vector<int>& deg, bool weighted) {
    if (!weighted) return 0.5;
    double dri = (double)phi[i].size() / (double)std::max(deg[i], 1);
    double drj = (double)phi[j].size() / (double)std::max(deg[j], 1);
    double denom = dri + drj;
    return denom == 0.0 ? 0.5 : dri / denom;
}

static int compute_eemb(State& st, const std::vector<std::vector<int>>& phi,
                         const std::vector<int>& inv_hw) {
    (void)phi;
    st.covered.clear();
    int count = 0;
    for (auto& e : st.H_edges) {
        int u = e.first, v = e.second;
        int i = inv_hw[u], j = inv_hw[v];
        if (i < 0 || j < 0 || i == j) continue;
        int a = std::min(i, j), b = std::max(i, j);
        int64_t key = (int64_t)a * st.N + b;
        if (!st.I_edges_set.count(key)) continue;
        if (st.covered.insert(key).second) count++;
    }
    return count;
}

static int local_swap_delta(State& st, const std::vector<std::vector<int>>& phi, int ie, int j) {
    const std::vector<int>& chain_ie = phi[ie];
    const std::vector<int>& chain_j = phi[j];
    int delta = 0;
    for (int m : st.I_adj[ie]) {
        if (m == j) continue;
        const std::vector<int>& chain_m = phi[m];
        bool before = chain_linked(st, chain_ie, chain_m);
        bool after = chain_linked(st, chain_j, chain_m);
        delta += (int)after - (int)before;
    }
    for (int m : st.I_adj[j]) {
        if (m == ie) continue;
        const std::vector<int>& chain_m = phi[m];
        bool before = chain_linked(st, chain_j, chain_m);
        bool after = chain_linked(st, chain_ie, chain_m);
        delta += (int)after - (int)before;
    }
    return delta;
}

static int local_shift_delta(State& st, const std::vector<std::vector<int>>& phi, int i, int j, int u) {
    std::vector<int> new_phi_i = remove_leaf(phi[i], u);
    std::vector<int> new_phi_j = phi[j];
    new_phi_j.push_back(u);

    int delta = 0;
    for (int m : st.I_adj[i]) {
        if (m == j) continue;
        const std::vector<int>& chain_m = phi[m];
        bool before = chain_linked(st, phi[i], chain_m);
        bool after = chain_linked(st, new_phi_i, chain_m);
        delta += (int)after - (int)before;
    }
    for (int m : st.I_adj[j]) {
        if (m == i) continue;
        const std::vector<int>& chain_m = phi[m];
        bool before = chain_linked(st, phi[j], chain_m);
        bool after = chain_linked(st, new_phi_j, chain_m);
        delta += (int)after - (int)before;
    }
    int a = std::min(i, j), b = std::max(i, j);
    if (st.I_edges_set.count((int64_t)a * st.N + b)) {
        bool before = chain_linked(st, phi[i], phi[j]);
        bool after = chain_linked(st, new_phi_i, new_phi_j);
        delta += (int)after - (int)before;
    }
    return delta;
}

static int gibbs_choose_swap_partner(State& st, const std::vector<std::vector<int>>& phi,
                                      int ie, const std::vector<int>& candidates,
                                      double T, PyRandom& rng) {
    if (candidates.size() == 1) return candidates[0];

    std::vector<int> deltas(candidates.size());
    for (size_t idx = 0; idx < candidates.size(); ++idx)
        deltas[idx] = local_swap_delta(st, phi, ie, candidates[idx]);

    if (T <= 0.0) {
        int best = 0;
        for (size_t idx = 1; idx < deltas.size(); ++idx)
            if (deltas[idx] > deltas[best]) best = (int)idx;
        return candidates[best];
    }

    std::vector<double> scaled(deltas.size());
    for (size_t idx = 0; idx < deltas.size(); ++idx) scaled[idx] = deltas[idx] / T;
    double m = *std::max_element(scaled.begin(), scaled.end());
    std::vector<double> weights(scaled.size());
    for (size_t idx = 0; idx < scaled.size(); ++idx) weights[idx] = std::exp(scaled[idx] - m);

    int pick = rng.weighted_choice(weights);
    return candidates[pick];
}

static std::pair<int, int> gibbs_choose_shift_target(
    State& st, const std::vector<std::vector<int>>& phi, int i, int u,
    const std::vector<std::pair<int, int>>& candidate_jv, double T, PyRandom& rng) {

    std::vector<int> unique_js;
    {
        std::unordered_set<int> seen;
        for (auto& jv : candidate_jv) {
            if (seen.insert(jv.first).second) unique_js.push_back(jv.first);
        }
    }

    int chosen_j;
    if (T <= 0.0 || unique_js.size() == 1) {
        std::vector<int> deltas(unique_js.size());
        for (size_t idx = 0; idx < unique_js.size(); ++idx)
            deltas[idx] = local_shift_delta(st, phi, i, unique_js[idx], u);
        int best = 0;
        for (size_t idx = 1; idx < deltas.size(); ++idx)
            if (deltas[idx] > deltas[best]) best = (int)idx;
        chosen_j = unique_js[best];
    } else {
        std::vector<int> deltas(unique_js.size());
        for (size_t idx = 0; idx < unique_js.size(); ++idx)
            deltas[idx] = local_shift_delta(st, phi, i, unique_js[idx], u);
        std::vector<double> scaled(deltas.size());
        for (size_t idx = 0; idx < deltas.size(); ++idx) scaled[idx] = deltas[idx] / T;
        double m = *std::max_element(scaled.begin(), scaled.end());
        std::vector<double> weights(scaled.size());
        for (size_t idx = 0; idx < scaled.size(); ++idx) weights[idx] = std::exp(scaled[idx] - m);
        int pick = rng.weighted_choice(weights);
        chosen_j = unique_js[pick];
    }

    std::vector<std::pair<int, int>> matches;
    for (auto& jv : candidate_jv) if (jv.first == chosen_j) matches.push_back(jv);
    return matches[rng.randint((int)matches.size())];
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The annealing loop (Algorithm 1)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
static void run_pssa_loop(
    State& st,
    std::vector<std::vector<int>>& phi,  // in/out, starts as phi0 and changes in place
    std::vector<int>& inv_hw,               // in/out
    int m_I,
    long long tmax,
    double T0, double alpha,
    double ps0, double ps_end,
    double pa0, double pa_end,
    bool weighted, bool gibbs,
    PyRandom& rng,
    double deadline,  // < 0 means no deadline
    bool verbose,
    long long log_every,
    std::vector<std::vector<int>>& phi_best_out,
    int& eemb_best_out
) {
    int cur_e = compute_eemb(st, phi, inv_hw);
    std::vector<std::vector<int>> phi_best = phi;
    int eemb_best = cur_e;

    if (eemb_best == m_I) {
        phi_best_out = phi_best;
        eemb_best_out = eemb_best;
        return;
    }

    int N = st.N;

    for (long long t = 0; t < tmax; ++t) {
        if (deadline >= 0.0 && (t % 1000 == 0) && now_epoch_seconds() > deadline) break;

        double T = T0 * std::pow(alpha, (double)t);
        double ps = ps0 + (ps_end - ps0) * (double)t / (double)tmax;
        double pa = pa0 + (pa_end - pa0) * (double)t / (double)tmax;

        if (verbose && (t % log_every == 0)) {
            std::printf("  t=%9lld (%3lld%%)  T=%6.3f  Eemb=%d/%d\n",
                        t, (100LL * t) / tmax, T, cur_e, m_I);
        }

        bool do_shift = rng.random() < ps;

        if (do_shift) {
            // shift
            std::vector<int> candidates;
            candidates.reserve(N);
            for (int idx = 0; idx < N; ++idx) if (phi[idx].size() > 1) candidates.push_back(idx);
            if (candidates.empty()) continue;
            int i = candidates[rng.randint((int)candidates.size())];

            std::vector<int> leaves_i = leaves(st, phi[i]);
            if (leaves_i.empty()) continue;
            int u = leaves_i[rng.randint((int)leaves_i.size())];
            bool allow_any = rng.random() < pa;

            std::vector<std::pair<int, int>> candidate_jv;
            for (int v : st.H_adj[u]) {
                int j = inv_hw[v];
                if (j == -1 || j == i) continue;
                if (allow_any || st.gp_lookup[u] == st.gp_lookup[v]) candidate_jv.push_back({j, v});
            }
            if (candidate_jv.empty()) continue;

            int j, v;
            if (gibbs) {
                auto pr = gibbs_choose_shift_target(st, phi, i, u, candidate_jv, T, rng);
                j = pr.first; v = pr.second;
            } else {
                auto& pick = candidate_jv[rng.randint((int)candidate_jv.size())];
                j = pick.first; v = pick.second;
            }

            double p_ij = shift_direction_prob(i, j, phi, st.deg, weighted);

            std::vector<int> saved_i = phi[i];
            std::vector<int> saved_j = phi[j];
            int hw_changed, old_owner, new_owner;

            if (rng.random() < p_ij) {
                phi[i] = remove_leaf(phi[i], u);
                attach_leaf(phi[j], u);
                hw_changed = u; old_owner = i; new_owner = j;
            } else {
                if (phi[j].size() < 2) continue;
                std::vector<int> leaves_j = leaves(st, phi[j]);
                if (std::find(leaves_j.begin(), leaves_j.end(), v) == leaves_j.end()) continue;
                phi[j] = remove_leaf(phi[j], v);
                attach_leaf(phi[i], v);
                hw_changed = v; old_owner = j; new_owner = i;
            }
            inv_hw[hw_changed] = new_owner;

            int prop_e = compute_eemb(st, phi, inv_hw);
            int delta = prop_e - cur_e;
            bool accept;
            if (T > 0.0) {
                accept = (delta < 0) ? (std::exp(std::min(delta / T, 0.0)) > rng.random()) : true;
            } else {
                accept = delta >= 0;
            }

            if (accept) {
                cur_e = prop_e;
                if (cur_e > eemb_best) { phi_best = phi; eemb_best = cur_e; }
                if (eemb_best == m_I) {
                    if (verbose) std::printf("  Embedding found at t=%lld\n", t);
                    break;
                }
            } else {
                phi[i] = saved_i;
                phi[j] = saved_j;
                inv_hw[hw_changed] = old_owner;
            }
        } else {
            // swap
            if (st.edges_I.empty()) continue;
            auto& e = st.edges_I[rng.randint((int)st.edges_I.size())];
            int ie = e.first, k = e.second;

            st.ctrB++;
            int seenStamp = st.ctrB;
            std::vector<int> adjacent_j;
            for (int hw_k : phi[k]) {
                for (int hw_nbr : st.H_adj[hw_k]) {
                    int j_cand = inv_hw[hw_nbr];
                    if (j_cand != -1 && j_cand != k && j_cand != ie) {
                        if (st.markB[j_cand] != seenStamp) {
                            st.markB[j_cand] = seenStamp;
                            adjacent_j.push_back(j_cand);
                        }
                    }
                }
            }
            if (adjacent_j.empty()) continue;

            int j;
            if (gibbs) {
                j = gibbs_choose_swap_partner(st, phi, ie, adjacent_j, T, rng);
            } else {
                j = adjacent_j[rng.randint((int)adjacent_j.size())];
            }

            std::vector<int> saved_ie = phi[ie];
            std::vector<int> saved_j = phi[j];

            phi[ie] = saved_j;
            phi[j] = saved_ie;
            for (int hw : phi[ie]) inv_hw[hw] = ie;
            for (int hw : phi[j]) inv_hw[hw] = j;

            int prop_e = compute_eemb(st, phi, inv_hw);
            int delta = prop_e - cur_e;
            bool accept;
            if (T > 0.0) {
                accept = (delta < 0) ? (std::exp(std::min(delta / T, 0.0)) > rng.random()) : true;
            } else {
                accept = delta >= 0;
            }

            if (accept) {
                cur_e = prop_e;
                if (cur_e > eemb_best) { phi_best = phi; eemb_best = cur_e; }
                if (eemb_best == m_I) {
                    if (verbose) std::printf("  Embedding found at t=%lld\n", t);
                    break;
                }
            } else {
                phi[ie] = saved_ie;
                phi[j] = saved_j;
                for (int hw : phi[ie]) inv_hw[hw] = ie;
                for (int hw : phi[j]) inv_hw[hw] = j;
            }
        }
    }

    phi_best_out = phi_best;
    eemb_best_out = eemb_best;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// pybind11 binding: run_pssa_core
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
static py::tuple run_pssa_core(
    std::vector<std::pair<int, int>> edges_I,        // logical, original ids
    std::vector<int> nodes_I,                        // sorted logical ids
    std::unordered_map<int, std::vector<int>> H_adj,  // native hw ids
    std::unordered_map<int, int> gp_lookup,  // hw id -> guiding pattern piece
    std::unordered_map<int, std::vector<int>> phi0,  // logical id -> super vertex
    long long tmax,
    double T0, double alpha,
    double ps0, double ps_end,
    double pa0, double pa_end,
    bool weighted, bool gibbs,
    long long seed,       // < 0 -> unseeded
    double deadline,       // < 0 -> None
    bool verbose,
    long long log_every
) {
    State st;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // hardware side
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int n_hw = 0;
    for (auto& kv : H_adj) {
        n_hw = std::max(n_hw, kv.first);
        for (int v : kv.second) n_hw = std::max(n_hw, v);
    }
    st.n_hw = n_hw;
    st.H_adj.assign(n_hw + 1, {});
    for (auto& kv : H_adj) st.H_adj[kv.first] = kv.second;

    st.gp_lookup.assign(n_hw + 1, -1);
    for (auto& kv : gp_lookup) st.gp_lookup[kv.first] = kv.second;

    for (int u = 1; u <= n_hw; ++u)
        for (int v : st.H_adj[u])
            if (v > u) st.H_edges.push_back({u, v});

    st.markA.assign(n_hw + 1, 0);
    st.markB.assign(n_hw + 1, 0);
    st.markC.assign(n_hw + 1, 0);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // logical side
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    int N = (int)nodes_I.size();
    st.N = N;
    std::unordered_map<int, int> id2idx;
    id2idx.reserve(N * 2);
    for (int idx = 0; idx < N; ++idx) id2idx[nodes_I[idx]] = idx;

    st.I_adj.assign(N, {});
    for (auto& e : edges_I) {
        int a = id2idx.at(e.first), b = id2idx.at(e.second);
        st.I_adj[a].push_back(b);
        st.I_adj[b].push_back(a);
        int lo = std::min(a, b), hi = std::max(a, b);
        st.I_edges_set.insert((int64_t)lo * N + hi);
        st.edges_I.push_back({a, b});
    }
    st.deg.assign(N, 0);
    for (int idx = 0; idx < N; ++idx) st.deg[idx] = (int)st.I_adj[idx].size();

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // initial phi / inv
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    std::vector<std::vector<int>> phi(N);
    for (auto& kv : phi0) {
        int idx = id2idx.at(kv.first);
        phi[idx] = kv.second;
    }
    std::vector<int> inv_hw(n_hw + 1, -1);
    for (int idx = 0; idx < N; ++idx)
        for (int u : phi[idx]) inv_hw[u] = idx;

    int m_I = (int)edges_I.size();

    PyRandom rng(seed);

    std::vector<std::vector<int>> phi_best;
    int eemb_best = 0;

    {
        py::gil_scoped_release release;
        run_pssa_loop(st, phi, inv_hw, m_I, tmax, T0, alpha, ps0, ps_end, pa0, pa_end,
                      weighted, gibbs, rng, deadline, verbose, log_every, phi_best, eemb_best);
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // convert phi_best back to {orig_id: [hw...]}
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    py::dict result;
    for (int idx = 0; idx < N; ++idx) {
        result[py::int_(nodes_I[idx])] = phi_best[idx];
    }
    return py::make_tuple(result, eemb_best);
}

PYBIND11_MODULE(pssa_core, m) {
    m.doc() = "C++ hot-loop core for PSSA (Path Super-vertex Simulated Annealing)";
    m.def("run_pssa_core", &run_pssa_core,
          py::arg("edges_I"), py::arg("nodes_I"), py::arg("H_adj"),
          py::arg("gp_lookup"), py::arg("phi0"),
          py::arg("tmax"), py::arg("T0"), py::arg("alpha"),
          py::arg("ps0"), py::arg("ps_end"), py::arg("pa0"), py::arg("pa_end"),
          py::arg("weighted"), py::arg("gibbs"), py::arg("seed"), py::arg("deadline"),
          py::arg("verbose"), py::arg("log_every"),
          "Run the PSSA Metropolis annealing loop in C++. Returns (phi, eemb).");
}
