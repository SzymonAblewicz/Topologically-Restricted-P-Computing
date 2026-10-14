// The B*-tree floorplanner and the A* search of place_and_route.py in C++, with the
// same results bit for bit, as it draws its random numbers from a copy of CPython's own
// generator and computes every float in the same order.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CPython's random.Random: MT19937, seeded and drawn exactly as there
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct PyRandom {
    static const int N = 624, M = 397;
    uint32_t mt[N];
    int index = N + 1;

    explicit PyRandom(uint64_t seed) {
        // random.seed(int): the absolute value, cut into 32-bit words from the right
        uint32_t key[2] = {(uint32_t)(seed & 0xffffffffu), (uint32_t)(seed >> 32)};
        init_by_array(key, key[1] ? 2 : 1);
    }

    void init_genrand(uint32_t s) {
        mt[0] = s;
        for (index = 1; index < N; index++)
            mt[index] = 1812433253U * (mt[index - 1] ^ (mt[index - 1] >> 30)) + (uint32_t)index;
    }

    void init_by_array(const uint32_t* key, size_t len) {
        init_genrand(19650218U);
        size_t i = 1, j = 0;
        for (size_t k = (N > len ? N : len); k; k--) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525U)) + key[j] + (uint32_t)j;
            i++; j++;
            if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
            if (j >= len) j = 0;
        }
        for (size_t k = N - 1; k; k--) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U)) - (uint32_t)i;
            i++;
            if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
        }
        mt[0] = 0x80000000U;
    }

    uint32_t genrand_uint32() {
        static const uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
        uint32_t y;
        if (index >= N) {
            int kk;
            for (kk = 0; kk < N - M; kk++) {
                y = (mt[kk] & 0x80000000U) | (mt[kk + 1] & 0x7fffffffU);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            for (; kk < N - 1; kk++) {
                y = (mt[kk] & 0x80000000U) | (mt[kk + 1] & 0x7fffffffU);
                mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            y = (mt[N - 1] & 0x80000000U) | (mt[0] & 0x7fffffffU);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1U];
            index = 0;
        }
        y = mt[index++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }

    // random()
    double random() {
        uint32_t a = genrand_uint32() >> 5, b = genrand_uint32() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    // getrandbits(k), k <= 32
    uint32_t getrandbits(int k) { return k == 0 ? 0 : genrand_uint32() >> (32 - k); }

    // _randbelow(n), which randrange(n) and choice() use
    int randbelow(int n) {
        int k = 0;
        for (uint32_t m = (uint32_t)n; m; m >>= 1) k++;          // n.bit_length()
        uint32_t r = getrandbits(k);
        while (r >= (uint32_t)n) r = getrandbits(k);
        return (int)r;
    }

    template <class T>
    const T& choice(const std::vector<T>& seq) { return seq[randbelow((int)seq.size())]; }

    template <class T>
    void shuffle(std::vector<T>& x) {
        for (int i = (int)x.size() - 1; i >= 1; i--) std::swap(x[i], x[randbelow(i + 1)]);
    }

    // sample(population, 2)
    std::pair<int, int> sample2(const std::vector<int>& population) {
        int n = (int)population.size(), result[2];
        if (n <= 21) {                                         // setsize for k = 2
            std::vector<int> pool(population);
            for (int i = 0; i < 2; i++) {
                int j = randbelow(n - i);
                result[i] = pool[j];
                pool[j] = pool[n - i - 1];
            }
        } else {
            int selected[2], n_sel = 0;
            for (int i = 0; i < 2; i++) {
                int j = randbelow(n);
                while (std::find(selected, selected + n_sel, j) != selected + n_sel) j = randbelow(n);
                selected[n_sel++] = j;
                result[i] = population[j];
            }
        }
        return {result[0], result[1]};
    }
};

// CPython's sum() of a list of floats: Neumaier's compensated sum, from the int 0
static double py_sum(const std::vector<double>& xs) {
    if (xs.empty()) return 0.0;
    double f = 0.0 + xs[0], c = 0.0;
    for (size_t i = 1; i < xs.size(); i++) {
        double x = xs[i], t = f + x;
        if (std::fabs(f) >= std::fabs(x)) c += (f - t) + x;
        else c += (x - t) + f;
        f = t;
    }
    if (c != 0.0 && std::isfinite(c)) f += c;
    return f;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// B*-tree: representation, decode, moves (clusters are 0..k-1)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct Tree {
    std::vector<int> parent, left, right;   // -1 = None
    int root = -1;
    std::vector<int> nodes;                 // the Python dict's keys, in its order
};

static std::pair<int, int> transform_rc(int r, int c, int L, int t) {
    switch (t) {
        case 0: return {r, c};
        case 1: return {c, L - 1 - r};
        case 2: return {L - 1 - r, L - 1 - c};
        case 3: return {L - 1 - c, r};
        case 4: return {r, L - 1 - c};
        case 5: return {L - 1 - r, c};
        case 6: return {c, r};
        default: return {L - 1 - c, L - 1 - r};
    }
}

static Tree build_caterpillar(const std::vector<std::pair<int, int>>& sizes, int k) {
    Tree t;
    t.parent.assign(k, -1); t.left.assign(k, -1); t.right.assign(k, -1);
    int prev = -1;
    for (auto& s : sizes) {
        int cid = s.first;
        t.nodes.push_back(cid);
        if (prev < 0) t.root = cid;
        else { t.left[prev] = cid; t.parent[cid] = prev; }
        prev = cid;
    }
    return t;
}

static Tree build_random_tree(const std::vector<std::pair<int, int>>& sizes, int k, PyRandom& rng) {
    std::vector<std::pair<int, int>> order(sizes);
    rng.shuffle(order);
    Tree t;
    t.parent.assign(k, -1); t.left.assign(k, -1); t.right.assign(k, -1);
    int cid0 = order[0].first;
    t.nodes.push_back(cid0);
    t.root = cid0;
    std::vector<int> placed = {cid0};
    for (size_t i = 1; i < order.size(); i++) {
        int cid = order[i].first;
        t.nodes.push_back(cid);
        std::vector<int> candidates;
        for (int c : placed)
            if (t.left[c] < 0 || t.right[c] < 0) candidates.push_back(c);
        int parent = rng.choice(candidates);
        if (t.left[parent] < 0 && (t.right[parent] >= 0 || rng.random() < 0.5)) t.left[parent] = cid;
        else t.right[parent] = cid;
        t.parent[cid] = parent;
        placed.push_back(cid);
    }
    return t;
}

static void set_parent_child(Tree& t, int parent, int old_child, int new_child) {
    if (parent < 0) return;
    if (t.left[parent] == old_child) t.left[parent] = new_child;
    else if (t.right[parent] == old_child) t.right[parent] = new_child;
}

static void subtree_cids(const Tree& t, int cid, std::vector<char>& out) {
    if (cid < 0) return;
    out[cid] = 1;
    if (t.left[cid] >= 0) subtree_cids(t, t.left[cid], out);
    if (t.right[cid] >= 0) subtree_cids(t, t.right[cid], out);
}

static void swap_positions(Tree& t, int a, int b) {
    int pa = t.parent[a], la = t.left[a], ra = t.right[a];
    int pb = t.parent[b], lb = t.left[b], rb = t.right[b];
    set_parent_child(t, pa, a, b);
    set_parent_child(t, pb, b, a);
    if (la >= 0) t.parent[la] = b;
    if (ra >= 0) t.parent[ra] = b;
    if (lb >= 0) t.parent[lb] = a;
    if (rb >= 0) t.parent[rb] = a;
    if (t.root == a) t.root = b;
    else if (t.root == b) t.root = a;
    t.parent[a] = pb; t.left[a] = lb; t.right[a] = rb;
    t.parent[b] = pa; t.left[b] = la; t.right[b] = ra;
}

static void attach(Tree& t, int cid, int parent, bool left_side) {
    if (parent < 0) { t.root = cid; return; }
    (left_side ? t.left : t.right)[parent] = cid;
    t.parent[cid] = parent;
}

static void delete_node(Tree& t, int cid, PyRandom& rng) {
    int parent = t.parent[cid], left = t.left[cid], right = t.right[cid];
    int promoted = -1, orphan = -1;
    if (left >= 0 && right >= 0) { promoted = left; orphan = right; }
    else if (left >= 0) promoted = left;
    else if (right >= 0) promoted = right;

    set_parent_child(t, parent, cid, promoted);
    if (promoted >= 0) t.parent[promoted] = parent;
    if (parent < 0) t.root = promoted;
    t.nodes.erase(std::find(t.nodes.begin(), t.nodes.end(), cid));
    t.parent[cid] = t.left[cid] = t.right[cid] = -1;

    if (orphan >= 0) {
        t.parent[orphan] = -1;
        std::vector<char> excluded(t.parent.size(), 0);
        subtree_cids(t, orphan, excluded);
        std::vector<std::pair<int, bool>> candidates;       // (parent, left side?)
        for (int n : t.nodes) {
            if (excluded[n]) continue;
            if (t.left[n] < 0) candidates.push_back({n, true});
            if (t.right[n] < 0) candidates.push_back({n, false});
        }
        auto pick = rng.choice(candidates);
        attach(t, orphan, pick.first, pick.second);
    }
}

static void insert_node(Tree& t, int cid, int parent, bool left_side) {
    t.nodes.push_back(cid);
    attach(t, cid, parent, left_side);
}

static void rotate_node(Tree& t, int cid) {
    int p = t.parent[cid];
    if (p < 0) return;
    int g = t.parent[p];
    if (t.left[p] == cid) {
        int b = t.right[cid];
        t.left[p] = b;
        if (b >= 0) t.parent[b] = p;
        t.right[cid] = p;
    } else {
        int b = t.left[cid];
        t.right[p] = b;
        if (b >= 0) t.parent[b] = p;
        t.left[cid] = p;
    }
    t.parent[p] = cid;
    t.parent[cid] = g;
    if (g < 0) t.root = cid;
    else if (t.left[g] == p) t.left[g] = cid;
    else t.right[g] = cid;
}

static void random_move(Tree& t, std::vector<int>& orient, const std::vector<int>& cids, PyRandom& rng) {
    double kind = rng.random();
    if (kind < 0.5 || t.nodes.size() < 3) {
        int c = rng.choice(cids);
        orient[c] = rng.randbelow(8);
        return;
    }
    if (kind < 0.70) {
        for (int attempt = 0; attempt < 5; attempt++) {
            auto ab = rng.sample2(cids);
            int a = ab.first, b = ab.second;
            if (t.parent[a] == b || t.parent[b] == a) continue;
            swap_positions(t, a, b);
            return;
        }
        return;
    }
    std::vector<int> non_root;
    for (int c : cids)
        if (c != t.root) non_root.push_back(c);
    if (non_root.empty()) return;
    if (kind < 0.85) {
        int c = rng.choice(non_root);
        delete_node(t, c, rng);
        std::vector<std::pair<int, bool>> targets;
        for (int n : t.nodes) {
            if (t.left[n] < 0) targets.push_back({n, true});
            if (t.right[n] < 0) targets.push_back({n, false});
        }
        auto pick = rng.choice(targets);
        insert_node(t, c, pick.first, pick.second);
        return;
    }
    rotate_node(t, rng.choice(non_root));
}

// decode(): preorder DFS + contour compaction; each block is an (L+gap)^2 cell
struct Layout {
    std::vector<int> row0, col0;
    int W = 0, H = 0;
};

struct Decoder {
    const Tree& t;
    const std::vector<int>& Ls;
    int gap;
    Layout out;
    std::vector<std::array<int, 3>> contour, nc, merged;    // (col_start, col_end, row_bottom)

    Decoder(const Tree& t_, const std::vector<int>& Ls_, int gap_) : t(t_), Ls(Ls_), gap(gap_) {
        out.row0.assign(Ls.size(), 0);
        out.col0.assign(Ls.size(), 0);
    }

    int query(int a, int b) const {
        int best = 0;
        for (auto& s : contour)
            if (s[0] < b && s[1] > a && s[2] > best) best = s[2];
        return best;
    }

    void update(int a, int b, int new_h) {
        nc.clear();
        for (auto& s : contour) {
            if (s[1] <= a || s[0] >= b) { nc.push_back(s); continue; }
            if (s[0] < a) nc.push_back({s[0], a, s[2]});
            if (s[1] > b) nc.push_back({b, s[1], s[2]});
        }
        nc.push_back({a, b, new_h});
        std::sort(nc.begin(), nc.end());
        merged.clear();
        for (auto& seg : nc) {
            if (!merged.empty() && merged.back()[1] == seg[0] && merged.back()[2] == seg[2])
                merged.back()[1] = seg[1];
            else
                merged.push_back(seg);
        }
        contour.swap(merged);
    }

    void visit(int cid, int col0) {
        int Lp = Ls[cid] + gap;
        int row0 = query(col0, col0 + Lp);
        out.row0[cid] = row0;
        out.col0[cid] = col0;
        update(col0, col0 + Lp, row0 + Lp);
        out.W = std::max(out.W, col0 + Lp);
        out.H = std::max(out.H, row0 + Lp);
        if (t.left[cid] >= 0) visit(t.left[cid], col0 + Lp);
        if (t.right[cid] >= 0) visit(t.right[cid], col0);
    }
};

static Layout decode(const Tree& t, const std::vector<int>& Ls, int gap) {
    Decoder d(t, Ls, gap);
    if (t.root >= 0) d.visit(t.root, 0);
    return d.out;
}

// One cut edge's term of the wire cost: its two border nodes' perimeter points
struct Wire {
    int ca, cb;
    std::vector<std::pair<int, int>> pts_u, pts_v;
};

static std::pair<int, int> border_pos(const std::vector<std::pair<int, int>>& pts, int L, int t,
                                      int row0, int col0) {
    std::pair<int, int> best;
    for (size_t i = 0; i < pts.size(); i++) {
        auto rc = transform_rc(pts[i].first, pts[i].second, L, t);
        std::pair<int, int> w = {row0 + rc.first, col0 + rc.second};
        if (i == 0 || w < best) best = w;
    }
    return best;
}

static double wire_cost(const std::vector<Wire>& wires, const std::vector<int>& Ls,
                        const Layout& pl, const std::vector<int>& orient) {
    double total = 0.0;
    for (auto& w : wires) {
        auto a = border_pos(w.pts_u, Ls[w.ca], orient[w.ca], pl.row0[w.ca], pl.col0[w.ca]);
        auto b = border_pos(w.pts_v, Ls[w.cb], orient[w.cb], pl.row0[w.cb], pl.col0[w.cb]);
        int cheb = std::max(std::abs(a.first - b.first), std::abs(a.second - b.second));
        total += std::max(0.0, cheb - 1.0);
    }
    return total;
}

struct State {
    double area, wire;
    Layout pl;
    int L;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// anneal(): floorplan_anneal()
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

py::tuple anneal(const std::vector<int>& Ls,
                 const std::vector<std::tuple<int, int, std::vector<std::pair<int, int>>,
                                              std::vector<std::pair<int, int>>>>& wire_terms,
                 int gap, int n_restarts, int n_moves, uint64_t seed, double lam) {
    const int k = (int)Ls.size();
    std::vector<int> cids(k);
    for (int i = 0; i < k; i++) cids[i] = i;
    std::vector<std::pair<int, int>> sizes, sizes_desc;
    for (int i = 0; i < k; i++) sizes.push_back({i, Ls[i]});
    sizes_desc = sizes;
    std::stable_sort(sizes_desc.begin(), sizes_desc.end(),
                     [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return -a.second < -b.second; });

    std::vector<Wire> wires;
    for (auto& w : wire_terms) wires.push_back({std::get<0>(w), std::get<1>(w), std::get<2>(w), std::get<3>(w)});

    auto area_and_wire = [&](const Tree& t, const std::vector<int>& orient) {
        State s;
        s.pl = decode(t, Ls, gap);
        s.L = std::max(std::max(s.pl.W, s.pl.H), 1);
        s.area = (double)s.L * (double)s.L;                  // float(L) ** 2, exact
        s.wire = wire_cost(wires, Ls, s.pl, orient);
        return s;
    };

    double best_cost = std::numeric_limits<double>::infinity();
    Layout best_pl;
    std::vector<int> best_orient;
    int best_L = 1;

    PyRandom rng_master(seed);
    for (int restart = 0; restart < n_restarts; restart++) {
        PyRandom rng((uint64_t)rng_master.randbelow(1 << 30));
        Tree tree = restart == 0 ? build_caterpillar(sizes_desc, k) : build_random_tree(sizes, k, rng);
        std::vector<int> orient(k, 0);
        State cur = area_and_wire(tree, orient);

        // burn-in: typical |delta| of area and of wire, to balance them
        int n_probe = std::min(60, std::max(10, n_moves / 10));
        Tree probe_tree = tree;
        std::vector<int> probe_orient = orient;
        double probe_area = cur.area, probe_wire = cur.wire;
        std::vector<double> area_deltas, wire_deltas;
        for (int i = 0; i < n_probe; i++) {
            random_move(probe_tree, probe_orient, cids, rng);
            State s = area_and_wire(probe_tree, probe_orient);
            area_deltas.push_back(std::fabs(s.area - probe_area));
            wire_deltas.push_back(std::fabs(s.wire - probe_wire));
            probe_area = s.area;
            probe_wire = s.wire;
        }
        double mean_area_delta = py_sum(area_deltas) / (double)area_deltas.size();
        double mean_wire_delta = py_sum(wire_deltas) / (double)wire_deltas.size();
        double lam_eff = mean_wire_delta > 1e-9 ? lam * (mean_area_delta / mean_wire_delta) : lam;

        double cur_cost = cur.area + lam_eff * cur.wire;
        if (cur_cost < best_cost) {
            best_cost = cur_cost; best_pl = cur.pl; best_orient = orient; best_L = cur.L;
        }

        std::vector<double> combined;
        for (int i = 0; i < n_probe; i++) combined.push_back(area_deltas[i] + lam_eff * wire_deltas[i]);
        double mean_delta = py_sum(combined) / (double)combined.size();
        double T0 = 1.0 > mean_delta * 2.0 ? 1.0 : mean_delta * 2.0;
        double T_end = 1e-3 > T0 / 200.0 ? 1e-3 : T0 / 200.0;

        for (int i = 0; i < n_moves; i++) {
            double frac = (double)i / (double)std::max(1, n_moves - 1);
            double T = T0 * std::pow(T_end / T0, frac);
            Tree backup_tree = tree;
            std::vector<int> backup_orient = orient;
            random_move(tree, orient, cids, rng);
            State s = area_and_wire(tree, orient);
            double new_cost = s.area + lam_eff * s.wire;
            double delta = new_cost - cur_cost;
            bool accept = delta <= 0 || rng.random() < std::exp(-delta / T);
            if (accept) {
                cur_cost = new_cost;
                if (new_cost < best_cost) {
                    best_cost = new_cost; best_pl = s.pl; best_orient = orient; best_L = s.L;
                }
            } else {
                tree = std::move(backup_tree);
                orient = std::move(backup_orient);
            }
        }
    }
    return py::make_tuple(best_pl.row0, best_pl.col0, best_orient, best_L);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// astar(): _astar_route() on KG(L), hardware vertices 1..L*L
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Scratch arrays reused from call to call; a stamp marks what this call set.
struct Scratch {
    std::vector<uint32_t> g_stamp, closed_stamp, source_stamp, target_stamp;
    std::vector<double> g;
    std::vector<int> parent;
    uint32_t stamp = 0;

    void begin(size_t n) {
        if (g.size() < n) {
            g_stamp.assign(n, 0); closed_stamp.assign(n, 0);
            source_stamp.assign(n, 0); target_stamp.assign(n, 0);
            g.assign(n, 0.0); parent.assign(n, 0);
            stamp = 0;
        }
        stamp++;
    }
};
static Scratch S;

py::object astar(int L, const std::vector<int>& sources, const std::vector<int>& targets,
                 py::array_t<uint8_t, py::array::c_style> free_,
                 py::array_t<int64_t, py::array::c_style> reserved_,
                 int64_t u, int64_t v, py::object hist_, py::object occ_, double pres_fac) {
    const uint8_t* free = free_.data();
    const int64_t* reserved = reserved_.data();
    const bool priced = !hist_.is_none();
    py::array_t<double, py::array::c_style> hist_arr;
    py::array_t<int32_t, py::array::c_style> occ_arr;
    const double* hist = nullptr;
    const int32_t* occ = nullptr;
    if (priced) {
        hist_arr = hist_.cast<py::array_t<double, py::array::c_style>>();
        occ_arr = occ_.cast<py::array_t<int32_t, py::array::c_style>>();
        hist = hist_arr.data();
        occ = occ_arr.data();
    }

    S.begin((size_t)L * L + 1);
    const uint32_t st = S.stamp;
    for (int w : targets) S.target_stamp[w] = st;

    std::vector<std::pair<int, int>> tgt_rc;
    for (int w : targets) tgt_rc.push_back({(w - 1) / L, (w - 1) % L});
    auto h = [&](int w) {
        int r = (w - 1) / L, c = (w - 1) % L, best = std::numeric_limits<int>::max();
        for (auto& t : tgt_rc) best = std::min(best, std::max(std::abs(r - t.first), std::abs(c - t.second)));
        return std::max(0, best - 1);
    };

    typedef std::tuple<double, long long, int> Entry;  // (f, tie, vertex), as heapq's tuples
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
    long long tie = 0;
    for (int w : sources) {
        S.source_stamp[w] = st;
        S.g_stamp[w] = st;
        S.g[w] = 0.0;
        heap.push(Entry((double)h(w), tie++, w));
    }

    while (!heap.empty()) {
        int a = std::get<2>(heap.top());
        heap.pop();
        if (S.closed_stamp[a] == st) continue;
        S.closed_stamp[a] = st;
        int ra = (a - 1) / L, ca = (a - 1) % L;
        for (int dr = -1; dr <= 1; dr++) {
            int r = ra + dr;
            if (r < 0 || r >= L) continue;
            for (int dc = -1; dc <= 1; dc++) {
                int c = ca + dc;
                if ((dr == 0 && dc == 0) || c < 0 || c >= L) continue;
                int b = r * L + c + 1;
                if (S.target_stamp[b] == st) {
                    py::list path;
                    std::vector<int> rev;
                    for (int cur = a; S.source_stamp[cur] != st; cur = S.parent[cur]) rev.push_back(cur);
                    for (auto it = rev.rbegin(); it != rev.rend(); ++it) path.append(*it);
                    return std::move(path);
                }
                if (S.closed_stamp[b] == st || !free[b]) continue;
                int64_t owner = reserved[b];
                if (owner >= 0 && owner != u && owner != v) continue;
                double step = priced ? (1.0 + hist[b]) * (1.0 + pres_fac * (double)occ[b]) : 1.0;
                double ng = S.g[a] + step;
                if (S.g_stamp[b] != st || ng < S.g[b]) {
                    S.g_stamp[b] = st;
                    S.g[b] = ng;
                    S.parent[b] = a;
                    heap.push(Entry(ng + h(b), tie++, b));
                }
            }
        }
    }
    return py::none();
}

// Test hooks: CPython-equivalent random draws and sum, checked against Python
static std::vector<double> _test_random(uint64_t seed, int n) {
    PyRandom r(seed);
    std::vector<double> out;
    for (int i = 0; i < n; i++) {
        out.push_back(r.random());
        out.push_back((double)r.randbelow(1 << 30));
        out.push_back((double)r.randbelow(8));
        out.push_back((double)r.randbelow(i % 50 + 1));
    }
    return out;
}

PYBIND11_MODULE(floorplan_core, m) {
    m.doc() = "C++ core of pssa_floorplan.py (ImprovedDnC_2): the B*-tree annealer and A*";
    m.def("anneal", &anneal, "floorplan_anneal(): returns (row0s, col0s, orientations, L)");
    m.def("astar", &astar, "_astar_route(): a path of free qubits, or None");
    m.def("_test_random", &_test_random);
    m.def("_test_sum", &py_sum);
    m.def("_test_pow_exp", [](double x, double y) { return std::make_pair(std::pow(x, y), std::exp(x)); });
    m.def("_test_sample_shuffle", [](uint64_t seed, int n) {
        PyRandom r(seed);
        std::vector<int> pop(n), out;
        for (int i = 0; i < n; i++) pop[i] = i;
        auto s = r.sample2(pop);
        out.push_back(s.first); out.push_back(s.second);
        r.shuffle(pop);
        out.insert(out.end(), pop.begin(), pop.end());
        out.push_back(r.choice(pop));
        return out;
    });
}
