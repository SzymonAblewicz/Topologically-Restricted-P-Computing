// The streaming end-to-end validation of a whole fallback embedding, for embeddings far too
// large to build (export_embedding.py writes its input). The lattice is swept row by row,
// with every hardware vertex generated from the export and checked against the row above:
//   (M1) every super vertex non-empty and connected, by a union-find over two rows at a time,
//   (M2) no hardware vertex in two super vertices, and none outside KG(L),
//   (M3) every edge of the graph realized by two adjacent hardware vertices.
// Memory O(L + input), time O(hardware vertices).
//
//   stream_validate <export dir>      (prints one JSON line; exit 0 iff valid)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;
typedef int32_t i32;
typedef int64_t i64;

static string dir;

template <class T> static vector<T> load(const char* name) {
    string p = dir + "/" + name;
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); i64 sz = ftell(f); fseek(f, 0, SEEK_SET);
    vector<T> v(sz / sizeof(T));
    if (sz && fread(v.data(), sizeof(T), v.size(), f) != v.size()) { fprintf(stderr, "short read %s\n", p.c_str()); exit(2); }
    fclose(f);
    return v;
}

static i64 meta_int(const string& js, const char* key) {
    string k = string("\"") + key + "\":";
    size_t p = js.find(k);
    if (p == string::npos) { fprintf(stderr, "meta: no %s\n", key); exit(2); }
    p += k.size();
    while (js[p] == ' ') p++;
    return atoll(js.c_str() + p);
}

// construction_fast.stripe_cols, exactly (one fold at each wall)
static inline i64 stripe_col(i64 j, i64 k, i64 S) {
    bool even = (j & 1) == 0;
    i64 x = even ? j + k - 1 : j - k + 1;
    if (even && x >= S) x = 2 * S - x - 1;
    if (!even && x < 0) x = -x - 1;
    return x;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: stream_validate <export dir>\n"); return 2; }
    dir = argv[1];
    auto t0 = chrono::steady_clock::now();
    string js;
    { FILE* f = fopen((dir + "/meta.json").c_str(), "rb"); if (!f) { fprintf(stderr, "no meta.json\n"); return 2; }
      char buf[65536]; size_t r; while ((r = fread(buf, 1, sizeof buf, f)) > 0) js.append(buf, r); fclose(f); }
    // meta.json is written with indent=1: "key": value
    for (size_t p; (p = js.find("\": ")) != string::npos;) js.erase(p + 2, 1);
    const i64 n = meta_int(js, "n"), L = meta_int(js, "L_plan"), rail = meta_int(js, "rail"),
              X0 = meta_int(js, "X0"), S = meta_int(js, "S");

    vector<i64> off = load<i64>("csr_off.bin");
    vector<i32> nbr = load<i32>("csr_nbr.bin");
    vector<i32> patch = load<i32>("patch.bin");     // (r, c, owner) sorted by r
    vector<i32> seg = load<i32>("seg.bin");         // (r0, c0, dr, dc, len, owner)
    vector<i32> stp = load<i32>("stripe.bin");      // (owner, j, reach)
    if ((i64)off.size() != n + 1) { fprintf(stderr, "csr size\n"); return 2; }
    const i64 np_ = patch.size() / 3, ns = seg.size() / 6, nt = stp.size() / 3;

    // straight runs, normalized to dr in {0, 1} (a run going up is the same cells reversed)
    struct Seg { i64 r0, c0, dr, dc, len; i32 own; };
    vector<Seg> segs; segs.reserve(ns);
    i64 rmax_seen = 0;
    for (i64 i = 0; i < ns; i++) {
        Seg s{seg[6*i], seg[6*i+1], seg[6*i+2], seg[6*i+3], seg[6*i+4], seg[6*i+5]};
        if (s.len <= 0) continue;
        if (s.dr < 0) { s.r0 += (s.len - 1) * s.dr; s.c0 += (s.len - 1) * s.dc; s.dr = -s.dr; s.dc = -s.dc; }
        if (s.dr == 0 && s.dc < 0) { s.c0 += (s.len - 1) * s.dc; s.dc = -s.dc; }
        segs.push_back(s);
        rmax_seen = max(rmax_seen, s.r0 + (s.dr ? s.len - 1 : 0));
    }
    sort(segs.begin(), segs.end(), [](const Seg& a, const Seg& b) { return a.r0 < b.r0; });

    // the chains of the routing structure in order, so the sweep reads the graph in order;
    // each dropped once past its reach
    vector<i64> order(nt);
    for (i64 i = 0; i < nt; i++) order[i] = i;
    sort(order.begin(), order.end(), [&](i64 a, i64 b) { return stp[3*a+1] < stp[3*b+1]; });
    i64 maxreach = -1;
    for (i64 i = 0; i < nt; i++) maxreach = max(maxreach, (i64)stp[3*i+2]);
    rmax_seen = max(rmax_seen, rail + 1 + maxreach);
    for (i64 i = 0; i < np_; i++) rmax_seen = max(rmax_seen, (i64)patch[3*i]);
    const i64 R = rmax_seen + 1;                 // rows to sweep
    const i64 W = max(L, (i64)1) + 2;            // owner arrays, with a guard column each side

    vector<i32> own[2] = {vector<i32>(W, -1), vector<i32>(W, -1)};
    vector<i32> touched[2];
    touched[0].reserve(W); touched[1].reserve(W);
    vector<i32> par(2 * W), alive(2 * W, -1), counted(2 * W, -1), rep(2 * W, -1), first(2 * W, -1);
    vector<i32> rootof;
    vector<i32> closed(n, 0);
    vector<uint8_t> realized(nbr.size(), 0);

    i64 cells = 0, conflicts = 0, self_dup = 0, oob = 0, adj_checks = 0;
    i64 rmin = INT64_MAX, rmx = -1, cmin = INT64_MAX, cmx = -1;
    char first_err[512] = "";

    auto find = [&](i32 x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; };

    i64 pi = 0, si = 0;
    vector<i64> active;                          // indices into segs alive at this row
    for (i64 r = 0; r < R; r++) {
        const int cur = r & 1, prv = cur ^ 1;
        vector<i32>& oc = own[cur];
        vector<i32>& tc = touched[cur];
        auto put = [&](i64 c, i32 o) {
            cells++;
            if (r < 0 || r >= L || c < 0 || c >= L) {
                if (!oob++ && !first_err[0]) snprintf(first_err, sizeof first_err, "cell (%lld,%lld) of chain %d outside KG(%lld)", (long long)r, (long long)c, o, (long long)L);
                return;
            }
            i32& slot = oc[c + 1];
            if (slot != -1) {
                if (slot == o) self_dup++; else conflicts++;
                if (!first_err[0]) snprintf(first_err, sizeof first_err, "cell (%lld,%lld) claimed by chains %d and %d", (long long)r, (long long)c, slot, o);
                return;
            }
            slot = o;
            tc.push_back((i32)c);
            rmin = min(rmin, r); rmx = max(rmx, r); cmin = min(cmin, c); cmx = max(cmx, c);
        };
        // patch cells of this row
        while (pi < np_ && patch[3*pi] == r) { put(patch[3*pi+1], patch[3*pi+2]); pi++; }
        // straight runs starting here join the active list
        while (si < (i64)segs.size() && segs[si].r0 == r) { active.push_back(si); si++; }
        size_t keep = 0;
        for (size_t q = 0; q < active.size(); q++) {
            const Seg& s = segs[active[q]];
            if (s.dr == 0) {
                for (i64 t = 0; t < s.len; t++) put(s.c0 + s.dc * t, s.own);
            } else {
                i64 t = r - s.r0;
                put(s.c0 + s.dc * t, s.own);
                if (t + 1 < s.len) active[keep++] = active[q];
            }
        }
        active.resize(keep);
        // the chains of the routing structure: index idx = r - rail - 1, one cell each within reach
        if (r >= rail + 1) {
            const i64 idx = r - rail - 1;
            size_t kp = 0;
            for (size_t q = 0; q < order.size(); q++) {
                const i64 t = order[q];
                if (stp[3*t+2] < idx) continue;
                put(X0 + stripe_col(stp[3*t+1], idx + 1, S), stp[3*t]);
                if (stp[3*t+2] > idx) order[kp++] = t;
            }
            order.resize(kp);
        }

        // union-find nodes for this row, then adjacency with the left cell and the row above
        const i32 base_c = cur * (i32)W, base_p = prv * (i32)W;
        for (i32 c : tc) par[base_c + c + 1] = base_c + c + 1;
        const vector<i32>& op = own[prv];
        for (i32 c : tc) {
            const i32 a = oc[c + 1], node = base_c + c + 1;
            const i32 nb_own[4] = {oc[c], op[c], op[c + 1], op[c + 2]};       // left, up-left, up, up-right
            const i32 nb_node[4] = {base_c + c, base_p + c, base_p + c + 1, base_p + c + 2};
            for (int k = 0; k < 4; k++) {
                const i32 b = nb_own[k];
                if (b < 0) continue;
                if (b == a) {
                    i32 x = find(node), y = find(nb_node[k]);
                    if (x != y) par[x] = y;
                } else {
                    adj_checks++;
                    for (i64 e = off[a]; e < off[a + 1]; e++)
                        if (nbr[e] == b) { realized[e] = 1; break; }
                }
            }
        }
        // components of the row above that did not continue into this row are closed
        const i32 stamp = (i32)(r & 0x7fffffff);
        for (i32 c : tc) alive[find(base_c + c + 1)] = stamp;
        for (i32 c : touched[prv]) {
            i32 x = find(base_p + c + 1);
            if (alive[x] != stamp && counted[x] != stamp) { counted[x] = stamp; closed[op[c + 1]]++; }
        }
        // re-root this row's components on cells of this row (each component's first cell),
        // so that no parent pointer leads into the row above, whose slots are reused next
        rootof.resize(tc.size());
        for (size_t q = 0; q < tc.size(); q++) rootof[q] = find(base_c + tc[q] + 1);
        for (size_t q = 0; q < tc.size(); q++) {
            const i32 x = rootof[q];
            if (rep[x] != stamp) { rep[x] = stamp; first[x] = base_c + tc[q] + 1; }
        }
        for (size_t q = 0; q < tc.size(); q++) par[base_c + tc[q] + 1] = first[rootof[q]];
        // clear the row above
        for (i32 c : touched[prv]) own[prv][c + 1] = -1;
        touched[prv].clear();
    }
    // after the last row, every remaining component closes
    {
        const int last = (R - 1) & 1;
        const i32 base = last * (i32)W, stamp = -2;
        for (i32 c : touched[last]) {
            i32 x = find(base + c + 1);
            if (counted[x] != stamp) { counted[x] = stamp; closed[own[last][c + 1]]++; }
        }
    }

    i64 empty = 0, split = 0, missing = 0, m_und = 0;
    for (i64 v = 0; v < n; v++) {
        if (closed[v] == 0) { empty++; if (!first_err[0]) snprintf(first_err, sizeof first_err, "chain %lld is empty", (long long)v); }
        else if (closed[v] > 1) { split++; if (!first_err[0]) snprintf(first_err, sizeof first_err, "chain %lld has %d components", (long long)v, closed[v]); }
    }
    for (i64 a = 0; a < n; a++)
        for (i64 e = off[a]; e < off[a + 1]; e++) {
            i64 b = nbr[e];
            if (b <= a) continue;
            m_und++;
            bool ok = realized[e];
            if (!ok) for (i64 f = off[b]; f < off[b + 1]; f++) if (nbr[f] == a) { ok = realized[f]; break; }
            if (!ok) { missing++; if (!first_err[0]) snprintf(first_err, sizeof first_err, "edge (%lld,%lld) not realized", (long long)a, (long long)b); }
        }
    const i64 L_final = (rmx < 0) ? 0 : max(rmx - rmin + 1, cmx - cmin + 1);
    const bool valid = !conflicts && !self_dup && !oob && !empty && !split && !missing;
    double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    printf("{\"valid\": %s, \"n\": %lld, \"m\": %lld, \"L_plan\": %lld, \"L_final\": %lld, \"cells\": %lld, "
           "\"rows\": %lld, \"conflicts\": %lld, \"self_dup\": %lld, \"out_of_bounds\": %lld, "
           "\"empty_chains\": %lld, \"split_chains\": %lld, \"edges_missing\": %lld, \"adjacent_pairs_checked\": %lld, "
           "\"seconds\": %.1f, \"first_error\": \"%s\"}\n",
           valid ? "true" : "false", (long long)n, (long long)m_und, (long long)L, (long long)L_final, (long long)cells,
           (long long)R, (long long)conflicts, (long long)self_dup, (long long)oob, (long long)empty, (long long)split,
           (long long)missing, (long long)adj_checks, secs, first_err);
    return valid ? 0 : 1;
}
