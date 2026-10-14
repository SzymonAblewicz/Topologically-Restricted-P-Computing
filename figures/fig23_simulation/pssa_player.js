/*
 * The live Figure 23 of the report: one recorded run of the fallback method, replayed.
 * Graph partitioning, PSSA wBC on every cluster at once, then the routing. The player
 * computes nothing about the algorithms; every state it draws was recorded.
 *
 *     const player = new PSSAPlayer(document.querySelector('#pssa'), data);
 *     player.start();
 *
 * Space plays and pauses, the arrows step, home and end jump, a click on a patch zooms
 * into it and escape leaves it.
 */
(function (global) {
  'use strict';

  // Every colour in this file comes from `pssa_palette.js`, which must be loaded first.
  if (!global.PSSA_PALETTE) {
    throw new Error('pssa_palette.js must be loaded before pssa_player.js');
  }

  function hexToRgb(h) {
    var o = h.charAt(0) === '#' ? 1 : 0;
    return [parseInt(h.substr(o, 2), 16),
            parseInt(h.substr(o + 2, 2), 16),
            parseInt(h.substr(o + 4, 2), 16)];
  }

  // Relative luminance (WCAG), to decide whether a colour needs a dark casing on paper.
  function luma(c) {
    return (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;
  }

  function rgbStr(c, a) {
    return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' +
           (a === undefined ? 1 : a) + ')';
  }
  function mix(a, b, t) {
    return [a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t].map(Math.round);
  }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function clamp01(t) { return t < 0 ? 0 : (t > 1 ? 1 : t); }
  function easeInOut(t) {
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  }
  function fmt(n) { return n.toLocaleString('en-US'); }
  // One decimal at most, and none when it rounds to a whole number: the run's last step,
  // 399,999, reads 400k rather than 400.0k.
  function kfmt(v) {
    function one(x) { return x.toFixed(1).replace(/\.0$/, ''); }
    if (v >= 1e6) return one(v / 1e6) + 'M';
    if (v >= 1e3) return one(v / 1e3) + 'k';
    return String(v);
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // surface palette
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  var PAL;
  var INK;        // headline marks, playheads, hardware edges
  var MUTED;      // axis labels, tick text
  var FAINT;      // gridlines, captions, empty-state marks
  var ALERT;      // unembedded edges
  var PAPER;
  var PAPER_STR;  // PAPER at full alpha, as a string
  var PLOT_T;     // T(t)
  var PLOT_E;     // E_emb(t)

  // Border vertices carry a mark of their own rather than one of the ink tiers.
  var BORDER = global.PSSA_BORDER_MARK;

  function usePalette(p) {
    PAL = p;
    INK = p.ink;
    MUTED = p.muted;
    FAINT = p.faint;
    ALERT = p.alert;
    PAPER = p.paper;
    PAPER_STR = rgbStr(PAPER, 1);
    PLOT_T = p.plotT;
    PLOT_E = p.plotE;
  }
  usePalette(global.PSSA_PALETTE);

  // One stroke weight for both link types, as a fraction of the cell pitch.
  var LINK_W = 0.09;

  // The cell pitch below which marks stop shrinking.
  var MARK_MIN_CELL = 6.0;
  // How far off a tile a click may land and still pick it, in screen px.
  var PICK_SLOP = 8;

  // An unembedded edge is a thin hairline at reduced opacity, not a heavy dashed rule.
  var MISS_W = 0.032;
  var MISS_A = 0.55;

  // Where in a `railIn` morph the perimeter marks start arriving.
  var RAIL_IN_FROM = 0.35;

  // How a cut edge is inked on each side of the handover.
  var CUT_G  = 0.34;
  var ARCS_A = 0.14;


  function ink(a)   { return rgbStr(INK, a); }
  function muted(a) { return rgbStr(MUTED, a); }
  function faint(a) { return rgbStr(FAINT, a); }
  function alert_(a) { return rgbStr(ALERT, a); }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // the player
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  function PSSAPlayer(container, data, opts) {
    opts = opts || {};
    this.container = container;
    this.data = data;
    if (data.meta.schema !== 5) {
      throw new Error('pssa_player.js expects trajectory schema 5; got ' +
                      data.meta.schema + '. Re-run record_trajectory.py.');
    }
    this.meta = data.meta;
    this.k = data.meta.k;
    this.nV = data.graph.node_id.length;

    // Playback is uniform in t, so the anneal's duration is a single number and everything
    // else (steps/s, ms per step) falls out of it.
    this.annealSeconds = opts.annealSeconds || 26;
    this.loop = opts.loop !== false;
    this.speed = opts.speed || 1;
    this.xmode = opts.xmode === 'log' ? 'log' : 'lin';
    this.ymode = opts.ymode === 'log' ? 'log' : 'lin';
    this.onPhase = opts.onPhase || null;
    // fitHeight: the CSS px the page can give the player, so the whole figure fits on one
    // screen.
    this.fitHeight = opts.fitHeight || null;
    // The base layer (see `_drawBase`).
    this.baseLayer = opts.baseLayer !== false;
    this.base = null;
    this.meshGen = 0;             // counts King's mesh rebuilds
    this.palGen = 0;              // counts recolourings

    // transport state: one clock, everything else derived from it
    this.clock = 0;
    this.playing = false;
    this.running = false;
    this.lastNow = null;
    this.hudFrame = null;
    this.zoomCid = null;          // which patch the camera is inside, if any

    this.reduced = !!(global.matchMedia &&
      global.matchMedia('(prefers-reduced-motion: reduce)').matches);

    this._buildModel();
    this._buildGeometry();
    this._buildStates();
    this._buildStage3();
    this._buildTimeline();
    this._buildDom();
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // model: the graph, the patches, the colours
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PSSAPlayer.prototype._buildModel = function () {
    var d = this.data, g = d.graph, i;

    this.pos0 = g.pos0;
    this.pos1 = g.pos1;
    this.label = g.label;
    this.isBorderV = g.border;
    this.graphEdges = g.edges;
    this.nodeId = g.node_id;

    // Cluster colours, and the per-vertex shades drawn from them, come from the active
    // palette rather than from the recording, because which colour a cluster is drawn in is
    // presentation and the recording is data.
    this.shade = new Array(this.nV);
    this._recolor();

    // Adjacency over the input graph, for the border test the player runs for itself rather
    // than trusting the recorded flags.
    this.adj = [];
    for (i = 0; i < this.nV; i++) this.adj.push([]);
    for (i = 0; i < this.graphEdges.length; i++) {
      var e = this.graphEdges[i];
      this.adj[e[0]].push(e[1]);
      this.adj[e[1]].push(e[0]);
    }

    this.patches = d.clusters.map(function (c) {
      return {
        cid: c.cid, L: c.L, nHw: c.L * c.L, origin: c.origin,
        nodes: c.nodes, edges: c.edges, border: c.border,
        anchors: c.anchors, perimeterList: c.perimeter,
        mI: c.m_I, eStart: c.e_start, eFinal: c.e_final, success: c.success,
        tEnd: c.t_end, tSplit: c.t_split, nMoves: c.n_moves,
        nSlow: c.n_slow_frames, nBlocked: c.n_blocked_total,
        seed: c.seed, T0: c.T0, alpha: c.alpha, tmax: c.tmax,
        attempts: c.attempts, src: c
      };
    });
    this.patchOf = {};
    this.patches.forEach(function (p) { this.patchOf[p.cid] = p; }, this);
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // geometry: per-patch King's adjacency, in lattice units
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PSSAPlayer.prototype._buildGeometry = function () {
    this.patches.forEach(function (p) {
      var L = p.L, n = p.nHw, u, r, c;
      p.row = new Int16Array(n + 1);
      p.col = new Int16Array(n + 1);
      for (u = 1; u <= n; u++) {
        p.row[u] = Math.floor((u - 1) / L);
        p.col[u] = (u - 1) % L;
      }
      // 8-neighbourhood, built from L rather than shipped in the trajectory
      p.nbr = [[]];
      for (u = 1; u <= n; u++) {
        r = p.row[u]; c = p.col[u];
        var list = [];
        for (var dr = -1; dr <= 1; dr++) {
          for (var dc = -1; dc <= 1; dc++) {
            if (!dr && !dc) continue;
            var rr = r + dr, cc = c + dc;
            if (rr < 0 || rr >= L || cc < 0 || cc >= L) continue;
            list.push(rr * L + cc + 1);
          }
        }
        p.nbr[u] = list;
      }
      p.isPerimeter = new Uint8Array(n + 1);
      p.perimeterList.forEach(function (u) { p.isPerimeter[u] = 1; });

      // Outward offset for perimeter nodes, used to place the rail.
      p.normal = {}; p.normalUnit = {};
      for (u = 1; u <= n; u++) {
        if (!p.isPerimeter[u]) continue;
        var nx = p.col[u] === 0 ? -1 : (p.col[u] === L - 1 ? 1 : 0);
        var ny = p.row[u] === 0 ? -1 : (p.row[u] === L - 1 ? 1 : 0);
        var m = Math.hypot(nx, ny) || 1;
        p.normal[u] = [nx, ny];
        p.normalUnit[u] = [nx / m, ny / m];
      }

      // world position of every hardware vertex, once, in lattice units
      p.wx = new Float32Array(n + 1);
      p.wy = new Float32Array(n + 1);
      for (u = 1; u <= n; u++) {
        p.wx[u] = p.col[u] + p.origin[1];
        p.wy[u] = -(p.row[u] + p.origin[0]);
      }
      p.cx = p.origin[1] + (L - 1) / 2;
      p.cy = -(p.origin[0] + (L - 1) / 2);
      // the patch's box in world units, used for hit-testing and for framing
      p.x0 = p.origin[1] - 0.6;
      p.x1 = p.origin[1] + L - 0.4;
      p.y1 = -(p.origin[0]) + 0.6;
      p.y0 = -(p.origin[0] + L - 1) - 0.6;
    });

    // World bounds over everything that is ever drawn, so one framing holds for the whole
    // piece and the camera never has to move between stages.
    var b = { x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity };
    function take(x, y) {
      if (x < b.x0) b.x0 = x;
      if (x > b.x1) b.x1 = x;
      if (y < b.y0) b.y0 = y;
      if (y > b.y1) b.y1 = y;
    }
    this.pos0.forEach(function (p) { take(p[0], p[1]); });
    this.pos1.forEach(function (p) { take(p[0], p[1]); });
    this.patches.forEach(function (p) {
      take(p.x0, p.y0); take(p.x1, p.y1);
    });
    this.world = b;
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // states: the scripted acts, then the recorded frames
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PSSAPlayer.prototype._ownFromChains = function (p, chains) {
    var own = new Int16Array(p.nHw + 1);
    for (var k in chains) {
      if (!Object.prototype.hasOwnProperty.call(chains, k)) continue;
      var v = parseInt(k, 10) + 1;         // slot 0 becomes owner 1
      chains[k].forEach(function (u) { own[u] = v; });
    }
    return own;
  };

  PSSAPlayer.prototype._buildStates = function () {
    var self = this;

    this.patches.forEach(function (p) {
      var c = p.src, u, i;

      p.acts = {
        split: self._ownFromChains(p, c.split),
        anchored: self._ownFromChains(p, c.anchored),
        final: self._ownFromChains(p, c.final)
      };

      // Slot -> colour, so a chain is drawn in its own vertex's shade.
      p.slotColor = [null];
      for (i = 0; i < p.nodes.length; i++) p.slotColor.push(self.shade[p.nodes[i]]);

      p.borderSlot = {};
      p.border.forEach(function (s) { p.borderSlot[s + 1] = true; });

      // Expand the frame diffs once, up front, so a dropped animation frame never has to
      // replay a diff chain to catch up.
      var frames = c.frames, nHw = p.nHw;
      var owns = new Array(frames.length);
      var cur = new Int16Array(nHw + 1);
      for (u = 0; u < nHw; u++) cur[u + 1] = c.frame0_owner[u] + 1;
      p.initialOwn = new Int16Array(cur);
      for (i = 0; i < frames.length; i++) {
        var f = frames[i];
        if (f.full) {
          for (u = 0; u < nHw; u++) cur[u + 1] = f.full[u] + 1;
        } else {
          for (var j = 0; j < f.d.length; j++) cur[f.d[j][0]] = f.d[j][1] + 1;
        }
        owns[i] = new Int16Array(cur);
      }
      p.frameOwn = owns;
      p.frames = frames;
      p.tArr = new Float64Array(frames.length);
      for (i = 0; i < frames.length; i++) p.tArr[i] = frames[i].t;

    });

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // aggregate Eemb across every cluster, sampled on one t axis
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    var tMax = Math.max(this.meta.t_end_max, 1);
    var N = 600, s;
    this.aggT = new Float64Array(N + 1);
    this.aggE = new Float64Array(N + 1);
    for (s = 0; s <= N; s++) {
      var t = tMax * s / N;
      var tot = 0;
      for (var pi = 0; pi < this.patches.length; pi++) {
        var p = this.patches[pi];
        var fi = p.frames.length ? this._frameAtT(p, t) : -2;
        tot += fi >= 0 ? p.frames[fi].e : (fi === -1 ? p.eStart : p.eFinal);
      }
      this.aggT[s] = t;
      this.aggE[s] = tot;
    }
    this.eStartTotal = this.patches.reduce(function (a, p) {
      return a + p.eStart;
    }, 0);
    this.mTotal = Math.max(this.meta.m_total, 1);

    // Temperature bounds across every cluster's schedule, so one right-hand axis serves all
    // of them.
    var hi = 0, lo = Infinity;
    this.patches.forEach(function (p) {
      hi = Math.max(hi, p.T0);
      lo = Math.min(lo, Math.max(p.T0 * Math.pow(p.alpha, p.tEnd), 1e-9));
    });
    this.tHi = hi;
    this.tLo = Math.max(lo, 1e-9);
    var lt = linTicks(this.tHi);
    this.tLinStep = lt.step; this.tLinTop = lt.top; this.tLinTicks = lt.ticks;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // graph partitioning: the partition found (Mt-KaHyPar has no event log)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    this.s1 = this.data.stage1;
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // routing: the construction
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  PSSAPlayer.prototype._buildStage3 = function () {
    var s3 = this.data.stage3, self = this;
    this.s3 = null;
    this.patches.forEach(function (p) { p.dx = 0; p.dy = 0; });
    if (!s3 || !s3.lanes) return;

    var L = s3.L, off = s3.offset;
    function wpos(h) {
      var i = h - 1;
      return [(i % L) + off, -(Math.floor(i / L) + off)];
    }

    // Where each patch is going: `d3` is its move from the ring to the grid, in world
    // units.
    var place = {};
    s3.placements.forEach(function (q) { place[q.cid] = q; });
    this.patches.forEach(function (p) {
      var q = place[p.cid];
      p.origin3 = q ? q.origin : p.origin;
      p.d3x = p.origin3[1] - p.origin[1];
      p.d3y = -(p.origin3[0] - p.origin[0]);
    });

    // The lanes, each in its own vertex's shade, since a lane is the tail of that vertex's
    // chain.
    var maxEsc = 0, maxTrunk = 0, maxFun = 0;
    var laneOf = {};
    var lanes = s3.lanes.map(function (ln) {
      var n = ln.esc.length + ln.fun.length + ln.trunk.length;
      // The dock, the perimeter hardware vertex the lane leaves from, leads the array.
      var w = new Float32Array(2 * (n + 1)), i = 0, j;
      function push(h) { var q = wpos(h); w[i++] = q[0]; w[i++] = q[1]; }
      push(ln.dock);
      for (j = 0; j < ln.esc.length; j++) push(ln.esc[j]);
      for (j = 0; j < ln.fun.length; j++) push(ln.fun[j]);
      for (j = 0; j < ln.trunk.length; j++) push(ln.trunk[j]);
      var o = {
        v: ln.v, cid: ln.cid, side: ln.side, stripe: ln.stripe,
        w: w, n: n,
        nEsc: ln.esc.length, nFun: ln.fun.length, nTrunk: ln.trunk.length,
        col: self.shade[ln.v]
      };
      if (o.nEsc > maxEsc) maxEsc = o.nEsc;
      if (o.nFun > maxFun) maxFun = o.nFun;
      if (o.nTrunk > maxTrunk) maxTrunk = o.nTrunk;
      laneOf[ln.v] = o;
      return o;
    });

    // Every cut edge, as the polyline the construction routes it along: `trunk[u][0..iu]`
    // then `trunk[v][iv..0]`, which is exactly the array `guaranteed_route_fast` returns
    // for it.
    var maxRoute = 1;
    var routes = [];
    var adjacent = [];              // cut edges whose patches already touch
    s3.contacts.forEach(function (c, ci) {
      var a = laneOf[c[0]], b = laneOf[c[1]];
      if (!a || !b || c[2] < 0 || c[3] < 0) { adjacent.push(ci); return; }
      var ia = Math.min(c[2], a.nTrunk - 1), ib = Math.min(c[3], b.nTrunk - 1);
      if (ia < 0 || ib < 0) return;
      var n = (ia + 1) + (ib + 1);
      var w = new Float32Array(2 * n), i = 0, j, base;
      // +1 for the dock that now leads every lane's point array.
      base = 2 * (1 + a.nEsc + a.nFun);
      for (j = 0; j <= ia; j++) { w[i++] = a.w[base + 2 * j]; w[i++] = a.w[base + 2 * j + 1]; }
      base = 2 * (1 + b.nEsc + b.nFun);
      for (j = ib; j >= 0; j--) { w[i++] = b.w[base + 2 * j]; w[i++] = b.w[base + 2 * j + 1]; }
      if (n > maxRoute) maxRoute = n;
      // `split` is the first cell on v's chain, so the contact, the one King's step where
      // the two chains meet, is the segment from `split - 1` to `split`.
      routes.push({ ci: ci, u: c[0], v: c[1], w: w, n: n, split: ia + 1,
                    need: Math.max(ia, ib) + 1 });
    });

    this.s3 = {
      L: L, offset: off, lanes: lanes, laneOf: laneOf, routes: routes,
      maxEsc: Math.max(maxEsc, 1), maxFun: Math.max(maxFun, 1),
      maxTrunk: Math.max(maxTrunk, 1), maxRoute: maxRoute,
      adjacent: adjacent,
      stats: s3.stats, ledger: s3.ledger, result: s3.result,
      // The lattice square itself, in world units.
      x0: off - 0.5, x1: L + off - 0.5,
      y0: -(L + off - 0.5), y1: -(off - 0.5)
    };
  };

  // Largest i with frames[i].t <= t, or 0.
  // The last recorded frame at or before step t, or -1 before the first one: frames[0].t is
  // hundreds of steps into the run, and until then the state is the initial placement
  // (p.initialOwn, e_start), not frame 0's.
  PSSAPlayer.prototype._frameAtT = function (p, t) {
    var a = p.tArr, lo = 0, hi = a.length - 1;
    if (hi < 0) return 0;
    if (t < a[0]) return -1;
    while (lo < hi) {
      var mid = (lo + hi + 1) >> 1;
      if (a[mid] <= t) lo = mid; else hi = mid - 1;
    }
    return lo;
  };

  // Round tick step and top for a linear temperature axis.
  function linTicks(hi) {
    var raw = hi / 5;
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var norm = raw / mag;
    var step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
    var top = Math.ceil(hi / step) * step;
    var ticks = [];
    for (var v = 0; v <= top * 1.0001; v += step) ticks.push(v);
    return { step: step, top: top, ticks: ticks };
  }

  function decadeTicks(lo, hi) {
    var out = [], e = Math.floor(Math.log(lo) / Math.LN10) - 1;
    var top = Math.ceil(Math.log(hi) / Math.LN10);
    for (; e <= top; e++) {
      for (var k = 0; k < 3; k++) {
        var v = [1, 2, 5][k] * Math.pow(10, e);
        if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
      }
    }
    return out;
  }

  // A round step for the left axis that keeps labels from colliding.
  function eStepFor(m, ph) {
    var want = Math.max(2, Math.floor(ph / 26));
    var raw = m / want;
    var mag = Math.pow(10, Math.floor(Math.log(Math.max(raw, 1)) / Math.LN10));
    var norm = raw / mag;
    return Math.max(1, (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag);
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // timeline
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PSSAPlayer.prototype._timeline = function () {
    var m = this.meta;

    return [
      { kind: 'act', space: 'graph', to: 'tangle', ms: 1700, hud: 'none',
        label: 'Input graph · random cubic network' },
      // Mt-KaHyPar is an external solver with no per-move state, so there is no
      // partitioning to replay: the grey graph, then the partition found, whole, at one
      // instant (FOUND_AT), held before the clusters part.
      { kind: 'act', space: 'graph', to: 'found', ms: 2400, hud: 'none',
        morph: true,
        label: 'Graph partitioning · fewest border nodes' },
      { kind: 'act', space: 'graph', to: 'separate', ms: 2600, hud: 'none',
        morph: true, label: 'Graph partitioning · fewest border nodes' },
      { kind: 'act', space: 'graph', to: 'shade', ms: 1000, hud: 'none',
        morph: true, label: 'PSSA wBC · border-aware initial placement' },
      { kind: 'act', space: 'bridge', to: 'split', ms: 2600, hud: 'none',
        morph: true, label: 'PSSA wBC · border-aware initial placement' },
      { kind: 'act', space: 'lattice', from: 'split', to: 'anchored', ms: 1300,
        morph: true, rail: true, railIn: true, hud: 'none',
        label: 'PSSA wBC · border-aware initial placement' },
      { kind: 'hold', space: 'lattice', act: 'anchored', ms: 1500, rail: true,
        hud: 'none' },
      { kind: 'anneal', space: 'lattice', hud: 'frame', label: 'PSSA wBC · simulated annealing' },
      { kind: 'act', space: 'lattice', from: null, to: 'final', ms: 1100,
        morph: true, rail: true, hud: 'last', label: 'PSSA wBC · terminal search' },
      { kind: 'hold', space: 'lattice', act: 'final', ms: 2200, rail: true,
        hud: 'last', label: m.success_all ? 'Clusters embedded' : 'Best found' }
    ].concat(this._timeline3());
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // the routing's beats
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  PSSAPlayer.prototype._timeline3 = function () {
    var s3 = this.s3;
    if (!s3) return [];
    var m = this.meta, r = s3.result, st = s3.stats;
    var pct = (100 * r.qubits / r.lattice).toFixed(1);
    var routing = 100 * (s3.ledger.escape + s3.ledger.funnel + s3.ledger.trunk)
                      / r.qubits;

    // One rate of cells per second for the whole descent, as the anneal keeps one rate of
    // steps per second for every cluster.
    var nDown = s3.maxEsc + s3.maxFun;        // tile to the head of its chain
    var nPlaza = s3.maxTrunk;                 // the head of its chain to its tail
    var LAY_MS = 13500;                       // the whole descent, at 1x
    var perCell = LAY_MS / Math.max(nDown + nPlaza, 1);
    this.cellsPerSecond = 1000 / perCell;

    return [
      { kind: 's3', s3: true, phase: 'place', space: 's3', ms: 3000,
        morph: true, hud: 's3',
        label: 'Routing · cut edges through the routing structure' },
      // Escape and funnel are one beat because they are one motion, from the tile to the
      // head of its chain.
      { kind: 's3', s3: true, phase: 'escape', space: 's3',
        ms: Math.round(nDown * perCell), hud: 's3',
        label: 'Routing · cut edges through the routing structure' },
      { kind: 's3', s3: true, phase: 'plaza', space: 's3',
        ms: Math.round(nPlaza * perCell), hud: 's3',
        label: 'Routing · cut edges through the routing structure' },
      // The last thing on screen before the loop restarts, so the reader gets a beat with
      // the finished embedding rather than being pulled straight back to the tangle.
      { kind: 's3', s3: true, phase: 'done', space: 's3', ms: 6400,
        hud: 's3',
        label: r.valid ? 'Final embedding' : 'Incomplete' }
    ];
  };

  PSSAPlayer.prototype._buildTimeline = function () {
    this.script = this._timeline();
    var i;

    // The uniform-in-t clock, now shared by every cluster: one steps-per-second for all of
    // them, so the anneal lasts as long as the longest run takes and a cluster that
    // finished earlier sits finished.
    var tMax = Math.max(this.meta.t_end_max, 1);
    this.stepsPerSecond = tMax / this.annealSeconds;
    this.msPerStep = (this.annealSeconds * 1000) / tMax;
    this.annealMs = this.annealSeconds * 1000;

    var at = 0;
    this.stageStartMs = new Float64Array(this.script.length + 1);
    this.annealStage = 0;
    for (i = 0; i < this.script.length; i++) {
      this.stageStartMs[i] = at;
      var st = this.script[i];
      st.durMs = st.kind === 'anneal' ? this.annealMs : st.ms;
      if (st.kind === 'anneal') this.annealStage = i;
      at += st.durMs;
    }
    this.stageStartMs[this.script.length] = at;
    this.totalMs = at;

    var a0 = this.stageStartMs[this.annealStage];
    // The partitioning ends where the clusters part, found by name, not position.
    var s1End = 0;
    for (i = 0; i < this.script.length; i++) {
      if (this.script[i].to === 'separate') { s1End = this.stageStartMs[i]; break; }
    }
    this.stage3Start = -1;
    for (i = 0; i < this.script.length; i++) {
      if (this.script[i].s3) { this.stage3Start = i; break; }
    }
    var s3At = this.stage3Start >= 0 ? this.stageStartMs[this.stage3Start] : at;
    // The method's three stages and nothing finer.
    this.segments = [
      { from: 0, to: s1End, label: 'graph partitioning' },
      { from: s1End, to: s3At, label: 'embedding clusters' }
    ];
    if (this.stage3Start >= 0) {
      this.segments.push({ from: s3At, to: at, label: 'routing structure' });
    }
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // camera
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PSSAPlayer.prototype._fitBox = function (x0, y0, x1, y1, pad) {
    var w = Math.max(x1 - x0, 1e-6), h = Math.max(y1 - y0, 1e-6);
    var s = Math.min((this.LATW - 2 * pad) / w, (this.LATH - 2 * pad) / h);
    return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, s: s };
  };

  PSSAPlayer.prototype._camAll = function () {
    var b = this.world;
    return this._fitBox(b.x0, b.y0, b.x1, b.y1, 14);
  };

  PSSAPlayer.prototype._camPatch = function (p) {
    // A little extra room below and around so the rail, the leaders and the arcs leaving
    // the patch are all inside the frame rather than clipped at its edge.
    var m = 1.4;
    var b = this._box(p);
    return this._fitBox(b.x0 - m, b.y0 - m, b.x1 + m, b.y1 + m, 18);
  };

  // The routing's framing: the whole L x L lattice.
  PSSAPlayer.prototype._camS3 = function (st) {
    var b = this.s3;
    if (!b) return this._camAll();
    var full = this._fitBox(b.x0, b.y0, b.x1, b.y1, 14);
    if (!st || st.phase !== 'place') return full;
    var local = this.clock - this.stageStartMs[this.stage];
    var e = easeInOut(st.ms > 0 ? clamp01(local / st.ms) : 1);
    var a = this._camAll();
    var cam = {
      cx: lerp(a.cx, full.cx, e),
      cy: lerp(a.cy, full.cy, e),
      s: Math.exp(lerp(Math.log(a.s), Math.log(full.s), e))
    };
    // Never frame less than where the patches are at this instant.
    var pad = 14, hx = 0, hy = 0;
    for (var i = 0; i < this.patches.length; i++) {
      var q = this.patches[i], dx = q.d3x * e, dy = q.d3y * e;
      hx = Math.max(hx, Math.abs(q.x0 + dx - cam.cx), Math.abs(q.x1 + dx - cam.cx));
      hy = Math.max(hy, Math.abs(q.y0 + dy - cam.cy), Math.abs(q.y1 + dy - cam.cy));
    }
    var fit = Math.min((this.LATW / 2 - pad) / Math.max(hx, 1e-6),
                       (this.LATH / 2 - pad) / Math.max(hy, 1e-6));
    if (fit < cam.s) cam.s = fit;
    return cam;
  };

  PSSAPlayer.prototype._camNow = function (now) {
    // The partitioning has no patches to be inside.
    var st = this.script[this.stage || 0];
    var inGraph = st && st.space === 'graph';
    var inS3 = st && st.space === 's3';
    var target;
    if (this.zoomCid !== null && !inGraph) {
      target = this._camPatch(this.patchOf[this.zoomCid]);
    } else if (inS3) {
      target = this._camS3(st);
    } else {
      target = this._camAll();
    }
    if (!this.cam) { this.cam = target; this.camFrom = target; this.camT0 = 0; }
    var d = this.camMs ? clamp01((now - this.camT0) / this.camMs) : 1;
    if (d >= 1) { this.cam = target; return this.cam; }
    var e = easeInOut(d);
    // Interpolate the scale geometrically: a linear ramp between two scales an order of
    // magnitude apart spends almost all its time near the larger one and reads as a jerk at
    // the end.
    this.cam = {
      cx: lerp(this.camFrom.cx, target.cx, e),
      cy: lerp(this.camFrom.cy, target.cy, e),
      s: Math.exp(lerp(Math.log(this.camFrom.s), Math.log(target.s), e))
    };
    return this.cam;
  };

  PSSAPlayer.prototype.zoomTo = function (cid) {
    if (cid !== null && !this.patchOf[cid]) return this;
    if (cid === this.zoomCid) return this;
    this.camFrom = this.cam || this._camAll();
    // Now, not the last frame the loop happened to draw.
    this.camT0 = (global.performance ? global.performance.now() : 0) ||
                 this.lastNow || 0;
    this.camMs = this.reduced ? 0 : 620;
    this.zoomCid = cid;
    this.container.classList.toggle('pssa-zoomed', cid !== null);
    if (!this.running) {                 // paused: still needs to animate out
      this.running = true;
      this.lastNow = null;
      global.requestAnimationFrame(this._tick.bind(this));
    }
    this._syncZoomUi();
    return this;
  };

  PSSAPlayer.prototype.sx = function (x) {
    return this.LATX + this.LATW / 2 + (x - this.cam.cx) * this.cam.s;
  };
  PSSAPlayer.prototype.sy = function (y) {
    return this.LATY + this.LATH / 2 - (y - this.cam.cy) * this.cam.s;
  };
  // Screen position of patch hardware vertex `u`.
  PSSAPlayer.prototype.qpos = function (p, u) {
    return [this.sx(p.wx[u] + p.dx), this.sy(p.wy[u] + p.dy)];
  };

  // A patch's box in world units, with that displacement applied.
  PSSAPlayer.prototype._box = function (p) {
    return { x0: p.x0 + p.dx, x1: p.x1 + p.dx,
             y0: p.y0 + p.dy, y1: p.y1 + p.dy };
  };
  // Screen position of graph vertex `v` at act progress.
  PSSAPlayer.prototype.vpos = function (v, mixT) {
    var a = this.pos0[v], b = this.pos1[v];
    if (mixT <= 0) return [this.sx(a[0]), this.sy(a[1])];
    if (mixT >= 1) return [this.sx(b[0]), this.sy(b[1])];
    return [this.sx(lerp(a[0], b[0], mixT)), this.sy(lerp(a[1], b[1], mixT))];
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // dom
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  var ICON = {
    play:  '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.4 13.2 8 4 13.6Z"/></svg>',
    pause: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4.4 2.6h2.7v10.8H4.4zM8.9 2.6h2.7v10.8H8.9z"/></svg>',
    prev:  '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.2 2.6h1.9v10.8H3.2z"/><path d="M13.2 2.6 6.2 8l7 5.4Z"/></svg>',
    next:  '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10.9 2.6h1.9v10.8h-1.9z"/><path d="M2.8 2.6 9.8 8l-7 5.4Z"/></svg>'
  };

  PSSAPlayer.prototype._buildDom = function () {
    var c = this.container, self = this;
    c.classList.add('pssa');
    // The palette's custom properties are declared on `.pssa`, so a rule on the element
    // wins over the value inherited from `<html>`; setting them inline here is what lets
    // the active palette override them.
    if (global.PSSA_PALETTE_APPLY) global.PSSA_PALETTE_APPLY(c, PAL);
    if (this.reduced) c.classList.add('pssa-reduced');
    c.setAttribute('tabindex', '0');
    c.setAttribute('role', 'group');
    c.setAttribute('aria-label',
      'Embedding pipeline replay. Space plays and pauses; left and right ' +
      'arrows step; click a cluster to zoom into it.');

    c.innerHTML =
      '<div class="pssa-stage">' +
        '<canvas class="pssa-canvas"></canvas>' +
        '<button type="button" class="pssa-back" hidden>&#8592; all clusters</button>' +
        '<div class="pssa-phase"><span class="pssa-phase-label"></span></div>' +
      '</div>' +

      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // transport
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      '<div class="pssa-transport">' +
        '<div class="pssa-tbtns">' +
          '<button type="button" class="pssa-btn pssa-play" data-act="play" ' +
            'aria-label="Play" title="Play / pause (space)">' +
            '<span class="i-play">' + ICON.play + '</span>' +
            '<span class="i-pause">' + ICON.pause + '</span></button>' +
          '<button type="button" class="pssa-btn" data-act="prev" ' +
            'aria-label="Step back one frame" title="Step back (left arrow)">' +
            ICON.prev + '</button>' +
          '<button type="button" class="pssa-btn" data-act="next" ' +
            'aria-label="Step forward one frame" title="Step forward (right arrow)">' +
            ICON.next + '</button>' +
        '</div>' +
        '<div class="pssa-scrub">' +
          '<canvas class="pssa-scrub-bg"></canvas>' +
          '<input class="pssa-range" type="range" min="0" max="1000" step="1" ' +
            'value="0" aria-label="Timeline position">' +
        '</div>' +
        '<div class="pssa-tmeta">' +
          '<div class="pssa-speed" role="group" aria-label="Playback speed">' +
            '<button type="button" data-speed="0.25">&#188;&#215;</button>' +
            '<button type="button" data-speed="0.5">&#189;&#215;</button>' +
            '<button type="button" data-speed="1" class="on">1&#215;</button>' +
            '<button type="button" data-speed="2">2&#215;</button>' +
            '<button type="button" data-speed="4">4&#215;</button>' +
          '</div>' +
        '</div>' +
      '</div>' +

      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // diagnostic plot
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      '<div class="pssa-plot">' +
        '<div class="pssa-plot-head">' +
          '<h4 class="pssa-plot-title"></h4>' +
          '<div class="pssa-scales">' +
            '<div class="pssa-xmode" role="group" ' +
                 'aria-label="Temperature axis scale">' +
              '<button type="button" data-y="lin">linear T</button>' +
              '<button type="button" data-y="log">log T</button>' +
            '</div>' +
            '<div class="pssa-xmode" role="group" aria-label="Time axis scale">' +
              '<button type="button" data-x="lin">linear t</button>' +
              '<button type="button" data-x="log">log t</button>' +
            '</div>' +
          '</div>' +
        '</div>' +
        '<canvas class="pssa-plot-canvas"></canvas>' +
      '</div>';

    this.canvas = c.querySelector('.pssa-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.plotEl = c.querySelector('.pssa-plot-canvas');
    this.plotCtx = this.plotEl.getContext('2d');
    this.scrubEl = c.querySelector('.pssa-scrub-bg');
    this.scrubCtx = this.scrubEl.getContext('2d');
    this.range = c.querySelector('.pssa-range');
    this.playBtn = c.querySelector('.pssa-play');
    this.phaseLabel = c.querySelector('.pssa-phase-label');
    this.plotTitle = c.querySelector('.pssa-plot-title');
    this.backBtn = c.querySelector('.pssa-back');

    this._bindTransport();
    this.setSpeed(this.speed);
    this.setXMode(this.xmode);
    this.setYMode(this.ymode);

    this._resize();
    global.addEventListener('resize', this._resize.bind(this));
  };

  PSSAPlayer.prototype._syncZoomUi = function () {
    if (this.backBtn) this.backBtn.hidden = (this.zoomCid === null);
  };

  // Everything in a box except its canvas: the box's own padding plus the outer height of
  // every other child.
  function boxChrome(box) {
    var gcs = global.getComputedStyle;
    if (!gcs) return 22;
    var s = gcs(box);
    var total = (parseFloat(s.paddingTop) || 0) + (parseFloat(s.paddingBottom) || 0);
    var kids = box.children || [];
    for (var i = 0; i < kids.length; i++) {
      var el = kids[i];
      if (!el || el.tagName === 'CANVAS') continue;
      if (typeof el.offsetHeight !== 'number') continue;
      var k = gcs(el);
      total += el.offsetHeight +
        (parseFloat(k.marginTop) || 0) + (parseFloat(k.marginBottom) || 0);
    }
    return total;
  }

  // A dpr change does not always arrive as a `resize` event -- dragging the window to a
  // display with different scaling changes the ratio while every CSS dimension stays put.
  PSSAPlayer.prototype._checkDpr = function () {
    var dpr = global.devicePixelRatio || 1;
    if (this._dpr === dpr) return;
    this._dpr = dpr;
    this._resize();
  };

  PSSAPlayer.prototype._resize = function () {
    var stage = this.container.querySelector('.pssa-stage');
    var w = stage.clientWidth || 640;
    // The figure lives in a square box at the top of the stage.
    var lat = Math.min(w, 720);
    // The phase caption gets a band of its own beneath that box.
    var capEl = this.container.querySelector('.pssa-phase');
    var cap = Math.max(72, (capEl && typeof capEl.offsetHeight === 'number')
      ? capEl.offsetHeight : 0);
    // On one screen: the box gives up height until stage and transport fit in what the page
    // has.
    if (this.fitHeight) {
      var tEl = this.container.querySelector('.pssa-transport');
      var tH = (tEl && typeof tEl.offsetHeight === 'number' && tEl.offsetHeight) || 48;
      lat = Math.min(lat, Math.floor(this.fitHeight() - cap - tH - 4));
    }
    lat = Math.max(260, lat);    // below this the ring is too small to read
    var h = lat + cap;
    var dpr = global.devicePixelRatio || 1;
    // A backing store is a whole number of device pixels, so `w * dpr` at a fractional
    // ratio is truncated.
    var bw = Math.max(1, Math.round(w * dpr));
    var bh = Math.max(1, Math.round(h * dpr));
    this.canvas.width = bw;
    this.canvas.height = bh;
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';
    this.bsx = bw / w; this.bsy = bh / h;
    this.ctx.setTransform(this.bsx, 0, 0, this.bsy, 0, 0);
    this.W = w; this.H = h;

    // The camera's viewport: the square box, not the whole canvas, so the caption band
    // below stays clear of the figure.
    this.LATX = 0; this.LATY = 0;
    this.LATW = w; this.LATH = lat;
    this.cam = null;
    this.mesh = null;

    // Same rounding, and the same scale-from-what-you-got, for the plot and the scrub
    // track.
    function fit(el, ctx, cssH, dflt) {
      var cw = el.clientWidth || dflt;
      var ew = Math.max(1, Math.round(cw * dpr));
      var eh = Math.max(1, Math.round(cssH * dpr));
      el.width = ew;
      el.height = eh;
      el.style.height = cssH + 'px';
      if (ctx) ctx.setTransform(ew / cw, 0, 0, eh / cssH, 0, 0);
      return cw;
    }
    // Side by side with the stage, the plot canvas takes the height the stage leaves after
    // the plot's own heading, so the two columns end level.
    var plotH = 212;
    var pBox = this.container.querySelector('.pssa-plot');
    if (pBox && typeof pBox.offsetTop === 'number' &&
        typeof stage.offsetTop === 'number' &&
        Math.abs(pBox.offsetTop - stage.offsetTop) < 4) {
      plotH = Math.max(212, Math.round(h - boxChrome(pBox)));
    }
    this.PW = fit(this.plotEl, this.plotCtx, plotH, w); this.PH = plotH;
    this.plotBg = null;

    this.BW = fit(this.scrubEl, this.scrubCtx, 18, 400); this.BH = 18;

    this._camNow(this.lastNow || 0);
    this._drawScrubTrack();
    if (this.lastDraw) this._render(this.lastDraw);
    this._drawPlot(this.hudFrame);
  };

  // The cell pitch in pixels: one lattice unit.
  PSSAPlayer.prototype._cell = function () { return this.cam.s; };

  // The pitch every mark is sized from, as opposed to the pitch every position is placed
  // at.
  PSSAPlayer.prototype._markCell = function () {
    return Math.max(this.cam.s, MARK_MIN_CELL);
  };

  // Every colour this instance holds, rebuilt from the active theme.
  PSSAPlayer.prototype._recolor = function () {
    var self = this, ids = this.nodeId, byCid = [], i, s, sh;

    this.clusterCol = PAL.clusterColors(this.k);
    for (i = 0; i < this.k; i++) byCid.push([]);
    for (i = 0; i < this.nV; i++) byCid[this.label[i]].push(i);
    for (i = 0; i < this.k; i++) {
      byCid[i].sort(function (a, b) {
        return String(ids[a]) < String(ids[b]) ? -1
             : (String(ids[a]) > String(ids[b]) ? 1 : 0);
      });
      // The whole cluster at once, because a theme may place its shades against each other
      // rather than deriving each one independently.
      sh = PAL.clusterShades(this.clusterCol[i], byCid[i].length);
      sh = this._orderShades(sh, i, byCid[i]);
      for (s = 0; s < byCid[i].length; s++) this.shade[byCid[i][s]] = sh[s];
    }

    // The two places a shade is copied rather than looked up.
    if (this.patches) {
      this.patches.forEach(function (p) {
        if (!p.slotColor) return;
        for (var j = 0; j < p.nodes.length; j++) {
          p.slotColor[j + 1] = self.shade[p.nodes[j]];
        }
      });
    }
    if (this.s3 && this.s3.lanes) {
      this.s3.lanes.forEach(function (ln) { ln.col = self.shade[ln.v]; });
    }

    // The rasterized layers with palette colours baked into them.
    this.mesh = null;
    this.s3Layer = null;
    this.palGen++;
  };

  // Which of a cluster's shades each of its vertices gets.
  PSSAPlayer.prototype._orderShades = function (shades, cid, verts) {
    var rule = PAL.assign || 'index';
    if (rule === 'index' || shades.length < 2) return shades;
    if (rule === 'stripe') return this._assignByStripe(shades, verts);
    if (rule === 'spread') return this._assignBySpread(shades, cid, verts);
    return shades;
  };

  // Border vertices in the order their lanes descend the routing structure.
  PSSAPlayer.prototype._assignByStripe = function (shades, verts) {
    var s3 = this.data.stage3, i;
    if (!s3 || !s3.lanes) return shades;
    if (!this._stripeOf) {
      this._stripeOf = {};
      for (i = 0; i < s3.lanes.length; i++) {
        this._stripeOf[s3.lanes[i].v] = s3.lanes[i].stripe;
      }
    }
    var stripeOf = this._stripeOf, toLab = global.PSSA_RGB_TO_LAB;
    var byL = shades.slice().sort(function (a, b) {
      return toLab(a)[0] - toLab(b)[0];
    });
    var lane = [], inner = [];
    for (i = 0; i < verts.length; i++) {
      if (stripeOf[verts[i]] === undefined) inner.push(i); else lane.push(i);
    }
    lane.sort(function (a, b) { return stripeOf[verts[a]] - stripeOf[verts[b]]; });

    var out = new Array(verts.length), taken = {}, pick;
    for (i = 0; i < lane.length; i++) {
      pick = lane.length > 1
        ? Math.round(i * (byL.length - 1) / (lane.length - 1)) : 0;
      while (taken[pick]) pick = (pick + 1) % byL.length;
      taken[pick] = true;
      out[lane[i]] = byL[pick];
    }
    var rest = [];
    for (i = 0; i < byL.length; i++) if (!taken[i]) rest.push(byL[i]);
    for (i = 0; i < inner.length; i++) out[inner[i]] = rest[i];
    return out;
  };

  // No vertex next to one of a similar colour.
  PSSAPlayer.prototype._assignBySpread = function (shades, cid, verts) {
    var src = null, i, j, k;
    for (i = 0; i < this.data.clusters.length; i++) {
      if (this.data.clusters[i].cid === cid) { src = this.data.clusters[i]; break; }
    }
    if (!src || !src.edges) return shades;

    var pos = {};
    for (i = 0; i < verts.length; i++) pos[verts[i]] = i;
    var adj = [];
    for (i = 0; i < verts.length; i++) adj.push([]);
    for (i = 0; i < src.edges.length; i++) {
      var a = pos[src.nodes[src.edges[i][0]]], b = pos[src.nodes[src.edges[i][1]]];
      if (a !== undefined && b !== undefined) { adj[a].push(b); adj[b].push(a); }
    }

    var order = [];
    for (i = 0; i < verts.length; i++) order.push(i);
    order.sort(function (x, y) { return adj[y].length - adj[x].length; });

    // Every distance this needs, once: n is at most 50, so 2,500 of them.
    var TAU = 3, n = shades.length, gap = global.PSSA_DE2000 || global.PSSA_DE;
    var D = [], W = [];
    for (i = 0; i < n; i++) {
      D.push(new Float64Array(n));
      W.push(new Float64Array(n));
      for (j = 0; j < n; j++) {
        D[i][j] = i === j ? 0 : (j < i ? D[j][i] : gap(shades[i], shades[j]));
        W[i][j] = Math.exp(-D[i][j] / TAU);
      }
    }

    var idx = new Array(verts.length), taken = {};
    for (i = 0; i < idx.length; i++) idx[i] = -1;
    for (k = 0; k < order.length; k++) {
      var v = order[k], best = -1, bestD = -1;
      for (j = 0; j < n; j++) {
        if (taken[j]) continue;
        var m = Infinity;
        for (i = 0; i < adj[v].length; i++) {
          var nb = idx[adj[v][i]];
          if (nb >= 0) m = Math.min(m, D[j][nb]);
        }
        if (m === Infinity) m = 0;          // no coloured neighbour yet
        if (m > bestD) { bestD = m; best = j; }
      }
      if (best < 0) break;
      idx[v] = best;
      taken[best] = true;
    }
    for (i = 0; i < idx.length; i++) if (idx[i] < 0) idx[i] = i;

    // Then pairwise swaps, kept while they lower sum exp(-dE / 3) over the joined pairs: a
    // soft minimum, so the closest joined pair is pushed apart first.
    var pass, u, w, t, improved;
    function cost(x, c) {
      var e = 0, q;
      for (q = 0; q < adj[x].length; q++) e += W[c][idx[adj[x][q]]];
      return e;
    }
    for (pass = 0; pass < 100; pass++) {
      improved = false;
      for (u = 0; u < idx.length; u++) {
        for (w = u + 1; w < idx.length; w++) {
          var before = cost(u, idx[u]) + cost(w, idx[w]);
          t = idx[u]; idx[u] = idx[w]; idx[w] = t;
          if (cost(u, idx[u]) + cost(w, idx[w]) < before - 1e-12) improved = true;
          else { idx[w] = idx[u]; idx[u] = t; }
        }
      }
      if (!improved) break;
    }
    var out = [];
    for (i = 0; i < idx.length; i++) out.push(shades[idx[i]]);
    return out;
  };

  // Switch theme, by name or by palette object.
  PSSAPlayer.prototype.setPalette = function (which) {
    var all = global.PSSA_PALETTES || {};
    var p = typeof which === 'string' ? all[which] : which;
    if (!p) return false;

    usePalette(p);
    if (global.PSSA_PALETTE_APPLY) {
      global.PSSA_PALETTE_APPLY(this.container, p);
      if (global.document && global.document.documentElement) {
        global.PSSA_PALETTE_APPLY(global.document.documentElement, p);
      }
    }
    this._recolor();
    // The same re-render `seek` does, so the state on screen is unchanged.
    this._apply(this.lastNow || 0, true);
    return true;
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // static layer: the King's mesh under every patch
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // What the two rasterized layers, the King's mesh and the lanes, are cached against.
  PSSAPlayer.prototype._camSig = function () {
    return Math.round(this.cam.cx * 100) + ',' + Math.round(this.cam.cy * 100) +
           ',' + Math.round(this.cam.s * 100) + ',' + this.W + 'x' + this.H +
           '@' + (global.devicePixelRatio || 1);
  };

  PSSAPlayer.prototype._drawMesh = function () {
    // The faint King's mesh under each patch never changes for a given camera, so it is
    // rasterized once into an offscreen canvas and blitted, and rebuilt only while the
    // camera is actually moving.
    var c = document.createElement('canvas');
    c.width = this.canvas.width; c.height = this.canvas.height;
    var x = c.getContext('2d');
    x.setTransform(this.bsx, 0, 0, this.bsy, 0, 0);
    var cell = this._cell(), mc = this._markCell(), self = this;
    var dotR = this.dotR();

    x.strokeStyle = ink(0.10);
    x.lineWidth = mc * 0.018;
    // `_visible` admits a patch that only overlaps the view, and zoomed into one tile a
    // neighbouring tile's cells are a whole cell pitch apart, so an edge of its mesh can
    // run far outside the canvas.
    x.beginPath();
    this.patches.forEach(function (p) {
      if (!self._visible(p)) return;
      for (var u = 1; u <= p.nHw; u++) {
        var a = self.qpos(p, u);
        var ai = self._onStage(a);
        for (var k = 0; k < p.nbr[u].length; k++) {
          var w = p.nbr[u][k];
          if (w < u) continue;
          var b = self.qpos(p, w);
          var bi = self._onStage(b);
          if (!ai && !bi) continue;
          var a2 = ai ? a : self._clipToView(b, a);
          var b2 = bi ? b : self._clipToView(a, b);
          x.moveTo(a2[0], a2[1]); x.lineTo(b2[0], b2[1]);
        }
      }
    });
    x.stroke();

    // bare hardware vertices
    x.fillStyle = ink(0.16);
    this.patches.forEach(function (p) {
      if (!self._visible(p)) return;
      for (var u = 1; u <= p.nHw; u++) {
        var q = self.qpos(p, u);
        if (!self._onStage(q)) continue;
        x.beginPath(); x.arc(q[0], q[1], dotR * 0.32, 0, 6.2832); x.fill();
      }
    });

    // the perimeter rail of each patch: the track every border super vertex must keep a
    // foot on.
    x.strokeStyle = ink(0.30);
    x.lineWidth = mc * 0.05;
    x.lineJoin = 'miter';
    x.lineCap = 'butt';
    this.patches.forEach(function (p) {
      if (!self._visible(p)) return;
      var ring = self._railLoop(p);
      // A closed square, so it is clipped by the context rather than corner by corner:
      // moving a corner to the edge of the view would bevel the very shape the rail exists
      // to show.
      if (ring.some(function (pt) { return !self._onStage(pt); })) {
        var m = 24;
        var cx0 = self.LATX - m, cx1 = self.LATX + self.LATW + m;
        var cy0 = self.LATY - m, cy1 = self.LATY + self.LATH + m;
        ring = ring.map(function (pt) {
          return [Math.max(cx0, Math.min(cx1, pt[0])),
                  Math.max(cy0, Math.min(cy1, pt[1]))];
        });
      }
      x.beginPath();
      ring.forEach(function (pt, i) {
        i ? x.lineTo(pt[0], pt[1]) : x.moveTo(pt[0], pt[1]);
      });
      x.closePath();
      x.stroke();
    });

    this.mesh = c;
    this.meshSig = this._camSig();
    this.meshGen++;
  };

  // Whether a patch's box is on screen at all.
  PSSAPlayer.prototype._visible = function (p) {
    var m = this.railOffset() * 2 + 8;
    var b = this._box(p);
    var x0 = this.sx(b.x0) - m, x1 = this.sx(b.x1) + m;
    var y0 = this.sy(b.y1) - m, y1 = this.sy(b.y0) + m;
    return x1 >= this.LATX && x0 <= this.LATX + this.LATW &&
           y1 >= this.LATY && y0 <= this.LATY + this.LATH;
  };

  // A vertex drawn in the graph stage, and the hardware vertex it becomes, are the same
  // mark at slightly different sizes.
  PSSAPlayer.prototype.vertR = function () {
    return this._markCell() * 0.24;
  };

  PSSAPlayer.prototype.railOffset = function () { return this._cell() * 0.48; };
  PSSAPlayer.prototype.dotR = function () {
    return this._markCell() * 0.20;
  };

  PSSAPlayer.prototype._railLoop = function (p) {
    var o = this.railOffset();
    var x0 = this.sx(p.origin[1] + p.dx) - o;
    var x1 = this.sx(p.origin[1] + p.L - 1 + p.dx) + o;
    var y0 = this.sy(-p.origin[0] + p.dy) - o;
    var y1 = this.sy(-(p.origin[0] + p.L - 1) + p.dy) + o;
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  };

  PSSAPlayer.prototype.railPoint = function (p, u) {
    var q = this.qpos(p, u), n = p.normal[u], o = this.railOffset();
    return [q[0] + n[0] * o, q[1] + n[1] * o];
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // per-frame rendering: the lattice layer
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Chain interiors: every King's-adjacent pair of hardware vertices owned by the same
  // super vertex.
  // `other`, during a morph, is the state on the other side of it, and `out` says which
  // side this one is: a link the two states share (both ends owned by the same super vertex
  // in both) is drawn once, by the incoming side, at full strength, so only what actually
  // changes cross-fades.
  PSSAPlayer.prototype._drawChains = function (p, own, alpha, spread, srcOf,
                                              other, out) {
    var ctx = this.ctx, cell = this._cell(), mc = this._markCell(), self = this;
    ctx.lineCap = 'round';
    var wide = mc * LINK_W;
    for (var u = 1; u <= p.nHw; u++) {
      var o = own[u];
      if (!o) continue;
      var a = this._qposSpread(p, u, o, spread, srcOf);
      var col = p.slotColor[o] || FAINT;
      // Glasbey optimizes distance between palette entries, not between an entry and the
      // page it is printed on, so a few of them sit within a few percent of white and
      // vanish.
      var needsCasing = luma(col) > 0.72;
      for (var k = 0; k < p.nbr[u].length; k++) {
        var w = p.nbr[u][k];
        if (w < u || own[w] !== o) continue;
        var la = alpha, ca = alpha;
        if (other && other[u] === o && other[w] === o) {
          if (out) continue;            // shared: the incoming side draws it
          la = ca = 1;
        } else if (other && needsCasing && other[u] && other[u] === other[w] &&
                   luma(p.slotColor[other[u]] || FAINT) > 0.72) {
          // another light chain on the same link: the colour changes, the ink casing around
          // it does not, so the casing is one shared mark, drawn by the outgoing side so it
          // lies beneath both colours
          ca = out ? 1 : 0;
        }
        if (la < MIN_ALPHA && ca < MIN_ALPHA) continue;   // nothing to see
        var b = this._qposSpread(p, w, o, spread, srcOf);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
        if (needsCasing && ca >= MIN_ALPHA) {
          ctx.strokeStyle = ink(0.30 * ca);
          ctx.lineWidth = wide + mc * 0.020;
          ctx.stroke();
        }
        if (la < MIN_ALPHA) continue;
        ctx.strokeStyle = rgbStr(col, la);
        ctx.lineWidth = wide;
        ctx.stroke();
      }
    }
  };

  PSSAPlayer.prototype._qposSpread = function (p, u, slot, spread, srcOf) {
    var q = this.qpos(p, u);
    if (spread === undefined || spread >= 1 || !srcOf) return q;
    var s = srcOf(p, slot);
    if (!s) return q;
    return [lerp(s[0], q[0], spread), lerp(s[1], q[1], spread)];
  };

  PSSAPlayer.prototype._chainsOf = function (p, own) {
    var m = {};
    for (var u = 1; u <= p.nHw; u++) {
      var o = own[u];
      if (!o) continue;
      (m[o] || (m[o] = [])).push(u);
    }
    return m;
  };

  // Logical edges of one cluster, drawn against the current placement.
  // One logical edge's mark under `own`: the King's-adjacent pair that embeds it, or, when
  // it is missing, the closest pair of its two chains.
  var MIN_ALPHA = 1 / 255;   // below one step of 8-bit alpha a mark is invisible
  function markKey(m) {
    var a = m.seg[0], b = m.seg[1];
    return (a < b ? a + '-' + b : b + '-' + a) + (m.embedded ? 'e' : 'm');
  }
  PSSAPlayer.prototype._edgeMark = function (p, own, ch, e) {
    var vi = e[0] + 1, vj = e[1] + 1;
    var ci = ch[vi], cj = ch[vj];
    if (!ci || !cj) return null;
    var pair = null;
    for (var a = 0; a < ci.length && !pair; a++) {
      var u = ci[a], nb = p.nbr[u];
      for (var k = 0; k < nb.length; k++) {
        if (own[nb[k]] === vj) { pair = [u, nb[k]]; break; }
      }
    }
    if (pair) return { seg: pair, embedded: true };
    var seg = null, best = Infinity;
    for (var q = 0; q < ci.length; q++) {
      for (var r = 0; r < cj.length; r++) {
        var x = ci[q], y = cj[r];
        var dd = Math.pow(p.row[x] - p.row[y], 2) +
                 Math.pow(p.col[x] - p.col[y], 2);
        if (dd < best) { best = dd; seg = [x, y]; }
      }
    }
    return seg ? { seg: seg, embedded: false } : null;
  };

  PSSAPlayer.prototype._drawLogicalEdges = function (p, own, alpha, withMissing,
                                                     spread, srcOf, sm, other, out) {
    var ctx = this.ctx, self = this, cell = this._cell();
    var mc = this._markCell();
    var ch = this._chainsOf(p, own);
    // sm = 0 means "still the input graph": every edge of I is drawn as the plain grey line
    // the graph stage drew it as, because at that instant no edge is embedded or missing
    // yet -- the chains have not unfolded, so the question has no answer.
    if (sm === undefined) sm = 1;
    var wEmb = lerp(mc * 0.024, mc * LINK_W, sm);
    var wMiss = lerp(mc * 0.024, mc * MISS_W, sm);
    var aEmb = lerp(0.20, 0.88, sm);
    var cMiss = mix(INK, ALERT, sm), aMiss = lerp(0.20, MISS_A, sm);
    ctx.lineWidth = wEmb;
    // Butt caps, not the round ones _drawChains leaves on the context.
    ctx.lineCap = 'butt';

    // A logical edge's mark is its segment and its kind: every embedded edge is drawn in
    // the same ink and every missing one in the same red, so two states draw the same mark
    // wherever they put the same kind of edge on the same pair of hardware vertices,
    // whichever edge of I it is.
    var otherMarks = null;
    if (other) {
      var chO = this._chainsOf(p, other);
      otherMarks = {};
      p.edges.forEach(function (e) {
        var mo = self._edgeMark(p, other, chO, e);
        if (mo) otherMarks[markKey(mo)] = true;
      });
    }

    p.edges.forEach(function (e) {
      var m = self._edgeMark(p, own, ch, e);
      if (!m) return;
      var seg = m.seg, pair = m.embedded ? seg : null;
      if (!pair && !withMissing) return;

      // During a morph, a mark the other side also draws is drawn once, by the incoming
      // side, at full strength.
      var ea = alpha;
      if (otherMarks && otherMarks[markKey(m)]) {
        if (out) return;
        ea = 1;
      }
      if (ea < MIN_ALPHA) return;

      var s = self._qposSpread(p, seg[0], own[seg[0]], spread, srcOf);
      var t = self._qposSpread(p, seg[1], own[seg[1]], spread, srcOf);
      ctx.save();
      if (pair) {
        // An embedded logical edge is the thing this whole run exists to produce, so it is
        // drawn as the strongest mark on the lattice.
        ctx.strokeStyle = ink(aEmb * ea);
      } else {
        // Fixed strength except during a scripted morph, where `alpha` is the only thing
        // that ever moves it.
        ctx.strokeStyle = rgbStr(cMiss, aMiss * ea);
        ctx.lineWidth = wMiss;
      }
      ctx.beginPath(); ctx.moveTo(s[0], s[1]); ctx.lineTo(t[0], t[1]); ctx.stroke();
      ctx.restore();
    });
  };

  // The rail: every perimeter hardware vertex held by a border super vertex gets a bar in
  // that vertex's colour, so Mode 2's permutations slide the bars around the loop.
  // `other` and `out` as in `_drawChains`: during a morph, a rail mark the two states share
  // (the same border super vertex on the same perimeter vertex, or a plain tick in both) is
  // drawn once, by the incoming side, at full strength, and only marks that change cross-
  // fade.
  PSSAPlayer.prototype._drawRail = function (p, own, alpha, other, out) {
    if (!other && alpha < MIN_ALPHA) return;   // nothing to see, so nothing drawn
    var mc = this._markCell();
    var ctx = this.ctx, cell = this._cell(), dotR = this.dotR();
    var thin = cell < 9;      // in the ring view there is no room for leaders
    for (var u = 1; u <= p.nHw; u++) {
      if (!p.isPerimeter[u]) continue;
      var o = own[u];
      // Position comes off the un-normalized offset, so the mark sits on the square;
      // direction comes off the unit one, so ticks and leaders are the same length on a
      // corner as on a side.
      var rp = this.railPoint(p, u), nrm = p.normalUnit[u];
      var isB = o && p.borderSlot[o];
      if (!isB && thin) continue;
      var ra = alpha;
      if (other) {
        var oo = other[u], isBo = oo && p.borderSlot[oo];
        if (isB ? (isBo && oo === o) : !isBo) {
          if (out) continue;
          ra = 1;
        }
      }
      if (ra < MIN_ALPHA) continue;
      var len = isB ? cell * 0.34 : cell * 0.12;
      var px = -nrm[1], py = nrm[0];
      var col = isB ? (p.slotColor[o] || FAINT) : null;

      if (isB && !thin) {
        // A leader from the hardware vertex out to its mark on the rail.
        var q = this.qpos(p, u), r0 = dotR * 1.95;
        ctx.strokeStyle = rgbStr(col, 0.42 * ra);
        ctx.lineWidth = mc * 0.028;
        ctx.beginPath();
        ctx.moveTo(q[0] + nrm[0] * r0, q[1] + nrm[1] * r0);
        ctx.lineTo(rp[0] - nrm[0] * cell * 0.05, rp[1] - nrm[1] * cell * 0.05);
        ctx.stroke();
      }

      ctx.strokeStyle = isB ? rgbStr(col, 0.95 * ra) : ink(0.22 * ra);
      ctx.lineWidth = isB ? mc * 0.13 : mc * 0.055;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(rp[0] - px * len / 2, rp[1] - py * len / 2);
      ctx.lineTo(rp[0] + px * len / 2, rp[1] + py * len / 2);
      ctx.stroke();

      if (isB && !thin) {   // ring the hardware vertex, matching the matplotlib figures
        var g = this.qpos(p, u);
        ctx.strokeStyle = rgbStr(BORDER, 0.95 * ra);
        ctx.lineWidth = mc * 0.055;
        ctx.beginPath();
        ctx.arc(g[0], g[1], dotR * 1.65, 0, 6.2832); ctx.stroke();
      }
    }
  };

  PSSAPlayer.prototype._drawQubits = function (p, own, prevOwn, t, spread,
                                              srcOf, sm) {
    var ctx = this.ctx, cell = this._cell(), mc = this._markCell();
    // At sm = 0 a hardware vertex is drawn exactly as the vertex it came from.
    var dotR = sm === undefined || sm >= 1 ? this.dotR()
             : lerp(this.vertR(), this.dotR(), sm);
    ctx.lineWidth = mc * 0.022;
    for (var u = 1; u <= p.nHw; u++) {
      var o = own[u], po = prevOwn ? prevOwn[u] : o;
      if (!o && !po) continue;              // unowned on both sides: absent
      // An unowned site resolves to the page, not to a dark chip: on paper "released" has
      // to mean "gone", and anything non-white left behind reads as a hardware vertex that
      // is still owned by something.
      var ca = po ? (p.slotColor[po] || FAINT) : PAPER;
      var cb = o ? (p.slotColor[o] || FAINT) : PAPER;
      var col = (o === po) ? cb : mix(ca, cb, t);
      var a = (!o && po) ? (1 - t) : ((o && !po) ? t : 1);
      if (a < MIN_ALPHA) continue;          // nothing to see, so nothing drawn
      var q = this._qposSpread(p, u, o || po, spread, srcOf);
      ctx.fillStyle = rgbStr(col, a);
      ctx.beginPath(); ctx.arc(q[0], q[1], dotR, 0, 6.2832); ctx.fill();
      // The outline fades with the fill.
      if (cell > 5) {
        ctx.strokeStyle = ink(0.45 * a);
        ctx.stroke();
      }
    }
  };

  // Where a cut edge attaches to a chain: the perimeter hardware vertex the vertex holds if
  // it holds one, and otherwise any hardware vertex of its chain.
  PSSAPlayer.prototype._attachMap = function (p, own, which) {
    var key = '_att' + (which || 'A');
    var m = p[key];
    if (!m || m.length !== p.nodes.length + 2) {
      m = p[key] = new Int16Array(p.nodes.length + 2);
    }
    var i;
    for (i = 0; i < m.length; i++) m[i] = 0;
    for (var u = 1; u <= p.nHw; u++) {
      var o = own[u];
      if (!o || o >= m.length) continue;
      if (!m[o] || (p.isPerimeter[u] && !p.isPerimeter[m[o]])) m[o] = u;
    }
    return m;
  };

  // Screen position of every vertex's attachment point, for every patch, once per frame.
  PSSAPlayer.prototype._buildAttach = function (st, spread, srcOf, t) {
    var self = this, out = {};
    this.patches.forEach(function (p) {
      var own = st.ownOf(p);
      var att = self._attachMap(p, own, 'A');
      var prev = st.prevOwnOf && t < 1
        ? self._attachMap(p, st.prevOwnOf(p), 'B') : null;
      var pos = new Array(p.nodes.length + 2);
      for (var slot = 1; slot <= p.nodes.length; slot++) {
        var u = att[slot];
        if (!u) { pos[slot] = null; continue; }
        var q = self._qposSpread(p, u, slot, spread, srcOf);
        if (prev && prev[slot] && prev[slot] !== u) {
          var r = self._qposSpread(p, prev[slot], slot, spread, srcOf);
          q = [lerp(r[0], q[0], t), lerp(r[1], q[1], t)];
        }
        pos[slot] = q;
      }
      out[p.cid] = pos;
    });
    return out;
  };

  // Cut edges: from the point one border vertex's chain is attached at to the point its
  // partner's chain is attached at.
  PSSAPlayer.prototype._drawCutArcs = function (attach, inkA, skip) {
    var ctx = this.ctx, self = this, cell = this._cell();
    var mc = this._markCell();
    var cross = this.data.graph.cross;
    this.cutLinks = 0;
    if (!cross.length || inkA <= 0.005) return;
    ctx.save();
    ctx.lineCap = 'round';
    // The same weight the graph stage draws a cut edge at, so the handover changes only the
    // ink, and only gradually.
    ctx.lineWidth = mc * 0.030;
    ctx.strokeStyle = ink(inkA);
    // slot lookup: global vertex -> (patch, slot)
    if (!this._slotOf) {
      this._slotOf = {};
      this.patches.forEach(function (p) {
        for (var i = 0; i < p.nodes.length; i++) {
          self._slotOf[p.nodes[i]] = { p: p, slot: i + 1 };
        }
      });
    }
    for (var i = 0; i < cross.length; i++) {
      if (skip && skip[i]) continue;
      var a = this._slotOf[cross[i][0]], b = this._slotOf[cross[i][1]];
      if (!a || !b) continue;
      // An arc with one end off screen still runs off the edge, which is what a cut edge
      // leaving a zoomed patch should look like.
      if (!this._visible(a.p) && !this._visible(b.p)) continue;
      var s = attach[a.p.cid] && attach[a.p.cid][a.slot];
      var e = attach[b.p.cid] && attach[b.p.cid][b.slot];
      if (!s || !e) continue;
      this.cutLinks++;
      var bothOn = this._visible(a.p) && this._visible(b.p);
      ctx.beginPath();
      if (bothOn) {
        // bow outward from the figure's centre, so arcs between neighbouring patches do not
        // all pile onto the same chord
        var mx = (s[0] + e[0]) / 2, my = (s[1] + e[1]) / 2;
        var cxp = this.sx(this.cam.cx), cyp = this.sy(this.cam.cy);
        var dx = mx - cxp, dy = my - cyp, d = Math.hypot(dx, dy) || 1;
        var bow = Math.min(Math.hypot(e[0] - s[0], e[1] - s[1]) * 0.16,
                           this.LATW * 0.06);
        ctx.moveTo(s[0], s[1]);
        ctx.quadraticCurveTo(mx + dx / d * bow, my + dy / d * bow, e[0], e[1]);
      } else {
        // The partner is off screen -- zoomed into one patch, most of them are.
        var on = this._visible(a.p) ? s : e, off = this._visible(a.p) ? e : s;
        ctx.moveTo(on[0], on[1]);
        var q = this._clipToView(on, off);
        ctx.lineTo(q[0], q[1]);
      }
      ctx.stroke();
    }
    ctx.restore();
  };

  // Walk from `from` toward `to` and stop at the edge of the viewport (plus a margin, so
  // the stroke's own width never reveals the cut).
  PSSAPlayer.prototype._onStage = function (q) {
    var m = 24;
    return q[0] >= this.LATX - m && q[0] <= this.LATX + this.LATW + m &&
           q[1] >= this.LATY - m && q[1] <= this.LATY + this.LATH + m;
  };

  PSSAPlayer.prototype._clipToView = function (from, to) {
    var m = 24;
    var x0 = this.LATX - m, x1 = this.LATX + this.LATW + m;
    var y0 = this.LATY - m, y1 = this.LATY + this.LATH + m;
    var dx = to[0] - from[0], dy = to[1] - from[1];
    var t = 1;
    if (dx > 1e-9) t = Math.min(t, (x1 - from[0]) / dx);
    else if (dx < -1e-9) t = Math.min(t, (x0 - from[0]) / dx);
    if (dy > 1e-9) t = Math.min(t, (y1 - from[1]) / dy);
    else if (dy < -1e-9) t = Math.min(t, (y0 - from[1]) / dy);
    t = Math.max(0, Math.min(1, t));
    return [from[0] + dx * t, from[1] + dy * t];
  };

  // The black ring that says "this vertex has an edge leaving its cluster".
  PSSAPlayer.prototype._drawBorderRings = function (p, attach, alpha, sm) {
    if (alpha <= 0.01 || !attach) return;
    var ctx = this.ctx, cell = this._cell(), mc = this._markCell();
    // 1.75 on a vertex and 1.65 on a hardware vertex are the radii the two stages already
    // used; interpolating between them is what keeps the ring the same mark across the
    // handover instead of two marks that swap.
    var dotR = sm === undefined || sm >= 1
      ? this.dotR() * 1.65
      : lerp(this.vertR() * 1.75, this.dotR() * 1.65, sm);
    ctx.strokeStyle = rgbStr(BORDER, 0.9 * alpha);
    ctx.lineWidth = mc * 0.05;
    for (var i = 0; i < p.border.length; i++) {
      var q = attach[p.border[i] + 1];
      if (!q) continue;
      ctx.beginPath();
      ctx.arc(q[0], q[1], dotR, 0, 6.2832);
      ctx.stroke();
    }
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // per-frame rendering: the graph layer
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // The input graph as vertices and edges at interpolated positions.
  PSSAPlayer.prototype._drawGraph = function (st) {
    var ctx = this.ctx, self = this, i;
    if (st.countCuts !== false) this.cutLinks = 0;
    var mixT = st.posMix || 0;
    var alpha = st.gAlpha === undefined ? 1 : st.gAlpha;
    if (alpha <= 0.01) return;

    var pts = new Array(this.nV);
    for (i = 0; i < this.nV; i++) pts[i] = this.vpos(i, mixT);

    var r = this.vertR();
    var cell = this._cell(), mc = this._markCell();

    // Zoomed into a patch, most of the graph is off screen.
    var pad = Math.max(40, r * 4);
    var on = new Uint8Array(this.nV);
    for (i = 0; i < this.nV; i++) {
      on[i] = (pts[i][0] > this.LATX - pad &&
               pts[i][0] < this.LATX + this.LATW + pad &&
               pts[i][1] > this.LATY - pad &&
               pts[i][1] < this.LATY + this.LATH + pad) ? 1 : 0;
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // edges: intra-cluster edges stay straight, cut edges bow as the clusters pull apart
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ctx.lineCap = 'butt';
    for (i = 0; i < this.graphEdges.length; i++) {
      var e = this.graphEdges[i];
      if (!on[e[0]] && !on[e[1]]) continue;
      var la = st.labels ? st.labels[e[0]] : -1;
      var lb = st.labels ? st.labels[e[1]] : -1;
      var cut = (la >= 0 && lb >= 0 && la !== lb);
      if (cut) this.cutLinks++;
      var s = pts[e[0]], t = pts[e[1]];
      ctx.strokeStyle = cut ? ink(CUT_G * alpha) : ink(0.20 * alpha);
      ctx.lineWidth = mc * (cut ? 0.030 : 0.024);
      ctx.beginPath();
      if (!on[e[0]] || !on[e[1]]) {
        // One end is outside the view (the bridge act, drawn while the camera is already
        // inside a patch).
        var from = on[e[0]] ? s : t, to = on[e[0]] ? t : s;
        var q = this._clipToView(from, to);
        ctx.moveTo(from[0], from[1]);
        ctx.lineTo(q[0], q[1]);
      } else if (cut && mixT > 0.02) {
        ctx.moveTo(s[0], s[1]);
        var mx = (s[0] + t[0]) / 2, my = (s[1] + t[1]) / 2;
        var cxp = this.sx(this.cam.cx), cyp = this.sy(this.cam.cy);
        var dx = mx - cxp, dy = my - cyp, d = Math.hypot(dx, dy) || 1;
        var bow = Math.min(Math.hypot(t[0] - s[0], t[1] - s[1]) * 0.16,
                           this.LATW * 0.06) * mixT;
        ctx.quadraticCurveTo(mx + dx / d * bow, my + dy / d * bow, t[0], t[1]);
      } else {
        ctx.moveTo(s[0], s[1]);
        ctx.lineTo(t[0], t[1]);
      }
      ctx.stroke();
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // vertices
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for (i = 0; i < this.nV; i++) {
      if (!on[i]) continue;
      var lab = st.labels ? st.labels[i] : -1;
      var col;
      if (lab < 0) col = FAINT;                              // no partition yet
      else if (st.shadeMix !== undefined) {                  // mid-act
        col = mix(this.clusterCol[lab], this.shade[i], st.shadeMix);
      } else if (st.shaded) col = this.shade[i];             // its own shade
      else col = this.clusterCol[lab];                       // cluster colour
      var p = pts[i];
      ctx.fillStyle = rgbStr(col, alpha);
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 6.2832); ctx.fill();
      if (cell > 5) {
        ctx.strokeStyle = ink(0.45 * alpha);
        ctx.lineWidth = mc * 0.022;
        ctx.stroke();
      }
      // Border vertices get the same ring as on the lattice, so the mark means the same
      // thing across the join.
      if (st.showBorder && st.isBorder && st.isBorder[i]) {
        ctx.strokeStyle = rgbStr(BORDER, 0.9 * alpha);
        ctx.lineWidth = mc * 0.05;
        ctx.beginPath(); ctx.arc(p[0], p[1], r * 1.75, 0, 6.2832); ctx.stroke();
      }
    }

  };

  // Border vertices of the labelling currently on screen: a vertex with at least one
  // neighbour outside its own cluster.
  PSSAPlayer.prototype._borderMask = function (labels) {
    var mask = new Uint8Array(this.nV), i, j;
    for (i = 0; i < this.nV; i++) {
      var a = this.adj[i], li = labels[i];
      if (li < 0) continue;
      for (j = 0; j < a.length; j++) {
        if (labels[a[j]] >= 0 && labels[a[j]] !== li) { mask[i] = 1; break; }
      }
    }
    return mask;
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // the render
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // The routing's lanes and routes.
  // How many cells of one lane are laid in this state.
  PSSAPlayer.prototype._laneLaid = function (ln, st) {
    if (st.nEsc < ln.nEsc) return Math.min(st.nEsc, ln.nEsc);
    if (st.nFun < ln.nFun) return ln.nEsc + Math.min(st.nFun, ln.nFun);
    return ln.nEsc + ln.nFun + Math.min(st.nTrunk, ln.nTrunk);
  };

  PSSAPlayer.prototype._drawStage3 = function (st) {
    var s3 = this.s3;
    if (!s3) return;
    var cell = this._cell(), i, n;

    // Once every lane is fully laid nothing in that layer changes again for a given camera,
    // so it is rasterized once and blitted, the same trick the King's mesh uses.
    var whole = this.zoomCid === null &&
                st.nEsc >= s3.maxEsc && st.nFun >= s3.maxFun &&
                st.nTrunk >= s3.maxTrunk;
    var drawn = 0;
    for (i = 0; i < s3.lanes.length; i++) drawn += this._laneLaid(s3.lanes[i], st);
    this.s3Cells = drawn;

    // Once every trunk is laid every contact exists, so the finished contacts go into the
    // same cached layer and the closing hold costs one blit.
    var layerHasRoutes = whole;

    if (whole) {
      var sig = this._camSig() + '|' + (layerHasRoutes ? 1 : 0);
      if (!this.s3Layer || this.s3LayerSig !== sig) {
        // Same size and scale as the stage canvas: see `_resize`.
        var c = document.createElement('canvas');
        c.width = this.canvas.width;
        c.height = this.canvas.height;
        var lx = c.getContext('2d');
        lx.setTransform(this.bsx, 0, 0, this.bsy, 0, 0);
        this._strokeLanes(lx, st, cell);
        if (layerHasRoutes) this._strokeRoutes(lx, st, cell);
        this.s3Layer = c;
        this.s3LayerSig = sig;
      }
      this.ctx.drawImage(this.s3Layer, 0, 0, this.W, this.H);
    } else {
      this.s3Layer = null;
      this._strokeLanes(this.ctx, st, cell);
    }

    if (layerHasRoutes) {
      this.s3Routes = s3.routes.length;
      return;
    }

    // The contacts, while the chains are still arriving at them, so the layer cannot be
    // cached yet.
    this.s3Routes = st.nTrunk > 0 ? this._strokeRoutes(this.ctx, st, cell) : 0;
  };

  // The cut edges, drawn where they are actually realized.
  PSSAPlayer.prototype._strokeRoutes = function (ctx, st, cell) {
    var mc = this._markCell();
    var s3 = this.s3, done = 0, i;
    var k = this.cam.s;
    var ox = this.LATX + this.LATW / 2 - this.cam.cx * k;
    var oy = this.LATY + this.LATH / 2 + this.cam.cy * k;
    ctx.save();
    ctx.beginPath();
    ctx.rect(this.LATX, this.LATY, this.LATW, this.LATH);
    ctx.clip();
    ctx.lineCap = 'butt';
    ctx.strokeStyle = ink(0.88);
    ctx.lineWidth = mc * LINK_W;
    // A contact is one cell long, so a segment with neither end in view has nothing of
    // itself inside it: unlike a cut arc, there is no long span to clip against the edge of
    // the frame.
    var vx0 = this.LATX - cell, vx1 = this.LATX + this.LATW + cell;
    var vy0 = this.LATY - cell, vy1 = this.LATY + this.LATH + cell;
    ctx.beginPath();
    for (i = 0; i < s3.routes.length; i++) {
      var r = s3.routes[i];
      if (st.nTrunk < r.need) continue;
      done++;
      var a = 2 * (r.split - 1), b = 2 * r.split;
      var ax = ox + r.w[a] * k, ay = oy - r.w[a + 1] * k;
      var bx = ox + r.w[b] * k, by = oy - r.w[b + 1] * k;
      if ((ax < vx0 || ax > vx1 || ay < vy0 || ay > vy1) &&
          (bx < vx0 || bx > vx1 || by < vy0 || by > vy1)) continue;
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
    }
    ctx.stroke();
    ctx.restore();
    return done;
  };

  // The viewport-clipped polyline walker, emitting one `lineTo` per straight run rather
  // than one per cell.
  PSSAPlayer.prototype._s3Clip = function (ctx, cell) {
    var k = this.cam.s;
    var ox = this.LATX + this.LATW / 2 - this.cam.cx * k;
    var oy = this.LATY + this.LATH / 2 + this.cam.cy * k;
    var vx0 = this.LATX - cell, vx1 = this.LATX + this.LATW + cell;
    var vy0 = this.LATY - cell, vy1 = this.LATY + this.LATH + cell;
    return function (w, n) {
      var open = false, px = 0, py = 0, pin = false, j, X, Y, cin;
      // The straight run in hand: its heading, and whether one is open.
      var run = false, rux = 0, ruy = 0, ux, uy;
      for (j = 0; j < n; j++) {
        X = ox + w[2 * j] * k; Y = oy - w[2 * j + 1] * k;
        cin = X >= vx0 && X <= vx1 && Y >= vy0 && Y <= vy1;
        if (j > 0 && (cin || pin)) {
          if (!open) { ctx.moveTo(px, py); open = true; run = false; }
          ux = w[2 * j] - w[2 * j - 2]; uy = w[2 * j + 1] - w[2 * j - 1];
          if (run && ux === rux && uy === ruy) {
            // the same heading: the run reaches one cell further, no point
          } else {
            if (run) ctx.lineTo(px, py);      // the run that just ended
            rux = ux; ruy = uy; run = true;
          }
        } else if (j > 0) {
          if (run) { ctx.lineTo(px, py); run = false; }
          open = false;
        }
        px = X; py = Y; pin = cin;
      }
      if (run) ctx.lineTo(px, py);            // the run the lane ends on
    };
  };

  // Every lane, as far as this state has laid it, onto whichever context.
  PSSAPlayer.prototype._strokeLanes = function (ctx, st, cell) {
    var mc = this._markCell();
    var s3 = this.s3, i, n;
    ctx.save();
    ctx.beginPath();
    ctx.rect(this.LATX, this.LATY, this.LATW, this.LATH);
    ctx.clip();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = mc * LINK_W;
    // How lane strokes combine where they overlap.
    ctx.globalCompositeOperation = PAL.routeComposite;
    var clip = this._s3Clip(ctx, cell);
    for (i = 0; i < s3.lanes.length; i++) {
      var ln = s3.lanes[i];
      n = this._laneLaid(ln, st);
      if (n <= 0) continue;
      ctx.strokeStyle = rgbStr(ln.col, PAL.routeAlpha);
      ctx.beginPath();
      clip(ln.w, n + 1);           // n cells, and the dock they leave from
      ctx.stroke();
    }
    ctx.restore();
  };

  // The King's mesh, as the first thing on a lattice frame.
  PSSAPlayer.prototype._blitMesh = function (ctx, st, wantMesh) {
    if (!wantMesh) {
      // nothing to blit: the routing pulls the camera back until the mesh is finer than a pixel
    } else if (st.meshAlpha === undefined || st.meshAlpha >= 1) {
      ctx.drawImage(this.mesh, 0, 0, this.W, this.H);
    } else {
      ctx.save();
      ctx.globalAlpha = st.meshAlpha;
      ctx.drawImage(this.mesh, 0, 0, this.W, this.H);
      ctx.restore();
    }
  };

  // One patch: its chains, logical edges, hardware vertices, rail and border rings, onto
  // `this.ctx`.
  PSSAPlayer.prototype._drawPatch = function (p, st, t, spread, srcOf, attach, sm) {
    var own = st.ownOf(p), prevOwn = st.prevOwnOf ? st.prevOwnOf(p) : null;
    // A morph: what the two states share is drawn once at full strength by the incoming
    // side, and only what differs cross-fades, the outgoing side's missing edges included,
    // so the act's first frame is the last frame before it and its last frame is the next.
    var morphing = !!(prevOwn && t < 1);
    if (morphing) {
      this._drawChains(p, prevOwn, 1 - t, spread, srcOf, own, true);
      if (st.edgesPrev) {
        this._drawLogicalEdges(p, prevOwn, 1 - t, true, spread, srcOf, sm,
                               own, true);
      }
    }
    this._drawChains(p, own, prevOwn ? t : 1, spread, srcOf,
                     morphing ? prevOwn : null, false);
    if (st.edges) {
      this._drawLogicalEdges(p, own, prevOwn ? t : 1, true, spread, srcOf, sm,
                             morphing && st.edgesPrev ? prevOwn : null, false);
    }
    this._drawQubits(p, own, prevOwn, t, spread, srcOf, sm);
    if (st.rail) {
      if (morphing && !st.railIn && st.railA === undefined) {
        // both states have a rail: shared marks once, changes cross-fade
        this._drawRail(p, prevOwn, 1 - t, own, true);
        this._drawRail(p, own, t, prevOwn, false);
      } else {
        this._drawRail(p, own, st.railA !== undefined ? st.railA
          : (st.railIn ? clamp01((t - RAIL_IN_FROM) / (1 - RAIL_IN_FROM)) : 1));
      }
    }
    this._drawBorderRings(p, attach[p.cid], 1, sm);
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // the base layer: the mesh and the patches, kept between frames
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  PSSAPlayer.prototype._newLayer = function () {
    return document.createElement('canvas');
  };

  // Device-pixel rectangle that holds every mark patch `p` can draw at this camera.
  PSSAPlayer.prototype._patchRect = function (p) {
    var cell = this._cell(), mc = this._markCell();
    var ext = Math.max(0.60 * cell + 0.065 * mc, 0.36 * mc) * 1.05;
    var xa = this.sx(p.origin[1] + p.dx), xb = this.sx(p.origin[1] + p.L - 1 + p.dx);
    var ya = this.sy(-p.origin[0] + p.dy), yb = this.sy(-(p.origin[0] + p.L - 1) + p.dy);
    var bw = this.canvas.width, bh = this.canvas.height;
    var x0 = Math.max(0, Math.floor((Math.min(xa, xb) - ext) * this.bsx) - 2);
    var x1 = Math.min(bw, Math.ceil((Math.max(xa, xb) + ext) * this.bsx) + 2);
    var y0 = Math.max(0, Math.floor((Math.min(ya, yb) - ext) * this.bsy) - 2);
    var y1 = Math.min(bh, Math.ceil((Math.max(ya, yb) + ext) * this.bsy) + 2);
    return [x0, y0, Math.max(0, x1 - x0), Math.max(0, y1 - y0)];
  };

  // Draw the mesh and every patch through the base layer and blit it onto the stage.
  PSSAPlayer.prototype._drawBase = function (st, t, attach, sm, wantMesh) {
    if (!this.baseLayer) return false;
    if (st.spread !== undefined && st.spread < 1) return false;
    if (st.prevOwnOf && t < 1) return false;

    var cam = this.cam, cv = this.canvas, i, p;
    var mAlpha = st.meshAlpha === undefined ? 1 : st.meshAlpha;
    var key = [cv.width, cv.height, this.W, this.H, this.bsx, this.bsy,
               this.LATX, this.LATY, this.LATW, this.LATH, cam.cx, cam.cy, cam.s,
               this.palGen, wantMesh ? this.meshGen : -1, wantMesh ? mAlpha : -1,
               t, sm, st.edges, st.edgesPrev, st.rail, st.railIn, st.railA];
    for (i = 0; i < this.patches.length; i++) {
      key.push(this.patches[i].dx, this.patches[i].dy);
    }
    key = key.join();

    var n = this.patches.length;
    var own = new Array(n), prev = new Array(n);
    for (i = 0; i < n; i++) {
      p = this.patches[i];
      own[i] = st.ownOf(p);
      prev[i] = st.prevOwnOf ? st.prevOwnOf(p) : null;
    }

    var B = this.base, redraw = [];
    var full = !B || B.key !== key;
    if (!full) {
      for (i = 0; i < n; i++) {
        if (B.own[i] !== own[i] || B.prev[i] !== prev[i]) redraw.push(i);
      }
      if (redraw.length && !B.disjoint) full = true;
    }

    var main = this.ctx;
    if (full) {
      if (!B) {
        B = this.base = { canvas: this._newLayer() };
        B.ctx = B.canvas.getContext('2d');
      }
      if (B.canvas.width !== cv.width || B.canvas.height !== cv.height) {
        B.canvas.width = cv.width;
        B.canvas.height = cv.height;
      }
      var x = B.ctx;
      x.setTransform(1, 0, 0, 1, 0, 0);
      x.clearRect(0, 0, cv.width, cv.height);
      x.setTransform(this.bsx, 0, 0, this.bsy, 0, 0);
      this._blitMesh(x, st, wantMesh);
      // which patches are drawn, and where each one's marks can reach
      B.vis = new Array(n); B.rect = new Array(n);
      for (i = 0; i < n; i++) {
        B.vis[i] = this._visible(this.patches[i]);
        B.rect[i] = B.vis[i] ? this._patchRect(this.patches[i]) : null;
      }
      B.disjoint = true;
      for (i = 0; i < n && B.disjoint; i++) {
        for (var j = i + 1; j < n; j++) {
          var a = B.rect[i], b = B.rect[j];
          if (a && b && a[0] < b[0] + b[2] && b[0] < a[0] + a[2] &&
              a[1] < b[1] + b[3] && b[1] < a[1] + a[3]) { B.disjoint = false; break; }
        }
      }
      redraw = [];
      for (i = 0; i < n; i++) if (B.vis[i]) redraw.push(i);
    }

    if (redraw.length) {
      this.ctx = B.ctx;
      try {
        for (var k = 0; k < redraw.length; k++) {
          i = redraw[k];
          if (!B.vis[i]) continue;
          if (!full) {
            // back to the mesh, in this patch's rectangle only
            var r = B.rect[i], bx = B.ctx;
            bx.save();
            bx.setTransform(1, 0, 0, 1, 0, 0);
            bx.clearRect(r[0], r[1], r[2], r[3]);
            if (wantMesh && r[2] > 0 && r[3] > 0) {
              if (mAlpha < 1) bx.globalAlpha = mAlpha;
              bx.drawImage(this.mesh, r[0], r[1], r[2], r[3], r[0], r[1], r[2], r[3]);
            }
            bx.restore();
          }
          this._drawPatch(this.patches[i], st, t, st.spread, null, attach, sm);
        }
      } finally {
        this.ctx = main;
      }
    }
    B.key = key; B.own = own; B.prev = prev;

    main.drawImage(B.canvas, 0, 0, this.W, this.H);
    return true;
  };

  PSSAPlayer.prototype._render = function (st) {
    this.lastDraw = st;
    this._checkDpr();
    var ctx = this.ctx, self = this;
    ctx.clearRect(0, 0, this.W, this.H);

    var t = st.t === undefined ? 1 : easeInOut(st.t);

    if (st.space === 'graph') {
      this._drawGraph(st);
      return;
    }

    // lattice and bridge: the mesh, then every patch
    var wantMesh = st.meshAlpha === undefined || st.meshAlpha > 0.01;
    if (wantMesh && (!this.mesh || this.meshSig !== this._camSig())) {
      this._drawMesh();
    }

    var spread = st.spread;
    var srcOf = null;
    if (spread !== undefined && spread < 1) {
      srcOf = function (p, slot) {
        if (!slot) return null;
        var gi = p.nodes[slot - 1];
        if (gi === undefined) return null;
        var b = self.pos1[gi];
        return [self.sx(b[0]), self.sy(b[1])];
      };
    }

    // Where every vertex's cut edges and border ring attach, this frame.
    var attach = this._buildAttach(st, spread, srcOf, t);
    // How far this state has moved from drawing the input graph to drawing the hardware.
    var sm = st.styleMix === undefined ? 1 : st.styleMix;

    // The mesh and the patches: from the base layer when it can hold them, otherwise
    // straight onto the stage, as they always were.
    if (!this._drawBase(st, t, attach, sm, wantMesh)) {
      this._blitMesh(ctx, st, wantMesh);
      this.patches.forEach(function (p) {
        if (!self._visible(p)) return;
        self._drawPatch(p, st, t, spread, srcOf, attach, sm);
      });
    }

    // The lanes and routes sit between the tiles and the cut arcs, in the order
    // draw_fallback rasterizes them.
    if (st.space === 's3') this._drawStage3(st);

    if (st.arcs) this._drawCutArcs(attach, lerp(CUT_G, st.arcs, sm), st.arcSkip);
    // A cut edge whose arc has retired is on screen as its route instead, so the running
    // count is arcs plus completed routes.
    if (st.space === 's3') this.cutLinks += this.s3Routes || 0;

    // The patch under the pointer gets a hairline box, so "these are clickable" is
    // discoverable without a legend telling anyone so.
    if (this.hoverCid !== null && this.hoverCid !== undefined &&
        this.zoomCid === null) {
      var hp = this.patchOf[this.hoverCid];
      if (hp) {
        ctx.strokeStyle = ink(0.34);
        ctx.lineWidth = 1;
        var m = this.railOffset() * 1.6;
        var hb = this._box(hp);
        ctx.strokeRect(this.sx(hb.x0) - m, this.sy(hb.y1) - m,
                       (hb.x1 - hb.x0) * this.cam.s + 2 * m,
                       (hb.y1 - hb.y0) * this.cam.s + 2 * m);
      }
    }
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // diagnostic plot
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // PLOT_T, PLOT_E and PAPER_STR are declared with the rest of the surface palette at the
  // top of this file, so a theme switch reaches them.

  PSSAPlayer.prototype._plotGeom = function () {
    var W = this.PW, H = this.PH;
    var padL = 56, padR = 56, padT = 36, padB = 40;
    return { W: W, H: H, padL: padL, padR: padR, padT: padT, padB: padB,
             pw: W - padL - padR, ph: H - padT - padB };
  };

  PSSAPlayer.prototype._xOf = function (g, t) {
    var tEnd = Math.max(this._tEndPlot(), 1);
    var f = this.xmode === 'lin'
      ? t / tEnd
      : Math.log(1 + Math.max(t, 0)) / Math.log(1 + tEnd);
    return g.padL + g.pw * clamp01(f);
  };

  PSSAPlayer.prototype._tEndPlot = function () {
    return this.zoomCid === null ? this.meta.t_end_max
                                 : this.patchOf[this.zoomCid].tEnd;
  };
  PSSAPlayer.prototype._mPlot = function () {
    return this.zoomCid === null ? this.mTotal : this.patchOf[this.zoomCid].mI;
  };

  PSSAPlayer.prototype._yE = function (g, e) {
    return g.padT + g.ph * (1 - e / Math.max(this._mPlot(), 1));
  };

  PSSAPlayer.prototype._yT = function (g, T) {
    if (this.ymode === 'lin') {
      return g.padT + g.ph * (1 - Math.max(0, T) / this.tLinTop);
    }
    var a = Math.log(this.tLo), b = Math.log(this.tHi);
    return g.padT + g.ph * (1 - (Math.log(T) - a) / (b - a));
  };

  PSSAPlayer.prototype._plotKey = function () {
    return this.PW + 'x' + this.PH + ':' + this.xmode + this.ymode + ':' +
           this.zoomCid;
  };

  // A label set in the report's fonts: italic symbols, upright subscripts, and words in the
  // text font `txt`.
  var MATH_FONT = "'STIX Two Text', 'Times New Roman', serif";
  function mathText(x, segs, x0, y0, size, txt, measure) {
    var w = 0, i, base = x.textBaseline;
    for (i = 0; i < segs.length; i++) {
      var t = segs[i][0], st = segs[i][1], dy = 0;
      if (st === 'text') x.font = txt;
      else if (st === 'i') x.font = 'italic ' + size + 'px ' + MATH_FONT;
      else if (st === 'r') x.font = size + 'px ' + MATH_FONT;
      else {
        x.font = (st === 'sup' ? 'italic ' : '') + Math.round(size * 0.7) + 'px ' + MATH_FONT;
        dy = st === 'sub' ? size * 0.28 : -size * 0.38;
      }
      if (!measure) {
        x.textAlign = 'left';
        x.textBaseline = base;
        x.fillText(t, x0 + w, y0 + dy);
      }
      w += x.measureText(t).width + (st === 'i' ? size * 0.06 : 0);
    }
    x.font = txt;
    return w;
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // the cluster embedding's plot
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PSSAPlayer.prototype._buildPlotBg = function () {
    var g = this._plotGeom(), self = this;
    // The plot canvas's own backing size and scale, for the same reason the stage layers
    // take the stage's: a layer blitted at anything but one to one resamples every hairline
    // on it.
    var c = document.createElement('canvas');
    c.width = this.plotEl.width; c.height = this.plotEl.height;
    var x = c.getContext('2d');
    x.setTransform(this.plotEl.width / g.W, 0, 0, this.plotEl.height / g.H, 0, 0);

    var txt = '13px ' + MATH_FONT;          // legend
    var txtS = '12px ' + MATH_FONT;         // ticks and axis labels
    var tEnd = Math.max(this._tEndPlot(), 1);
    var m = this._mPlot();

    x.font = txtS; x.textBaseline = 'middle'; x.textAlign = 'right';
    var eStep = eStepFor(m, g.ph);
    for (var e = 0; e <= m; e += eStep) {
      var y = this._yE(g, e);
      x.strokeStyle = ink(0.07); x.lineWidth = 1;
      x.beginPath();
      x.moveTo(g.padL, Math.round(y) + 0.5);
      x.lineTo(g.padL + g.pw, Math.round(y) + 0.5);
      x.stroke();
      x.fillStyle = muted(0.85);
      x.fillText(kfmt(e), g.padL - 7, y);
    }

    x.textBaseline = 'middle';

    // right (temperature) ticks
    x.textAlign = 'left';
    var linT = this.ymode === 'lin';
    var tt = linT ? this.tLinTicks : decadeTicks(this.tLo, this.tHi);
    var tdec = linT
      ? Math.max(0, -Math.floor(Math.log(this.tLinStep) / Math.LN10 + 1e-9)) : 1;
    for (var i = 0; i < tt.length; i++) {
      var yv = this._yT(g, tt[i] || this.tLo);
      x.strokeStyle = rgbStr(hexToRgb(PLOT_T), 0.16);
      x.beginPath();
      x.moveTo(g.padL + g.pw, Math.round(yv) + 0.5);
      x.lineTo(g.padL + g.pw + 4, Math.round(yv) + 0.5);
      x.stroke();
      x.fillStyle = rgbStr(hexToRgb(PLOT_T), 0.95);
      x.fillText(linT ? tt[i].toFixed(tdec)
                      : (tt[i] < 1 ? tt[i].toFixed(1) : String(tt[i])),
                 g.padL + g.pw + 8, yv);
    }

    // x ticks
    x.textAlign = 'center'; x.textBaseline = 'top';
    var xs = [];
    if (this.xmode === 'lin') {
      var stepT = Math.pow(10, Math.floor(Math.log(tEnd) / Math.LN10));
      if (tEnd / stepT < 3) stepT /= 2;
      for (var v = 0; v <= tEnd * 1.0001; v += stepT) xs.push(v);
    } else {
      xs.push(0);
      for (var p = 0; Math.pow(10, p) <= tEnd; p++) xs.push(Math.pow(10, p));
    }
    xs.push(tEnd);
    for (i = 0; i < xs.length; i++) {
      var px = this._xOf(g, xs[i]);
      if (i < xs.length - 1 && Math.abs(px - this._xOf(g, tEnd)) < 26) continue;
      x.strokeStyle = ink(0.12);
      x.beginPath();
      x.moveTo(Math.round(px) + 0.5, g.padT);
      x.lineTo(Math.round(px) + 0.5, g.padT + g.ph + 4);
      x.stroke();
      x.fillStyle = i === xs.length - 1 ? muted(0.95) : muted(0.75);
      x.fillText(xs[i] === 0 ? '0' : kfmt(xs[i]), px, g.padT + g.ph + 8);
    }

    x.strokeStyle = ink(0.26); x.lineWidth = 1;
    x.beginPath();
    x.moveTo(g.padL + 0.5, g.padT);
    x.lineTo(g.padL + 0.5, g.padT + g.ph + 0.5);
    x.lineTo(g.padL + g.pw + 0.5, g.padT + g.ph + 0.5);
    x.stroke();

    // the faint full-run curves
    this._plotSeries(x, g, tEnd, 0.20);

    x.textBaseline = 'bottom';
    x.fillStyle = faint(1);
    var axT = [['annealing step ', 'text'], ['t', 'i'],
               [this.xmode === 'lin' ? '   (linear scale)' : '   (log scale)', 'text']];
    mathText(x, axT, g.padL + g.pw / 2 - mathText(x, axT, 0, 0, 13, txtS, true) / 2,
             g.H - 4, 13, txtS);

    x.save();
    x.translate(8, g.padT + g.ph / 2);
    x.rotate(-Math.PI / 2);
    x.textBaseline = 'top';
    x.fillStyle = rgbStr(hexToRgb(PLOT_E), 0.95);
    var axE = [['E', 'i'], ['emb', 'sub'], ['  (logical edges embedded)', 'text']];
    x.translate(-mathText(x, axE, 0, 0, 13, txtS, true) / 2, 0);   // centred
    mathText(x, axE, 0, -1, 13, txtS);
    x.restore();

    x.save();
    x.translate(g.W - 5, g.padT + g.ph / 2);
    x.rotate(Math.PI / 2);
    x.textAlign = 'center'; x.textBaseline = 'top';
    x.fillStyle = rgbStr(hexToRgb(PLOT_T), 0.95);
    x.fillText(this.ymode === 'lin' ? 'temperature   (linear scale)'
                                    : 'temperature   (log scale)', 0, 0);
    x.restore();

    // legend
    x.font = txt; x.textAlign = 'left'; x.textBaseline = 'middle';
    var lx = g.padL + 2, ly = 13;
    function key(col, segs) {
      x.strokeStyle = col; x.lineWidth = 2;
      x.beginPath(); x.moveTo(lx, ly); x.lineTo(lx + 14, ly); x.stroke();
      x.fillStyle = col;
      lx += 19 + mathText(x, segs, lx + 19, ly + 0.5, 14, txt) + 18;
    }
    var Tt = [['T', 'i'], ['(', 'r'], ['t', 'i'], [')', 'r']];
    key(PLOT_T, this.zoomCid === null ? Tt.concat([[' ∀ ', 'r'], ['clusters', 'text']])
      : Tt.concat([[' = ', 'r'], ['T', 'i'], ['0', 'sub'], [' · ', 'r'],
                   ['α', 'i'], ['t', 'sup']]));
    var Et = [['E', 'i'], ['emb', 'sub'], ['(', 'r'], ['t', 'i'], [')', 'r']];
    key(PLOT_E, this.zoomCid === null ? [['Σ', 'r']].concat(Et) : Et);

    this.plotBg = c;
    this._plotBgKey = this._plotKey();
    return c;
  };

  // Both series, drawn up to annealing step `tNow`.
  PSSAPlayer.prototype._plotSeries = function (x, g, tNow, alpha) {
    var self = this, i, px;
    var xEnd = this._xOf(g, tNow);
    var tEnd = Math.max(this._tEndPlot(), 1);
    var steps = Math.max(2, Math.round(g.pw));

    // temperature is analytic, so sample it per pixel column rather than per frame: smooth
    // in both axis modes and independent of frame density.
    var list = this.zoomCid === null ? this.patches : [this.patchOf[this.zoomCid]];
    var many = list.length > 1;
    x.lineWidth = alpha > 0.5 ? (many ? 1.0 : 1.8) : 1.0;
    // Every cluster's schedule goes into one path and is stroked once.
    x.strokeStyle = rgbStr(hexToRgb(PLOT_T), alpha * (many ? 0.55 : 1));
    x.beginPath();
    for (var pi = 0; pi < list.length; pi++) {
      var p = list[pi], started = false;
      for (i = 0; i <= steps; i++) {
        px = g.padL + (g.pw * i) / steps;
        if (px > xEnd + 0.5) break;
        var frac = i / steps;
        var t = this.xmode === 'lin' ? frac * tEnd
                                     : Math.pow(1 + tEnd, frac) - 1;
        if (t > p.tEnd) break;           // this cluster's run has ended
        var T = p.T0 * Math.pow(p.alpha, t);
        var py = this._yT(g, Math.max(T, this.tLo));
        if (started) { x.lineTo(px, py); }
        else { x.moveTo(px, py); started = true; }
      }
    }
    x.stroke();

    // E_emb is data.
    x.strokeStyle = rgbStr(hexToRgb(PLOT_E), alpha);
    x.lineWidth = alpha > 0.5 ? 1.8 : 1.2;
    x.beginPath();
    if (this.zoomCid === null) {
      x.moveTo(this._xOf(g, 0), this._yE(g, this.eStartTotal));
      for (i = 0; i < this.aggT.length; i++) {
        if (this.aggT[i] > tNow) break;
        x.lineTo(this._xOf(g, this.aggT[i]), this._yE(g, this.aggE[i]));
      }
    } else {
      // A cluster can have no frames at all: if its initial placement is already a legal
      // embedding the anneal accepts nothing and the recorder writes an empty `frames` with
      // `t_end` 0.
      var q = this.patchOf[this.zoomCid];
      x.moveTo(this._xOf(g, 0), this._yE(g, q.eStart));
      if (q.frames.length && this._frameAtT(q, tNow) >= 0) {
        var upto = this._frameAtT(q, tNow);
        var step = Math.max(1, Math.floor(q.frames.length /
                                          Math.max(g.pw * 1.5, 1)));
        for (i = 0; i <= upto; i += step) {
          x.lineTo(this._xOf(g, q.frames[i].t), this._yE(g, q.frames[i].e));
        }
        x.lineTo(this._xOf(g, q.frames[upto].t), this._yE(g, q.frames[upto].e));
      }
    }
    x.stroke();
  };

  // The plot is the cluster embedding's, for the whole piece.
  PSSAPlayer.prototype._drawPlot = function (tNow) {
    var x = this.plotCtx;
    if (!x || !this.PW || this.PW < 120) return;
    var g = this._plotGeom();
    if (g.pw < 40 || g.ph < 40) return;
    if (this.plotTitle) {
      this.plotTitle.innerHTML = 'Annealing schedule and cost function ' +
        '<span class="pssa-math">(<i>E</i><sub>emb</sub>)</span>';
    }
    if (!this.plotBg || this._plotBgKey !== this._plotKey()) this._buildPlotBg();

    // Everything the plot is drawn from, apart from the recording itself.
    var fonts = global.document && global.document.fonts;
    var fst = fonts ? fonts.status : '';
    var key = [tNow, this.plotEl.width, this.plotEl.height, this._plotBgKey,
               this.palGen, fst, x.lineJoin, x.lineCap, x.miterLimit,
               x.globalAlpha, x.globalCompositeOperation].join();
    if (fst !== 'loading' && this.plotBg === this._plotDrawnBg &&
        key === this._plotDrawnKey) return;
    this._plotDrawnBg = this.plotBg;
    this._plotDrawnKey = key;

    x.clearRect(0, 0, g.W, g.H);
    x.drawImage(this.plotBg, 0, 0, g.W, g.H);
    if (tNow === null || tNow === undefined) return;

    this._plotSeries(x, g, tNow, 0.95);

    // playhead
    var px = this._xOf(g, tNow);
    x.strokeStyle = ink(0.55); x.lineWidth = 1;
    x.beginPath();
    x.moveTo(Math.round(px) + 0.5, g.padT);
    x.lineTo(Math.round(px) + 0.5, g.padT + g.ph);
    x.stroke();

    var eNow, Tnow;
    if (this.zoomCid === null) {
      eNow = this._aggEAt(tNow);
      var live = this.patches.filter(function (p) { return tNow <= p.tEnd; });
      Tnow = live.length
        ? live.reduce(function (a, p) {
            return a + p.T0 * Math.pow(p.alpha, tNow);
          }, 0) / live.length
        : this.tLo;
    } else {
      var q = this.patchOf[this.zoomCid];
      var qi = q.frames.length ? this._frameAtT(q, tNow) : -2;
      eNow = qi >= 0 ? q.frames[qi].e : (qi === -1 ? q.eStart : q.eFinal);
      Tnow = q.T0 * Math.pow(q.alpha, Math.min(tNow, q.tEnd));
    }

    var yE = this._yE(g, eNow), yT = this._yT(g, Math.max(Tnow, this.tLo));
    x.fillStyle = PLOT_E;
    x.beginPath(); x.arc(px, yE, 2.8, 0, 6.2832); x.fill();
    x.fillStyle = PLOT_T;
    x.beginPath(); x.arc(px, yT, 2.8, 0, 6.2832); x.fill();

    // Value labels, flipped to whichever side has room.
    x.font = '12px ' + MATH_FONT;
    x.textBaseline = 'middle';
    var right = px < g.padL + g.pw * 0.72;
    x.textAlign = right ? 'left' : 'right';
    var dx = right ? 7 : -7;
    // Above its dot, or below it when there is no room above: clamping it into the plot
    // instead pushed a label near the top down onto its own curve.
    var lo = g.padT + 8, hi = g.padT + g.ph - 8;
    function place(y) { return y - 12 >= lo ? y - 12 : Math.min(hi, y + 12); }
    var lE = place(yE), lT = place(yT);
    if (Math.abs(lE - lT) < 14) {
      var mid = (lE + lT) / 2;
      lE = mid + (yE <= yT ? -8 : 8);
      lT = mid + (yE <= yT ? 8 : -8);
    }
    lE = Math.max(lo, Math.min(hi, lE));
    lT = Math.max(lo, Math.min(hi, lT));
    // A paper-coloured halo under each, so a label stays legible wherever a curve or
    // gridline runs behind it.
    x.lineJoin = 'round';
    x.lineWidth = 3;
    x.strokeStyle = rgbStr(PAPER, 0.92);
    var sE = kfmt(eNow), sT = Tnow.toFixed(Tnow < 1 ? 3 : 2);
    x.strokeText(sE, px + dx, lE);
    x.fillStyle = PLOT_E;
    x.fillText(sE, px + dx, lE);
    x.strokeText(sT, px + dx, lT);
    x.fillStyle = PLOT_T;
    x.fillText(sT, px + dx, lT);
  };

  PSSAPlayer.prototype._aggEAt = function (t) {
    var n = this.aggT.length - 1;
    var i = Math.round(clamp01(t / Math.max(this.meta.t_end_max, 1)) * n);
    return this.aggE[Math.max(0, Math.min(n, i))];
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // scrub track
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PSSAPlayer.prototype._drawScrubTrack = function () {
    var x = this.scrubCtx;
    if (!x || !this.BW) return;
    var W = this.BW, H = this.BH;
    x.clearRect(0, 0, W, H);
    var barH = 6, y0 = (H - barH) / 2;
    var total = this.totalMs || 1;

    x.fillStyle = ink(0.07);
    x.fillRect(0, y0, W, barH);
    var a0 = this.stageStartMs[this.annealStage];
    var xa = W * a0 / total, span = W * this.annealMs / total;
    x.fillStyle = ink(0.13);                 // the anneal, a shade darker
    x.fillRect(xa, y0, span, barH);
    if (span > 4) {
      x.beginPath();
      x.moveTo(xa, y0 + barH);
      for (var i = 0; i < this.aggT.length; i++) {
        var px = xa + span * (this.aggT[i] / Math.max(this.meta.t_end_max, 1));
        x.lineTo(px, y0 + barH - (this.aggE[i] / this.mTotal) * barH);
      }
      x.lineTo(xa + span, y0 + barH);
      x.closePath();
      x.fillStyle = rgbStr(hexToRgb(PLOT_E), 0.42);
      x.fill();
    }

    x.strokeStyle = rgbStr(PAPER, 0.9);
    x.lineWidth = 1;
    this.segments.forEach(function (s) {
      if (s.from <= 0) return;
      var px = Math.round(W * s.from / total) + 0.5;
      x.beginPath(); x.moveTo(px, y0 - 1); x.lineTo(px, y0 + barH + 1); x.stroke();
    });
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // hud
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PSSAPlayer.prototype._setPhase = function (label) {
    if (this.phaseLabel.textContent !== label) {
      this.phaseLabel.textContent = label;
      if (this.onPhase) this.onPhase(label);
    }
  };

  // The plot is the one readout left: drawn up to the frame's step, to the end of the
  // anneal once routing is on screen, and empty before it.
  PSSAPlayer.prototype._updateHud = function (st) {
    var t = st.space === 's3' ? this.meta.t_end_max : st.tNow;
    // Kept for `_resize`, which redraws the plot at the step on screen.
    this.hudFrame = t === undefined ? null : t;
    this._drawPlot(this.hudFrame);
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // resolve the clock to a drawn state
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PSSAPlayer.prototype._apply = function (now, seeking) {
    var script = this.script, s = 0, self = this;
    while (s < script.length - 1 && this.clock >= this.stageStartMs[s + 1]) s++;
    this.stage = s;
    var st = script[s];
    var local = this.clock - this.stageStartMs[s];
    // The routing's displacement is per-frame draw state.
    if (st.space !== 's3') {
      for (var pi = 0; pi < this.patches.length; pi++) {
        this.patches[pi].dx = 0;
        this.patches[pi].dy = 0;
      }
    }
    this._camNow(now);

    var draw;

    if (st.space === 's3') {
      draw = this._stage3State(st, local);
      this._setPhase(st.label);
    } else if (st.space === 'graph') {
      draw = this._graphState(st, local);
      // each graph act's own title, set every frame, so the one a later act left behind
      // never survives a loop or a scrub back
      this._setPhase(st.label || '');
    } else if (st.kind === 'anneal') {
      var tNow = Math.min(local / this.msPerStep, this.meta.t_end_max);
      draw = {
        space: 'lattice', rail: true, edges: true, t: 1, arcs: ARCS_A,
        tNow: tNow,
        ownOf: function (p) {
          if (!p.frames.length) return p.acts.anchored;
          var fi = self._frameAtT(p, tNow);
          return fi >= 0 ? p.frameOwn[fi] : p.initialOwn;
        },
        prevOwnOf: null
      };
      this._setPhase(st.label);
    } else {
      draw = this._latticeState(st, local);
      if (st.label) this._setPhase(st.label);
    }

    this.lastState = draw;
    this.tNow = draw.tNow === undefined ? null : draw.tNow;
    this._render(draw);
    this._updateHud(draw);
  };

  // Graph-space acts: tangle, the clusters found, separation, shading.
  var FOUND_AT = 0.25;
  PSSAPlayer.prototype._graphState = function (st, local) {
    var i;
    var prog = st.ms > 0 ? clamp01(local / st.ms) : 1;
    var labels = new Int16Array(this.nV);

    // The tangle, and the opening of 'found' before the partition arrives: the input graph
    // with no partition on it yet.
    if (st.to === 'tangle' ||
        (st.to === 'found' && local < FOUND_AT * st.ms)) {
      for (i = 0; i < this.nV; i++) labels[i] = -1;
      return { space: 'graph', labels: labels, posMix: 0, tNow: null };
    }

    // The partition as Mt-KaHyPar and T-FM left it, the recorded labels.
    for (i = 0; i < this.nV; i++) labels[i] = this.label[i];
    var mask = this._borderMask(labels);
    var base = { space: 'graph', labels: labels, isBorder: mask,
                 s1T: this.s1.T_end, tNow: null };

    if (st.to === 'found') {
      // Mt-KaHyPar returns the partition whole, so it arrives whole: at one instant every
      // vertex takes its cluster's colour, every border node its ring and every cut edge
      // its darker line, all recorded, nothing faded, and the partition holds until the
      // clusters part.
      base.posMix = 0;
      base.showBorder = true;
      return base;
    }
    if (st.to === 'separate') {
      // The clusters pull apart to where their patches will be.
      base.posMix = easeInOut(prog);
      base.showBorder = true;
      return base;
    }
    // shade: each vertex moves from its cluster's colour to its own shade.
    base.posMix = 1;
    base.showBorder = true;
    base.shadeMix = easeInOut(prog);
    return base;
  };

  // The routing's beats.
  PSSAPlayer.prototype._stage3State = function (st, local) {
    var s3 = this.s3, self = this;
    var prog = st.ms > 0 ? clamp01(local / st.ms) : 1;
    var ph = st.phase;
    var placed = ph === 'place' ? easeInOut(prog) : 1;

    this.patches.forEach(function (p) {
      p.dx = p.d3x * placed;
      p.dy = p.d3y * placed;
    });

    // One cell counter, at one rate.
    var nEsc = 0, nFun = 0, nTrunk = 0;
    if (ph === 'escape') {
      var down = Math.round(prog * (s3.maxEsc + s3.maxFun));
      nEsc = Math.min(down, s3.maxEsc);
      nFun = Math.max(0, down - s3.maxEsc);
    } else if (ph !== 'place') {
      nEsc = s3.maxEsc;
      nFun = s3.maxFun;
      nTrunk = ph === 'plaza' ? Math.round(prog * s3.maxTrunk) : s3.maxTrunk;
    }

    // The per-tile King's mesh is scenery, and it is legible only when a cell is a few
    // pixels across.
    var lod = clamp01((this._cell() - 4) / 8);
    var meshA = ph === 'place' ? Math.max(1 - placed, lod) : lod;

    // The bowed cut arcs are the abstraction the piece has carried since the clusters
    // separated: "these two vertices have to end up connected".
    var skip = null;
    if (ph !== 'place') {
      skip = new Uint8Array(this.data.graph.cross.length);
      s3.adjacent.forEach(function (ci) { skip[ci] = 1; });
      s3.routes.forEach(function (r) { if (r.need <= nTrunk) skip[r.ci] = 1; });
    }

    return {
      space: 's3', phase: ph, t: 1, rail: meshA > 0.01, railA: meshA,
      edges: true, arcs: ARCS_A, arcSkip: skip,
      meshAlpha: meshA, placed: placed,
      nEsc: nEsc, nFun: nFun, nTrunk: nTrunk,
      tNow: this.meta.t_end_max,
      ownOf: function (p) { return p.acts.final; },
      prevOwnOf: null
    };
  };

  // Lattice-space acts, and the bridge that gets there.
  PSSAPlayer.prototype._latticeState = function (st, local) {
    var self = this;
    var t = st.morph && st.ms > 0 ? clamp01(local / st.ms) : 1;
    var labels = this.data.graph.label;
    var mask = this._borderMask(labels);

    if (st.space === 'bridge') {
      // The one act where both spaces are live.
      var e = easeInOut(t);
      return {
        space: 'lattice', t: 1, spread: e, meshAlpha: e, styleMix: e,
        labels: labels, isBorder: mask,
        showBorder: true, edges: true, rail: false, arcs: ARCS_A,
        tNow: null, s1T: this.s1.T_end,
        ownOf: function (p) { return p.acts.split; },
        prevOwnOf: null
      };
    }

    if (st.kind === 'hold') {
      return {
        space: 'lattice', t: 1, rail: !!st.rail, edges: true,
        arcs: ARCS_A,
        tNow: st.hud === 'last' ? this.meta.t_end_max : null,
        ownOf: function (p) { return p.acts[st.act]; },
        prevOwnOf: null
      };
    }

    // an 'act': cut or morph between two named states
    var prevOwnOf = null;
    if (st.morph) {
      prevOwnOf = st.from
        ? function (p) { return p.acts[st.from]; }
        : function (p) {
            return p.frames.length ? p.frameOwn[p.frames.length - 1]
                                   : p.acts.anchored;
          };
    }
    return {
      space: 'lattice', t: st.morph ? t : 1, rail: !!st.rail,
      railIn: !!st.railIn, edges: true, edgesPrev: true,
      arcs: ARCS_A,
      tNow: st.hud === 'last' ? this.meta.t_end_max : null,
      ownOf: function (p) { return p.acts[st.to]; },
      prevOwnOf: prevOwnOf
    };
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // transport
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PSSAPlayer.prototype.play = function () {
    if (this.playing) return this;
    if (this.clock >= this.totalMs - 0.5) this.clock = 0;
    this.playing = true;
    this._scrubbing = false;
    this._syncPlayBtn();
    if (!this.running) {
      this.running = true;
      this.lastNow = null;
      global.requestAnimationFrame(this._tick.bind(this));
    }
    return this;
  };

  PSSAPlayer.prototype.pause = function () {
    this.playing = false;
    this._syncPlayBtn();
    return this;
  };

  PSSAPlayer.prototype.toggle = function () {
    return this.playing ? this.pause() : this.play();
  };

  PSSAPlayer.prototype.seek = function (ms) {
    this.clock = Math.max(0, Math.min(this.totalMs, ms));
    this._apply(this.lastNow || 0, true);
    this._syncTransport();
    return this;
  };

  // One step is one recorded frame of the cluster being watched, and one act outside the
  // anneal.
  PSSAPlayer.prototype.step = function (delta) {
    this.pause();
    var s = this.stage, st = this.script[s];
    if (st.kind === 'anneal') {
      var local = this.clock - this.stageStartMs[s];
      var tNow = local / this.msPerStep;
      var tNext;
      if (this.zoomCid !== null && this.patchOf[this.zoomCid].frames.length) {
        var p = this.patchOf[this.zoomCid];
        var i = this._frameAtT(p, tNow) + delta;
        if (i >= 0 && i < p.frames.length) {
          tNext = p.frames[i].t;
        } else {
          return this.seek(delta > 0 ? this.stageStartMs[s + 1]
                                     : this.stageStartMs[s] - 0.01);
        }
      } else {
        tNext = tNow + delta * this.meta.t_end_max / 1000;
      }
      if (tNext < 0) return this.seek(this.stageStartMs[s] - 0.01);
      if (tNext > this.meta.t_end_max) return this.seek(this.stageStartMs[s + 1]);
      return this.seek(this.stageStartMs[s] + tNext * this.msPerStep + 0.01);
    }
    var loc = this.clock - this.stageStartMs[s];
    if (delta > 0) return this.seek(this.stageStartMs[s + 1]);
    return this.seek(loc > 1 ? this.stageStartMs[s]
                             : this.stageStartMs[Math.max(s - 1, 0)]);
  };

  // Annealing steps per second of wall clock at the current multiplier.
  PSSAPlayer.prototype.stepRate = function () {
    return this.stepsPerSecond * this.speed;
  };

  PSSAPlayer.prototype.setSpeed = function (mult) {
    this.speed = mult;
    var c = this.container;
    if (c.querySelectorAll) {
      c.querySelectorAll('.pssa-speed button').forEach(function (b) {
        if (parseFloat(b.getAttribute('data-speed')) === mult) b.classList.add('on');
        else b.classList.remove('on');
      });
    }
    this._syncTransport();
    return this;
  };

  PSSAPlayer.prototype.setXMode = function (mode) {
    this.xmode = mode === 'lin' ? 'lin' : 'log';
    this.plotBg = null;
    var c = this.container;
    if (c.querySelectorAll) {
      var want = this.xmode;
      c.querySelectorAll('.pssa-xmode button[data-x]').forEach(function (b) {
        if (b.getAttribute('data-x') === want) b.classList.add('on');
        else b.classList.remove('on');
      });
    }
    this._drawPlot(this.tNow);
    return this;
  };

  PSSAPlayer.prototype.setYMode = function (mode) {
    this.ymode = mode === 'log' ? 'log' : 'lin';
    this.plotBg = null;
    var c = this.container;
    if (c.querySelectorAll) {
      var want = this.ymode;
      c.querySelectorAll('.pssa-xmode button[data-y]').forEach(function (b) {
        if (b.getAttribute('data-y') === want) b.classList.add('on');
        else b.classList.remove('on');
      });
    }
    this._drawPlot(this.tNow);
    return this;
  };

  PSSAPlayer.prototype._syncPlayBtn = function () {
    if (!this.playBtn) return;
    if (this.playing) this.playBtn.classList.add('on');
    else this.playBtn.classList.remove('on');
    if (this.playBtn.setAttribute) {
      this.playBtn.setAttribute('aria-label', this.playing ? 'Pause' : 'Play');
    }
  };

  PSSAPlayer.prototype._syncTransport = function () {
    var r = this.range;
    if (r) {
      if (!this._scrubbing) r.value = String(Math.round(this.clock));
      if (r.setAttribute) r.setAttribute('aria-valuetext', this._positionText());
    }
  };

  PSSAPlayer.prototype._positionText = function () {
    var st = this.script[this.stage];
    if (st && st.kind === 'anneal') {
      return 'annealing step ' + fmt(Math.round(this.tNow || 0));
    }
    return st ? (st.label || 'set-up') : '—';
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // pointer: click a patch to go into it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Which cluster a point on the figure picks, or null.
  PSSAPlayer.prototype._patchAt = function (px, py) {
    if (!this.cam) return null;
    var tol = Math.max(this.railOffset() * 1.6, PICK_SLOP);
    var best = null, bestD = Infinity, i;
    for (i = 0; i < this.patches.length; i++) {
      var p = this.patches[i];
      var b = this._box(p);
      var x0 = this.sx(b.x0), x1 = this.sx(b.x1);
      var y1 = this.sy(b.y0), y0 = this.sy(b.y1);
      // Distance from the point to the box, per axis, zero inside it.
      var dx = Math.max(x0 - px, 0, px - x1);
      var dy = Math.max(y0 - py, 0, py - y1);
      if (dx > tol || dy > tol) continue;
      var d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = p.cid; }
    }
    return best;
  };

  // What a click on the figure means.
  PSSAPlayer.prototype._clickFigure = function (px, py) {
    var st = this.script[this.stage];
    var noPatches = !st || st.space === 'graph';
    if (this.zoomCid !== null && noPatches) return this.zoomTo(null);
    if (noPatches) return this;
    var cid = this._patchAt(px, py);
    return this.zoomTo(this.zoomCid !== null && cid === this.zoomCid
      ? null : cid);
  };

  // A pointer event, in the units the figure is laid out in.
  PSSAPlayer.prototype._localPoint = function (ev) {
    var r = this.canvas.getBoundingClientRect();
    var x = ev.clientX - r.left, y = ev.clientY - r.top;
    if (r.width > 0 && this.W > 0) x *= this.W / r.width;
    if (r.height > 0 && this.H > 0) y *= this.H / r.height;
    return [x, y];
  };

  PSSAPlayer.prototype._bindPointer = function () {
    var self = this, cv = this.canvas;
    if (!cv || !cv.addEventListener) return;

    function local(ev) { return self._localPoint(ev); }
    cv.addEventListener('pointermove', function (ev) {
      // Only on the lattice: during the partitioning there are no patches, and a hover box
      // over empty space would promise something untrue.
      var st = self.script[self.stage];
      if (!st || st.space === 'graph' || self.zoomCid !== null) {
        if (self.hoverCid !== null) { self.hoverCid = null; cv.style.cursor = ''; }
        return;
      }
      var q = local(ev);
      var cid = self._patchAt(q[0], q[1]);
      if (cid !== self.hoverCid) {
        self.hoverCid = cid;
        cv.style.cursor = cid === null ? '' : 'pointer';
        if (!self.playing && self.lastDraw) self._render(self.lastDraw);
      }
    });
    cv.addEventListener('pointerleave', function () {
      self.hoverCid = null; cv.style.cursor = '';
    });
    cv.addEventListener('click', function (ev) {
      var q = local(ev);
      self._clickFigure(q[0], q[1]);
    });
    if (this.backBtn) {
      this.backBtn.addEventListener('click', function () { self.zoomTo(null); });
    }
  };

  PSSAPlayer.prototype._bindTransport = function () {
    var self = this, c = this.container;
    if (!c.querySelectorAll) return;

    c.querySelectorAll('.pssa-btn').forEach(function (b) {
      b.addEventListener('click', function () {
        var a = b.getAttribute('data-act');
        if (a === 'play') self.toggle();
        else if (a === 'prev') self.step(-1);
        else if (a === 'next') self.step(1);
      });
    });

    c.querySelectorAll('.pssa-speed button').forEach(function (b) {
      b.addEventListener('click', function () {
        self.setSpeed(parseFloat(b.getAttribute('data-speed')));
      });
    });

    c.querySelectorAll('.pssa-xmode button[data-x]').forEach(function (b) {
      b.addEventListener('click', function () {
        self.setXMode(b.getAttribute('data-x'));
      });
    });

    c.querySelectorAll('.pssa-xmode button[data-y]').forEach(function (b) {
      b.addEventListener('click', function () {
        self.setYMode(b.getAttribute('data-y'));
      });
    });

    var r = this.range;
    if (r) {
      r.max = String(Math.max(1, Math.round(this.totalMs)));
      r.addEventListener('input', function () {
        self._scrubbing = true;
        self.pause();
        self.seek(parseFloat(r.value));
      });
      r.addEventListener('change', function () { self._scrubbing = false; });
      r.addEventListener('pointerup', function () { self._scrubbing = false; });
      r.addEventListener('keyup', function () { self._scrubbing = false; });
    }

    this._bindPointer();

    c.addEventListener('pointerenter', function () { self._hover = true; });
    c.addEventListener('pointerleave', function () { self._hover = false; });

    var doc = global.document;
    if (doc && doc.addEventListener) {
      doc.addEventListener('keydown', function (ev) { self._onKey(ev); });
    }

    // The player only claims the keyboard when it is plausibly what the reader is looking
    // at: focused, hovered, or filling the viewport.
    this._inView = true;
    if (typeof global.IntersectionObserver === 'function') {
      this._inView = false;
      var io = new global.IntersectionObserver(function (entries) {
        self._inView = entries[entries.length - 1].intersectionRatio >= 0.4;
      }, { threshold: [0, 0.4, 1] });
      io.observe(c);
    }
  };

  PSSAPlayer.prototype._keyActive = function (ev) {
    var t = ev.target;
    // Never steal keys from a form field -- except our own scrub bar, whose native arrow
    // step is one millisecond of a minute-long clock and so means nothing.
    if (t && t.tagName && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) &&
        t !== this.range) return false;
    if (t && t.isContentEditable) return false;
    var doc = global.document;
    if (doc && doc.activeElement && this.container.contains &&
        this.container.contains(doc.activeElement)) return true;
    return !!(this._hover || this._inView);
  };

  PSSAPlayer.prototype._onKey = function (ev) {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (!this._keyActive(ev)) return;
    var k = ev.key;
    if (k === ' ' || k === 'Spacebar' || k === 'k') this.toggle();
    else if (k === 'ArrowRight') this.step(ev.shiftKey ? 25 : 1);
    else if (k === 'ArrowLeft') this.step(ev.shiftKey ? -25 : -1);
    else if (k === 'Home') { this.pause(); this.seek(0); }
    else if (k === 'End') { this.pause(); this.seek(this.totalMs); }
    else if (k === 'Escape' && this.zoomCid !== null) this.zoomTo(null);
    else return;
    if (ev.preventDefault) ev.preventDefault();
  };

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // run loop
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PSSAPlayer.prototype.start = function () {
    this.clock = 0;
    if (this.reduced) {
      // Reduced motion: land on the finished embeddings rather than animating toward them.
      this.clock = this.totalMs;
      this._apply(0, true);
      this._syncTransport();
      return this;
    }
    this._apply(0, true);
    this._syncTransport();
    return this.play();
  };

  PSSAPlayer.prototype.stop = function () {
    this.playing = false;
    this.running = false;
    this._syncPlayBtn();
    return this;
  };

  PSSAPlayer.prototype._tick = function (now) {
    if (!this.running) return;
    var dt = this.lastNow === null ? 0 : (now - this.lastNow);
    this.lastNow = now;

    if (this.playing) {
      this.clock += Math.min(dt, 100) * this.speed;
      if (this.clock >= this.totalMs) {
        if (this.loop) {
          this.clock -= this.totalMs;
          if (this.clock >= this.totalMs) this.clock = 0;
        } else {
          this.clock = this.totalMs;
          this.playing = false;
          this._syncPlayBtn();
        }
      }
    }

    // `running === true` is the player's promise that a frame is scheduled, and `zoomTo` /
    // `play` both read it to decide whether they need to start the loop or whether one is
    // already turning.
    try {
      this._apply(now, false);
      this._syncTransport();
    } catch (err) {
      this.running = false;
      this.playing = false;
      this._syncPlayBtn();
      throw err;
    }

    // Keep running while the camera is still easing, even when paused, so a zoom that
    // starts on a paused frame still arrives.
    var camMoving = this.camMs &&
      (now - this.camT0) < this.camMs;
    if (!this.playing && !camMoving) { this.running = false; return; }
    global.requestAnimationFrame(this._tick.bind(this));
  };

  global.PSSAPlayer = PSSAPlayer;
}(window));
