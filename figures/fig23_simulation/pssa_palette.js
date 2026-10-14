/*
 * The colours of the live Figure 23, in one place: the two themes (Paul Tol's muted
 * scheme and the report's own figure colours), the page chrome, and the colour maths
 * that gives every vertex of a cluster its own shade.
 */
(function (global) {
  'use strict';

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // colour space
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  function hexToRgb(h) {
    var o = h.charAt(0) === '#' ? 1 : 0;
    return [parseInt(h.substr(o, 2), 16),
            parseInt(h.substr(o + 2, 2), 16),
            parseInt(h.substr(o + 4, 2), 16)];
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  // HSV, matching Python's `colorsys` branch for branch.
  function rgbToHsv(r, g, b) {
    var mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn, h = 0;
    if (d !== 0) {
      if (mx === r) h = ((g - b) / d) % 6;
      else if (mx === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h /= 6;
      if (h < 0) h += 1;
    }
    return [h, mx === 0 ? 0 : d / mx, mx];
  }

  function hsvToRgb(h, s, v) {
    if (s === 0) return [v, v, v];
    var i = Math.floor(h * 6), f = h * 6 - i;
    var p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    switch (i % 6) {
      case 0: return [v, t, p];
      case 1: return [q, v, p];
      case 2: return [p, v, t];
      case 3: return [p, q, v];
      case 4: return [t, p, v];
      default: return [v, p, q];
    }
  }

  function to8bit(c) {
    return [clamp(Math.round(c[0] * 255), 0, 255),
            clamp(Math.round(c[1] * 255), 0, 255),
            clamp(Math.round(c[2] * 255), 0, 255)];
  }

  // CIE L*a*b*, D65, and its polar form LCh, which the machined cluster grid is laid out
  // in.
  var Xn = 0.95047, Yn = 1.0, Zn = 1.08883;
  var D = 6 / 29;

  function srgbToLinear(u) {
    u /= 255;
    return u <= 0.04045 ? u / 12.92 : Math.pow((u + 0.055) / 1.055, 2.4);
  }
  function linearToSrgb(u) {
    return 255 * (u <= 0.0031308 ? 12.92 * u
                                 : 1.055 * Math.pow(u, 1 / 2.4) - 0.055);
  }
  function labF(t) { return t > D * D * D ? Math.cbrt(t) : t / (3 * D * D) + 4 / 29; }
  function labFInv(t) { return t > D ? t * t * t : 3 * D * D * (t - 4 / 29); }

  function rgbToLab(c) {
    var R = srgbToLinear(c[0]), G = srgbToLinear(c[1]), B = srgbToLinear(c[2]);
    var fx = labF((0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / Xn);
    var fy = labF((0.2126729 * R + 0.7151522 * G + 0.0721750 * B) / Yn);
    var fz = labF((0.0193339 * R + 0.1191920 * G + 0.9503041 * B) / Zn);
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
  }

  // Lab -> sRGB, unclamped, so a caller can tell whether the colour exists.
  function labToRgbRaw(L, a, b) {
    var fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
    var X = labFInv(fx) * Xn, Y = labFInv(fy) * Yn, Z = labFInv(fz) * Zn;
    return [linearToSrgb( 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z),
            linearToSrgb(-0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z),
            linearToSrgb( 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z)];
  }

  function inGamut(L, a, b) {
    var c = labToRgbRaw(L, a, b);
    return c[0] >= -0.5 && c[0] <= 255.5 && c[1] >= -0.5 && c[1] <= 255.5 &&
           c[2] >= -0.5 && c[2] <= 255.5;
  }

  // The largest chroma still inside sRGB at this lightness and hue.
  function maxChroma(L, h) {
    var rad = h * Math.PI / 180, lo = 0, hi = 140, i, m;
    for (i = 0; i < 22; i++) {
      m = (lo + hi) / 2;
      if (inGamut(L, m * Math.cos(rad), m * Math.sin(rad))) lo = m; else hi = m;
    }
    return lo;
  }

  function lchToRgb(L, C, h) {
    var rad = h * Math.PI / 180, c = Math.min(C, maxChroma(L, h));
    return labToRgbRaw(L, c * Math.cos(rad), c * Math.sin(rad))
      .map(function (v) { return clamp(Math.round(v), 0, 255); });
  }

  function rgbToLch(c) {
    var L = rgbToLab(c);
    return { L: L[0], C: Math.sqrt(L[1] * L[1] + L[2] * L[2]),
             h: (Math.atan2(L[2], L[1]) * 180 / Math.PI + 360) % 360 };
  }

  // Perceptual distance between two sRGB colours, as plain Euclidean dE76 in Lab.
  function dE(a, b) {
    var p = rgbToLab(a), q = rgbToLab(b);
    return Math.sqrt((p[0] - q[0]) * (p[0] - q[0]) +
                     (p[1] - q[1]) * (p[1] - q[1]) +
                     (p[2] - q[2]) * (p[2] - q[2]));
  }

  // CIEDE2000 between two Lab colours (Sharma, Wu and Dalal 2005).
  function dE2000Lab(l1, l2) {
    var r = Math.PI / 180, P7 = Math.pow(25, 7);
    var C1 = Math.sqrt(l1[1] * l1[1] + l1[2] * l1[2]);
    var C2 = Math.sqrt(l2[1] * l2[1] + l2[2] * l2[2]);
    var Cb7 = Math.pow((C1 + C2) / 2, 7);
    var G = 0.5 * (1 - Math.sqrt(Cb7 / (Cb7 + P7)));
    var a1 = l1[1] * (1 + G), a2 = l2[1] * (1 + G);
    var C1p = Math.sqrt(a1 * a1 + l1[2] * l1[2]);
    var C2p = Math.sqrt(a2 * a2 + l2[2] * l2[2]);
    var h1 = Math.atan2(l1[2], a1) / r, h2 = Math.atan2(l2[2], a2) / r;
    if (h1 < 0) h1 += 360;
    if (h2 < 0) h2 += 360;
    var dL = l2[0] - l1[0], dC = C2p - C1p, dh = 0, hb = h1 + h2;
    if (C1p * C2p !== 0) {
      dh = h2 - h1;
      if (dh > 180) dh -= 360; else if (dh < -180) dh += 360;
      hb = Math.abs(h1 - h2) > 180
        ? (h1 + h2 < 360 ? (h1 + h2 + 360) / 2 : (h1 + h2 - 360) / 2)
        : (h1 + h2) / 2;
    }
    var dH = 2 * Math.sqrt(C1p * C2p) * Math.sin(dh * r / 2);
    var Lb = (l1[0] + l2[0]) / 2, Cbp = (C1p + C2p) / 2, Cbp7 = Math.pow(Cbp, 7);
    var T = 1 - 0.17 * Math.cos((hb - 30) * r) + 0.24 * Math.cos(2 * hb * r) +
            0.32 * Math.cos((3 * hb + 6) * r) - 0.20 * Math.cos((4 * hb - 63) * r);
    var dT = 30 * Math.exp(-((hb - 275) / 25) * ((hb - 275) / 25));
    var RC = 2 * Math.sqrt(Cbp7 / (Cbp7 + P7));
    var SL = 1 + 0.015 * (Lb - 50) * (Lb - 50) / Math.sqrt(20 + (Lb - 50) * (Lb - 50));
    var SC = 1 + 0.045 * Cbp, SH = 1 + 0.015 * Cbp * T;
    var RT = -Math.sin(2 * dT * r) * RC;
    var x = dL / SL, y = dC / SC, z = dH / SH;
    return Math.sqrt(x * x + y * y + z * z + RT * y * z);
  }
  function dE2000(a, b) { return dE2000Lab(rgbToLab(a), rgbToLab(b)); }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // the report palette: colorcet's glasbey_hv
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // All 256 entries, in palette order.
  var GLASBEY =
    '30a2dafc4f30e5ae386d904f17becf9467bdd627281f77b4e377c28c564bbcbd' +
    '220043000fffa9c6bdffb80080ffb7b36126ffffff9aaec9ab00867c553a0094' +
    'fcff00bf00ab720091ff0001be8a00457bc8826fff1f83dd00ff057400888fff' +
    'ffb6f4ce85ffbeb4bea5608995d3ffff80028b2945ada06d53458bc8ffd9aa46' +
    '00ff798f83d3719400f5ebd09b00634affdc007eaba3f500c665332900667804' +
    'e3c8a737aec5dbe14d6eff9b9301cd586befdefe795a00b4ff9205875184206f' +
    '3c9605657300f1a06cd06827d796ab895dff2b55b96e7cbbe7d5d37c3b0180b1' +
    '7dc8d97d00e83b7cb2ffff55ffa427211de4ff7daf3b7b4b91e0ff48cda897be' +
    '63c489cdce414a0105a79dcf8c37fff8d0b544ffcf4993cfa4df94d400a794da' +
    '2da5588de3b6a4a99d6c5cb7ff7e5eafbed82ac4ffa6683df691fe874b64ff0c' +
    '4b215e234292ffb14f41004e53ad4167d6fffd7fb5d1a9b969ff96cbc87495ff' +
    'd063a978ff03c863e7bed4d4e3d0897c27cddcffaa676b323474ff5ea9009bb0' +
    '71ffdd785c3850659bcc00b3015f92aabdbe017f9904dd97873a2cf0968e75c6' +
    'aaccdc09af8557d800759d3f81d94500dd67545fff79d5b17362265ebaa23dd9' +
    'f2b3a4a9fface8db995901ac00e247822fcbc3ad00c5b6fd57647096d2728d07' +
    '1530a0d1c1e2c985d000a279b2a9cff90000b0e9ff939e50d92e554761010059' +
    'ff7740b5ace460674525a9e49ad962f68e7dcfffbd939affb9a7c2fff46200e5' +
    'f0ffb89ca4ff9f358c2900726b32df824eaf7bd5bc2d00c7a3ff004d28c4c68e' +
    'e048d7e7e965e5c10b00f4f19f5ba24c41b765338ea98a36';

  var N_GLASBEY = 200;

  // A port of draw_embedding.get_color(v, n).
  function getColor(v, n) {
    var i = (Math.round(v) - 1) % N_GLASBEY;
    if (i < 0) i += N_GLASBEY;
    var o = i * 6;
    return [
      parseInt(GLASBEY.substr(o, 2), 16),
      parseInt(GLASBEY.substr(o + 2, 2), 16),
      parseInt(GLASBEY.substr(o + 4, 2), 16)
    ];
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // per-vertex shades
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // The three golden-angle sequences the recording spreads a cluster's vertices with.
  var G1 = 0.6180339887, G2 = 0.3819660113, G3 = 0.2469796037;
  var HUE_JITTER_DEG = 25.0;   // max hue deviation from the cluster's base hue

  function hueJitter(h0, idx) {
    var h = (h0 + (((idx * G1) % 1.0) - 0.5) * (2 * HUE_JITTER_DEG / 360.0)) % 1.0;
    return h < 0 ? h + 1 : h;
  }

  // A port of the recording's vertex shades, constant for constant, so the `paper` palette
  // reproduces `trajectory.json`'s own `graph.shade`.
  function shadeCategorical(base, idx) {
    var hsv = rgbToHsv(base[0] / 255, base[1] / 255, base[2] / 255);
    return to8bit(hsvToRgb(
      hueJitter(hsv[0], idx),
      0.55 + ((idx * G2) % 1.0) * 0.45,     // 0.55-1.00
      0.42 + ((idx * G3) % 1.0) * 0.53      // 0.42-0.95
    ));
  }

  // The candidates one cluster's vertices are chosen from: four lightnesses, three chromas
  // and three hues around the cluster's own colour.
  // The lightness band a vertex shade may sit in, and the chroma ceiling it may reach.
  var SHADE_L_MIN = 32, SHADE_L_MAX = 74;
  var CHROMA_CAP = 40, CHROMA_OF_MAX = 0.90;

  // Half-width of a cluster's lightness window, and the one knob that trades contrast
  // inside a tile against the lightness gradient across the tiles.
  // The lattice a cluster's vertices are chosen from, as four numbers.
  var SHADE_DEFAULTS = {
    dl: 13,
    // How many hue levels the lattice carries, and the floor under a shade's chroma.
    nh: 3,
    minC: 0,
    // The chroma floor at the top of the lightness band, if it differs from `minC` at the
    // bottom.
    minCTop: undefined,
    // An optional corridor the hue window must stay inside, as [hLo, hHi] in Lab hue
    // degrees.
    hLo: undefined,
    hHi: undefined,
    // The hue the corridor's upper end collapses toward as lightness rises, and the band
    // itself.
    hHiTop: undefined,
    lmin: SHADE_L_MIN,
    lmax: SHADE_L_MAX,
    // 15 degrees either side, against a 30-degree gap between cluster hues.
    dh: 15,
    fr: [0.38, 0.69, 1.00],
    cap: CHROMA_CAP * 1.15,
    ofMax: CHROMA_OF_MAX
  };

  function shadeOpts(over) { return extend(SHADE_DEFAULTS, over || {}); }

  // A window of fixed width, slid to sit as near the base as the band allows.
  function shadeLattice(b, o) {
    var loL = clamp(b.L - o.dl, o.lmin, o.lmax - 2 * o.dl);
    var hiL = loL + 2 * o.dl;
    var out = [], li, ci, k;
    for (li = 0; li < 4; li++) {
      var L = loL + (hiL - loL) * li / 3;
      // The hue window, slid to sit inside the corridor when there is one -- the same trick
      // the lightness window uses, so a base near an edge of the corridor keeps its full
      // spread rather than having it clipped.
      var hHi = o.hHi;
      if (o.hHiTop !== undefined) {
        var t = clamp((L - o.lmin) / (o.lmax - o.lmin), 0, 1);
        hHi = o.hHi + (o.hHiTop - o.hHi) * t * t;
      }
      var hSpan = 2 * o.dh, hLo = b.h - o.dh;
      if (o.hLo !== undefined) {
        hSpan = Math.min(2 * o.dh, hHi - o.hLo);
        hLo = clamp(b.h - hSpan / 2, o.hLo, hHi - hSpan);
      }
      for (ci = 0; ci < o.fr.length; ci++) {
        for (k = 0; k < o.nh; k++) {
          var h = hLo + (o.nh > 1 ? hSpan * k / (o.nh - 1) : hSpan / 2);
          var room = Math.min(o.cap, o.ofMax * maxChroma(L, h));
          var floor = o.minCTop === undefined ? o.minC
            : o.minC + (o.minCTop - o.minC) *
              clamp((L - o.lmin) / (o.lmax - o.lmin), 0, 1);
          out.push(lchToRgb(L, Math.min(room, Math.max(floor, room * o.fr[ci])), h));
        }
      }
    }
    return out;
  }

  // Greedy farthest-point: seed with the candidate nearest the cluster's own colour, then
  // repeatedly take whichever is furthest from everything already taken.
  function farthestPoints(cands, n, seed) {
    // Everything is compared in Lab, so convert once rather than inside the inner loop.
    var lab = cands.map(rgbToLab), origin = rgbToLab(seed);
    var gap = dE2000Lab;
    var i, j, best = 0, bestD = Infinity, d;
    for (i = 0; i < lab.length; i++) {
      d = gap(lab[i], origin);
      if (d < bestD) { bestD = d; best = i; }
    }
    var chosen = [best], taken = {};
    taken[best] = true;
    // Each candidate's distance to the nearest chosen one, kept up to date as picks are
    // made rather than recomputed from scratch.
    var near = lab.map(function (x) { return gap(x, lab[best]); });
    while (chosen.length < n && chosen.length < lab.length) {
      var pick = -1, pickD = -1;
      for (i = 0; i < lab.length; i++) {
        if (!taken[i] && near[i] > pickD) { pickD = near[i]; pick = i; }
      }
      if (pick < 0) break;
      chosen.push(pick);
      taken[pick] = true;
      for (i = 0; i < lab.length; i++) near[i] = Math.min(near[i], gap(lab[i], lab[pick]));
    }
    var out = [];
    // A cluster larger than the lattice would wrap.
    for (i = 0; i < n; i++) out.push(cands[chosen[i % chosen.length]]);
    return out;
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // the palettes
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // The border-vertex ring, and the one colour in the piece that is not a theme choice.
  var BORDER_MARK = [0, 0, 0];

  // Shallow merge, so the two machined themes can share one chrome block instead of
  // repeating forty custom properties between them.
  function extend(a, b) {
    var out = {}, k;
    for (k in a) if (Object.prototype.hasOwnProperty.call(a, k)) out[k] = a[k];
    for (k in b) if (Object.prototype.hasOwnProperty.call(b, k)) out[k] = b[k];
    return out;
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // the academic themes
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  var ACADEMIC_CHROME = {
    shadeRule: 'placed',
    assign: 'index',

    ink:   [15, 23, 42],
    muted: [71, 85, 105],
    faint: [148, 163, 184],
    alert: [193, 32, 32],
    paper: [255, 255, 255],

    // Source-over at full opacity.
    routeComposite: 'source-over',
    routeAlpha: 1,

    css: {
      '--bg': '#ffffff',
      '--panel': '#fbfbfc',
      '--line': '#dfe3ea',
      '--chrome': '#f4f5f7',
      '--chrome-hi': '#e6e9ef',
      '--ink': '#101828',
      '--ink-dim': '#4a5568',
      '--ink-faint': '#8792a6',
      '--alert': '#c12020',
      '--good': '#12805a',
      '--focus': '#0072B2',
      '--page-bg': '#ffffff',
      '--page-panel': '#fafbfc',
      '--page-ink': '#101828',
      '--page-text': '#3f4a5c',
      '--page-dim': '#4a5568',
      '--page-faint': '#8792a6',
      '--page-rule': '#e7eaf0',
      '--page-rule-soft': '#e2e6ee',
      '--page-rule-strong': '#c8d0dd',
      '--page-hair': '#eef0f4',
      '--page-dot': '#b6bfcd',
      '--page-mono': '#2f3b4f',
      '--page-mono-bg': '#f2f4f8',
      '--fail-border': '#e8c9c9',
      '--fail-bg': '#fdf7f7',
      '--fail-ink': '#a01c1c'
    }
  };

  // The lightness band a reused hue may be moved within.
  var CAT_L_MIN = 30, CAT_L_MAX = 84;

  // The shade lattice these themes use.
  var CAT_SHADE = { dl: 14, dh: 21, nh: 5, minC: 6,
                    fr: [0.25, 0.50, 0.75, 1.00], cap: 70, ofMax: 0.96,
                    lmin: 32, lmax: 84 };

  function catTheme(name, label, hexes, plotT, plotE) {
    var o = shadeOpts(CAT_SHADE);
    return extend(ACADEMIC_CHROME, {
      name: name,
      label: label,
      plotT: plotT,
      plotE: plotE,
      clusterColors: function (k) {
        var n = hexes.length, tiers = Math.ceil(k / n), out = [], i, t;
        var Ls = [];
        for (t = 0; t < tiers; t++) {
          Ls.push(tiers > 1
            ? CAT_L_MIN + (CAT_L_MAX - CAT_L_MIN) * t / (tiers - 1)
            : (CAT_L_MIN + CAT_L_MAX) / 2);
        }
        for (i = 0; i < k; i++) {
          var b = rgbToLch(hexToRgb(hexes[i % n])), tier = Math.floor(i / n);
          // Whichever tier sits closest to this hue's own lightness carries the published
          // colour untouched.
          var mine = Math.ceil((k - (i % n)) / n), near = 0;
          for (t = 1; t < mine; t++) {
            if (Math.abs(Ls[t] - b.L) < Math.abs(Ls[near] - b.L)) near = t;
          }
          out.push(tier === near ? hexToRgb(hexes[i % n])
                                 : lchToRgb(Ls[tier], b.C, b.h));
        }
        return out;
      },
      clusterShades: function (base, n) {
        return farthestPoints(shadeLattice(rgbToLch(base), o), n, base);
      }
    });
  }


  // The two themes, in the order the toolbar lists them.
  var PALETTES = {
    // Paul Tol's muted scheme, the usual choice in physics and astronomy figures.
    tolMuted: extend(catTheme('tolMuted', 'Paul Tol Muted',
      ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
       '#DDCC77', '#CC6677', '#882255', '#AA4499'],
      '#CC6677', '#332288'), {
      // No two joined vertices of a cluster in similar shades: see `_assignBySpread` in
      // pssa_player.js.
      assign: 'spread'
    }),

    // The report's own figures: glasbey cluster colours (`draw_embedding.get_color`) and
    // golden-angle shades, in index order as the figures do.
    paper: extend(catTheme('paper', 'Glasbey', [], '#CC6677', '#332288'), {
      shadeRule: 'jitter',
      clusterColors: function (k) {
        var out = [], i;
        for (i = 0; i < k; i++) out.push(getColor(i + 1, k));
        return out;
      },
      clusterShades: function (base, n) {
        var out = [], i;
        for (i = 0; i < n; i++) out.push(shadeCategorical(base, i));
        return out;
      }
    })
  };

  // Which theme the page opens on: Paul Tol's muted scheme, the one the page uses.
  var ACTIVE = PALETTES.tolMuted;

  // Write a palette's custom properties onto one element, inline, so they beat the literals
  // the stylesheets declare.
  function applyTo(el, pal) {
    if (!el || !el.style || !el.style.setProperty) return;
    var css = (pal || ACTIVE).css, key;
    for (key in css) {
      if (Object.prototype.hasOwnProperty.call(css, key)) {
        el.style.setProperty(key, css[key]);
      }
    }
  }

  if (typeof document !== 'undefined' && document.documentElement) {
    applyTo(document.documentElement, ACTIVE);
  }

  // Lab distance and Lab itself, exported for checking the palette.
  global.PSSA_DE = dE;
  global.PSSA_DE2000 = dE2000;
  global.PSSA_RGB_TO_LAB = rgbToLab;
  global.PSSA_BORDER_MARK = BORDER_MARK;
  global.PSSA_PALETTES = PALETTES;
  global.PSSA_PALETTE = ACTIVE;
  global.PSSA_PALETTE_APPLY = applyTo;
  global.__pssaGetColor = getColor;   // the matplotlib port, exported for checking

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PALETTES: PALETTES, ACTIVE: ACTIVE, applyTo: applyTo,
                       dE: dE, rgbToLab: rgbToLab,
                       BORDER_MARK: BORDER_MARK };
  }
}(typeof window !== 'undefined' ? window : globalThis));
