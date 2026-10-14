/*
 * Records the live Figure 23 to a video. From the page or the console, pssaExport()
 * records the figure at 1x, in real time with the tab in the foreground, and downloads
 * a .webm; pssaExport({ layout: 'stage+plot' }) records the top row. Optional: the
 * player does not depend on it.
 */
(function (global) {
  'use strict';

  var MIMES = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm'
  ];

  function pickMime() {
    if (typeof MediaRecorder === 'undefined') return null;
    for (var i = 0; i < MIMES.length; i++) {
      if (MediaRecorder.isTypeSupported(MIMES[i])) return MIMES[i];
    }
    return null;
  }

  // The canvas that actually gets recorded.
  function target(player, layout, background, scale) {
    var stage = player.canvas;
    var plot = player.plotEl;
    var wide = layout === 'stage+plot' && plot;
    var gap = wide ? Math.round(16 * (global.devicePixelRatio || 1)) : 0;

    var w0 = wide ? stage.width + gap + plot.width : stage.width;
    var h0 = wide ? Math.max(stage.height, plot.height) : stage.height;

    // Supersampling: the figure is fine line art, which a video codec's 4:2:0 chroma
    // desaturates, and a larger frame loses less of it.
    if (!scale) {
      scale = 2;
      while (scale > 1 && Math.max(w0, h0) * scale > 3840) scale -= 1;
    }
    var c = document.createElement('canvas');
    c.width = Math.round(w0 * scale);
    c.height = Math.round(h0 * scale);
    var x = c.getContext('2d', { alpha: false });
    x.imageSmoothingEnabled = true;
    x.imageSmoothingQuality = 'high';
    return {
      canvas: c,
      scale: scale,
      draw: function () {
        x.fillStyle = background || '#ffffff';
        x.fillRect(0, 0, c.width, c.height);
        x.drawImage(stage, 0, 0, stage.width * scale, stage.height * scale);
        if (wide) {
          x.drawImage(plot, (stage.width + gap) * scale, 0,
                      plot.width * scale, plot.height * scale);
        }
      }
    };
  }

  function stamp() {
    var d = new Date(), p = function (n) { return (n < 10 ? '0' : '') + n; };
    return d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate()) + '-' +
           p(d.getHours()) + p(d.getMinutes());
  }

  function save(blob, name) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    // Revoked late: some browsers read the URL asynchronously after the click.
    setTimeout(function () { URL.revokeObjectURL(url); }, 60000);
  }

  function record(player, opts) {
    opts = opts || {};
    player = player || global.pssaPlayer;

    if (!player || !player.canvas) {
      return Promise.reject(new Error(
        'no PSSAPlayer to record. Pass one in, or set window.pssaPlayer.'));
    }
    var mime = pickMime();
    if (!mime || !player.canvas.captureStream) {
      return Promise.reject(new Error(
        'this browser has no MediaRecorder/captureStream. Chrome, Edge and ' +
        'Firefox all do; Safari did not until recently.'));
    }

    var fps = opts.fps || 60;
    var speed = opts.speed || 1;
    var t = target(player, opts.layout || 'stage', opts.background, opts.scale);
    var stream = t.canvas.captureStream(fps);
    var rec = new MediaRecorder(stream, {
      mimeType: mime,
      // Line art needs far more bits than video of the world does: there is no motion blur
      // and no soft gradient for the codec to hide error in, and every edge is a hard one.
      videoBitsPerSecond: opts.bitrate || 60000000
    });
    var chunks = [];
    rec.ondataavailable = function (e) { if (e.data && e.data.size) chunks.push(e.data); };

    // The transport is left exactly as it was found, whether this finishes or throws:
    // recording should not quietly change where the reader was.
    var wasLoop = player.loop;
    var wasSpeed = player.speed;
    var wasClock = player.clock;
    var wasPlaying = player.playing;
    function restore() {
      player.pause();
      player.loop = wasLoop;
      player.setSpeed(wasSpeed);
      player.seek(wasClock);
      if (wasPlaying) player.play();
    }

    return new Promise(function (resolve, reject) {
      var raf = null;
      rec.onerror = function (e) {
        if (raf) cancelAnimationFrame(raf);
        restore();
        reject(e.error || new Error('MediaRecorder failed'));
      };
      rec.onstop = function () {
        restore();
        var blob = new Blob(chunks, { type: mime });
        if (opts.download !== false) {
          save(blob, opts.filename ||
            ('pssa-' + (opts.layout || 'stage') + '-' + stamp() + '.webm'));
        }
        resolve(blob);
      };

      player.pause();
      player.loop = false;               // stop at the end rather than restart
      player.setSpeed(speed);
      player.seek(0);

      // One frame of lead-in, so the recorder's first frame is the piece's first frame
      // rather than whatever was on screen when it was called.
      requestAnimationFrame(function () {
        t.draw();
        rec.start();
        player.play();
        var watch = function () {
          t.draw();
          // The player stops itself at the end when `loop` is off.
          if (!player.playing || player.clock >= player.totalMs - 1) {
            t.draw();                    // include the final held frame
            setTimeout(function () { rec.stop(); }, 250);
            return;
          }
          raf = requestAnimationFrame(watch);
        };
        raf = requestAnimationFrame(watch);
      });
    });
  }

  // The lossless path: one PNG per frame, straight out of the canvas.
  function frames(player, opts) {
    opts = opts || {};
    player = player || global.pssaPlayer;
    if (!player) {
      return Promise.reject(new Error('no PSSAPlayer to record'));
    }
    if (!global.showDirectoryPicker) {
      return Promise.reject(new Error(
        'this browser has no File System Access API. Chrome and Edge do; ' +
        'in others use pssaExport() instead.'));
    }
    var fps = opts.fps || 30;
    var t = target(player, opts.layout || 'stage', opts.background, opts.scale);
    var total = Math.ceil((player.totalMs / 1000) * fps);

    var wasLoop = player.loop, wasClock = player.clock;
    player.pause();
    player.loop = false;

    return global.showDirectoryPicker({ mode: 'readwrite' }).then(function (dir) {
      var i = 0;
      function step() {
        if (i >= total) {
          player.loop = wasLoop;
          player.seek(wasClock);
          console.log('pssaFrames: wrote ' + total + ' frames at ' + fps +
                      ' fps, ' + t.canvas.width + ' x ' + t.canvas.height);
          return total;
        }
        player.seek((i * 1000) / fps);
        t.draw();
        var name = ('00000' + i).slice(-5) + '.png';
        var blob;
        return new Promise(function (res) { t.canvas.toBlob(res, 'image/png'); })
          .then(function (b) {
            blob = b;
            return dir.getFileHandle(name, { create: true });
          })
          .then(function (fh) { return fh.createWritable(); })
          .then(function (w) {
            return w.write(blob).then(function () { return w.close(); });
          })
          .then(function () {
            if (i % 60 === 0) console.log('  ' + i + ' / ' + total);
            i++;
            return step();
          });
      }
      return step();
    });
  }

  global.pssaExport = function (opts) { return record(null, opts); };
  global.pssaExport.record = record;
  global.pssaFrames = function (opts) { return frames(null, opts); };
  global.pssaFrames.frames = frames;

  // ffmpeg, once you have it.
}(typeof window !== 'undefined' ? window : this));
