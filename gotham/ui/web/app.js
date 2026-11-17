"use strict";
var bridge = null;
var G = window.G = {};
var S = { wordDisplayTime: 750, wordAnimationTime: 150 };
var state = {};

function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
  return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
function send(name, args) { if (bridge) bridge.action(JSON.stringify({ name: name, args: args || {} })); }
$.fn.textWidth = function () {
  var org = $(this).html(); $(this).html("<span>" + org + "</span>");
  var w = $(this).find("span:first").width(); $(this).html(org); return w;
};

new QWebChannel(qt.webChannelTransport, function (ch) {
  bridge = ch.objects.bridge;
  var hole = document.getElementById("hole");
  function report() {
    var r = hole.getBoundingClientRect();
    bridge.hole(Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height));
  }
  new ResizeObserver(report).observe(hole);
  window.addEventListener("resize", report);
  report();
  bridge.ready();
});

var sayQueue = [], talking = false, idleTimer = null;
function idleBlink() {
  clearTimeout(idleTimer);
  if (talking) return;
  var t = $("#triangle");
  t.css("opacity", t.css("opacity") === "0" ? 1 : 0);
  idleTimer = setTimeout(idleBlink, 500);
}
G.say = function (text) {
  text = String(text || "").trim();
  if (!text) return;
  sayQueue.push(text);
  if (!talking) nextSay();
};
function nextSay() {
  if (!sayQueue.length) { talking = false; $("#triangle").removeClass("speaking"); idleBlink(); return; }
  talking = true;
  clearTimeout(idleTimer);
  var words = sayQueue.shift().split(/\s+/);
  var p = $("#message-container p"), hr = $("#message-container hr");
  $("#triangle").css("opacity", 1).addClass("speaking");
  var t = 0;
  words.forEach(function (word) {
    var wt = S.wordDisplayTime;
    if (word.length > 8) wt *= word.length / 8;
    setTimeout(function () {
      p.addClass("hidden-word").text(word);
      hr.stop().animate({ width: (p.textWidth() + 18) + "px" }, {
        duration: S.wordAnimationTime, start: function () { p.removeClass("hidden-word"); } });
    }, t + S.wordAnimationTime);
    t += wt;
  });
  setTimeout(function () {
    p.html("&nbsp;");
    $("#triangle").removeClass("speaking");
    hr.stop().animate({ width: "30px" }, { duration: S.wordAnimationTime, done: function () {
      setTimeout(nextSay, 250); } });
  }, t + S.wordDisplayTime);
}
G.status = function (text) {
  $("#voice-status").html(text ? '<span class="invert">' + esc(text) + ' <span class="blink">_</span></span>' : "");
};
G.busy = function (b) { $("#cmd-input").prop("disabled", !!b); if (!b) $("#cmd-input").focus(); };

var alertTimer = null;
G.alert = function (title, text) {
  $("#alertbox .ab-title").text(title);
  $("#alertbox .ab-text").html(esc(text) + ' <span class="blink">_</span>');
  $("#alertbox").removeClass("hidden").css("opacity", 1);
  $("body").addClass("alerting");
  clearTimeout(alertTimer);
  alertTimer = setTimeout(function () {
    $("#alertbox").css("opacity", 0);
    setTimeout(function () { $("#alertbox").addClass("hidden"); $("body").removeClass("alerting"); }, 300);
  }, 5000);
};

G.log = function (lines) {
  var box = $("#log");
  lines.forEach(function (l) {
    box.append('<div class="logline ' + (l.cls || "") + '"><span class="ts">' + esc(l.t) + '</span> <span class="k">' +
      esc(l.k) + '</span> ' + esc(l.text || "") + "</div>");
  });
  var kids = box.children();
  if (kids.length > 300) kids.slice(0, kids.length - 300).remove();
  box.scrollTop(box[0].scrollHeight);
};

$("#cmd-input").on("keydown", function (e) {
  if (e.key === "Enter") {
    var v = this.value.trim();
    if (v && bridge) { bridge.command(v); this.value = ""; }
  } else if (e.key === "Escape") { this.blur(); }
});
$(document).on("keydown", function (e) {
  var typing = document.activeElement && document.activeElement.tagName === "INPUT";
  if (e.key === "/" && !typing) { e.preventDefault(); $("#cmd-input").focus(); }
});

$(document).on("click", "[data-action]", function (e) {
  e.stopPropagation();
  var el = $(this), args = {};
  $.each(this.attributes, function () {
    if (this.name.indexOf("data-") === 0 && this.name !== "data-action") args[this.name.slice(5)] = this.value;
  });
  send(el.attr("data-action"), args);
});
$(document).on("click", ".mode", function () { send("mode", { mode: $(this).attr("data-mode") }); });

function kv(k, v, cls) {
  return '<div class="kv"><div class="k">' + esc(k) + ':</div><div class="v"><span class="' + (cls || "") + '">' +
    esc(v) + "</span></div></div>";
}
function bar(q) { q = Math.max(0, Math.min(1, q || 0));
  return '<div class="bar' + (q < 0.45 ? " low" : "") + '"><i style="width:' + (q * 100).toFixed(0) + '%"></i></div>'; }
function fmtT(s) { s = Math.max(0, Math.round(s || 0)); return ("0" + Math.floor(s / 60)).slice(-2) + ":" + ("0" + s % 60).slice(-2); }

G.state = function (s) {
  state = s;
  ["live", "images", "memory"].forEach(function (m) { document.body.classList.toggle("mode-" + m, s.mode === m); });
  $(".mode").each(function () { $(this).toggleClass("on", $(this).attr("data-mode") === s.mode); });
  renderSources(s); renderLayers(s); renderSystem(s); renderMemSum(s);
  if (s.mode === "live") {
    $("#lab,#persondetail").html("");
    renderFeed(s); renderSubjects(s); renderInspector(s);
  } else if (s.mode === "memory") {
    $("#feedblock,#subjects,#inspector,#lab").html("");
    if (s.memory) renderMemory(s.memory);
  } else {
    $("#feedblock,#subjects,#inspector,#persondetail").html("");
    renderImages(s); renderLab(s);
  }
};

function renderSources(s) {
  var h = "";
  (s.feeds || []).forEach(function (f) {
    var bad = f.status !== "live" && f.status !== "paused";
    h += '<div class="row' + (f.id === s.selected ? " sel" : "") + '" data-action="select" data-feed="' + f.id + '">' +
      '<span class="nm">' + esc(f.name) + " · " + esc(f.label) + "</span>" +
      '<span class="st' + (bad ? " bad" : "") + '">' + esc(f.status.toUpperCase()) + "</span></div>";
  });
  (s.sources || []).forEach(function (c) {
    if (c.open) return;
    h += '<div class="row" data-action="open_camera" data-index="' + c.index + '">' +
      '<span class="nm">CAM_' + c.index + " · " + esc(c.name) + "</span>" +
      '<span class="st ' + (c.available ? "dim" : "bad") + '">' + (c.available ? (c.w ? c.w + "x" + c.h : "READY") : "BUSY") +
      "</span></div>";
  });
  if (!h) h = '<div class="mini">NO CAMERAS DETECTED</div>';
  $("#sources").html(h);
}

function renderLayers(s) {
  var L = s.layers || {}, h = "";
  Object.keys(L).forEach(function (k) {
    if (k === "style") return;
    h += '<span class="layer' + (L[k] ? " on" : "") + '" data-action="layer" data-layer="' + k + '">' + k.toUpperCase() + "</span>";
  });
  h += '<div class="btnrow"><span class="btn' + (L.style === "samaritan" ? " on" : "") + '" data-action="style" data-style="samaritan">SAMARITAN</span>' +
       '<span class="btn' + (L.style === "machine" ? " on" : "") + '" data-action="style" data-style="machine">MACHINE</span></div>';
  $("#layers").html(h);
}

function renderSystem(s) {
  var y = s.system || {};
  $("#system").html(
    "CPU_LOAD ...... " + esc(y.cpu) + "%<br>" +
    "MEM_SYS ....... " + esc(y.ram) + "%<br>" +
    "MEM_PROC ...... " + esc(y.proc) + " MB<br>" +
    "INFERENCE ..... " + esc(y.provider) + "<br>" +
    "GPU ........... " + esc(y.gpu) + "<br>" +
    "OBSERVER ...... " + esc(y.observer) + "<br>" +
    "WATCHES ....... " + esc(y.watches) + "<br>" +
    "UPTIME ........ " + esc(y.uptime));
}

function renderFeed(s) {
  var f = (s.feeds || []).filter(function (x) { return x.id === s.selected; })[0];
  if (!f) { $("#feedblock").html('<div class="blk"><div class="t">NO FEED SELECTED</div><div class="c mini">SELECT OR OPEN AN INPUT FEED</div><div class="f"></div></div>'); return; }
  var mode = f.tracking ? "TRACKING_" : (f.status === "live" ? "OBSERVING" : f.status.toUpperCase());
  var h = '<div class="blk"><div class="t"><span>' + esc(f.name) + '</span><span class="sm">' + esc(f.res) + "</span></div><div class=\"c\">" +
    kv("SOURCE", f.label) + kv("STATUS", f.status.toUpperCase(), f.status === "live" ? "ok" : "hot") +
    kv("MODE", mode, f.tracking ? "hot" : "") +
    kv("RATE", f.fps + " / " + f.afps + " FPS") + kv("LATENCY", f.lat + " MS") +
    kv("LINK", "RECON " + f.reconnects + " · DROP " + f.dropped + (f.faults ? " · FAULTS " + f.faults : "")) +
    spark(f.spark) +
    (f.tracking ? kv("SESSION", fmtT(f.session_s) + " · " + f.total + " SUBJ") : "") +
    '<div class="btnrow">' +
    '<span class="btn ' + (f.tracking ? "red" : "") + '" data-action="tracking" data-feed="' + f.id + '">' + (f.tracking ? "END SESSION" : "START TRACKING") + "</span>" +
    '<span class="btn' + (f.paused ? " on" : "") + '" data-action="pause" data-feed="' + f.id + '">' + (f.paused ? "RESUME" : "PAUSE") + "</span>" +
    '<span class="btn' + (f.small ? " on" : "") + '" data-action="small" data-feed="' + f.id + '">SMALL FACES</span>' +
    '<span class="btn' + (f.landmarks ? " on" : "") + '" data-action="landmarks" data-feed="' + f.id + '">LANDMARKS</span>' +
    '<span class="btn" data-action="expand" data-feed="' + f.id + '">EXPAND</span>' +
    '<span class="btn" data-action="snapshot" data-feed="' + f.id + '">SNAPSHOT</span>' +
    '<span class="btn red" data-action="close" data-feed="' + f.id + '">DISCONNECT</span>' +
    "</div></div><div class=\"f\"></div></div>";
  $("#feedblock").html(h);
}

function renderSubjects(s) {
  var subs = s.subjects || [];
  var f = (s.feeds || []).filter(function (x) { return x.id === s.selected; })[0];
  if (!f) { $("#subjects").html(""); return; }
  var h = '<div class="sec-title">SUBJECTS · ' + subs.filter(function (x) { return x.visible; }).length + " IN VIEW</div>";
  if (!f.tracking) h += '<div class="mini">' + f.faces + " FACE(S) DETECTED · START A TRACKING SESSION TO ASSIGN SUBJECT IDS</div>";
  subs.forEach(function (t) {
    var foc = s.focus && s.focus[0] === s.selected && s.focus[1] === t.id;
    h += '<div class="subj' + (foc ? " focus" : "") + (t.state === "lost" ? " gone" : "") + '" data-action="focus" data-feed="' + s.selected + '" data-subject="' + t.id + '">' +
      '<div class="hd"><b>SUBJECT ' + esc(t.label) + "</b><span>" + esc(t.state.toUpperCase()) + "</span></div>" +
      (t.person ? '<div class="who"><b>' + esc(t.person.toUpperCase()) + "</b> " + (t.mem === "new" ? "NEW PROFILE" : "KNOWN " + t.pscore) + "</div>"
                : ((t.mem === "matching" || t.mem === "uncertain") ? '<div class="who">IDENTIFYING...</div>' : "")) +
      '<div class="meta">IN SESSION ' + fmtT(t.age) + " · REACQ " + t.reacq + (t.gaze ? " · GAZE " + esc(t.gaze) : "") + "</div>" +
      bar(t.quality) + "</div>";
  });
  $("#subjects").html(h);
}

function renderInspector(s) {
  var d = s.inspector;
  if (!d) { $("#inspector").html(""); return; }
  var h = '<div class="blk"><div class="t red"><span>SUBJECT ' + esc(d.label) + '</span><span class="sm">' + esc(d.state.toUpperCase()) + "</span></div><div class=\"c\">";
  if (d.person) {
    h += kv("KNOWN AS", d.person.toUpperCase() + (d.mem === "new" ? " (NEW)" : " · " + d.pscore), "hot");
    h += '<div class="btnrow" style="margin:0 0 6px"><span class="btn" data-action="mem_open" data-person="' + d.pid + '">OPEN PROFILE</span></div>';
  } else if (d.mem) h += kv("MEMORY", "IDENTIFYING...");
  if (d.pose) h += kv("HEAD POSE", "Y " + d.pose[0] + "° P " + d.pose[1] + "° R " + d.pose[2] + "°");
  if (d.gaze) h += kv("GAZE", d.gaze, d.at_cam ? "hot" : "");
  if (d.eyes) h += kv("EYES", d.eyes);
  h += kv("FACE SIZE", d.size + " PX");
  h += kv("DETECTION", d.score);
  if (d.expr) h += kv("EXPRESSION", d.expr);
  h += kv("VISIBILITY", d.trunc ? "PARTIAL (FRAME EDGE)" : "FULL");
  Object.keys(d.quality || {}).forEach(function (k) {
    h += '<div class="kv"><div class="k">' + esc(k.toUpperCase()) + '</div><div class="v">' + bar(d.quality[k]) + "</div></div>";
  });
  h += '<div class="btnrow"><span class="btn" data-action="focus" data-feed="' + d.feed + '" data-subject="">CLEAR FOCUS</span>' +
       '<span class="btn red" data-action="suppress" data-feed="' + d.feed + '" data-subject="' + d.id + '">EXCLUDE</span></div>';
  h += '</div><div class="f"></div></div>';
  $("#inspector").html(h);
}

function renderImages(s) {
  var h = "";
  (s.images || []).forEach(function (im) {
    h += '<div class="row' + (im.sel ? " sel" : "") + '" data-action="select_image" data-index="' + im.idx + '">' +
      '<span class="nm">' + esc(im.name) + '</span><span class="st' + (im.faces ? "" : " dim") + '">' +
      (im.pending ? "…" : im.faces + " FACE" + (im.faces === 1 ? "" : "S")) + "</span></div>";
  });
  $("#imagelist").html(h || '<div class="mini">NO IMAGES LOADED</div>');
}

function renderLab(s) {
  var lab = s.lab || {}, h = "";
  if (lab.compare) {
    var c = lab.compare;
    h += '<div class="blk"><div class="t red"><span>VISUAL COMPARISON</span><span class="sm">A / B</span></div><div class="c">' +
      '<div class="cmp-imgs"><div><img src="' + c.a_img + '">A · ' + esc(c.a_name) + '</div><div><img src="' + c.b_img + '">B · ' + esc(c.b_name) + "</div></div>";
    if (c.shape) h += '<img class="cmp-shape" src="' + c.shape + '">';
    (c.rows || []).forEach(function (r) {
      h += '<div class="kv"><div class="k">' + esc(r[0]) + '</div><div class="v"><span>' + esc(r[1]) + "</span> <span>" + esc(r[2]) + '</span> <span class="' + (r[4] ? "hot" : "") + '">Δ ' + esc(r[3]) + "</span></div></div>";
    });
    h += '<div class="note">' + esc(c.note) + "</div>";
    h += '<div class="btnrow"><span class="btn" data-action="clear_compare">CLEAR A/B</span></div></div><div class="f"></div></div>';
  }
  var dets = lab.detections || [];
  h += '<div class="sec-title">DETECTIONS · ' + dets.length + "</div>";
  if (dets.length) h += '<div class="mini">CLICK: INSPECT · SHIFT+CLICK: SET A / B FOR COMPARISON</div>';
  dets.forEach(function (d) {
    h += '<div class="det' + (d.ab === "A" ? " a" : d.ab === "B" ? " b" : "") + '" data-key="' + esc(d.key) + '">' +
      '<img src="' + d.thumb + '"><div class="info">' + (d.ab ? '<span class="tagab">' + d.ab + "</span> " : "") +
      esc(d.title) + (d.match ? ' <span class="mm">' + esc(d.match) + "</span>" : "") +
      "<br>CONF " + d.score + " · " + d.size + "PX<br>YAW " + d.yaw + "° · Q " + d.q + "<br>" + bar(d.qv) + "</div></div>";
  });
  if (lab.detail) {
    var x = lab.detail;
    h += '<div class="blk"><div class="t"><span>' + esc(x.title) + '</span><span class="sm">INSPECT</span></div><div class="c">' +
      '<img class="cmp-shape" src="' + x.img + '">';
    (x.rows || []).forEach(function (r) { h += kv(r[0], r[1]); });
    if (x.can_remember) h += '<div class="btnrow"><span class="btn" data-action="remember" data-key="' + esc(x.key) + '">REMEMBER THIS FACE</span></div>';
    if (x.matches && x.matches.length) {
      h += '<div class="kv"><div class="k">MEMORY SEARCH:</div></div>';
      x.matches.forEach(function (m) {
        h += '<div class="alike' + (m.verdict === "MATCH" ? " likely" : "") + '">' + (m.thumb ? '<img src="' + m.thumb + '">' : "") +
          '<div class="ai"><b>' + esc(m.name) + "</b> " + esc(m.verdict) + " " + esc(m.score) + "<br>SEEN " + m.seen + "&times; &middot; LAST " + esc(m.last) +
          '</div><span class="btn" data-action="mem_open" data-person="' + m.id + '">OPEN</span></div>';
      });
    }
    h += '</div><div class="f"></div></div>';
  }
  $("#lab").html(h);
}
$(document).on("click", ".det", function (e) {
  send(e.shiftKey ? "set_ab" : "inspect", { key: $(this).attr("data-key") });
});

var bootIdx = 0;
G.bootStart = function () {
  $("#boot-left").css({ width: "50%" }).animate({ width: "20%" }, 900);
  $("#boot-right").css({ width: "50%" }).animate({ width: "20%" }, 900, function () {
    $("#loading-pane-container").animate({ opacity: 0.8 }, 200).animate({ opacity: 0.1 }, 80).animate({ opacity: 1 }, 200);
    $("#loading-name, #loading-bar-container").delay(300).animate({ opacity: 1 }, 250);
  });
};
G.bootLine = function (line) {
  var i = bootIdx++;
  $("#boot-log").append('<span class="r">[[</span>' + (i < 10 ? " " : "x") + i + (i < 10 ? " " : "") + '<span class="r">]</span> ' + esc(line) + "<br>");
};
G.bootLoad = function (name, pct) {
  $("#loading-bar").stop().animate({ width: pct + "%" }, 250, function () { $("#loading-name").text(name); });
};
G.bootDone = function (cb) {
  $("#loading-pane-container").delay(400).animate({ opacity: 0 }, 500);
  $("#loading-connected").delay(900).animate({ opacity: 0.8 }, 200).animate({ opacity: 0.1 }, 80).animate({ opacity: 1 }, 200)
    .delay(800).animate({ opacity: 0 }, 400, function () {
      $("#boot").fadeOut(600, function () { if (bridge) bridge.booted(); idleBlink(); G.say("WHAT ARE YOUR COMMANDS ?"); });
    });
  $("#boot-left, #boot-right").delay(1200).animate({ width: "0%" }, 1100);
};
idleBlink();

function renderMemSum(s) {
  var m = s.memory_summary || {};
  if (!m.available) { $("#memsum").html("UNAVAILABLE"); return; }
  $("#memsum").html(
    "STATUS ........ " + (m.upgrading !== null && m.upgrading !== undefined ? '<span style="color:#ff4a4a">UPGRADING ' + m.upgrading + "%</span>"
                                : (m.enabled ? "RECOGNISING" : '<span style="color:#ff4a4a">PAUSED</span>')) + "<br>" +
    "PROFILES ...... " + m.profiles + "<br>SIGHTINGS ..... " + m.sightings +
    '<div class="btnrow"><span class="btn" data-action="mode" data-mode="memory">OPEN</span>' +
    '<span class="btn' + (m.enabled ? "" : " on") + '" data-action="mem_toggle">' + (m.enabled ? "PAUSE" : "RESUME") + "</span></div>");
}

var thumbCache = {}, memQuery = "", memSort = "recent", memShow = 120, lastMem = null, lastHeadHtml = "", lastReviewHtml = "";
function renderMemory(m) {
  if (!m.available) { $("#memview").html('<div class="mem-empty">FACE MEMORY UNAVAILABLE</div>'); $("#persondetail").html(""); lastGalleryHtml = ""; return; }
  (m.people || []).forEach(function (p) { if (p.thumb) thumbCache[p.id] = p.thumb; });
  lastMem = m;
  if (m.page) { renderPage(m.page); renderRecent(m); return; }
  var back = lastPageId !== null;
  lastPageId = null; lastPageHtml = "";
  if (back || !document.getElementById("memgrid")) {
    $("#memview").html('<div id="memtop"></div><div class="mem-tools"><input id="memsearch" type="text" spellcheck="false" autocomplete="off" ' +
      'placeholder="SEARCH P-CODE OR LABEL"><span class="sortlbl">SORT</span>' +
      '<span class="btn" data-msort="recent">LAST SEEN</span><span class="btn" data-msort="seen">MOST SEEN</span>' +
      '<span class="btn" data-msort="new">NEWEST</span><span class="btn" data-msort="named">LABELLED / WATCHLIST</span>' +
      '<span class="cnt" id="memcount"></span></div><div id="memreview"></div><div id="memgrid"></div>');
    $("#memsearch").val(memQuery);
    lastGalleryHtml = lastHeadHtml = lastReviewHtml = "";
  }
  var h = '<div class="mem-head"><span class="ttl">FACE_MEMORY</span><span class="meta">' + m.stats.profiles + " PROFILES · " +
    m.stats.sightings + ' VISITS · ENCRYPTED · LOCAL ONLY</span><span style="flex:1"></span>' +
    '<span class="btn' + (m.enabled ? "" : " on") + '" data-action="mem_toggle">' + (m.enabled ? "PAUSE RECOGNITION" : "RESUME RECOGNITION") + "</span>" +
    (m.dup_count ? '<span class="btn' + (m.reviewing ? " on" : "") + '" data-action="mem_review">' + m.dup_count + " POSSIBLE DUPLICATE" + (m.dup_count > 1 ? "S" : "") + "</span>" : "") +
    (m.certain ? '<span class="btn" data-action="mem_merge_certain">MERGE ' + m.certain + " CERTAIN</span>" : "") +
    (m.prunable ? '<span class="btn" data-action="mem_prune">PRUNE ' + m.prunable + "</span>" : "") +
    '<span class="btn red" data-action="mem_clear">CLEAR ALL</span></div>';
  if (h !== lastHeadHtml) { $("#memtop").html(h); lastHeadHtml = h; }
  var r = "";
  if (m.reviewing && m.review.length) {
    r += '<div class="review">';
    m.review.forEach(function (d) {
      r += '<div class="dup">' + (d.a_thumb ? '<img src="' + d.a_thumb + '">' : '<div class="noimg"></div>') +
        (d.b_thumb ? '<img src="' + d.b_thumb + '">' : '<div class="noimg"></div>') + '<div class="di"><b>' + esc(d.a_name) + "</b> &harr; <b>" +
        esc(d.b_name) + "</b>" + (d.certain ? ' <span class="cert">CERTAIN</span>' : "") + "<br>BEST VIEW PAIR " + esc(d.score) + " &middot; AS WHOLES " + esc(d.whole) +
        '</div><span class="btn" data-action="mem_merge_into" data-person="' + d.a +
        '" data-target="' + d.b + '">SAME PERSON</span><span class="btn" data-action="mem_dismiss" data-person="' + d.a + '" data-target="' + d.b +
        '">DIFFERENT</span></div>';
    });
    r += "</div>";
  }
  if (r !== lastReviewHtml) { $("#memreview").html(r); lastReviewHtml = r; }
  renderGallery(back);
  renderPerson(m.detail);
}
function renderGallery(back) {
  var m = lastMem;
  if (!m || !document.getElementById("memgrid")) return;
  $("[data-msort]").each(function () { $(this).toggleClass("on", $(this).attr("data-msort") === memSort); });
  var raw = memQuery.trim().toLowerCase(), all = m.people || [];
  var code = /^p-?\d+$/.test(raw) ? raw.replace(/^p-?/, "") : (/^\d+$/.test(raw) ? raw : null);
  var list = all.filter(function (p) {
    if (!raw) return true;
    if (code !== null)
      return p.code.slice(2).indexOf(code) === 0 || p.id === parseInt(code, 10);
    return (p.label || "").toLowerCase().indexOf(raw) >= 0;
  });
  var key = { recent: function (p) { return -p.ls; }, seen: function (p) { return -p.sightings * 1e10 - p.ls; },
              "new": function (p) { return -p.fs; },
              named: function (p) { return (p.alert ? 0 : p.label ? 1 : 2) * 1e10 - p.ls; } }[memSort];
  list = list.slice().sort(function (a, b) { return key(a) - key(b); });
  var shown = list.slice(0, memShow), h = "";
  $("#memcount").text(list.length === all.length ? all.length + " PROFILES" : list.length + " OF " + all.length);
  if (!all.length) {
    h = '<div class="mem-empty">NO PROFILES YET<br><span style="font-size:11px;font-family:Lekton">' +
        "START A TRACKING SESSION ON A FEED: PEOPLE ARE REMEMBERED AUTOMATICALLY.<br>" +
        "IN IMAGES, INSPECT A FACE AND PRESS REMEMBER.</span></div>";
  } else if (!list.length) {
    h = '<div class="mem-empty">NO PROFILE MATCHES &ldquo;' + esc(memQuery) + '&rdquo;</div>';
  } else {
    h = '<div class="mem-grid">';
    shown.forEach(function (p) {
      var t = thumbCache[p.id];
      h += '<div class="pcard' + (p.id === m.selected ? " sel" : "") + '" data-action="mem_page" data-person="' + p.id + '" title="OPEN PROFILE">' +
        '<div class="ph">' + (t ? '<img src="' + t + '">' : "") + (p.live ? '<span class="live">IN VIEW</span>' : "") +
        (p.alert ? '<span class="watch">WATCHLIST</span>' : "") + '<span class="open">OPEN PROFILE &rsaquo;</span></div>' +
        '<div class="nm"><b>' + esc(p.code) + "</b> " + esc(p.label) + "</div>" +
        '<div class="mt">SEEN ' + p.sightings + "× · LAST " + esc(p.last) + "</div></div>";
    });
    h += "</div>";
    if (list.length > shown.length) h += '<div class="btnrow mem-more"><span class="btn" data-mmore="1">SHOW ' + Math.min(120, list.length - shown.length) +
      " MORE (" + (list.length - shown.length) + " HIDDEN)</span></div>";
  }
  if (h !== lastGalleryHtml || back) {
    var sc = back ? galleryScroll : $("#memview").scrollTop();
    $("#memgrid").html(h);
    $("#memview").scrollTop(sc);
    lastGalleryHtml = h;
  }
}
$(document).on("input", "#memsearch", function () { memQuery = this.value; memShow = 120; renderGallery(); });
$(document).on("keydown", "#memsearch", function (e) { if (e.key === "Escape") { this.value = ""; memQuery = ""; renderGallery(); this.blur(); } });
$(document).on("click", "[data-msort]", function (e) { e.stopPropagation(); memSort = $(this).attr("data-msort"); memShow = 120; renderGallery(); });
$(document).on("click", "[data-mmore]", function (e) { e.stopPropagation(); memShow += 120; renderGallery(); });

var lastPageId = null, lastPage = null, galleryScroll = 0, tlSort = "recent", tlAll = false, lastPageHtml = "", lastGalleryHtml = "";
function pblk(title, sub, body, cls) {
  return '<section class="pblk ' + (cls || "") + '"><div class="pt"><span>' + title + '</span><span class="sm">' + (sub || "") +
    '</span></div><div class="pc">' + body + '</div><div class="pf"></div></section>';
}
function pkv(k, v, cls) { return '<div class="pkv"><span class="k">' + esc(k) + '</span><span class="v ' + (cls || "") + '">' + esc(v) + "</span></div>"; }
function nl2br(t) { return esc(t).split("\n").join("<br>"); }
function renderPage(d) {
  if (lastPageId !== d.id) {
    if (lastPageId === null) galleryScroll = $("#memview").scrollTop();
    lastPageId = d.id; tlAll = false; lastPageHtml = ""; lastGalleryHtml = "";
    $("#memview").scrollTop(0);
  }
  lastPage = d;
  var live = d.live && d.live.length;
  var h = '<div class="pp"><div class="pp-head">' +
    '<span class="btn" data-action="mem_back">&lsaquo; GALLERY</span>' +
    '<span class="pp-code">' + esc(d.code) + '</span><span class="pp-name">' + esc(d.label ? d.label.toUpperCase() : "UNLABELLED") + "</span>" +
    (live ? '<span class="pp-live">IN VIEW &middot; ' + esc(d.live.join(", ")) + ' <span class="blink">_</span></span>' : "") +
    (d.alert ? '<span class="pp-watch">WATCHLIST</span>' : "") +
    '<span style="flex:1"></span>' +
    '<span class="btn" data-action="mem_rename" data-person="' + d.id + '">LABEL</span>' +
    '<span class="btn" data-action="mem_notes" data-person="' + d.id + '">NOTES</span>' +
    '<span class="btn' + (d.alert ? " on" : "") + '" data-action="mem_alert" data-person="' + d.id + '">' + (d.alert ? "ALERTING WHEN SEEN" : "ALERT WHEN SEEN") + "</span>" +
    (live ? '<span class="btn red" data-action="mem_locate" data-person="' + d.id + '">LOCATE</span>' : "") +
    '<span class="btn" data-action="mem_merge" data-person="' + d.id + '">MERGE INTO...</span>' +
    '<span class="btn red" data-action="mem_forget" data-person="' + d.id + '">FORGET</span></div>';

  var idb = '<div class="pp-photo">' + (d.thumb ? '<img src="' + d.thumb + '">' : "") + '<span class="tag">' + esc(d.code) + "</span>" +
    (live ? '<span class="tag live">IN VIEW</span>' : "") + "</div>";
  d.stats.forEach(function (r) { idb += pkv(r[0], r[1], r[0] === "POSSIBLE SIGHTINGS" && r[1] !== "0" ? "hot" : ""); });

  var f = d.face, fb = "";
  fb += f.wire ? '<img class="pp-wire" src="' + f.wire + '">'
               : '<div class="pp-nowire">NO FRONTAL MESH CAPTURED YET<br><span>STORED THE FIRST TIME THIS PERSON FACES A CAMERA, CLOSE ENOUGH FOR THE DENSE MESH</span></div>';
  if (f.signature) fb += '<div class="pp-sub">BIOMETRIC SIGNATURE &middot; ' + esc(d.model.toUpperCase()) + '</div><img class="pp-sig" src="' + f.signature + '">';
  fb += '<div class="pp-sub">ANGLE COVERAGE</div><div class="bins">';
  f.bin_names.forEach(function (n, i) { fb += '<span class="' + (f.bins[i] ? "on" : "") + '">' + n + (f.bins[i] ? " &times;" + f.bins[i] : "") + "</span>"; });
  fb += "</div>";
  var mixed = f.stray ? f.stray + " VIEW" + (f.stray > 1 ? "S" : "") + " LOOK LIKE SOMEONE ELSE" : (f.pair_disagrees ? "ITS 2 VIEWS DISAGREE" : "");
  if (f.coherence !== null) fb += pkv("PROFILE CONSISTENCY", f.coherence.toFixed(2) + (mixed ? " · " + mixed : ""), mixed ? "hot" : "");
  if (f.metrics.length) {
    fb += '<div class="pp-sub">PROPORTIONS (&times; EYE DISTANCE)</div><div class="pp-metrics">';
    f.metrics.forEach(function (r) { fb += '<div><span>' + esc(r[0]) + "</span><b>" + esc(r[1]) + "</b></div>"; });
    fb += "</div>";
  }

  var ab = '<div class="pp-sub">TIME OF DAY</div><div class="hours">';
  var nowH = new Date().getHours();
  d.hours.forEach(function (v, i) {
    ab += '<i class="' + (i === nowH ? "now" : "") + '" style="height:' + Math.max(2, v * 100).toFixed(0) + '%" title="' + ("0" + i).slice(-2) + ':00"></i>';
  });
  ab += '</div><div class="hours-x"><span>00</span><span>06</span><span>12</span><span>18</span><span>24</span></div>';
  var dm = Math.max.apply(null, d.days.map(function (x) { return x.n; }).concat([1]));
  ab += '<div class="pp-sub">LAST 14 DAYS</div><div class="days">';
  d.days.forEach(function (x) {
    ab += '<div title="' + esc(x.label) + ": " + x.n + '"><i style="height:' + (x.n ? Math.max(8, x.n / dm * 100) : 0).toFixed(0) + '%"></i><span>' + esc(x.d) + "</span></div>";
  });
  ab += '</div><div class="pp-sub">SOURCES</div>';
  if (!d.sources.length) ab += '<div class="pp-none">NO SIGHTINGS RECORDED YET</div>';
  d.sources.forEach(function (x) {
    ab += '<div class="src"><b>' + esc(x.source.toUpperCase().slice(0, 28)) + "</b><span>" + x.n + "&times; &middot; " + esc(x.time) + " &middot; " + esc(x.last) + "</span></div>";
  });

  h += '<div class="pp-grid">' + pblk("IDENTITY", esc(d.code), idb, "id") + pblk("FACE DESIGN", "478-POINT MESH", fb, "face") +
       pblk("ACTIVITY PATTERN", d.sources.length + " SOURCE" + (d.sources.length === 1 ? "" : "S"), ab, "act") + "</div>";

  var vb = '<div class="pp-views">';
  d.views.forEach(function (v) {
    vb += '<div class="vw' + (v.stray ? " stray" : "") + '" title="' + (v.agree !== null ? "AGREES " + esc(v.agree) + " WITH THE OTHER VIEWS" : "") + '">' +
      '<img src="' + v.thumb + '"><span>' + esc(v.yaw) + "&deg; &middot; Q" + esc(v.q) + "</span>" +
      (v.agree !== null ? '<span class="ag">' + (v.stray ? "NOT THEM? " : "FIT ") + esc(v.agree) + "</span>" : "") +
      '<span class="rm" data-action="mem_view_remove" data-view="' + v.id + '" data-person="' + d.id + '">REMOVE</span></div>';
  });
  vb += d.views.length ? "</div>" : '<div class="pp-none">NO VIEWS STORED</div></div>';
  h += pblk("LEARNED VIEWS", d.views.length + " DESCRIPTORS &middot; LEFT PROFILE &rarr; RIGHT PROFILE", vb);

  var bb = '<div class="pp-best">';
  d.best.forEach(function (x, i) {
    bb += '<div class="bc"><div class="ph">' + (x.thumb ? '<img src="' + x.thumb + '">' : "") + '<span class="rk">#' + (i + 1) + '</span><span class="sc">' + esc(x.score) +
      '</span></div><div class="bi"><b>' + esc(x.source.toUpperCase().slice(0, 22)) + "</b><br>" + esc(x.from) + "<br>" +
      (x.kind === "live" ? esc(x.dur) : esc(x.kind.toUpperCase())) + "</div></div>";
  });
  bb += d.best.length ? "</div>" : '<div class="pp-none">NO SIGHTINGS YET</div></div>';
  h += pblk("HIGHEST-CONFIDENCE SIGHTINGS", "BEST MATCH SCORES", bb);

  var cb = "";
  if (!d.candidates.length) cb = '<div class="pp-none">NO OPEN POSSIBLE SIGHTINGS &middot; NEAR MISSES (SIMILARITY &ge; ' + d.thresholds.possible.toFixed(2) +
    ", BELOW THE " + d.thresholds.match.toFixed(2) + " MATCH LINE) APPEAR HERE FOR REVIEW</div>";
  d.candidates.forEach(function (c) {
    cb += '<div class="cand">' + (c.thumb ? '<img src="' + c.thumb + '">' : '<div class="noimg"></div>') +
      '<div class="ci"><b>' + esc(c.source.toUpperCase().slice(0, 24)) + "</b> " + esc(c.when) + " &middot; " + esc(c.ago) +
      "<br>SUBJECT " + esc(c.label) + (c.other ? ' &middot; ENROLLED AS <span class="lnk" data-action="mem_page" data-person="' + c.other_id + '">' + esc(c.other) + "</span>" : "") +
      '<div class="meter"><i style="width:' + (c.strength * 100).toFixed(0) + '%"></i><span>SIMILARITY ' + esc(c.score) + "</span></div></div>" +
      '<span class="btn" data-action="cand_confirm" data-cand="' + c.id + '" data-person="' + d.id + '">' + (c.other ? "SAME PERSON &middot; MERGE" : "CONFIRM") + "</span>" +
      '<span class="btn red" data-action="cand_reject" data-cand="' + c.id + '" data-person="' + d.id + '">NOT THEM</span></div>';
  });
  h += pblk("POSSIBLE SIGHTINGS", d.candidates.length + " AWAITING REVIEW", cb, d.candidates.length ? "hotblk" : "");

  var rows = d.timeline.slice();
  if (tlSort === "confidence") rows.sort(function (a, b) { return b.s - a.s; });
  var shown = tlAll ? rows : rows.slice(0, 60);
  var tb = '<div class="tl-tools"><span class="btn' + (tlSort === "recent" ? " on" : "") + '" data-tl="recent">MOST RECENT</span>' +
    '<span class="btn' + (tlSort === "confidence" ? " on" : "") + '" data-tl="confidence">HIGHEST CONFIDENCE</span></div>';
  if (!rows.length) tb += '<div class="pp-none">NO SIGHTINGS YET (WRITTEN LIVE WHILE THEY ARE IN VIEW)</div>';
  shown.forEach(function (x) {
    tb += '<div class="tl ' + esc(x.kind) + '">' + (x.thumb ? '<img src="' + x.thumb + '">' : '<div class="noimg"></div>') +
      '<div class="ti"><b>' + esc(x.source.toUpperCase().slice(0, 30)) + "</b> " + esc(x.kind === "live" ? "" : x.kind.toUpperCase()) +
      "<br>" + esc(x.from) + (x.kind === "live" ? " &rarr; " + esc(x.to) + " &middot; " + esc(x.dur) + " &middot; " + x.frames + " FRAMES" : "") + "</div>" +
      '<div class="tsc"><div class="meter"><i style="width:' + Math.max(0, Math.min(100, x.s * 100)).toFixed(0) + '%"></i></div>' + esc(x.score) + "</div></div>";
  });
  if (rows.length > shown.length) tb += '<div class="btnrow"><span class="btn" data-tlall="1">SHOW ALL ' + rows.length + "</span></div>";
  h += pblk("SIGHTING TIMELINE", rows.length + " RECORDED", tb);

  var lb = "";
  if (!d.similar.length) lb = '<div class="pp-none">NO OTHER PROFILES</div>';
  d.similar.forEach(function (o) {
    lb += '<div class="alike' + (o.likely ? " likely" : "") + '">' + (o.thumb ? '<img src="' + o.thumb + '">' : "") +
      '<div class="ai"><b class="lnk" data-action="mem_page" data-person="' + o.id + '">' + esc(o.name) + "</b> " + esc(o.code) + "<br>SIMILARITY " + esc(o.score) +
      (o.likely ? " &middot; PROBABLY SAME PERSON" : "") + '</div><span class="btn" data-action="mem_merge_into" data-person="' + d.id + '" data-target="' + o.id + '">MERGE</span></div>';
  });
  var nb = d.notes ? '<div class="pp-notes">' + nl2br(d.notes) + "</div>" : '<div class="pp-none">NO NOTES</div>';
  nb += '<div class="btnrow"><span class="btn" data-action="mem_notes" data-person="' + d.id + '">EDIT NOTES</span></div>';
  var wb = "";
  if (!d.associates.length) wb = '<div class="pp-none">NEVER IN VIEW TOGETHER WITH ANOTHER PROFILE</div>';
  d.associates.forEach(function (o) {
    wb += '<div class="alike">' + (o.thumb ? '<img src="' + o.thumb + '">' : "") + '<div class="ai"><b class="lnk" data-action="mem_page" data-person="' + o.id + '">' +
      esc(o.name) + "</b> " + esc(o.code) + "<br>" + o.visits + " VISIT" + (o.visits === 1 ? "" : "S") + " TOGETHER &middot; " + esc(o.time) + " IN VIEW TOGETHER</div></div>";
  });
  h += '<div class="pp-three">' + pblk("OFTEN SEEN WITH", "SAME CAMERA, SAME TIME", wb) + pblk("LOOK-ALIKES", "CLOSEST OTHER PROFILES", lb) +
       pblk("OPERATOR NOTES", "LOCAL ONLY", nb) + "</div>";
  h += '<div class="pp-foot">ENCRYPTED AT REST (DPAPI) &middot; STORED ON THIS MACHINE ONLY &middot; MODEL ' + esc(d.model.toUpperCase()) + "</div></div>";
  if (h !== lastPageHtml) {
    var sc = $("#memview").scrollTop();
    $("#memview").html(h).scrollTop(sc);
    lastPageHtml = h;
  }
}
function renderRecent(m) {
  var h = '<div class="blk"><div class="t red"><span>PROFILE PAGE</span><span class="sm">ESC: GALLERY</span></div><div class="c mini">' +
    'CLICK ANOTHER PROFILE TO SWITCH.</div><div class="f"></div></div>';
  h += '<div class="sec-title">RECENTLY SEEN</div>';
  (m.recent || []).forEach(function (p) {
    h += '<div class="recent' + (m.page && p.id === m.page.id ? " sel" : "") + '" data-action="mem_page" data-person="' + p.id + '">' +
      (p.thumb ? '<img src="' + p.thumb + '">' : '<div class="noimg"></div>') + '<div><b>' + esc(p.code) + "</b> " + esc(p.name !== p.code ? p.name : "") +
      "<br><span>" + (p.live ? "<em>IN VIEW</em>" : esc(p.ago)) + "</span></div></div>";
  });
  $("#persondetail").html(h);
}
$(document).on("click", "[data-tl]", function (e) {
  e.stopPropagation(); tlSort = $(this).attr("data-tl"); lastPageHtml = ""; if (lastPage) renderPage(lastPage);
});
$(document).on("click", "[data-tlall]", function (e) {
  e.stopPropagation(); tlAll = true; lastPageHtml = ""; if (lastPage) renderPage(lastPage);
});
$(document).on("keydown", function (e) {
  if (e.key === "Escape" && state.mode === "memory" && lastPageId !== null && !(document.activeElement && document.activeElement.tagName === "INPUT")) send("mem_back");
});

function renderPerson(d) {
  if (!d) {
    $("#persondetail").html('<div class="blk"><div class="t">NO PROFILE SELECTED</div><div class="c mini">SELECT A PERSON TO SEE THEIR HISTORY</div><div class="f"></div></div>');
    return;
  }
  var h = '<div class="blk"><div class="t red"><span>' + esc((d.label || d.code).toUpperCase()) + '</span><span class="sm">' + esc(d.code) + '</span></div><div class="c">' +
    (d.thumb ? '<img class="pd-photo" src="' + d.thumb + '">' : "") +
    kv("LABEL", d.label || "-") + kv("FIRST SEEN", d.first) + kv("LAST SEEN", d.last) + kv("SIGHTINGS", d.sightings) +
    kv("IN VIEW NOW", d.live ? d.live.join(", ") : "NO", d.live ? "hot" : "") +
    '<div class="kv"><div class="k">VIEWS LEARNED:</div></div><div class="bins">';
  d.bin_names.forEach(function (n, i) { h += '<span class="' + (d.bins[i] ? "on" : "") + '">' + n + "</span>"; });
  h += "</div>";
  if (d.views && d.views.length) {
    h += '<div class="views">';
    d.views.forEach(function (v) { h += '<div><img src="' + v.thumb + '"><span>' + esc(v.yaw) + "&deg;</span></div>"; });
    h += "</div>";
  }
  if (d.similar && d.similar.length) {
    h += '<div class="kv"><div class="k">LOOK-ALIKES:</div></div>';
    d.similar.forEach(function (o) {
      h += '<div class="alike' + (o.likely ? " likely" : "") + '">' + (o.thumb ? '<img src="' + o.thumb + '">' : "") +
        '<div class="ai"><b>' + esc(o.name) + "</b> " + esc(o.code) + "<br>SIMILARITY " + esc(o.score) +
        (o.likely ? " &middot; PROBABLY SAME" : "") + '</div><span class="btn" data-action="mem_merge_into" data-person="' + d.id +
        '" data-target="' + o.id + '">MERGE</span></div>';
    });
  }
  h += '<div class="btnrow"><span class="btn" data-action="mem_rename" data-person="' + d.id + '">LABEL</span>' +
    '<span class="btn" data-action="mem_merge" data-person="' + d.id + '">MERGE INTO...</span>' +
    '<span class="btn red" data-action="mem_forget" data-person="' + d.id + '">FORGET</span></div></div><div class="f"></div></div>';
  h += '<div class="sec-title">SEEN HISTORY · ' + d.history.length + "</div>";
  if (!d.history.length) h += '<div class="mini">NO COMPLETED SIGHTINGS YET (WRITTEN WHEN THEY LEAVE VIEW)</div>';
  d.history.forEach(function (x) {
    h += '<div class="hist ' + esc(x.kind) + '">' + (x.thumb ? '<img src="' + x.thumb + '">' : "") + '<div class="hi"><b>' +
      esc(x.source.toUpperCase().slice(0, 26)) + "</b><br>" + esc(x.from) +
      (x.kind === "live" ? " &rarr; " + esc(x.to) + " · " + esc(x.dur) : " · PHOTO") + "<br>MATCH " + esc(x.score) + "</div></div>";
  });
  $("#persondetail").html(h);
}

function spark(v) {
  if (!v || v.length < 2) return "";
  var w = 220, h = 26, mx = Math.max(1, Math.max.apply(null, v));
  var pts = v.map(function (n, i) { return (i * w / (v.length - 1)).toFixed(1) + "," + (h - 2 - n * (h - 4) / mx).toFixed(1); }).join(" ");
  return '<div class="kv"><div class="k">IN VIEW 5M:</div><div class="v"><span>NOW ' + v[v.length - 1] + " · PEAK " + mx +
    '</span></div></div><svg class="spark" viewBox="0 0 ' + w + " " + h + '" preserveAspectRatio="none"><polyline points="' + pts +
    '" fill="none" stroke="#fff" stroke-width="1.2"/></svg>';
}
