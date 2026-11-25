# -*- coding: utf-8 -*-
"""Local desktop app that turns voice notes into Persian transcript text."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import av

_av_open = av.open


def _open_compat(*args, **kwargs):
    """PyAV shim: drop unsupported metadata_errors for newer av builds."""
    kwargs.pop("metadata_errors", None)
    return _av_open(*args, **kwargs)


av.open = _open_compat

from faster_whisper import WhisperModel  # noqa: E402

HOST = "127.0.0.1"
ROOT = Path(__file__).resolve().parent
PORT_FILE = Path(tempfile.gettempdir()) / "pardaz-voice-port.txt"
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
MODELS = {
    "medium": "medium",
    "large-v3": "large-v3",
    "small": "small",
}
# Persian initial prompt steers Whisper toward Persian script in the transcript.
PROMPT = (
    "گفتار فارسی محاوره‌ای. متن را با خط فارسی، املای درست و نشانه‌گذاری طبیعی بنویس. "
    "کلمه‌ها را به لاتین ننویس مگر نام خاص انگلیسی باشد."
)

_lock = threading.Lock()
_model = None
_model_key = None
_httpd = None


def weights_dir(name):
    """Return the newest local Hugging Face snapshot that contains model.bin."""
    root = HF_HUB / f"models--Systran--faster-whisper-{name}" / "snapshots"
    if not root.is_dir():
        return None
    ready = []
    for snap in root.iterdir():
        weights = snap / "model.bin"
        if weights.is_file() and weights.stat().st_size > 1_000_000:
            ready.append(weights)
    if not ready:
        return None
    return str(max(ready, key=lambda path: path.stat().st_mtime).parent)


def resolve_model(key):
    """Pick a usable local model; fall back to small when the requested one is missing."""
    name = MODELS.get(key, "small")
    found = weights_dir(name)
    if found:
        return found, name
    small = weights_dir("small")
    if small:
        return small, "small"
    raise RuntimeError("Speech model was not found on this system.")


def load_model(key):
    """Load (or reuse) a Faster-Whisper model on CPU with int8 compute."""
    global _model, _model_key
    target, used = resolve_model(key)
    with _lock:
        if _model is not None and _model_key == used:
            return _model, used
        _model = WhisperModel(target, device="cpu", compute_type="int8")
        _model_key = used
        return _model, used


def transcribe(path, key, on_progress=None):
    """Transcribe one audio/video file to Persian text and report progress 0-100."""
    def report(pct):
        if on_progress is None:
            return
        on_progress(max(0, min(100, int(pct))))

    report(0)
    model, used = load_model(key)
    segments, info = model.transcribe(
        str(path),
        language="fa",
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 450},
        condition_on_previous_text=True,
        initial_prompt=PROMPT,
    )
    duration = float(info.duration or 0)
    paragraphs = []
    buf = []
    prev_end = None
    last_pct = 0
    for seg in segments:
        if duration > 0:
            pct = int(min(99, round(100 * float(seg.end) / duration)))
            if pct > last_pct:
                last_pct = pct
                report(pct)
        text = (seg.text or "").strip()
        if not text:
            continue
        if prev_end is not None and seg.start - prev_end > 1.15 and buf:
            paragraphs.append(" ".join(buf))
            buf = []
        buf.append(text)
        prev_end = seg.end
    if buf:
        paragraphs.append(" ".join(buf))
    body = "\n\n".join(paragraphs).strip()
    report(100)
    return body or "No speech was detected in this file.", duration, used != MODELS.get(key, key)


PAGE = r"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>تبدیل ویس به متن فارسی</title>
<link rel="icon" href="/logo.png">
<style>
  @font-face {
    font-family: Estedad;
    src: url("/fonts/Estedad.woff2") format("woff2");
    font-weight: 100 900;
    font-style: normal;
    font-display: block;
  }
  :root {
    color-scheme: dark;
    --bg: #0D1225;
    --ink: #f3f6fb;
    --muted: #93a0bd;
    --cyan: #39ADDA;
    --line: rgba(57, 173, 218, .22);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; color: var(--ink);
    font-family: Estedad, Tahoma, sans-serif;
    font-weight: 420;
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
    background:
      radial-gradient(900px 420px at 100% -10%, rgba(57,173,218,.18), transparent 60%),
      radial-gradient(700px 380px at -10% 110%, rgba(57,173,218,.08), transparent 55%),
      var(--bg);
  }
  header {
    position: sticky; top: 0; z-index: 5;
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    padding: 16px 28px;
    background: rgba(13,18,37,.88);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--line);
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .mark {
    width: 42px; height: 42px; border-radius: 12px; overflow: hidden; flex: none;
  }
  .mark img { width: 100%; height: 100%; display: block; object-fit: cover; }
  .name-line { display: flex; align-items: baseline; gap: 8px; }
  .family { color: var(--cyan); font-weight: 800; font-size: 15px; }
  h1 { margin: 0; font-size: 20px; font-weight: 700; letter-spacing: -.01em; }
  .brand small { display: block; color: var(--muted); font-size: 12.5px; font-weight: 450; margin-top: 3px; }
  .tools { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  select, button { font: inherit; }
  select {
    background: #161C34; color: var(--ink); border: 1px solid var(--line);
    border-radius: 999px; padding: 8px 14px;
  }
  button {
    border: 1px solid var(--line); background: #161C34; color: var(--ink);
    border-radius: 999px; padding: 8px 14px; cursor: pointer; font-weight: 600; font-size: 14px;
  }
  button:hover { border-color: rgba(57,173,218,.55); }
  button.ghost { background: transparent; }
  main { width: min(920px, calc(100% - 40px)); margin: 0 auto; padding: 28px 0 48px; }
  .stage {
    min-height: 280px; border-radius: 28px; padding: 42px 28px;
    display: grid; place-items: center; text-align: center;
    background: linear-gradient(180deg, rgba(22,28,52,.92), rgba(13,18,37,.78));
    border: 1px dashed rgba(57,173,218,.45);
    box-shadow: inset 0 0 80px rgba(57,173,218,.06);
  }
  .stage h2 { margin: 10px 0 6px; font-size: 30px; font-weight: 650; }
  .stage p { margin: 0; color: var(--muted); line-height: 2; font-weight: 450; font-size: 16px; }
  .wave { display: flex; gap: 5px; align-items: flex-end; height: 36px; }
  .wave i {
    width: 5px; border-radius: 99px; background: var(--cyan); display: block;
    animation: bar 1.1s ease-in-out infinite;
  }
  .wave i:nth-child(1) { height: 12px; animation-delay: 0s; }
  .wave i:nth-child(2) { height: 26px; animation-delay: .1s; }
  .wave i:nth-child(3) { height: 18px; animation-delay: .2s; }
  .wave i:nth-child(4) { height: 32px; animation-delay: .15s; }
  .wave i:nth-child(5) { height: 14px; animation-delay: .25s; }
  @keyframes bar { 50% { transform: scaleY(.45); } }
  .stage.is-hidden, #feed:empty { display: none; }
  #feed { display: flex; flex-direction: column; gap: 14px; margin-top: 18px; }
  article {
    position: relative; overflow: hidden;
    background: rgba(22, 28, 52, .92);
    border: 1px solid var(--line); border-radius: 22px; padding: 18px 20px 16px;
    box-shadow: 0 18px 40px rgba(0,0,0,.22);
    animation: rise .35s ease;
  }
  article::before {
    content: ""; position: absolute; inset: 0 auto 0 0; width: 3px;
    background: linear-gradient(#39ADDA, transparent);
  }
  @keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
  .top { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
  .who { display: flex; gap: 12px; align-items: center; min-width: 0; }
  .num {
    width: 36px; height: 36px; flex: none; border-radius: 12px;
    display: grid; place-items: center; font-weight: 800; color: #0D1225;
    background: var(--cyan);
  }
  .who strong { display: block; font-size: 15.5px; font-weight: 650; }
  .who span, .pill { color: var(--muted); font-size: 12.5px; font-weight: 500; }
  .pill {
    border: 1px solid var(--line); border-radius: 999px; padding: 4px 10px; white-space: nowrap;
  }
  article[data-state="run"] .pill { color: var(--cyan); border-color: rgba(57,173,218,.45); }
  article[data-state="done"] .pill { color: #8ef0c2; border-color: rgba(142,240,194,.35); }
  article[data-state="err"] .pill { color: #ffb4a8; border-color: rgba(255,180,168,.4); }
  .body {
    margin: 16px 0 0; white-space: pre-wrap;
    line-height: 2.25; font-size: 18px; font-weight: 450;
  }
  .waitline { margin: 14px 0 0; color: var(--muted); }
  .meter {
    margin-top: 12px; height: 8px; border-radius: 99px; overflow: hidden;
    background: rgba(57,173,218,.14);
  }
  .meter i {
    display: block; height: 100%; width: 0%; border-radius: inherit;
    background: linear-gradient(270deg, #39ADDA, #1E7EAB);
    transition: width .35s ease;
  }
  .meter i.is-wait { width: 100%; opacity: .45; animation: pulse 1.1s ease-in-out infinite; }
  @keyframes pulse { 50% { opacity: .15; } }
  .card-actions { display: flex; gap: 8px; margin-top: 12px; }
  .card-actions button { padding: 6px 12px; font-size: 13px; }
  .veil {
    display: none; position: fixed; inset: 14px; z-index: 20;
    border-radius: 28px; border: 2px dashed var(--cyan);
    background: rgba(13, 18, 37, .86);
    backdrop-filter: blur(8px);
    align-items: center; justify-content: center; text-align: center;
  }
  .veil.on { display: flex; }
  .veil strong { display: block; font-size: 32px; font-weight: 650; margin-bottom: 8px; }
  .veil span { color: var(--muted); }
  footer {
    position: sticky; bottom: 0; padding: 10px 28px 14px;
    color: var(--muted); font-size: 13px;
    background: linear-gradient(transparent, var(--bg) 40%);
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="mark"><img src="/logo.png" alt="پرداز"></div>
    <div>
      <div class="name-line">
        <span class="family">پرداز</span>
        <h1>تبدیل ویس</h1>
      </div>
      <small>هر ویس، یک کارت متن فارسی</small>
    </div>
  </div>
  <div class="tools">
    <select id="model" aria-label="دقت">
      <option value="small" selected>سریع</option>
      <option value="medium">دقیق</option>
      <option value="large-v3">دقیق‌ترین</option>
    </select>
    <button id="pick" type="button">افزودن ویس</button>
    <button id="copy" class="ghost" type="button">کپی همه</button>
    <button id="save" class="ghost" type="button">ذخیره</button>
    <button id="wipe" class="ghost" type="button">پاک کردن</button>
    <button id="quit" class="ghost" type="button">بستن</button>
  </div>
</header>
<main>
  <section class="stage" id="stage">
    <div>
      <div class="wave" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></div>
      <h2>ویس‌ها را همین‌جا رها کنید</h2>
      <p>هر تعداد که بخواهید، با هم. برای هر ویس یک کارت جدا ساخته می‌شود و متنش داخل همان کارت می‌آید.</p>
    </div>
  </section>
  <div id="feed"></div>
</main>
<div class="veil" id="veil"><div><strong>رها کنید</strong><span>همهٔ ویس‌ها با هم پذیرفته می‌شوند</span></div></div>
<footer id="status">آماده. ویس را بکشید و روی پنجره رها کنید.</footer>
<input id="files" type="file" multiple hidden accept="audio/*,video/*,.ogg,.opus,.mp3,.wav,.m4a,.webm,.flac,.aac">
<script>
const queue = [];
let seq = 0;
let busy = false;
const feed = document.getElementById("feed");
const stage = document.getElementById("stage");
const status = document.getElementById("status");
const veil = document.getElementById("veil");
const input = document.getElementById("files");

function fa(n) { return String(n).replace(/\d/g, d => "۰۱۲۳۴۵۶۷۸۹"[d]); }
function clock(seconds) {
  const m = Math.floor((seconds || 0) / 60);
  const s = Math.round((seconds || 0) % 60);
  return fa(m) + ":" + fa(String(s).padStart(2, "0"));
}
function paintStage() {
  stage.classList.toggle("is-hidden", queue.length > 0);
}
function ensureCard(item) {
  if (item.el) return item.el;
  const art = document.createElement("article");
  art.dataset.state = "wait";
  art.innerHTML =
    '<div class="top"><div class="who"><b class="num"></b><div><strong></strong><span class="meta"></span></div></div><span class="pill"></span></div>' +
    '<div class="slot"></div><div class="card-actions"></div>';
  item.el = art;
  feed.append(art);
  return art;
}
function paint(item) {
  const art = ensureCard(item);
  art.dataset.state = item.state;
  art.querySelector(".num").textContent = fa(item.id);
  art.querySelector("strong").textContent = item.file.name;
  const sizeKb = Math.max(1, Math.round(item.file.size / 1024));
  art.querySelector(".meta").textContent = fa(sizeKb) + " کیلوبایت";
  const pill = art.querySelector(".pill");
  const slot = art.querySelector(".slot");
  const actions = art.querySelector(".card-actions");
  actions.innerHTML = "";
  if (item.state === "wait") {
    pill.textContent = "در صف";
    slot.innerHTML = '<p class="waitline">منتظر شنیدن این ویس…</p>';
  } else if (item.state === "run") {
    slot.innerHTML = '<p class="waitline"></p><div class="meter" role="progressbar"><i></i></div>';
    art.scrollIntoView({ behavior: "smooth", block: "nearest" });
    paintProgress(item);
  } else if (item.state === "done") {
    pill.textContent = clock(item.duration);
    slot.innerHTML = "";
    const p = document.createElement("p");
    p.className = "body";
    p.textContent = item.text;
    slot.append(p);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "کپی این متن";
    btn.onclick = async () => {
      await navigator.clipboard.writeText(item.text);
      status.textContent = "متن «" + item.file.name + "» کپی شد.";
    };
    actions.append(btn);
  } else {
    pill.textContent = "خطا";
    slot.innerHTML = "";
    const p = document.createElement("p");
    p.className = "body";
    p.textContent = item.text;
    slot.append(p);
  }
  paintStage();
}
function paintProgress(item) {
  const art = item.el;
  if (!art || item.state !== "run") return;
  const pct = Math.max(0, Math.min(99, item.progress | 0));
  const pill = art.querySelector(".pill");
  const line = art.querySelector(".waitline");
  const bar = art.querySelector(".meter i");
  if (pill) pill.textContent = fa(pct) + "٪";
  if (line) {
    line.textContent = pct > 0 ? fa(pct) + "٪ از این ویس شنیده شد" : "در حال شنیدن این ویس…";
  }
  if (bar) {
    bar.classList.toggle("is-wait", pct <= 0);
    bar.style.width = pct > 0 ? pct + "%" : "";
    bar.parentElement.setAttribute("aria-valuenow", String(pct));
  }
  const finished = queue.filter(q => q.state === "done" || q.state === "err").length;
  status.textContent = "در حال تبدیل " + fa(finished + 1) + " از " + fa(queue.length) + " — " + fa(pct) + "٪";
}
async function readJob(res, item) {
  if (!res.ok) {
    let message = "خطا";
    try {
      const data = await res.json();
      if (data.error) message = data.error;
    } catch (e) {}
    throw new Error(message);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let final = null;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      const msg = JSON.parse(line);
      if (msg.error) throw new Error(msg.error);
      if (typeof msg.progress === "number") {
        item.progress = msg.progress;
        paintProgress(item);
      }
      if (msg.done) final = msg;
    }
  }
  if (!final) throw new Error("پاسخ ناقص بود");
  return final;
}
function add(list) {
  let n = 0;
  for (const file of list) {
    if (!file || file.size === 0 && file.type === "") continue;
    if (queue.some(q => q.file.name === file.name && q.file.size === file.size && q.file.lastModified === file.lastModified)) continue;
    const item = { id: ++seq, file, state: "wait", text: "", duration: 0, progress: 0, el: null };
    queue.push(item);
    paint(item);
    n++;
  }
  if (n) {
    status.textContent = fa(queue.length) + " ویس در صف است. تبدیل همین حالا شروع می‌شود.";
    pump();
  }
}
async function pump() {
  if (busy) return;
  busy = true;
  const model = document.getElementById("model").value;
  let usedFallback = false;
  while (true) {
    const item = queue.find(q => q.state === "wait");
    if (!item) break;
    item.state = "run";
    paint(item);
    item.progress = 0;
    paintProgress(item);
    try {
      const res = await fetch("/transcribe?model=" + encodeURIComponent(model) + "&name=" + encodeURIComponent(item.file.name), {
        method: "POST",
        body: item.file
      });
      const data = await readJob(res, item);
      item.state = "done";
      item.text = data.text;
      item.duration = data.duration;
      if (data.fallback) usedFallback = true;
    } catch (err) {
      item.state = "err";
      item.text = err.message;
    }
    paint(item);
  }
  busy = false;
  const ok = queue.filter(q => q.state === "done").length;
  if (!ok) status.textContent = "آماده.";
  else if (usedFallback) status.textContent = "تمام شد. مدل انتخاب‌شده روی دستگاه نصب نیست؛ متن با مدل سریع ساخته شد.";
  else status.textContent = "تمام شد. متن " + fa(ok) + " ویس آماده است.";
}
function allText() {
  return queue.filter(q => q.state === "done").map(q => q.file.name + "\n" + q.text).join("\n\n");
}
document.getElementById("pick").onclick = () => input.click();
input.onchange = () => { add(input.files); input.value = ""; };
document.getElementById("copy").onclick = async () => {
  const text = allText();
  if (!text) return;
  await navigator.clipboard.writeText(text);
  status.textContent = "همهٔ متن‌ها کپی شد.";
};
document.getElementById("save").onclick = () => {
  const text = allText();
  if (!text) return;
  const blob = new Blob([text + "\n"], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "متن-ویس‌ها.txt";
  a.click();
};
document.getElementById("wipe").onclick = () => {
  if (busy) return;
  queue.length = 0;
  feed.innerHTML = "";
  paintStage();
  status.textContent = "پاک شد. ویس تازه را رها کنید.";
};
document.getElementById("quit").onclick = () => { fetch("/shutdown").finally(() => window.close()); };

let depth = 0;
window.addEventListener("dragenter", e => {
  if (![...e.dataTransfer.types].includes("Files")) return;
  depth++;
  veil.classList.add("on");
});
window.addEventListener("dragover", e => { e.preventDefault(); });
window.addEventListener("dragleave", () => {
  depth = Math.max(0, depth - 1);
  if (!depth) veil.classList.remove("on");
});
window.addEventListener("drop", e => {
  e.preventDefault();
  depth = 0;
  veil.classList.remove("on");
  if (e.dataTransfer.files && e.dataTransfer.files.length) add(e.dataTransfer.files);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    """Local HTTP API that serves the UI and streams transcription progress."""

    def log_message(self, fmt, *args):
        return

    def _send(self, code, body, content_type):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, "ok", "text/plain; charset=utf-8")
            return
        if self.path.startswith("/shutdown"):
            self._send(200, "bye", "text/plain; charset=utf-8")
            threading.Thread(target=lambda: os._exit(0), daemon=True).start()
            return
        if self.path.startswith("/logo.png"):
            logo = ROOT / "logo.png"
            if logo.is_file():
                self._send(200, logo.read_bytes(), "image/png")
                return
            self._send(404, "no", "text/plain; charset=utf-8")
            return
        if self.path.startswith("/fonts/"):
            name = Path(unquote(self.path.split("?", 1)[0])).name
            font = ROOT / "fonts" / name
            if font.is_file() and font.suffix.lower() == ".woff2":
                self._send(200, font.read_bytes(), "font/woff2")
                return
            self._send(404, "no", "text/plain; charset=utf-8")
            return
        self._send(200, PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        """Accept a raw audio body and stream NDJSON progress + final text."""
        if not self.path.startswith("/transcribe"):
            self._send(404, "no", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        params = {}
        for part in query.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k] = unquote(v)
        name = params.get("name") or "voice"
        key = params.get("model") or "small"
        if key not in MODELS:
            key = "small"
        safe = Path(name).name or "voice"
        dest = Path(tempfile.gettempdir()) / "pardaz-voice-in" / (uuid.uuid4().hex[:8] + "-" + safe)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        self.protocol_version = "HTTP/1.1"
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(obj):
            data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
            self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
            self.wfile.write(data)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        try:
            text, duration, fallback = transcribe(dest, key, lambda pct: emit({"progress": pct}))
            emit({
                "done": True,
                "name": safe,
                "duration": duration,
                "text": text,
                "fallback": fallback,
            })
        except Exception:
            err = traceback.format_exc().splitlines()[-1]
            try:
                emit({"error": err})
            except OSError:
                return
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                return


def edge_path():
    """Locate Microsoft Edge for the app-window fallback launcher."""
    candidates = [
        os.environ.get("PROGRAMFILES(X86)", "") + r"\Microsoft\Edge\Application\msedge.exe",
        os.environ.get("PROGRAMFILES", "") + r"\Microsoft\Edge\Application\msedge.exe",
        os.environ.get("LOCALAPPDATA", "") + r"\Microsoft\Edge\Application\msedge.exe",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return ""


def shell_exe():
    """Path to the optional WebView2 desktop host executable."""
    return ROOT / "shell" / "VoiceToText.exe"


def install_shell():
    """Write host.json and register the desktop shell shortcut when available."""
    exe = shell_exe()
    if not exe.is_file():
        return
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    payload = {
        "appId": "Pardaz.VoiceToText",
        "shortcut": "VoiceToText",
        "title": "Voice to Persian Text",
        "width": 1180,
        "height": 840,
        "script": str(ROOT / "app.py"),
        "workdir": str(ROOT),
        "icon": str(ROOT / "app.ico"),
        "pythonw": str(pythonw if pythonw.is_file() else Path(sys.executable)),
        "portFile": PORT_FILE.name,
        "noWindowEnv": "VOICETOTEXT_NO_WINDOW",
    }
    (exe.parent / "host.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run(
        [str(exe), "--install"],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def open_window(port):
    """Open the desktop shell, Edge app mode, or the system browser."""
    if os.environ.get("VOICETOTEXT_NO_WINDOW"):
        return
    exe = shell_exe()
    if exe.is_file():
        subprocess.Popen(
            [str(exe)],
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return
    url = f"http://{HOST}:{port}"
    edge = edge_path()
    if edge:
        subprocess.Popen(
            [edge, f"--app={url}", "--window-size=1180,840", "--disable-extensions"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    import webbrowser
    webbrowser.open(url)


def already_up():
    """Return the live server port if a previous instance is still healthy."""
    if not PORT_FILE.exists():
        return 0
    try:
        port = int(PORT_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return 0
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/health", timeout=0.6) as res:
            if res.read().strip() == b"ok":
                return port
    except Exception:
        return 0
    return 0


def main():
    """Start the local server once, reuse it if already running, and open the UI."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    running = already_up()
    if running:
        install_shell()
        open_window(running)
        return
    install_shell()
    server = ThreadingHTTPServer((HOST, 0), Handler)
    port = server.server_address[1]
    PORT_FILE.write_text(str(port), encoding="utf-8")
    open_window(port)
    try:
        server.serve_forever()
    finally:
        try:
            PORT_FILE.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
