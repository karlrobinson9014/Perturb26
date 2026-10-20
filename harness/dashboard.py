"""Perturb26 operator dashboard (pm2 app perturb26-dashboard; humans only, like the rest of harness/).

Watch the loop's agent sessions live (stream-json transcripts in logs/), see the ledger,
best score, spend and harness log, and steer the loop: standing directives, a queue of
one-shot ideas/plans for the next iterations, and pause / resume / stop.

    .venv/bin/python harness/dashboard.py --port 18080      # binds 127.0.0.1 only

Expose it through the instance's authenticated reverse proxy (Caddy, /etc/portal.yaml), never
directly. Writes are limited to $OPERATOR_DIR (outside the repo) and the STOP / PAUSE flags.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
STATUS = ROOT / "status"
OPERATOR_DIR = Path(os.environ.get("OPERATOR_DIR", str(Path.home() / "perturb26-operator")))
SESSION_RE = re.compile(r"^(iter-\d{4}-(build|research)|review-\d{4})\.jsonl$")
QUEUE_RE = re.compile(r"^[\w.-]+\.md$")
MAX_TEXT = 4000


def _read(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return default


def _tsv(path: Path, last: int) -> list[dict]:
    lines = _read(path).splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]][-last:]


def _tail(path: Path, n: int) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 64_000))
            return "\n".join(handle.read().decode("utf-8", "replace").splitlines()[-n:])
    except OSError:
        return ""


def _pm2() -> list[dict]:
    pm2 = shutil.which("pm2")
    if not pm2:
        candidates = sorted(Path("/opt/nvm/versions/node").glob("*/bin/pm2"))
        pm2 = str(candidates[-1]) if candidates else None
    if not pm2:
        return []
    try:
        env = dict(os.environ)
        env["PATH"] = f"{Path(pm2).parent}:{env.get('PATH', '')}"
        raw = subprocess.run([pm2, "jlist"], capture_output=True, text=True, timeout=10, env=env).stdout
        apps = json.loads(raw[raw.find("["):]) if "[" in raw else []
    except Exception:
        return []
    out = []
    for app in apps:
        env = app.get("pm2_env", {})
        if not str(app.get("name", "")).startswith("perturb26"):
            continue
        out.append({"name": app.get("name"), "status": env.get("status"), "restarts": env.get("restart_time"),
                    "uptime_s": int((time.time() * 1000 - env.get("pm_uptime", 0)) / 1000) if env.get("pm_uptime") else None})
    return out


def _gpu() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def _sessions() -> list[dict]:
    items = []
    for path in LOGS.glob("*.jsonl"):
        if not SESSION_RE.match(path.name):
            continue
        st = path.stat()
        items.append({"name": path.name, "size": st.st_size, "mtime": st.st_mtime,
                      "live": time.time() - st.st_mtime < 45,
                      "has_prompt": (LOGS / path.name.replace(".jsonl", ".prompt.md")).exists()})
    return sorted(items, key=lambda x: x["mtime"], reverse=True)


def _clip(text: str, limit: int = MAX_TEXT) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [{len(text) - limit} more chars]"


def _events(line: str) -> list[dict]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return []
    kind = event.get("type")
    if kind == "system" and event.get("subtype") == "init":
        servers = [f"{s.get('name')}:{s.get('status')}" for s in event.get("mcp_servers", [])]
        return [{"kind": "init", "text": f"model={event.get('model')} tools={len(event.get('tools', []))} mcp={', '.join(servers)}"}]
    if kind == "assistant":
        out = []
        for item in event.get("message", {}).get("content", []):
            if item.get("type") == "text":
                out.append({"kind": "text", "text": _clip(item.get("text", ""))})
            elif item.get("type") == "thinking":
                out.append({"kind": "thinking", "text": _clip(item.get("thinking", ""), 2000)})
            elif item.get("type") == "tool_use":
                out.append({"kind": "tool", "name": item.get("name"),
                            "text": _clip(json.dumps(item.get("input", {}), indent=1), 2500)})
        return out
    if kind == "user":
        out = []
        content = event.get("message", {}).get("content", [])
        for item in content if isinstance(content, list) else []:
            if item.get("type") != "tool_result":
                continue
            body = item.get("content")
            if isinstance(body, list):
                body = "\n".join(part.get("text", "") for part in body if isinstance(part, dict))
            out.append({"kind": "result", "error": bool(item.get("is_error")), "text": _clip(str(body or ""), 3000)})
        return out
    if kind == "result":
        return [{"kind": "final", "text": f"{event.get('subtype')}: turns={event.get('num_turns')} "
                                          f"cost=${event.get('total_cost_usd', 0) or 0:.2f} "
                                          f"duration={round((event.get('duration_ms') or 0) / 60000, 1)} min"}]
    return []


def _session(name: str, offset: int) -> dict:
    path = LOGS / name
    if not SESSION_RE.match(name) or not path.exists():
        return {"error": "no such session"}
    with path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read(2_000_000)
    end = chunk.rfind(b"\n")
    if end < 0:
        return {"events": [], "offset": offset, "live": time.time() - path.stat().st_mtime < 45}
    events = []
    for line in chunk[: end + 1].decode("utf-8", "replace").splitlines():
        events.extend(_events(line))
    return {"events": events, "offset": offset + end + 1, "live": time.time() - path.stat().st_mtime < 45}


def _operator() -> dict:
    def items(sub: str, last: int | None = None) -> list[dict]:
        paths = sorted((OPERATOR_DIR / sub).glob("*.md"))
        if last:
            paths = sorted(paths, key=lambda p: p.stat().st_mtime)[-last:][::-1]
        return [{"name": p.name, "text": _read(p)} for p in paths]

    return {"directives": _read(OPERATOR_DIR / "directives.md"), "queue": items("queue"),
            "active": items("active"), "done": items("done", 20)}


def _state() -> dict:
    best = {}
    try:
        best = json.loads(_read(STATUS / "best.json") or "{}")
    except json.JSONDecodeError:
        pass
    iterations = _tsv(STATUS / "iterations.tsv", 400)
    spend = sum(float(r.get("cost_usd") or 0) for r in iterations if (r.get("cost_usd") or "").replace(".", "", 1).isdigit())
    sessions = _sessions()
    return {
        "now": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "pm2": _pm2(), "gpu": _gpu(),
        "stop": (ROOT / "STOP").exists(), "pause": (ROOT / "PAUSE").exists(),
        "best": {k: v for k, v in best.items() if k != "dev_result"},
        "ledger": _tsv(STATUS / "results.tsv", 40)[::-1],
        "iterations": iterations[-40:][::-1],
        "spend_usd": round(spend, 2),
        "harness_log": _tail(LOGS / "harness.log", 80),
        "sessions": sessions[:60],
        "operator": _operator(),
    }


def _control(action: str) -> dict:
    if action == "pause":
        (ROOT / "PAUSE").touch()
    elif action == "resume":
        (ROOT / "PAUSE").unlink(missing_ok=True)
    elif action == "stop":
        (ROOT / "STOP").touch()
    elif action == "start":
        subprocess.Popen(["bash", str(ROOT / "harness" / "ctl.sh"), "start"], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    else:
        return {"error": f"unknown action {action}"}
    return {"ok": True}


def _add_queue(text: str, mode: str) -> dict:
    text = text.strip()
    if not text:
        return {"error": "empty"}
    if mode not in ("build", "research"):
        return {"error": "mode must be build or research"}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())[:40].strip("-") or "item"
    path = OPERATOR_DIR / "queue" / f"{stamp}-{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"mode: {mode}\n{text}\n", encoding="utf-8")
    return {"ok": True, "name": path.name}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # quiet
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/":
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if url.path == "/api/state":
            return self._json(_state())
        if url.path == "/api/session":
            return self._json(_session(query.get("name", [""])[0], int(query.get("offset", ["0"])[0] or 0)))
        if url.path == "/api/prompt":
            name = query.get("name", [""])[0]
            if not SESSION_RE.match(name):
                return self._json({"error": "bad name"}, 400)
            return self._json({"text": _read(LOGS / name.replace(".jsonl", ".prompt.md"), "(no prompt saved)")})
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        length = min(int(self.headers.get("Content-Length") or 0), 200_000)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)
        path = urlparse(self.path).path
        if path == "/api/directives":
            OPERATOR_DIR.mkdir(parents=True, exist_ok=True)
            (OPERATOR_DIR / "directives.md").write_text(str(data.get("text", "")).strip() + "\n", encoding="utf-8")
            return self._json({"ok": True})
        if path == "/api/queue":
            return self._json(_add_queue(str(data.get("text", "")), str(data.get("mode", "research"))))
        if path == "/api/queue/delete":
            name = str(data.get("name", ""))
            if not QUEUE_RE.match(name):
                return self._json({"error": "bad name"}, 400)
            (OPERATOR_DIR / "queue" / name).unlink(missing_ok=True)
            return self._json({"ok": True})
        if path == "/api/control":
            return self._json(_control(str(data.get("action", ""))))
        self._json({"error": "not found"}, 404)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perturb26 Loop</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1c2128;--mute:#5f6b7a;--line:#dde2e8;--accent:#2457c5;--ok:#1d7f45;--warn:#a86400;--bad:#b4232c;--code:#f0f2f5;--tool:#eef3fd;--res:#f4f6f8}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1216;--panel:#171b21;--ink:#e3e7ec;--mute:#93a0b0;--line:#2a313a;--accent:#7aa2ff;--ok:#4cc38a;--warn:#e0a341;--bad:#ff6b73;--code:#1f252d;--tool:#1b2333;--res:#1c2127}}
:root[data-theme="dark"]{--bg:#0f1216;--panel:#171b21;--ink:#e3e7ec;--mute:#93a0b0;--line:#2a313a;--accent:#7aa2ff;--ok:#4cc38a;--warn:#e0a341;--bad:#ff6b73;--code:#1f252d;--tool:#1b2333;--res:#1c2127}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:2}
h1{font-size:16px;margin:0 8px 0 0}h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute);margin:0 0 8px}
.chip{padding:2px 8px;border-radius:999px;border:1px solid var(--line);font-size:12px;white-space:nowrap}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.mute{color:var(--mute)}
button,select{font:inherit;padding:4px 10px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
main{display:grid;grid-template-columns:280px minmax(0,1fr) 760px;gap:12px;padding:12px 16px}
@media (max-width:1400px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
.list{max-height:60vh;overflow:auto}.list div{padding:6px 8px;border-radius:6px;cursor:pointer;display:flex;justify-content:space-between;gap:8px}
.list div:hover,.list div.sel{background:var(--code)}
#feed{height:72vh;overflow:auto;display:flex;flex-direction:column;gap:6px}
.ev{border-radius:6px;padding:6px 8px;white-space:pre-wrap;word-break:break-word;font-size:13px}
.ev.text{background:transparent;border-left:3px solid var(--accent)}.ev.thinking{color:var(--mute);font-style:italic;border-left:3px solid var(--line)}
.ev.tool{background:var(--tool);font-family:ui-monospace,Menlo,monospace;font-size:12px}.ev.result{background:var(--res);font-family:ui-monospace,Menlo,monospace;font-size:12px;max-height:240px;overflow:auto}
.ev.result.err{border-left:3px solid var(--bad)}.ev.final,.ev.init{color:var(--mute);font-size:12px}
.ev .tag{font-weight:600;font-family:system-ui;font-size:11px;color:var(--mute);display:block;margin-bottom:2px}
textarea{width:100%;min-height:90px;font:13px/1.4 ui-monospace,Menlo,monospace;background:var(--code);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:8px}
.row{display:flex;gap:8px;align-items:center;margin-top:6px;flex-wrap:wrap}
.item{border:1px solid var(--line);border-radius:6px;padding:6px 8px;margin-top:6px;white-space:pre-wrap;font-size:12px;font-family:ui-monospace,Menlo,monospace}
table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:3px 6px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
pre.log{max-height:260px;overflow:auto;background:var(--code);padding:8px;border-radius:6px;font-size:11px;white-space:pre-wrap}
.sec{margin-top:14px}.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:12px}
</style></head><body>
<header>
  <h1>Perturb26 loop</h1>
  <span id="chips"></span>
  <span style="flex:1"></span>
  <button onclick="ctl('pause')">Pause</button><button onclick="ctl('resume')">Resume</button>
  <button onclick="if(confirm('Stop the loop after the current iteration?'))ctl('stop')">Stop</button>
  <button class="primary" onclick="ctl('start')">Start</button>
</header>
<main>
  <section class="panel">
    <h2>Sessions</h2><div id="sessions" class="list"></div>
    <div class="sec"><h2>Best</h2><div id="best" class="kv"></div></div>
  </section>
  <section class="panel">
    <div class="row" style="margin:0 0 8px"><h2 style="margin:0" id="title">Session</h2><span style="flex:1"></span>
      <label class="mute"><input type="checkbox" id="follow" checked> follow live</label>
      <label class="mute"><input type="checkbox" id="think"> thinking</label>
      <button onclick="showPrompt()">Prompt</button></div>
    <div id="feed"></div>
  </section>
  <section class="panel">
    <h2>Operator directives (every iteration)</h2>
    <textarea id="dir"></textarea>
    <div class="row"><button class="primary" onclick="saveDir()">Save directives</button><span id="dirmsg" class="mute"></span></div>
    <div class="sec"><h2>Queue an idea / plan (next iteration)</h2>
      <textarea id="qtext" placeholder="e.g. Try 8 lanes with targeted searches on the top classes from a 2 s -logit_true discovery run; compare to best on --limit 16 first."></textarea>
      <div class="row"><select id="qmode"><option>research</option><option>build</option></select>
        <button class="primary" onclick="addQ()">Add to queue</button></div>
      <div id="queue"></div></div>
    <div class="sec"><h2>Done</h2><div id="done"></div></div>
  </section>
  <section class="panel" style="grid-column:1/-1">
    <h2>Ledger</h2><div style="overflow:auto"><table id="ledger"></table></div>
    <div class="sec"><h2>Harness log</h2><pre class="log" id="hlog"></pre></div>
  </section>
</main>
<script>
let cur=null, off=0, dirDirty=false;
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function api(p,body){const r=await fetch(p,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});return r.json()}
async function ctl(a){await api('api/control',{action:a});refresh()}
async function saveDir(){await api('api/directives',{text:$('dir').value});dirDirty=false;$('dirmsg').textContent='saved '+new Date().toLocaleTimeString()}
$('dir').addEventListener('input',()=>{dirDirty=true;$('dirmsg').textContent='unsaved'});
async function addQ(){const t=$('qtext').value;if(!t.trim())return;await api('api/queue',{text:t,mode:$('qmode').value});$('qtext').value='';refresh()}
async function delQ(n){await api('api/queue/delete',{name:n});refresh()}
async function showPrompt(){if(!cur)return;const r=await api('api/prompt?name='+encodeURIComponent(cur));alert(r.text||r.error)}
function pick(name){cur=name;off=0;$('feed').innerHTML='';$('title').textContent=name;document.querySelectorAll('#sessions div').forEach(d=>d.classList.toggle('sel',d.dataset.n===name));poll()}
async function poll(){if(!cur)return;const r=await api('api/session?name='+encodeURIComponent(cur)+'&offset='+off);if(r.error)return;off=r.offset;
  const f=$('feed');const atBottom=f.scrollHeight-f.scrollTop-f.clientHeight<80;
  for(const e of r.events){if(e.kind==='thinking'&&!$('think').checked)continue;const d=document.createElement('div');
    d.className='ev '+e.kind+(e.error?' err':'');
    const tag=e.kind==='tool'?'tool: '+e.name:e.kind==='result'?(e.error?'tool error':'tool result'):e.kind==='text'?'agent':e.kind;
    d.innerHTML='<span class="tag">'+esc(tag)+'</span>'+esc(e.text);f.appendChild(d)}
  if(r.events.length&&$('follow').checked&&atBottom)f.scrollTop=f.scrollHeight}
function ago(t){const s=Date.now()/1000-t;return s<90?Math.round(s)+'s':s<5400?Math.round(s/60)+'m':Math.round(s/3600)+'h'}
async function refresh(){const s=await api('api/state');
  const chips=[];for(const a of s.pm2)chips.push(`<span class="chip ${a.status==='online'?'ok':'warn'}">${esc(a.name)}: ${esc(a.status)}</span>`);
  if(!s.pm2.length)chips.push('<span class="chip warn">pm2: no perturb26 apps</span>');
  if(s.pause)chips.push('<span class="chip warn">PAUSED</span>');if(s.stop)chips.push('<span class="chip bad">STOP set</span>');
  chips.push(`<span class="chip">spend $${s.spend_usd}</span>`);if(s.gpu)chips.push(`<span class="chip mute">GPU ${esc(s.gpu)}</span>`);
  $('chips').innerHTML=chips.join(' ');
  $('sessions').innerHTML=s.sessions.map(x=>`<div data-n="${esc(x.name)}" class="${x.name===cur?'sel':''}" onclick="pick('${esc(x.name)}')"><span>${x.live?'<b class="ok">● </b>':''}${esc(x.name.replace('.jsonl',''))}</span><span class="mute">${ago(x.mtime)}</span></div>`).join('')||'<span class="mute">no sessions yet</span>';
  if(!cur&&s.sessions.length)pick(s.sessions[0].name);
  else if($('follow').checked&&s.sessions.length&&s.sessions[0].live&&s.sessions[0].name!==cur)pick(s.sessions[0].name);
  $('best').innerHTML=Object.entries(s.best).map(([k,v])=>`<span class="mute">${esc(k)}</span><span>${esc(v)}</span>`).join('')||'<span class="mute">none yet</span>';
  if(!dirDirty)$('dir').value=s.operator.directives;
  const q=[...s.operator.active.map(x=>({...x,active:1})),...s.operator.queue];
  $('queue').innerHTML=q.map(x=>`<div class="item">${x.active?'<b class="ok">in progress</b>\n':''}${esc(x.text)}${x.active?'':`<div class="row"><button onclick="delQ('${esc(x.name)}')">Remove</button></div>`}</div>`).join('');
  $('done').innerHTML=s.operator.done.map(x=>`<div class="item mute">${esc(x.text)}</div>`).join('')||'<span class="mute">none</span>';
  const cols=['ts','iter','mode','commit','dev_score','holdout_score','decision','reason'];
  $('ledger').innerHTML='<tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr>'+s.ledger.map(r=>'<tr>'+cols.map(c=>`<td class="${c==='decision'?(r[c]==='accept'?'ok':r[c]==='discard'?'bad':''):''}">${esc(r[c])}</td>`).join('')+'</tr>').join('');
  $('hlog').textContent=s.harness_log;}
refresh();setInterval(refresh,5000);setInterval(poll,2000);
</script></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    for sub in ("queue", "active", "done"):
        (OPERATOR_DIR / sub).mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard on http://{args.host}:{args.port} (operator dir {OPERATOR_DIR})", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
