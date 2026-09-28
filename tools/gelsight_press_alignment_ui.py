#!/usr/bin/env python3
"""Local side-by-side press reference and live GelSight Mini alignment UI."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_DATASET = Path("/home/suhang/datasets/octopi_fabric_v2_latest_sessions")
DEFAULT_SESSION = Path("F082/sessions/front/session_001")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--camera", help="UVC source; auto-detects GelSight Mini when omitted")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/gelsight_press_alignment"))
    parser.add_argument("--target-force", type=float, help="reference peak force; defaults to session metadata")
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def detect_camera() -> str:
    candidates = sorted(glob.glob("/dev/v4l/by-id/*GelSight*video-index0"))
    if candidates:
        return candidates[0]
    for candidate in ("/dev/video14", "/dev/video4", "/dev/video0"):
        if Path(candidate).exists():
            return candidate
    return "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._GelSight_Mini_R0B_2DWF-0RJM_2DWF0RJM-video-index0"


def force_segments(rows: list[tuple[float, float]], threshold: float = 1.0) -> list[dict]:
    segments: list[dict] = []
    start: float | None = None
    peak_force = float("-inf")
    peak_time = 0.0
    for timestamp, force in rows:
        if force >= threshold:
            if start is None:
                start = timestamp
                peak_force, peak_time = force, timestamp
            elif force > peak_force:
                peak_force, peak_time = force, timestamp
        elif start is not None:
            if timestamp - start >= 0.12:
                segments.append(
                    {"contact_start": start, "contact_end": timestamp, "peak_force": peak_force, "peak_time": peak_time}
                )
            start = None
            peak_force = float("-inf")
    return segments


def prepare_reference(dataset: Path, session_relative: Path, output_dir: Path, target_force: float | None) -> dict:
    session = (dataset / session_relative).resolve()
    video = session / "gelsight.mp4"
    force_csv = session / "nano17.csv"
    meta_path = session / "session_meta.json"
    for required in (video, force_csv, meta_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target = target_force if target_force is not None else float(meta.get("force", {}).get("target_peak_force_N", 15.0))
    rows: list[tuple[float, float]] = []
    with force_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("valid", "True").lower() == "true":
                rows.append((float(row["t_monotonic_sec"]), float(row["normal_force_N"])))
    segments = force_segments(rows)
    if not segments:
        raise RuntimeError(f"no press segment found in {force_csv}")
    selected = min(segments, key=lambda item: abs(item["peak_force"] - target))
    clip_start = max(0.0, selected["peak_time"] - 0.8)
    clip_end = selected["peak_time"] + 0.8
    curve = [
        {"t": round(timestamp - clip_start, 5), "force": round(force, 4)}
        for timestamp, force in rows
        if clip_start <= timestamp <= clip_end
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    clip = output_dir / "reference_press.mp4"
    poster = output_dir / "reference_peak.jpg"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{clip_start:.6f}", "-i", str(video), "-t", f"{clip_end - clip_start:.6f}",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(clip),
    ]
    subprocess.run(command, check=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{selected['peak_time']:.6f}", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(poster),
        ],
        check=True,
    )
    payload = {
        "fabric_id": meta.get("fabric_id"),
        "side": meta.get("side"),
        "session_id": meta.get("session_id"),
        "source": str(session),
        "target_force": target,
        "peak_force": round(selected["peak_force"], 3),
        "peak_time_source": round(selected["peak_time"], 5),
        "contact_start": round(selected["contact_start"] - clip_start, 5),
        "contact_end": round(selected["contact_end"] - clip_start, 5),
        "peak_time": round(selected["peak_time"] - clip_start, 5),
        "duration": round(clip_end - clip_start, 5),
        "curve": curve,
        "press_count_in_session": len(segments),
        "video": str(clip.resolve()),
        "poster": str(poster.resolve()),
    }
    (output_dir / "reference.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GelSight 压头位置对照</title>
<style>
:root{color-scheme:dark;--bg:#080b10;--panel:#10151d;--line:#263242;--text:#edf2f7;--muted:#8fa1b5;--cyan:#55d6be;--amber:#ffbd59;--red:#ff6b6b}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% -20%,#1a2735 0,var(--bg) 48%);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif;min-height:100vh}
header{height:62px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 24px;background:#0b1017dd;backdrop-filter:blur(12px)}
.brand{font-size:17px;font-weight:700;letter-spacing:.02em}.status{display:flex;align-items:center;gap:9px;color:var(--muted)}.dot{width:9px;height:9px;border-radius:50%;background:var(--amber);box-shadow:0 0 12px currentColor}.dot.ok{background:var(--cyan)}
main{padding:18px 22px 24px;max-width:1800px;margin:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:linear-gradient(145deg,#121923,#0d1219);border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 12px 40px #0005}
.cardhead{display:flex;align-items:center;justify-content:space-between;height:48px;padding:0 15px;border-bottom:1px solid var(--line)}.title{font-weight:650}.tag{font:11px ui-monospace,monospace;color:var(--muted);background:#080c12;padding:5px 8px;border:1px solid #202b38;border-radius:6px}
.viewport{aspect-ratio:4/3;position:relative;background:#020304;display:grid;place-items:center;overflow:hidden}.viewport video,.viewport img.feed{width:100%;height:100%;object-fit:contain;display:block}.feed{background:#020304}
.overlay{position:absolute;inset:0;pointer-events:none}.cross:before,.cross:after{content:"";position:absolute;background:#55d6beaa}.cross:before{left:50%;top:8%;bottom:8%;width:1px}.cross:after{top:50%;left:8%;right:8%;height:1px}.ring{position:absolute;left:50%;top:50%;width:34%;aspect-ratio:1;border:1px dashed #55d6beaa;border-radius:50%;transform:translate(-50%,-50%)}.gridline{position:absolute;background:#ffffff25}.v1{left:33.33%;top:0;bottom:0;width:1px}.v2{left:66.66%;top:0;bottom:0;width:1px}.h1{top:33.33%;left:0;right:0;height:1px}.h2{top:66.66%;left:0;right:0;height:1px}
.empty{position:absolute;text-align:center;color:var(--muted);max-width:330px;padding:20px}.empty strong{display:block;color:var(--amber);font-size:15px;margin-bottom:6px}
.controls{display:flex;align-items:center;gap:10px;padding:12px 14px;border-top:1px solid var(--line)}button{background:#182333;border:1px solid #2b3b50;color:var(--text);border-radius:8px;padding:8px 12px;cursor:pointer}button:hover{border-color:#507090}.grow{flex:1}.readout{font:12px ui-monospace,monospace;color:var(--muted)}
.chart{height:100px;border-top:1px solid var(--line);position:relative;background:#090d13}.chart canvas{width:100%;height:100%}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}.metric{background:#0e141c;border:1px solid var(--line);border-radius:11px;padding:12px 14px}.metric label{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}.metric b{display:block;font-size:19px;margin-top:4px}.hint{margin-top:14px;color:var(--muted);background:#0c1219;border-left:3px solid var(--cyan);padding:10px 13px;border-radius:4px}.warn{color:var(--amber)}
@media(max-width:950px){.grid{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}
</style></head>
<body>
<header><div class="brand">GelSight · 压头位置对照</div><div class="status"><span id="dot" class="dot"></span><span id="cameraStatus">正在连接 GelSight Mini…</span></div></header>
<main>
<div class="grid">
 <section class="card"><div class="cardhead"><span class="title">参考按压时序</span><span id="refTag" class="tag">载入中</span></div>
  <div class="viewport"><video id="refVideo" muted playsinline></video><div class="overlay"><i class="gridline v1"></i><i class="gridline v2"></i><i class="gridline h1"></i><i class="gridline h2"></i><i class="cross"></i><i class="ring"></i></div></div>
  <div class="chart"><canvas id="forceChart"></canvas></div>
  <div class="controls"><button id="play">▶ 播放</button><button id="peak">跳到峰值</button><label><input id="loop" type="checkbox" checked> 循环</label><span class="grow"></span><span id="time" class="readout">0.00 s</span></div>
 </section>
 <section class="card"><div class="cardhead"><span class="title">实时 GelSight Mini</span><span id="liveTag" class="tag">2DWF0RJM</span></div>
  <div class="viewport"><img id="live" class="feed" alt="GelSight 实时画面"><div id="empty" class="empty"><strong>等待相机画面</strong>若设备刚插入，请等待数秒；画面会自动重连。</div><div class="overlay"><i class="gridline v1"></i><i class="gridline v2"></i><i class="gridline h1"></i><i class="gridline h2"></i><i class="cross"></i><i class="ring"></i></div></div>
  <div class="controls"><button id="reconnect">重新连接</button><label><input id="guides" type="checkbox" checked> 对位网格</label><span class="grow"></span><span class="readout">中心十字 + 1/3 网格</span></div>
 </section>
</div>
<div class="metrics"><div class="metric"><label>参考布料</label><b id="fabric">—</b></div><div class="metric"><label>参考峰值</label><b id="force">—</b></div><div class="metric"><label>接触区间</label><b id="contact">—</b></div><div class="metric"><label>相机源</label><b id="source" style="font-size:13px">—</b></div></div>
<div class="hint"><span class="warn">调整建议：</span>先在右侧无接触画面下把压头投影移到中心环内，再轻触；对照左侧“峰值帧”的接触斑位置与形状。此界面只做视觉对位，不会控制机械臂或施加压力。</div>
</main>
<script>
let meta=null;const v=document.querySelector('#refVideo'),canvas=document.querySelector('#forceChart'),ctx=canvas.getContext('2d');
function draw(){if(!meta)return;const dpr=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);let max=Math.max(...meta.curve.map(x=>x.force),1),min=Math.min(...meta.curve.map(x=>x.force),0);let x=t=>t/meta.duration*w,y=f=>h-12-(f-min)/(max-min)*(h-25);ctx.strokeStyle='#263242';ctx.beginPath();ctx.moveTo(0,y(0));ctx.lineTo(w,y(0));ctx.stroke();ctx.strokeStyle='#55d6be';ctx.lineWidth=2;ctx.beginPath();meta.curve.forEach((p,i)=>(i?ctx.lineTo(x(p.t),y(p.force)):ctx.moveTo(x(p.t),y(p.force))));ctx.stroke();ctx.strokeStyle='#ffbd59';ctx.beginPath();ctx.moveTo(x(v.currentTime),0);ctx.lineTo(x(v.currentTime),h);ctx.stroke();ctx.fillStyle='#8fa1b5';ctx.font='11px ui-monospace';ctx.fillText('法向力 / N',8,14)}
fetch('/api/reference').then(r=>r.json()).then(m=>{meta=m;v.src='/reference.mp4';document.querySelector('#refTag').textContent=`${m.fabric_id} · ${m.side} · ${m.session_id}`;document.querySelector('#fabric').textContent=`${m.fabric_id} / ${m.side}`;document.querySelector('#force').textContent=`${m.peak_force.toFixed(2)} N`;document.querySelector('#contact').textContent=`${m.contact_start.toFixed(2)}–${m.contact_end.toFixed(2)} s`;draw()});
v.ontimeupdate=()=>{document.querySelector('#time').textContent=`${v.currentTime.toFixed(2)} s`;draw()};v.onended=()=>{if(document.querySelector('#loop').checked){v.currentTime=0;v.play()}};
document.querySelector('#play').onclick=e=>{if(v.paused){v.play();e.currentTarget.textContent='❚❚ 暂停'}else{v.pause();e.currentTarget.textContent='▶ 播放'}};
document.querySelector('#peak').onclick=()=>{v.currentTime=meta.peak_time;v.pause();document.querySelector('#play').textContent='▶ 播放'};
const img=document.querySelector('#live'),empty=document.querySelector('#empty'),dot=document.querySelector('#dot'),status=document.querySelector('#cameraStatus');
function connect(){empty.style.display='block';status.textContent='正在连接 GelSight Mini…';dot.classList.remove('ok');img.src='/live.mjpeg?ts='+Date.now()}
img.onload=()=>{empty.style.display='none';status.textContent='GelSight Mini 实时 · 已连接';dot.classList.add('ok')};img.onerror=()=>{empty.style.display='block';status.textContent='相机未就绪 · 3 秒后重试';dot.classList.remove('ok');setTimeout(connect,3000)};
document.querySelector('#reconnect').onclick=connect;document.querySelector('#guides').onchange=e=>document.querySelectorAll('.overlay').forEach(x=>x.style.display=e.target.checked?'block':'none');
fetch('/api/status').then(r=>r.json()).then(s=>{document.querySelector('#source').textContent=s.camera;document.querySelector('#liveTag').textContent=s.serial||'GelSight Mini'}).finally(connect);
addEventListener('resize',draw);addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();document.querySelector('#play').click()}if(e.key.toLowerCase()==='p')document.querySelector('#peak').click()});
</script></body></html>"""


class AlignmentServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], reference: dict, output_dir: Path, camera: str):
        super().__init__(address, AlignmentHandler)
        self.reference = reference
        self.output_dir = output_dir
        self.camera = camera


class AlignmentHandler(BaseHTTPRequestHandler):
    server: AlignmentServer

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stdout.write(f"[ui] {self.address_string()} {fmt % args}\n")

    def send_bytes(self, payload: bytes, content_type: str, cache: str = "no-store") -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
        elif path == "/api/reference":
            self.send_bytes(json.dumps(self.server.reference, ensure_ascii=False).encode(), "application/json")
        elif path == "/api/status":
            status = {"camera": self.server.camera, "serial": "2DWF0RJM", "exists": Path(self.server.camera).exists()}
            self.send_bytes(json.dumps(status).encode(), "application/json")
        elif path == "/reference.mp4":
            self.send_file(self.server.output_dir / "reference_press.mp4", "video/mp4")
        elif path == "/live.mjpeg":
            self.stream_camera()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def send_file(self, path: Path, content_type: str | None = None) -> None:
        data = path.read_bytes()
        self.send_bytes(data, content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream", "public,max-age=3600")

    def stream_camera(self) -> None:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "v4l2", "-input_format", "mjpeg", "-video_size", "640x480", "-framerate", "30",
            "-i", self.server.camera, "-vf", "fps=20", "-an", "-q:v", "4", "-f", "mpjpeg", "pipe:1",
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=ffmpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            assert process.stdout is not None
            while chunk := process.stdout.read(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> int:
    args = parse_args()
    camera = args.camera or detect_camera()
    reference = prepare_reference(args.dataset.resolve(), args.session, args.output_dir.resolve(), args.target_force)
    server = AlignmentServer((args.host, args.port), reference, args.output_dir.resolve(), camera)
    url = f"http://{args.host}:{args.port}"
    print(
        f"Reference: {reference['fabric_id']}/{reference['side']} peak={reference['peak_force']:.2f} N "
        f"at source t={reference['peak_time_source']:.3f}s\nCamera: {camera}\nUI: {url}",
        flush=True,
    )
    if args.open_browser:
        threading.Timer(
            0.8,
            lambda: subprocess.Popen(
                ["google-chrome", "--new-window", f"--app={url}", "--window-size=1500,980"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            ),
        ).start()
    shutdown = lambda *_: threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    server.serve_forever()
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
