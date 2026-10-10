"""Webcam fallback for the demo laptop: a local camera, the same pipeline and session script, cues on the speaker.

Capture: cv2.VideoCapture(--camera) where --camera is a device index (default 0), an RTSP/http URL, or a file path
(a file is paced at its own frame rate and is the synthetic camera used for tests; --loop repeats it). A capture
thread reads at whatever rate the source gives and keeps only the NEWEST frame; the main thread samples it at 10 fps by
wall clock (same rule as coach_agent.py: never a backlog), converts BGR->RGB and runs FramePipeline + Session with all
the agent's flags (--plan / --exercise / auto, --params, --thr, --counter, --tempo, ...).

Audio: the same CuePlayer and queue policy as the LiveKit track, but the sink is the laptop speaker through the
`sounddevice` package (PortAudio; the macOS wheel bundles it). When sounddevice is missing or has no output device the
cues are written instead to --audio-out (a wav with each cue at the wall-clock time it would have played) and the
status line says so. --show opens a cv2 window with the pose landmarks, the count, the session state and the last cue
(guarded: without a display it prints once and carries on; press q in the window to quit). Ctrl-C exits cleanly.

Run:  .venv/bin/python -m facetime.webcam_local --camera 0 --plan "squats:10,pushups:8" --show
      .venv/bin/python -m facetime.webcam_local --camera data/tmp/0011_jumpingjacks.mp4 --plan "jumping jacks:10" --audio wav
"""
import argparse, asyncio, json, os, sys, threading, time, wave
import numpy as np, cv2
from facetime import DATA, SAMPLE_RATE
from facetime.pipeline import FramePipeline, load_json_arg, FPS, COUNTERS
from facetime.session import Session
from facetime.audio_out import CuePlayer, BEEP

CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
               (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32), (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
               (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)]

class SpeakerSource:
    """CuePlayer sink on the default output device; the blocking write paces playback in real time."""
    def __init__(self):
        import sounddevice as sd
        self.stream = sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=480); self.stream.start()
        self.name = f'speaker: {sd.query_devices(sd.default.device[1])["name"]}'
    async def capture_frame(self, chunk):
        await asyncio.get_running_loop().run_in_executor(None, self.stream.write, np.asarray(chunk, np.int16).tobytes())
    async def wait_for_playout(self):
        await asyncio.sleep(0.05)
    def close(self):
        self.stream.stop(); self.stream.close()

class WavSource:
    """Fallback sink: emulates real-time playout (so the queue policy behaves the same) and writes the cues at their
    wall-clock times into one wav at exit."""
    def __init__(self, path):
        self.path = path; self.t0 = time.time(); self.chunks = []; self.name = f'wav: {path}'
    async def capture_frame(self, chunk):
        self.chunks.append((time.time(), np.asarray(chunk, np.int16).copy())); await asyncio.sleep(len(chunk) / SAMPLE_RATE)
    async def wait_for_playout(self):
        pass
    def close(self):
        if not self.chunks: return
        end = max(t for t, _ in self.chunks) - self.t0 + 1.0; out = np.zeros(int(end * SAMPLE_RATE) + SAMPLE_RATE, np.int16)
        for t, c in self.chunks:
            i = int((t - self.t0) * SAMPLE_RATE); out[i:i + len(c)] = c[:max(0, len(out) - i)]
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        w = wave.open(self.path, 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE); w.writeframes(out.tobytes()); w.close()
        print(f'[audio] wrote {self.path} ({len(out) / SAMPLE_RATE:.1f} s timeline, {len(self.chunks) * 0.02:.1f} s of cue audio)', flush=True)

def make_sink(mode, path):
    if mode in ('auto', 'speaker'):
        try:
            return SpeakerSource()
        except Exception as e:
            if mode == 'speaker': raise SystemExit(f'speaker output unavailable: {e}')
            print(f'[audio] no speaker output ({type(e).__name__}: {str(e)[:80]}); cues go to {path}', flush=True)
    return WavSource(path)

class Camera(threading.Thread):
    """Reads the source as fast as it delivers (a file at its own fps) and keeps the newest frame only."""
    def __init__(self, src, loop_file):
        super().__init__(daemon=True, name='camera')
        self.src = int(src) if str(src).isdigit() else src; self.is_file = isinstance(self.src, str) and os.path.exists(self.src)
        self.loop_file = loop_file; self.latest = None; self.seq = 0; self.stop = threading.Event(); self.ended = False; self.fps = None
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise SystemExit(f'cannot open camera/source {src!r}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.w, self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    def run(self):
        t0 = time.time(); k = 0
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file and self.loop_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); t0 = time.time(); k = 0; continue
                self.ended = True; break
            self.seq += 1; self.latest = (frame, time.time(), self.seq); k += 1
            if self.is_file and self.fps > 0:                         # pace a file like a live camera
                due = t0 + k / self.fps; now = time.time()
                if now < due: time.sleep(due - now)
        self.cap.release()

def draw(frame, lm, text_lines):
    h, w = frame.shape[:2]
    if lm is not None:
        pts = [(int(x * w), int(y * h)) for x, y, *_ in lm]
        for a, b in CONNECTIONS:
            if lm[a][3] > 0.3 and lm[b][3] > 0.3: cv2.line(frame, pts[a], pts[b], (0, 220, 120), 2)
        for (x, y), v in zip(pts, lm[:, 3]):
            if v > 0.3: cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
    for i, t in enumerate(text_lines):
        cv2.putText(frame, t, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4); cv2.putText(frame, t, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--camera', default='0', help='device index, RTSP/http URL or a file path (synthetic camera)')
    ap.add_argument('--loop', action='store_true', help='repeat a file source'); ap.add_argument('--show', action='store_true')
    ap.add_argument('--audio', default='auto', choices=['auto', 'speaker', 'wav']); ap.add_argument('--audio-out', default=os.path.join(DATA, 'tmp', 'webcam_cues.wav'))
    ap.add_argument('--exercise', default=None); ap.add_argument('--plan', default=None)
    ap.add_argument('--model', default=os.path.join(DATA, 'models', 'clf_pose.pt')); ap.add_argument('--pose', default='full')
    ap.add_argument('--params', default='{}'); ap.add_argument('--thr', default=None)
    ap.add_argument('--rest-s', type=float, default=5.0); ap.add_argument('--set-timeout', type=float, default=75.0)
    ap.add_argument('--tempo', default=None); ap.add_argument('--auto-classes', default='demo', choices=['demo', 'all'])
    ap.add_argument('--counter', default='v2', choices=['v2', 'v2plain', 'e1']); ap.add_argument('--no-announce', dest='announce', action='store_false')
    ap.add_argument('--trigger', default='auto', help='learned correction trigger: a model path, auto (data/models/trigger_lr.json if present) or none')
    ap.add_argument('--log', default=os.path.join(DATA, 'tmp', 'webcam_events.jsonl')); ap.add_argument('--max-s', type=float, default=None)
    ap.add_argument('--no-auto-lite', dest='auto_lite', action='store_false')
    ap.add_argument('--latency-probe', action='store_true', help='beep immediately when both wrists go above the head (after 1 s below)')
    ap.add_argument('--gesture-next', action='store_true', help='a T-pose held 1.5 s says "Okay, moving on" and closes the set / skips the rest / starts the exercise')
    ap.add_argument('--beep-every', type=float, default=None, help='beep at every N s wall-clock multiple (cuts speech)')
    ap.add_argument('--audio-queue-ms', type=int, default=60)
    a = ap.parse_args()
    # validate the plan / exercise names and load the models before touching the camera or the audio device
    from facetime.audio_out import CueBank
    bank = CueBank()
    pipe = FramePipeline(a.model, None, a.pose, load_json_arg(a.params, {}), load_json_arg(a.thr), rep_params=COUNTERS[a.counter], trigger=a.trigger, gesture_next=a.gesture_next)
    def say(text, intent, meta): loop.call_soon_threadsafe(player.say, text, intent, meta)
    sess = Session(pipe, say, plan=a.plan, exercise=a.exercise, announce=a.announce, rest_s=a.rest_s, set_timeout_s=a.set_timeout,
                   tempo=load_json_arg(a.tempo, None), auto_classes=a.auto_classes, clip_s=bank.duration, gesture_next=a.gesture_next)
    cam = Camera(a.camera, a.loop)
    print(f'[camera] {a.camera!r}: {cam.w}x{cam.h} at {cam.fps:.1f} fps ({"file" if cam.is_file else "live"}); sampling at {FPS} fps newest-frame', flush=True)
    sink = make_sink(a.audio, a.audio_out)                       # on the main thread so a failure exits cleanly
    loop = asyncio.new_event_loop(); threading.Thread(target=loop.run_forever, daemon=True, name='audio-loop').start()
    os.makedirs(os.path.dirname(a.log) or '.', exist_ok=True); logf = open(a.log, 'a'); state = dict(last_cue='', played=0)
    def on_record(rec):
        logf.write(json.dumps(rec) + '\n'); logf.flush()
        if rec['intent'] == 'probe' and rec.get('t_push'):
            state['probe'] = (1000 * (rec['t_push'] - rec['t_detect']), 1000 * (rec['t_push'] - rec['t_recv']))
            print(f"[probe] arms up: detect -> beep pushed {state['probe'][0]:.0f} ms, frame -> beep pushed {state['probe'][1]:.0f} ms (on-box)", flush=True); return
        if rec['intent'] == 'beep' and rec.get('t_push'):
            print(f"[beep] scheduled {time.strftime('%H:%M:%S', time.localtime(rec['t_sched']))}, pushed +{1000 * (rec['t_push'] - rec['t_sched']):.0f} ms", flush=True); return
        if rec['status'] == 'played': state['played'] += 1; state['last_cue'] = rec['text']
        print(f"[cue] {time.strftime('%H:%M:%S')} {rec['status']:8} [{rec['intent']:10}] {rec['text']!r} {rec.get('why_dropped', '')}", flush=True)
    player = CuePlayer(bank=bank, source=sink, on_record=on_record, queue_ms=a.audio_queue_ms)
    asyncio.run_coroutine_threadsafe(player.run(), loop)
    async def beeper():
        while True:
            t_next = (int(time.time() // a.beep_every) + 1) * a.beep_every
            await asyncio.sleep(max(0.0, t_next - time.time())); player.say_now(BEEP, 'beep', dict(t_sched=t_next, t_decision=time.time()))
    if a.beep_every: asyncio.run_coroutine_threadsafe(beeper(), loop)
    probe_said = [False]; import collections as _c; recv_hist = _c.deque(maxlen=4)
    print(f'[webcam] mode {sess.mode} ({a.plan or a.exercise or "auto-detect"}), counter {a.counter}, trigger {pipe.trigger_path or "off"}, pose {pipe.pose_variant}, audio {sink.name}', flush=True)
    show = a.show
    if show and not have_display():
        print('[show] no display (DISPLAY/WAYLAND_DISPLAY unset); continuing without the window', flush=True); show = False
    cam.start(); tick = 1.0 / FPS; nxt = time.monotonic(); last_seq = 0; t_start = time.monotonic(); done_at = None
    win = dict(t=time.monotonic(), n=0, pose=0.0, clf=0.0, seq0=0, ok=0); processed = 0; dropped = 0; note = ''
    try:
        while True:
            now = time.monotonic()
            if now < nxt: time.sleep(nxt - now)
            nxt += tick
            if time.monotonic() > nxt + tick: nxt = time.monotonic() + tick
            if a.max_s and time.monotonic() - t_start > a.max_s: print(f'[webcam] --max-s {a.max_s} reached'); break
            if cam.ended and cam.latest is not None and cam.latest[2] == last_seq: print('[webcam] source ended'); break
            if sess.done:
                done_at = done_at or time.monotonic()
                if time.monotonic() - done_at > 4: print('[webcam] plan finished'); break
            slot = cam.latest
            if slot is None or slot[2] == last_seq: continue
            frame, t_recv, seq = slot; dropped += seq - last_seq - 1; last_seq = seq
            r = pipe.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); t_dec = time.time(); processed += 1; recv_hist.append(t_recv)
            win['n'] += 1; win['pose'] += r['pose_ms']; win['clf'] += r['clf_ms']; win['ok'] += r['ok']
            meta = dict(frame_seq=seq, t_capture=t_recv, t_recv=t_recv, t_decision=t_dec, pose_ms=round(r['pose_ms'], 1), clf_ms=round(r['clf_ms'], 1))
            if a.latency_probe:
                if not probe_said[0] and time.monotonic() - t_start >= 2.5: probe_said[0] = True; loop.call_soon_threadsafe(player.say, 'Probe ready, raise both arms', 'announce', dict(meta))
                if r.get('probe_fire'):
                    span = int(r.get('probe_span', 1)); t_first = recv_hist[-span] if len(recv_hist) >= span else t_recv
                    loop.call_soon_threadsafe(player.say_now, BEEP, 'probe', dict(meta, t_recv=t_first, t_detect=time.time()))
            sess.on_frame(r, meta)
            if show:
                lines = [f"{pipe.exercise or sess.mode}  count {sess.last_count}  {sess.state}", f"last cue: {state['last_cue']}",
                         f"pose {r['pose_ms']:.0f} ms  {'ok' if r['ok'] else 'no pose'}  {pipe.pose_variant}"]
                try:
                    cv2.imshow('coach', draw(frame, pipe.last_landmarks, lines))
                    if cv2.waitKey(1) & 0xFF == ord('q'): print('[webcam] q pressed'); break
                except cv2.error as e:
                    print(f'[show] no display ({str(e).splitlines()[0][:90]}); continuing without the window', flush=True); show = False
            el = time.monotonic() - win['t']
            if el >= 5.0:
                recv = (cam.seq - win['seq0']) / el; fps = win['n'] / el; n = max(win['n'], 1); pose_ms = win['pose'] / n
                if a.auto_lite and pipe.pose_variant != 'lite' and recv >= 8 and fps < 8.5 and pose_ms > 60:
                    pipe.switch_pose('lite'); note = ' POSE->lite (full too slow for 10 fps here)'
                print(f"[status] camera {recv:4.1f} fps, processed {fps:4.1f} fps, pose {pose_ms:5.1f} ms ({pipe.pose_variant}), clf {win['clf'] / n:4.1f} ms/frame, "
                      f"dropped {dropped}, pose ok {win['ok']}/{win['n']}, cues played {state['played']} dropped {player.n_dropped} underruns {player.underruns}, "
                      f"session {sess.state}/{pipe.exercise or '-'} count {sess.last_count}, audio {sink.name.split(':')[0]}"
                      + (f", probe on-box {state['probe'][1]:.0f} ms" if state.get('probe') else '') + note, flush=True)
                win = dict(t=time.monotonic(), n=0, pose=0.0, clf=0.0, seq0=cam.seq, ok=0)
    except KeyboardInterrupt:
        print('[webcam] interrupted', flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f'[webcam] stopped on an error: {type(e).__name__}: {e}', flush=True)
    cam.stop.set()
    for _ in range(40):
        if not player.playing and player.pending is None: break
        time.sleep(0.1)
    loop.call_soon_threadsafe(player.stop); time.sleep(0.2); sink.close(); pipe.close(); logf.close()
    try: cv2.destroyAllWindows()
    except Exception: pass
    pl = [x for x in player.records if x['status'] == 'played']
    print(f'[webcam] done: frames {cam.seq} captured, {processed} processed, {dropped} dropped; pose {pipe.pose_ms_sum / max(pipe.n, 1):.1f} ms/frame; '
          f'cues played {len(pl)}, dropped {player.n_dropped}; log {a.log}', flush=True)
    for x in pl: print(f"   {time.strftime('%H:%M:%S', time.localtime(x['t_push']))} +{x['t_push'] - t_wall0:6.2f}s [{x['intent']:10}] {x['text']!r}")

def have_display():
    """cv2's Qt/GTK backend aborts the process (not a Python exception) without a display, so check first."""
    return sys.platform == 'darwin' or sys.platform.startswith('win') or bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))

t_wall0 = time.time()
if __name__ == '__main__':
    main()
