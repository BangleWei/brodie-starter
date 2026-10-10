"""The coach as a LiveKit participant (tiers b, c, d share this file).

Modes: --plan "squats:10,pushups:8,..." (demo default: scripted session, see facetime/session.py), --exercise X
(one exercise, no target) or neither (auto-detect the exercise from the classifier head). --params / --thr still
shape the Coach policy; --tempo overrides the seconds-per-rep targets for the tempo cues.

Contract: joins the room as AGENT_IDENTITY (default "brodie-agent") with a self-minted token, subscribes to the
video track named "facetime-remote-video" from BRIDGE_IDENTITY (default "chert-bridge"); any other participant's
video track is used too, with a log line, since the Chert bridge's identity and track name are assumptions; a track
that is republished (camera switch, rejoin) is picked up again and the session continues (under --once the agent leaves
only when no video comes back within 10 s). Publishes one audio track "coach-audio" (24 kHz mono s16, 20 ms frames) fed by
facetime/audio_out.py from the pre-rendered cue bank.

Frames: rtc.VideoStream delivers every decoded frame; only the NEWEST is kept in a one-slot buffer. A worker
thread samples that slot at 10 fps by wall clock (ticks with no new frame are idle, extra frames are dropped,
never a backlog), converts to RGB (short side 288, aspect kept), runs MediaPipe + Coach (facetime/pipeline.py)
and hands events to the player on the event loop. Each cue is one JSON line in --log with wall-clock times:
t_capture (the frame's timestamp_us when it is epoch-like, else receive time), t_recv, t_decision (Coach.step
returned), t_say, t_push (first audio frame pushed), t_done, pose_ms, clf_ms, status.

Immediacy: frames are processed as they arrive (>= 90 ms apart, --tick-sampling for the old fixed ticks); the audio queue
is 60 ms (--audio-queue-ms); --latency-probe beeps the instant both wrists go above the head and prints the on-box delay;
--beep-every N beeps on a wall-clock schedule; --video-quality low asks a simulcasting publisher for its smallest layer.
Rep counter: --counter v2 (default) is RepCounter2 with the demo guards added after the 2026-10-08 Mac tests (facetime.pipeline
REP_PARAMS_V2: squats vote only over hip/knee/shoulder signals, pushups only over elbow/shoulder/wrist signals, a body-state gate
(upright for squats and lunges, horizontal for pushups), 0.8 s of standing still before the first squat or lunge after a start
or a visibility pause, no squat while the hips move sideways faster than one torso length per second (walking into place) and no
first lunge from the arms alone); --counter v2plain is the counter exactly as it ran on the Mac (no guards); --counter e1 the old PCA counter.
Gesture command (opt-in): --gesture-next makes a T-pose (both arms straight out to the sides at shoulder height, facing the
camera) held for 1.5 s say "Okay, moving on" and close the current set (then the rest and the next exercise as usual), skip a
rest straight to the next exercise, or start an exercise still in its warm-up wait; in --exercise mode it ends the session; it
fires at most once per 5 s and jumping jacks do not trigger it (facetime/probe.py TPose, facetime/session.py). Off by default,
nothing changes without it; in --standby it may arrive per call in "args".
Run:  .venv/bin/python -m facetime.coach_agent --room rtcoach-test-abc --plan "squats:10,pushups:8" (or --exercise squats) \
        [--model data/models/clf_pose.pt] [--pose full|lite|heavy] [--params '{}'] [--thr thr.json] \
        [--log data/tmp/agent.jsonl] [--once] [--max-s 120] [--no-announce] [--no-auto-lite]
SIGINT exits cleanly (disconnect, summary). If "full" cannot hold 10 fps the pose model drops to "lite" and the
status line says so.
Standby (warm start; the Modal side holds the process ready before the call): --standby loads everything that does not need
the room (imports, pose model, classifier, trigger, cue bank), prints "[agent] standby ready in X.X s" and blocks on ONE JSON
line on stdin: {"room": str (required), "plan": str|null, "exercise": str|null, "max_s": number|null, "agent_token": str|null,
"args": [extra CLI tokens]|null}. room/plan/exercise/max_s override the CLI (plan and exercise exclusive), agent_token is
exported as RTCOACH_AGENT_TOKEN, args are re-parsed onto the namespace (pipeline flags such as --pose are fixed by then); then it
prints "[agent] standby released: room R mode M" and runs as if started with those flags. Empty line, EOF or bad JSON: exit 3.
"""
import argparse, collections, asyncio, json, logging, os, signal, sys, threading, time
_T0 = time.time()                        # standby timer fallback when /proc is unavailable; the heavy imports follow
import numpy as np
from livekit import rtc
from facetime import DATA, AGENT_IDENTITY, BRIDGE_IDENTITY, VIDEO_TRACK_NAME
from facetime.pipeline import FramePipeline, load_json_arg, FPS, COUNTERS
from facetime.audio_out import CuePlayer, CueBank, BEEP
ONCE_GRACE_S = 10.0          # --once: wait this long for a republished video track before leaving
import numpy as _np
from facetime.session import Session
from facetime.mint_token import mint

log = logging.getLogger('coach_agent')

def frame_to_rgb(frame):
    if frame.type != rtc.VideoBufferType.RGB24:
        frame = frame.convert(rtc.VideoBufferType.RGB24)
    buf = np.frombuffer(frame.data, np.uint8)
    h, w = frame.height, frame.width
    if len(buf) == h * w * 3:
        return buf.reshape(h, w, 3)
    stride = len(buf) // h                                  # padded rows
    return buf.reshape(h, stride)[:, :w * 3].reshape(h, w, 3)

class Agent:
    def __init__(self, a, loop, standby=False):
        """The room-independent, expensive part: cue bank and FramePipeline (pose model, classifier, trigger). The per-call part
        (setup_call: player, session, log, stamp reader) follows at once, unless standby=True: main() runs it after the release."""
        self.a = a; self.loop = loop
        self.room = rtc.Room(loop=loop)
        self.bank = CueBank()
        self.pipe = FramePipeline(a.model, None, a.pose, load_json_arg(a.params, {}), load_json_arg(a.thr), rep_params=COUNTERS[a.counter], trigger=a.trigger)
        self.player = self.session = self.logf = self.stamp_reader = None
        if not standby: self.setup_call(a)

    def setup_call(self, a):
        """Everything that depends on the room or on the per-call flags (plan/exercise, --once, --log, --read-stamp, --latency-probe,
        --audio-queue-ms): built from the released namespace so the standby overrides take effect."""
        self.a = a
        self.player = CuePlayer(bank=self.bank, on_record=self.log_cue, queue_ms=a.audio_queue_ms)
        gesture_next = bool(getattr(a, 'gesture_next', False)); self.pipe.set_gesture_next(gesture_next)   # per call, so a standby release can switch it on
        self.session = Session(self.pipe, self.say_from_worker, plan=a.plan, exercise=a.exercise, announce=a.announce,
                               rest_s=a.rest_s, set_timeout_s=a.set_timeout, tempo=load_json_arg(a.tempo, None), auto_classes=a.auto_classes, clip_s=self.bank.duration,
                               gesture_next=gesture_next)
        self.done_at = None
        self.latest = None; self.seq = 0; self.processed = 0; self.dropped = 0; self.idle_ticks = 0
        self.frame_event = threading.Event(); self.delays = []; self.recv_hist = collections.deque(maxlen=4); self.frame_size = None; self.last_probe = None; self.n_beeps = 0; self.probe_said = False
        self.stop = threading.Event(); self.done = asyncio.Event()
        self.video_task = None; self.video_from = None
        os.makedirs(os.path.dirname(a.log) or '.', exist_ok=True); self.logf = open(a.log, 'a')
        self.note = ''; self.ts_mode = None; self.first_frame_t = None; self.cues_played = 0; self.worker_error = None
        self.stamp_reader = None
        if getattr(a, 'read_stamp', False):                      # opt-in: decode the bridge's send-time stamp from each processed frame
            from facetime.frame_stamp import Reader
            self.stamp_reader = Reader(getattr(a, 'stamp_t0', None), getattr(a, 'stamp_log', None))

    # ---- cues ----
    def log_cue(self, rec):
        rec['room'] = self.a.room; self.logf.write(json.dumps(rec) + '\n'); self.logf.flush()
        if rec.get('status') == 'played' and rec.get('intent') not in ('probe', 'beep'): self.last_cue = (rec.get('text'), time.time())   # the mirror's bottom bar
        if rec['intent'] == 'probe' and rec.get('t_push'):
            d2p = 1000 * (rec['t_push'] - rec['t_detect']); r2p = 1000 * (rec['t_push'] - rec['t_recv']); self.last_probe = (d2p, r2p)
            print(f"[probe] arms up at {time.strftime('%H:%M:%S')}.{int(rec['t_detect'] * 1000) % 1000:03d}: detect -> beep pushed {d2p:.0f} ms, "
                  f"frame received -> beep pushed {r2p:.0f} ms (on-box; the phone hears it one network leg later)", flush=True); return
        if rec['intent'] == 'beep' and rec.get('t_push'):
            self.n_beeps += 1; print(f"[beep] #{self.n_beeps} scheduled {time.strftime('%H:%M:%S', time.localtime(rec['t_sched']))}.000, "
                                     f"pushed +{1000 * (rec['t_push'] - rec['t_sched']):.0f} ms (t_push {rec['t_push']:.3f})", flush=True); return
        if rec['status'] == 'played':
            self.cues_played += 1
            lat = '' if rec.get('t_decision') is None else f"  decision->push {1000 * (rec['t_push'] - rec['t_decision']):.0f} ms"
            print(f"[cue] {time.strftime('%H:%M:%S')} [{rec['intent']:10}] {rec['text']!r}{lat}", flush=True)
        else:
            print(f"[cue] {rec['status']} [{rec['intent']}] {rec['text']!r} {rec.get('why_dropped', '')}", flush=True)

    # ---- video ----
    def on_track_subscribed(self, track, pub, participant):
        if track.kind != rtc.TrackKind.KIND_VIDEO or (self.video_task is not None and not self.video_task.done()):
            return
        if participant.identity == AGENT_IDENTITY:
            return
        if pub.name != VIDEO_TRACK_NAME and participant.identity != BRIDGE_IDENTITY:   # the Chert bridge's identity and track name are assumptions
            print(f'[agent] video track {pub.name!r} from {participant.identity!r} is not the expected bridge track; using it anyway (only other video in the room)', flush=True)
        print(f'[agent] subscribed to video {pub.name!r} from {participant.identity!r} ({pub.width}x{pub.height}, simulcast {getattr(pub, 'simulcasted', False)})', flush=True)
        if self.a.video_quality != 'none' and getattr(pub, 'simulcasted', False):
            q = rtc.VideoQuality.Value('VIDEO_QUALITY_' + self.a.video_quality.upper())   # protobuf enum: VIDEO_QUALITY_LOW/MEDIUM/HIGH
            try: pub.set_video_quality(q); print(f'[agent] asked the SFU for the {self.a.video_quality.upper()} simulcast layer (the agent resizes to 288 px anyway)', flush=True)
            except Exception as e: print(f'[agent] set_video_quality failed: {e}', flush=True)
        elif self.a.video_quality != 'none':
            print('[agent] publisher does not simulcast: full resolution will be received', flush=True)
        self.video_from = participant.identity
        self.video_task = asyncio.create_task(self.read_video(track))

    async def read_video(self, track):
        stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)
        try:
            async for ev in stream:
                self.seq += 1; self.latest = (ev.frame, ev.timestamp_us, time.time(), self.seq); self.frame_event.set()
        finally:
            await stream.aclose()
        print('[agent] video stream ended', flush=True)
        self.video_task = None; self.video_from = None            # a republished track (camera switch, rejoin) is picked up again
        if self.a.once: self.loop.call_later(ONCE_GRACE_S, self._done_if_no_video)

    def _done_if_no_video(self):
        """--once: leave only if nobody republished video within the grace period (a camera switch or rejoin keeps the session)."""
        if self.video_task is None or self.video_task.done():
            print(f'[agent] no video for {ONCE_GRACE_S:.0f} s after the stream ended; leaving (--once)', flush=True); self.done.set()

    def _done_if_never_video(self):
        """--no-video-s N: a call whose video publisher never joined (a dropped Chert call, a bridge that never connected) must not
        hold the container for max_s; leave if no frame has arrived N seconds after connecting."""
        if self.first_frame_t is None and not self.done.is_set():
            print(f'[agent] no video {self.a.no_video_s:.0f} s after connecting; leaving (--no-video-s)', flush=True); self.done.set()

    def on_participant_disconnected(self, p):
        print(f'[agent] participant left: {p.identity}', flush=True)
        if self.a.once and p.identity == self.video_from: self.loop.call_later(ONCE_GRACE_S, self._done_if_no_video)

    def capture_time(self, ts_us, t_recv):
        """Use the frame timestamp only when it is on the wall clock; libwebrtc usually gives a monotonic-ish value."""
        if self.ts_mode is None:
            self.ts_mode = 'epoch' if abs(ts_us / 1e6 - t_recv) < 5 else 'recv'
            print(f'[agent] frame timestamp_us={ts_us} ({self.ts_mode}); capture time = {"frame ts" if self.ts_mode == "epoch" else "receive wall clock"}', flush=True)
        return ts_us / 1e6 if self.ts_mode == 'epoch' else t_recv

    def say_from_worker(self, text, intent, meta):
        self.loop.call_soon_threadsafe(self.player.say, text, intent, meta)

    # ---- worker thread: 10 fps by wall clock, newest frame only ----
    def worker(self):
        try:
            self._worker()
        except BaseException as e:
            log.exception('worker thread died: %r', e)
            print(f'[agent] worker thread died: {type(e).__name__}: {e}; exiting', flush=True)
            self.worker_error = e; self.loop.call_soon_threadsafe(self.done.set)

    def _worker(self):
        tick = 1.0 / FPS; nxt = time.monotonic(); last_seq = 0; last_proc = -1e9
        win = dict(t=time.monotonic(), n=0, pose=0.0, clf=0.0, seq0=0, idle=0, ok=0, d=[])
        while not self.stop.is_set():
            if self.a.tick_sampling:                                  # old mode: fixed 100 ms ticks (average 50 ms phase delay)
                now = time.monotonic()
                if now < nxt: time.sleep(nxt - now)
                nxt += tick
                if time.monotonic() > nxt + tick: nxt = time.monotonic() + tick
                slot = self.latest
            else:                                                     # arrival-aligned: process a frame as it arrives, >= 90 ms apart
                if not self.frame_event.wait(timeout=0.05): slot = None
                else:
                    self.frame_event.clear(); slot = self.latest
                    if slot is not None and slot[3] != last_seq:
                        since = time.monotonic() - last_proc
                        if since < 0.09:                              # too soon after the last one: wait for the 90 ms mark, then take the newest frame
                            time.sleep(0.09 - since); slot = self.latest; self.frame_event.clear()
            if slot is None or slot[3] == last_seq:
                win['idle'] += 1; self.idle_ticks += 1
            else:
                frame, ts_us, t_recv, seq = slot
                self.dropped += seq - last_seq - 1; last_seq = seq; last_proc = time.monotonic()
                t_cap = self.capture_time(ts_us, t_recv)
                if self.first_frame_t is None: self.first_frame_t = t_recv; print('[agent] first frame', frame.width, 'x', frame.height, flush=True)
                if self.frame_size != (frame.width, frame.height):
                    if self.frame_size is not None: print(f'[agent] received resolution changed {self.frame_size} -> {(frame.width, frame.height)}', flush=True)
                    self.frame_size = (frame.width, frame.height)
                rgb = frame_to_rgb(frame)
                if self.stamp_reader: self.stamp_reader(rgb, seq, t_recv, time.time())   # measurement runs: bridge send-time stamp per processed frame
                r = self.pipe.process(rgb)                          # an exception here ends the run (see worker)
                if getattr(self.a, 'mirror', False): self.mirror_src = (rgb, self.pipe.last_landmarks)   # newest processed frame for the mirror task
                t_dec = time.time(); self.processed += 1; self.delays.append(t_dec - t_recv); win['d'].append(t_dec - t_recv); self.recv_hist.append(t_recv)
                win['n'] += 1; win['pose'] += r['pose_ms']; win['clf'] += r['clf_ms']; win['ok'] += r['ok']
                meta = dict(frame_seq=seq, ts_us=ts_us, t_capture=t_cap, t_recv=t_recv, t_decision=t_dec,
                            pose_ms=round(r['pose_ms'], 1), clf_ms=round(r['clf_ms'], 1), pose_model=self.pipe.pose_variant)
                if self.a.latency_probe:
                    if not self.probe_said and t_recv - self.first_frame_t >= 2.5:      # after the session's own opening line
                        self.probe_said = True; self.loop.call_soon_threadsafe(self.player.say, 'Probe ready, raise both arms', 'announce', dict(meta))
                    if r.get('probe_fire'):                                      # on-box number counts from the FIRST up frame's receipt
                        span = int(r.get('probe_span', 1)); t_first = self.recv_hist[-span] if len(self.recv_hist) >= span else t_recv
                        self.loop.call_soon_threadsafe(self.player.say_now, BEEP, 'probe', dict(meta, t_recv=t_first, t_detect=time.time()))
                self.session.on_frame(r, meta)
                if self.session.done and self.a.plan:
                    if self.done_at is None: self.done_at = time.monotonic()
                    elif time.monotonic() - self.done_at > 4: self.loop.call_soon_threadsafe(self.done.set); break
            el = time.monotonic() - win['t']
            if el >= 5.0:
                recv = (self.seq - win['seq0']) / el; fps = win['n'] / el; n = max(win['n'], 1)
                pose_ms = win['pose'] / n; clf_ms = win['clf'] / n
                if (self.a.auto_lite and self.pipe.pose_variant != 'lite' and recv >= 8 and fps < 8.5 and pose_ms > 60):
                    self.pipe.switch_pose('lite'); self.note = ' POSE->lite (full too slow for 10 fps here)'
                d50 = 1000 * float(_np.median(win['d'])) if win['d'] else float('nan')
                probe = f", probe on-box {self.last_probe[1]:.0f} ms (detect->push {self.last_probe[0]:.0f})" if self.last_probe else ''
                print(f"[status] recv {recv:4.1f} fps, processed {fps:4.1f} fps, recv->decision p50 {d50:.0f} ms, pose {pose_ms:5.1f} ms ({self.pipe.pose_variant}), "
                      f"clf {clf_ms:4.1f} ms/frame, dropped {self.dropped} (idle {win['idle']}), pose ok {win['ok']}/{win['n']}, "
                      f"cues played {self.cues_played} dropped {self.player.n_dropped} underruns {self.player.underruns}, session {self.session.state}"
                      f"/{self.pipe.exercise or '-'} count {self.session.last_count}{probe}{self.note}", flush=True)
                win = dict(t=time.monotonic(), n=0, pose=0.0, clf=0.0, seq0=self.seq, idle=0, ok=0, d=[])

    async def beeper(self):
        """--beep-every N: a beep at every wall-clock multiple of N seconds (cuts whatever plays), logged with t_sched/t_push,
        so a recording on the far end can time the audio leg alone against the agent log."""
        n = self.a.beep_every
        while not self.done.is_set():
            t_next = (int(time.time() // n) + 1) * n
            await asyncio.sleep(max(0.0, t_next - time.time()))
            self.player.say_now(BEEP, 'beep', dict(t_sched=t_next, t_decision=time.time()))

    # ---- lifecycle ----
    async def start(self):
        """Player task + worker thread + signal handlers (no room needed: loopback_inproc.py uses this too)."""
        self.player_task = asyncio.create_task(self.player.run())
        self.beep_task = asyncio.create_task(self.beeper()) if self.a.beep_every else None
        self.thread = threading.Thread(target=self.worker, daemon=True, name='coach-worker'); self.thread.start()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try: self.loop.add_signal_handler(sig, lambda: (print('[agent] signal, stopping', flush=True), self.done.set()))
            except (NotImplementedError, RuntimeError): pass

    async def wait(self):
        try:
            await asyncio.wait_for(self.done.wait(), timeout=self.a.max_s)
        except asyncio.TimeoutError:
            print(f'[agent] --max-s {self.a.max_s} reached', flush=True)

    async def avatar_loop(self):
        """--avatar: Brodie's face (facetime/avatar.py) as a 360x640 video track at --avatar-fps; the mouth moves while a cue plays.
        Chert's bridge shows the agent's published video to the FaceTime caller. Any failure here is logged and leaves the call alone."""
        try:
            from facetime.avatar import Avatar
            av = Avatar(); fps = max(1.0, float(getattr(self.a, 'avatar_fps', 10.0) or 10.0))
            source = rtc.VideoSource(av.W, av.H)
            track = rtc.LocalVideoTrack.create_video_track('coach-video', source)
            await self.room.local_participant.publish_track(track, rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA, simulcast=False, video_encoding=rtc.VideoEncoding(max_framerate=int(fps), max_bitrate=600_000)))
            print(f'[agent] published video track coach-video {av.W}x{av.H} at {fps:g} fps (avatar)', flush=True)
            period = 1.0 / fps; next_t = self.loop.time()
            while not self.stop.is_set():
                speaking = self.player.playing is not None
                frame = rtc.VideoFrame(av.W, av.H, rtc.VideoBufferType.RGB24, av.frame_bytes(speaking=speaking)).convert(rtc.VideoBufferType.I420)
                source.capture_frame(frame)
                next_t += period; await asyncio.sleep(max(0.0, next_t - self.loop.time()))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f'[agent] avatar stopped: {e!r}', flush=True)

    async def mirror_loop(self):
        """--mirror: the caller's frame with the pose, the count, the last cue and Brodie's face inset (facetime/mirror.py), 720x1280 at
        --avatar-fps. Rendered on the event loop from the worker's newest processed frame (a few ms a frame)."""
        try:
            from facetime.avatar import Avatar
            from facetime.mirror import compose, W, H
            av = Avatar(); fps = max(1.0, float(getattr(self.a, 'avatar_fps', 10.0) or 10.0))
            source = rtc.VideoSource(W, H)
            track = rtc.LocalVideoTrack.create_video_track('coach-video', source)
            await self.room.local_participant.publish_track(track, rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA, simulcast=False, video_encoding=rtc.VideoEncoding(max_framerate=int(fps), max_bitrate=1_500_000)))
            print(f'[agent] published video track coach-video {W}x{H} at {fps:g} fps (mirror)', flush=True)
            period = 1.0 / fps; next_t = self.loop.time(); last_id = None; img = None
            while not self.stop.is_set():
                speaking = self.player.playing is not None
                src = self.mirror_src
                if src is not None and id(src) != last_id:
                    last_id = id(src); rgb, lm = src
                    img = compose(rgb, lm, self.session.last_count, getattr(self.session, 'target', None), self.pipe.exercise, self.session.state,
                                  self.last_cue[0], self.last_cue[1], av.frame(speaking=speaking))
                elif src is None:
                    img = av.frame(speaking=speaking)   # no caller frame yet: the plain face
                    img = np.ascontiguousarray(np.array(img.resize((W, H))))
                if img is not None:
                    source.capture_frame(rtc.VideoFrame(W, H, rtc.VideoBufferType.RGB24, np.ascontiguousarray(img).tobytes()).convert(rtc.VideoBufferType.I420))
                next_t += period; await asyncio.sleep(max(0.0, next_t - self.loop.time()))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f'[agent] mirror stopped: {e!r}', flush=True)

    async def shutdown(self):
        self.stop.set(); self.thread.join(timeout=2)
        if getattr(self, 'avatar_task', None): self.avatar_task.cancel()
        for _ in range(40):                                      # let the cue in flight finish (max 4 s)
            if not self.player.playing and self.player.pending is None: break
            await asyncio.sleep(0.1)
        if getattr(self, 'beep_task', None): self.beep_task.cancel()
        self.player.stop(); await asyncio.gather(self.player_task, return_exceptions=True)
        if self.video_task: self.video_task.cancel()
        self.pipe.close(); self.logf.close()
        if self.stamp_reader: self.stamp_reader.close()

    def summary(self):
        pl = [r for r in self.player.records if r['status'] == 'played']
        if self.delays:
            d = _np.array(self.delays) * 1000
            print(f"[agent] receive -> decision delay ({'tick' if self.a.tick_sampling else 'arrival-aligned'} sampling, n={len(d)}): p50 {_np.percentile(d, 50):.0f} ms, p90 {_np.percentile(d, 90):.0f} ms, max {d.max():.0f} ms; "
                  f"audio underruns {self.player.underruns} (queue {self.a.audio_queue_ms} ms)", flush=True)
        print(f'[agent] done: frames received {self.seq}, processed {self.processed}, dropped {self.dropped}, idle ticks {self.idle_ticks}; '
              f'pose {self.pipe.pose_ms_sum / max(self.pipe.n, 1):.1f} ms/frame ({self.pipe.pose_variant}), '
              f'clf {self.pipe.clf.total_ms / max(self.pipe.clf.n, 1):.1f} ms/call, {self.pipe.clf_ms_sum / max(self.pipe.n, 1):.1f} ms/frame; '
              f'cues played {len(pl)}, dropped {self.player.n_dropped}; log {self.a.log}'
              + (f'; {self.stamp_reader.summary()} -> {self.a.stamp_log}' if self.stamp_reader else ''), flush=True)

    async def run(self):
        a = self.a
        self.room.on('track_subscribed', self.on_track_subscribed)
        self.room.on('participant_disconnected', self.on_participant_disconnected)
        self.room.on('disconnected', lambda *x: (print('[agent] room disconnected', x, flush=True), self.done.set()))
        await self.room.connect(os.environ['LIVEKIT_URL'], mint(AGENT_IDENTITY, a.room), rtc.RoomOptions(auto_subscribe=True))
        print(f'[agent] connected to {a.room} as {AGENT_IDENTITY}; mode {self.session.mode} '
              f'({a.plan or a.exercise or "auto-detect"}), counter {a.counter}, trigger {self.pipe.trigger_path or "off"}, pose {self.pipe.pose_variant}, participants {list(self.room.remote_participants)}', flush=True)
        await self.room.local_participant.publish_track(self.player.track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        print('[agent] published audio track coach-audio', flush=True)
        self.mirror_src = None; self.last_cue = (None, 0.0)
        self.avatar_task = asyncio.create_task(self.mirror_loop() if getattr(a, 'mirror', False) else self.avatar_loop()) if (getattr(a, 'avatar', False) or getattr(a, 'mirror', False)) else None
        if a.no_video_s and a.no_video_s > 0: self.loop.call_later(a.no_video_s, self._done_if_never_video)
        await self.start(); await self.wait(); await self.shutdown()
        await self.room.disconnect(); self.summary()
        if self.worker_error is not None: raise SystemExit(f'agent stopped on an error: {self.worker_error!r}')

def build_parser(room_required=True):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--room', required=room_required, default='inproc')
    ap.add_argument('--exercise', default=None, help='single-exercise mode (LR2CLF key or squats/pushups/lunges/jumping jacks)')
    ap.add_argument('--plan', default=None, help='workout plan mode, e.g. "squats:10,pushups:8,jumping jacks:20,lunges:10"; neither flag = auto-detect')
    ap.add_argument('--rest-s', type=float, default=5.0); ap.add_argument('--set-timeout', type=float, default=75.0)
    ap.add_argument('--tempo', default=None, help='JSON (or .json file) overriding seconds-per-rep targets, e.g. {"squats":2.0}')
    ap.add_argument('--auto-classes', default='demo', choices=['demo', 'all'], help='auto mode: declare only the four demo exercises (families) or any LR2CLF class')
    ap.add_argument('--counter', default='v2', choices=['v2', 'v2plain', 'e1'], help='rep counter: RepCounter2 with the demo guards (default), v2plain = RepCounter2 as in the 2026-10-08 Mac tests (no mask, gate or settle), or the E1 pose-PCA counter')
    ap.add_argument('--trigger', default='auto', help='learned correction trigger: a model path, auto (data/models/trigger_lr.json if present) or none')
    ap.add_argument('--model', default=os.path.join(DATA, 'models', 'clf_pose.pt')); ap.add_argument('--pose', default='full')
    ap.add_argument('--params', default='{}'); ap.add_argument('--thr', default=None)
    ap.add_argument('--log', default=os.path.join(DATA, 'tmp', 'agent_events.jsonl'))
    ap.add_argument('--once', action='store_true', help='exit when the video publisher leaves')
    ap.add_argument('--max-s', type=float, default=None, help='exit after this many seconds')
    ap.add_argument('--no-video-s', type=float, default=0.0, help='exit if no video frame has arrived this many seconds after connecting (0 = never; the Modal webhook passes 120 so a dropped call does not hold the warm container)')
    ap.add_argument('--no-announce', dest='announce', action='store_false', help="skip the \"Let's do <exercise>\" line")
    ap.add_argument('--no-auto-lite', dest='auto_lite', action='store_false')
    ap.add_argument('--read-stamp', action='store_true', help='decode the bridge send-time stamp from each processed frame (facetime/frame_stamp.py)')
    ap.add_argument('--stamp-t0', type=float, default=None, help='the epoch the bridge stamps from; with it every frame logs its one-way video time')
    ap.add_argument('--stamp-log', default=os.path.join(DATA, 'tmp', 'agent_stamps.jsonl'), help='one json line per processed frame')
    ap.add_argument('--latency-probe', action='store_true', help='beep immediately when both wrists go above the head (after 1 s below); logs detect/push times')
    ap.add_argument('--gesture-next', action='store_true', help='a T-pose held 1.5 s says "Okay, moving on" and closes the set / skips the rest / starts the exercise (module docstring)')
    ap.add_argument('--beep-every', type=float, default=None, help='beep at every N s wall-clock multiple (cuts speech) to time the audio leg against the log')
    ap.add_argument('--audio-queue-ms', type=int, default=60, help='AudioSource queue (60; go to 100 if the status line shows underruns)')
    ap.add_argument('--avatar', action='store_true', help="publish Brodie's face as a video track \"coach-video\" (facetime/avatar.py; the FaceTime caller sees it, mouth moves while a cue plays)")
    ap.add_argument('--avatar-fps', type=float, default=10.0)
    ap.add_argument('--mirror', action='store_true', help="publish the caller's own frame with the pose, the count and the last cue drawn on it (facetime/mirror.py, v3 look, Brodie's face as an inset) as the coach's video track instead of the plain avatar")
    ap.add_argument('--tick-sampling', action='store_true', help='old 100 ms tick sampling instead of arrival-aligned (for comparison)')
    ap.add_argument('--video-quality', default='low', choices=['low', 'medium', 'high', 'none'], help='simulcast layer to request from the SFU when the publisher simulcasts')
    ap.add_argument('--standby', action='store_true', help='load everything, print "[agent] standby ready", then wait for one JSON line on stdin with the room (module docstring)')
    return ap

PIPELINE_FLAGS = ('model', 'pose', 'params', 'thr', 'counter', 'trigger')   # baked into the FramePipeline at standby: a per-call change is ignored

def process_age():
    """Seconds since this process started (the standby ready line): /proc on Linux, else since this module was imported."""
    try:
        with open('/proc/self/stat') as f: start_ticks = int(f.read().rsplit(')', 1)[1].split()[19])    # field 22: start, clock ticks since boot
        with open('/proc/uptime') as f: uptime = float(f.read().split()[0])
        return uptime - start_ticks / os.sysconf('SC_CLK_TCK')
    except (OSError, ValueError, IndexError):
        return time.time() - _T0

def standby_abort(why):
    print(f'[agent] standby aborted: {why}', flush=True); sys.exit(3)

def standby_release(ap, a):
    """--standby: say we are ready, block on ONE JSON line on stdin and apply it to the namespace (contract in the module
    docstring). Returns the namespace to run with; exits 3 on an empty line, EOF, invalid JSON or a bad field."""
    print(f'[agent] standby ready in {process_age():.1f} s', flush=True)
    try: line = sys.stdin.readline() if sys.stdin is not None else ''
    except KeyboardInterrupt: standby_abort('interrupted')
    if not line.strip(): standby_abort('EOF on stdin' if not line else 'empty line on stdin')
    try: req = json.loads(line)
    except ValueError as e: standby_abort(f'invalid JSON: {e}')
    if not isinstance(req, dict) or not isinstance(req.get('room'), str) or not req['room'].strip():
        standby_abort('need a JSON object with a non-empty "room" string')
    for k, t, what in (('plan', str, 'a string'), ('exercise', str, 'a string'), ('max_s', (int, float), 'a number'),
                       ('agent_token', str, 'a string'), ('args', list, 'a list of strings')):
        if req.get(k) is not None and (not isinstance(req[k], t) or isinstance(req[k], bool) or (k == 'args' and not all(isinstance(x, str) for x in req[k]))):
            standby_abort(f'"{k}" must be {what} or null')
    if req.get('plan') is not None and req.get('exercise') is not None: standby_abort('"plan" and "exercise" are mutually exclusive')
    before = {k: getattr(a, k) for k in PIPELINE_FLAGS}
    if req.get('args'):
        try: ap.parse_args(req['args'], namespace=a)                       # extra per-call flags, e.g. --latency-probe, --beep-every 10
        except SystemExit: standby_abort(f'bad "args" {req["args"]!r}')
    a.room = req['room'].strip()
    if req.get('plan') is not None: a.plan, a.exercise = req['plan'], None
    if req.get('exercise') is not None: a.exercise, a.plan = req['exercise'], None
    if req.get('max_s') is not None: a.max_s = float(req['max_s'])
    if req.get('agent_token') is not None: os.environ['RTCOACH_AGENT_TOKEN'] = req['agent_token']   # what the webhook exports today (the agent self-mints, mint_token.py)
    for k in PIPELINE_FLAGS:
        if getattr(a, k) != before[k]:
            print(f'[agent] standby: --{k.replace("_", "-")} {getattr(a, k)!r} arrived per call but the pipeline is already built; keeping {before[k]!r}', flush=True)
            setattr(a, k, before[k])
    print(f'[agent] standby released: room {a.room} mode {"plan" if a.plan else ("single" if a.exercise else "auto")}', flush=True)
    return a

def main():
    standby = '--standby' in sys.argv[1:]
    ap = build_parser(room_required=not standby)          # in standby the room arrives with the release
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(name)s %(levelname)s %(message)s')
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    agent = Agent(a, loop, standby=standby)
    if standby:
        try: a = standby_release(ap, a)
        except SystemExit: agent.pipe.close(); raise
        agent.setup_call(a)
    loop.run_until_complete(agent.run())

if __name__ == '__main__':
    main()
