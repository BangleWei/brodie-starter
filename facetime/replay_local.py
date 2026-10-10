"""Tier (a): run the live pipeline on a local mp4 at wall-clock 10 fps, no LiveKit.

Modes as in coach_agent: --exercise X (single), --plan "squats:10,..." (scripted session) or neither (auto-detect).
--demo-rehearsal runs the plan (default "squats:10,pushups:8,jumping jacks:20,lunges:10") on the mp4 and writes the
whole spoken script to a wav (--audio-out, default data/tmp/rehearsal.wav) so it can be heard end to end;
combine with --fast to render it quicker than real time.

Frames come from rtcoach.video.read_frames (ffmpeg -> RGB), are fed at wall-clock rate (frames are skipped
if the box falls behind, exactly like the agent), through live MediaPipe and the Coach. Prints each cue with
its clip time and wall time, and the per-frame compute. --audio-out mixes the cached PCM clips at their cue
times into a wav (plus the "Let's do <exercise>" announcement at t=0) so you can listen to the result.

Run:  .venv/bin/python -m facetime.replay_local --video data/tmp/0013_squats.mp4 --exercise squats \
        [--model data/models/clf_pose.pt] [--pose full] [--params '{}'] [--thr thr.json] [--audio-out out.wav] [--fast]
"""
import argparse, json, os, time, wave
import numpy as np
from facetime import DATA, SAMPLE_RATE
from facetime.pipeline import FramePipeline, load_json_arg, FPS, COUNTERS
from facetime.audio_out import CueBank
from facetime.session import Session
from rtcoach.video import read_frames

def mix_audio(path, cues, bank, total_s):
    """cues: list of (t_s, text). Writes a 24 kHz mono wav with each clip at its time, pushed later if the previous
    clip is still playing (like the one-slot player), 200 ms apart."""
    n = int((total_s + 4) * SAMPLE_RATE); out = np.zeros(n, np.int16); free = 0
    for t, text in cues:
        pcm = bank.get(text)
        if pcm is None: continue
        i = max(int(t * SAMPLE_RATE), free); j = min(n, i + len(pcm)); out[i:j] = pcm[:j - i]; free = j + int(0.2 * SAMPLE_RATE)
    w = wave.open(path, 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE); w.writeframes(out.tobytes()); w.close()

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--video', required=True); ap.add_argument('--exercise', default=None)
    ap.add_argument('--plan', default=None); ap.add_argument('--demo-rehearsal', action='store_true')
    ap.add_argument('--rest-s', type=float, default=5.0); ap.add_argument('--set-timeout', type=float, default=75.0)
    ap.add_argument('--tempo', default=None); ap.add_argument('--no-announce', dest='announce', action='store_false')
    ap.add_argument('--auto-classes', default='demo', choices=['demo', 'all']); ap.add_argument('--counter', default='v2', choices=['v2', 'v2plain', 'e1'])
    ap.add_argument('--trigger', default='auto', help='learned correction trigger: a model path, auto (data/models/trigger_lr.json if present) or none')
    ap.add_argument('--model', default=os.path.join(DATA, 'models', 'clf_pose.pt'))
    ap.add_argument('--pose', default='full'); ap.add_argument('--params', default='{}'); ap.add_argument('--thr', default=None)
    ap.add_argument('--audio-out', default=None); ap.add_argument('--log', default=None)
    ap.add_argument('--fast', action='store_true', help='no wall-clock pacing (process every frame as fast as possible)')
    ap.add_argument('--gesture-next', action='store_true', help='a T-pose held 1.5 s says "Okay, moving on" and closes the set / skips the rest (facetime/probe.py TPose)')
    ap.add_argument('--events', default=None, metavar='PATH', help='also write the versioned JSONL event stream (facetime/perception.py) for the recording; the printed cues, --log and --audio-out are unchanged')
    a = ap.parse_args()
    if a.demo_rehearsal:
        a.plan = a.plan or 'squats:10,pushups:8,jumping jacks:20,lunges:10'; a.audio_out = a.audio_out or os.path.join(DATA, 'tmp', 'rehearsal.wav')
    fr = read_frames(a.video); n = len(fr)
    print(f'{a.video}: {n} frames {fr.shape[2]}x{fr.shape[1]}, {n / FPS:.1f} s at {FPS} fps')
    pipe = FramePipeline(a.model, None, a.pose, load_json_arg(a.params, {}), load_json_arg(a.thr), rep_params=COUNTERS[a.counter], trigger=a.trigger, gesture_next=a.gesture_next)
    bank = CueBank(); cues = []; logf = open(a.log, 'w') if a.log else None
    def say(text, intent, meta):
        d = bank.duration(text); t_clip = meta['t_clip']
        print(f"  t={t_clip:6.2f}s wall+{meta['t_decision'] - t0:6.2f}s  [{intent:10}] {text!r}  ({meta.get('why', '')})" + ('' if d else '  NO AUDIO CLIP'))
        cues.append((t_clip, text))
        if logf: logf.write(json.dumps(dict(text=text, intent=intent, **meta)) + '\n')
    per = None
    if a.events:   # --events: the same pipeline and session, wrapped by the events-only perception service (off by default)
        from facetime.perception import Perception, Config
        per = Perception(Config.from_args(a), pipe=pipe, say=say, clip_s=bank.duration, t0=0.0); per.log_to(a.events); sess = per.session
    else:
        sess = Session(pipe, say, plan=a.plan, exercise=a.exercise, announce=a.announce, rest_s=a.rest_s, set_timeout_s=a.set_timeout, tempo=load_json_arg(a.tempo, None), auto_classes=a.auto_classes, clip_s=bank.duration, gesture_next=a.gesture_next)
    print(f'mode {sess.mode} ({a.plan or a.exercise or "auto-detect"}), counter {a.counter}, trigger {pipe.trigger_path or 'off'}, pose {pipe.pose_variant}' + (', gesture-next on' if a.gesture_next else ''))
    t0 = time.time(); i = 0; processed = 0; skipped = 0
    while i < n:
        if not a.fast:
            due = t0 + i / FPS; now = time.time()
            if now < due: time.sleep(due - now)
            elif now - due > 1.0 / FPS:          # behind: jump to the newest frame, like the live agent
                j = min(n - 1, int((now - t0) * FPS)); skipped += j - i; i = j
        if per is not None:
            per.process_frame(fr[i], t_recv=i / FPS, t_ms=i * 1000.0 / FPS, meta=dict(frame=i, t_clip=i / FPS)); processed += 1; i += 1; continue
        r = pipe.process(fr[i], t_ms=i * 1000.0 / FPS); processed += 1; t_dec = time.time()   # fixed timestamps: reproducible replays
        sess.on_frame(r, dict(frame=i, t_clip=i / FPS, t_decision=t_dec, pose_ms=round(r['pose_ms'], 1), clf_ms=round(r['clf_ms'], 1)))
        i += 1
    el = time.time() - t0
    print(f'processed {processed}/{n} frames in {el:.1f} s ({processed / el:.1f} fps), skipped {skipped}, pose ok {pipe.n_ok}/{pipe.n}; '
          f'pose {pipe.pose_ms_sum / max(pipe.n, 1):.1f} ms/frame, classifier {pipe.clf.total_ms / max(pipe.clf.n, 1):.1f} ms/call '
          f'({pipe.clf.n} calls, {pipe.clf_ms_sum / max(pipe.n, 1):.1f} ms/frame amortised)')
    missing = sorted({t for _, t in cues if bank.get(t) is None})
    if missing: print(f'{len(missing)} spoken strings have no audio clip yet (run facetime.cues_tts): {missing}')
    if a.audio_out:
        mix_audio(a.audio_out, cues, bank, n / FPS)
        print('wrote', a.audio_out)
    if per is not None:
        s = per.summary(); per.close(); print(f'wrote {a.events}: {sum(s["events"].values())} events {s["events"]}')
    pipe.close()
    if logf: logf.close()

if __name__ == '__main__':
    main()
