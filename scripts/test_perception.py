"""Test facetime/perception.py against the demo coach's own code path on the squats clip (plan squats:10).

1. Reference: FramePipeline + Session exactly as facetime/replay_local.py runs them (fixed t_ms), recording every count increment
   (frame, n) and the Session's cue timeline (t, intent, text).
2. Perception in events-only mode on the same frames, with every audio constructor patched to raise: CuePlayer, CueBank,
   rtc.AudioSource, rtc.LocalAudioTrack.create_audio_track.
Asserts: the rep events equal the reference counts (frame and n); the cue events equal the reference timeline; every event
carries the schema keys; no audio object was created; the JSONL file has one valid line per event.

Run:  .venv/bin/python scripts/test_perception.py [--video data/tmp/0013_squats.mp4] [--plan squats:10]
"""
import argparse, json, os, sys, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from facetime.pipeline import FramePipeline, load_json_arg, FPS, COUNTERS
from facetime.session import Session
from facetime.perception import Perception, Config, EVENT_SCHEMA_VERSION, EVENT_TYPES
from facetime import audio_out
from rtcoach.video import read_frames


def reference(fr, cfg):
    """replay_local's path: pipeline + session, no audio; clip durations from the bank manifest only."""
    from facetime.perception import cue_durations
    pipe = FramePipeline(cfg.model, None, cfg.pose, load_json_arg(cfg.params, {}), load_json_arg(cfg.thr), rep_params=COUNTERS[cfg.counter], trigger=cfg.trigger)
    timeline = []
    def say(text, intent, meta): timeline.append((meta['t_session'], intent, text))
    sess = Session(pipe, say, plan=cfg.plan, exercise=cfg.exercise, clip_s=cue_durations())
    counts = []; last = 0; coach_id = None
    for i in range(len(fr)):
        r = pipe.process(fr[i], t_ms=i * 1000.0 / FPS)
        cid = id(pipe.coach) if pipe.coach is not None else None
        if cid != coach_id: coach_id = cid; last = 0
        c = r['count']
        if c < last: last = 0
        for n in range(last + 1, c + 1): counts.append((i, n))
        last = c
        sess.on_frame(r, dict(frame=i, t_clip=i / FPS))
    pipe.close()
    return counts, timeline


class NoAudio(AssertionError):
    pass


def forbid(obj, name, what):
    def boom(*a, **k): raise NoAudio(f'{what} was constructed in events-only mode')
    old = getattr(obj, name); setattr(obj, name, boom); return old


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--video', default=os.path.join(ROOT, 'data', 'tmp', '0013_squats.mp4')); ap.add_argument('--plan', default='squats:10')
    a = ap.parse_args()
    fr = read_frames(a.video); cfg = Config(plan=a.plan)
    print(f'{a.video}: {len(fr)} frames, plan {a.plan}')
    t = time.time(); ref_counts, ref_timeline = reference(fr, cfg); print(f'reference: {len(ref_counts)} counts, {len(ref_timeline)} cues in {time.time() - t:.1f} s')

    from livekit import rtc
    saved = [(audio_out.CuePlayer, '__init__', forbid(audio_out.CuePlayer, '__init__', 'CuePlayer')),
             (audio_out.CueBank, '__init__', forbid(audio_out.CueBank, '__init__', 'CueBank')),
             (rtc, 'AudioSource', forbid(rtc, 'AudioSource', 'rtc.AudioSource')),
             (rtc.LocalAudioTrack, 'create_audio_track', forbid(rtc.LocalAudioTrack, 'create_audio_track', 'rtc.LocalAudioTrack'))]
    path = os.path.join(tempfile.mkdtemp(prefix='perception_test_'), 'events.jsonl')
    try:
        per = Perception(cfg, t0=0.0); per.log_to(path)
        evs = []; t = time.time()
        for i in range(len(fr)):
            evs += per.process_frame(fr[i], t_recv=i / FPS, t_ms=i * 1000.0 / FPS)
        per.close()
    finally:
        for obj, name, old in saved: setattr(obj, name, old)
    print(f'perception: {len(evs)} events in {time.time() - t:.1f} s: {per.summary()["events"]}')

    failures = []
    reps = [(e['frame'], e['n']) for e in evs if e['type'] == 'rep']
    if reps != ref_counts: failures.append(f'rep events {reps} != reference counts {ref_counts}')
    cues = [(e['t_session'], e['intent'], e['text']) for e in evs if e['type'] == 'cue']
    if cues != ref_timeline: failures.append(f'cue events {cues} != reference timeline {ref_timeline}')
    for e in evs:
        for k in ('v', 't', 't_wall', 'seq', 'type'):
            if k not in e: failures.append(f'event without {k}: {e}'); break
        if e['v'] != EVENT_SCHEMA_VERSION or e['type'] not in EVENT_TYPES: failures.append(f'bad v/type: {e}')
    if [e['seq'] for e in evs] != list(range(len(evs))): failures.append('seq is not 0..n-1')
    if not any(e['type'] == 'rep' and e.get('confidence') and e['confidence'].get('agree') is not None for e in evs): failures.append('no rep event carries a confidence')
    if hasattr(per, 'player') or hasattr(per, 'track') or hasattr(per, 'source'): failures.append('Perception holds an audio attribute')
    lines = [json.loads(l) for l in open(path)]
    if len(lines) != len(evs) or [l['seq'] for l in lines] != [e['seq'] for e in evs]: failures.append(f'jsonl has {len(lines)} lines for {len(evs)} events')
    print(f'reps: {reps}')
    print(f'cues: {[(t, i, x) for t, i, x in cues]}')
    if failures:
        print('FAIL'); [print('  ' + f[:400]) for f in failures]; sys.exit(1)
    print(f'PASS: {len(reps)} rep events match replay_local counts, {len(cues)} cue events match the session timeline, no audio object created, {len(lines)} JSONL lines')


if __name__ == '__main__':
    main()
