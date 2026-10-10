"""End-to-end latency from a phone recording of a coaching session: rep bottom (pose on the recorded video) -> count word heard
(onset in the recorded audio). Works on a screen recording of the phone page (its audio carries the coach) or on a second
phone's film of the demo phone (screen plus speaker). A FaceTime screen recording has NO audio (iOS does not capture it).

Run:  .venv/bin/python scripts/e2e_from_recording.py <recording.mp4> --crop W:H:X:Y --t0 11 --t1 36 --exercise squats
  --crop: the camera region inside the recording (ffmpeg crop syntax; e.g. the phone page: 396:576:96:472 on a 588x1280 recording)
  --t0/--t1: the window (recording seconds) of ONE set of one exercise; cue onsets are matched to the rep bottom within 2.5 s,
  so pick a set with few corrections, or read the per-rep lines and discard pairs where a correction came first.
Prints the rep bottoms, the audio onsets, bottom -> onset per rep and the p50. The counter's own reversal delay (about 0.3 s
after the bottom) is inside that number; subtract it for the transport part.
"""
import argparse, os, subprocess, sys, wave
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from facetime.pipeline import FramePipeline
from facetime import DATA

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('recording'); ap.add_argument('--crop', required=True, help='W:H:X:Y of the camera view inside the recording')
    ap.add_argument('--t0', type=float, required=True); ap.add_argument('--t1', type=float, required=True); ap.add_argument('--exercise', required=True)
    ap.add_argument('--model', default=os.path.join(DATA, 'models', 'clf_pose.pt')); ap.add_argument('--onset-dbfs', type=float, default=-35.0)
    ap.add_argument('--work', default=os.path.join(DATA, 'tmp', 'e2e_rec'))
    a = ap.parse_args(); os.makedirs(a.work, exist_ok=True)
    crop = os.path.join(a.work, 'crop.mp4'); wavp = os.path.join(a.work, 'rec.wav')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', a.recording, '-vf', f'crop={a.crop},fps=10,scale=288:-2', '-an', crop], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', a.recording, '-vn', '-ac', '1', '-ar', '16000', wavp], check=True)
    w = wave.open(wavp); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    fr = int(0.02 * sr); n = len(x) // fr; db = 20 * np.log10(np.sqrt((x[:n * fr].reshape(n, fr) ** 2).mean(1)) + 1e-9)
    if np.percentile(db, 99) < -80: print('the recording has no audio (a FaceTime screen recording?): film the phone with a second phone instead'); return
    ons, below = [], 0
    for i, v in enumerate(db):
        if v > a.onset_dbfs and below >= 7: ons.append(i * 0.02)
        below = below + 1 if v <= a.onset_dbfs else 0
    ons = np.array(ons)
    pipe = FramePipeline(a.model, None, 'full', {}, None, rep_params={'v2': True, 'select': 'vote'})
    pipe.set_exercise(a.exercise, params=dict(start_line=False, praise_after=0), armed=True)
    cap = cv2.VideoCapture(crop); i = 0; hip, ts = [], []
    while True:
        ok, f = cap.read()
        if not ok: break
        t = i / 10.0
        if a.t0 <= t <= a.t1:
            pipe.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), t_ms=i * 100); lm = pipe.last_landmarks
            hip.append(((lm[23][1] + lm[24][1]) / 2) if lm is not None else np.nan); ts.append(t)
        i += 1
    h = np.array(hip, float); ts = np.array(ts); h[np.isnan(h)] = np.nanmean(h); hs = np.convolve(h, np.ones(3) / 3, 'same')
    tr = [ts[j] for j in range(2, len(hs) - 2) if hs[j] >= hs[j - 2:j + 3].max() and hs[j] - min(hs[max(0, j - 15):j].min(), hs[j:j + 15].min()) > 0.04]
    m = []
    for t in tr:
        if not m or t - m[-1] > 0.8: m.append(float(t))
    print('rep bottoms (recording s):', [round(t, 1) for t in m])
    print('audio onsets in the window:', [round(float(o), 2) for o in ons if a.t0 <= o <= a.t1 + 2])
    d = []
    for t in m:
        nxt = ons[(ons > t) & (ons < t + 2.5)]
        if len(nxt): d.append(nxt[0] - t); print(f'bottom {t:6.1f} -> onset {nxt[0]:6.2f}  {1000 * (nxt[0] - t):5.0f} ms')
    if d:
        d = 1000 * np.array(d); print(f'bottom -> cue heard: n={len(d)} p50 {np.percentile(d, 50):.0f} ms, min {d.min():.0f}, max {d.max():.0f} (the counter fires about 300 ms after the bottom; the rest is transport and playout)')

if __name__ == '__main__':
    main()
