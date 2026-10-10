"""Timestamp-in-frame: measure the video leg (bridge -> SFU -> agent) by drawing the send time into each frame.

Opt-in helpers; nothing here runs unless fake_bridge.py gets --stamp or coach_agent.py gets --read-stamp.
Pattern: 32 black/white blocks in the top-left corner, 2 rows of 16, block side = frame width // 18 (16 px on a 288 px
wide frame; proportional, so it survives the codec and a proportional rescale). Bits (MSB first, row-major): a 4-bit
checksum (xor of the seven nibbles) then 28 bits of milliseconds since the epoch t0 (74 h of range). Reading samples the
central half of each block (mean > 127 = 1) and returns None for a frame without a stamp, an ambiguous block (mean in
96..160) or a checksum mismatch, so an unstamped frame never yields a number.

Epoch: both ends get the same --stamp-t0 (any wall-clock value chosen by the orchestrator, a little in the past); the
bridge stamps (send wall clock - t0), the agent computes (receive wall clock - t0 - stamp), so t0 cancels and what remains
is the one-way time plus the clock skew between the two hosts (zero when both run in one process, loopback_inproc).
Without --stamp-t0 the bridge uses its own start time, prints it once and writes it to the onsets json as stamp_t0, and
report() joins offline. The per-cue "frame sent -> audio onset" is skew-free in every setup: both ends are measured on
the bridge's clock. The stamp covers a 32 px band at the top of the frame, so use it for measurement runs only.

Report: .venv/bin/python -m facetime.frame_stamp --stamps data/tmp/agent_stamps.jsonl --agent-log data/tmp/agent.jsonl \
          --onsets data/tmp/onsets.json [--t0 <epoch>]
"""
import argparse, json, time
import numpy as np

BITS, ROWS, DIV, MS_BITS = 32, 2, 18, 28
MASK = (1 << MS_BITS) - 1

def _geom(w):
    return max(4, w // DIV), BITS // ROWS

def _checksum(ms):
    x = 0
    for i in range(7):
        x ^= (ms >> (4 * i)) & 0xF
    return x

def encode(ms):
    ms = int(ms) & MASK
    return (_checksum(ms) << MS_BITS) | ms

def decode(v):
    ms = v & MASK
    return ms if (v >> MS_BITS) == _checksum(ms) else None

def stamp(rgb, ms):
    """Copy of rgb (H,W,3 uint8) with `ms` drawn as the block pattern in the top-left corner."""
    out = np.array(rgb, copy=True); b, per_row = _geom(rgb.shape[1]); v = encode(ms)
    for k in range(BITS):
        r, c = divmod(k, per_row)
        out[r * b:(r + 1) * b, c * b:(c + 1) * b] = 255 if (v >> (BITS - 1 - k)) & 1 else 0
    return out

def read(rgb):
    """Milliseconds from the pattern, or None (no stamp, ambiguous block or checksum mismatch)."""
    b, per_row = _geom(rgb.shape[1]); q = b // 4; v = 0
    if rgb.shape[0] < ROWS * b or rgb.shape[1] < per_row * b:
        return None
    for k in range(BITS):
        r, c = divmod(k, per_row)
        m = float(rgb[r * b + q:(r + 1) * b - q, c * b + q:(c + 1) * b - q].mean())
        if 96 < m < 160:
            return None
        v = (v << 1) | int(m > 127)
    if v in (0, (1 << BITS) - 1):          # a flat black or white corner is not a stamp (both would pass the checksum)
        return None
    return decode(v)

class Stamper:
    """Bridge side: stamper(frame, now) draws ms since t0."""
    def __init__(self, t0=None):
        self.t0 = float(t0) if t0 else time.time()
    def __call__(self, rgb, now=None):
        return stamp(rgb, ((now if now is not None else time.time()) - self.t0) * 1000)

class Reader:
    """Agent side: one jsonl record per processed frame: seq, stamp_ms, t_recv, t_proc, wait_ms and (with t0) video_ms."""
    def __init__(self, t0=None, path=None):
        self.t0 = float(t0) if t0 else None; self.n = 0; self.n_ok = 0; self.last_ms = None
        self.f = open(path, 'a') if path else None
    def __call__(self, rgb, seq, t_recv, t_proc):
        ms = read(rgb); self.n += 1
        rec = dict(seq=seq, stamp_ms=ms, t_recv=round(t_recv, 4), t_proc=round(t_proc, 4), wait_ms=round((t_proc - t_recv) * 1000, 1))
        if ms is not None:
            self.n_ok += 1; self.last_ms = ms
            if self.t0:
                rec['video_ms'] = round((t_recv - (self.t0 + ms / 1000)) * 1000, 1)   # bridge send -> agent receive
        if self.f:
            self.f.write(json.dumps(rec) + '\n')
        return ms
    def close(self):
        if self.f: self.f.close()
    def summary(self):
        return f'stamps read {self.n_ok}/{self.n} frames'

def _q(v):
    p = np.percentile(v, [50, 90]); return f'p50/p90 {p[0]:.0f}/{p[1]:.0f} ms'

def report(stamps_path, agent_log=None, onsets_path=None, t0=None):
    """Video one-way, sampling wait, and per cue: frame sent -> decision -> onset (needs the cue log and the onsets)."""
    recs = [json.loads(l) for l in open(stamps_path) if l.strip()]
    on = json.load(open(onsets_path)) if onsets_path else None
    t0 = float(t0) if t0 else (on or {}).get('stamp_t0')
    ok = [r for r in recs if r.get('stamp_ms') is not None]
    print(f'frames processed {len(recs)}, stamp read on {len(ok)} ({100 * len(ok) / max(len(recs), 1):.0f}%)')
    if not ok:
        return
    wait = [r['t_proc'] - r['t_recv'] for r in ok]
    print(f'agent sampling wait (received -> picked by the 10 fps worker): {_q(np.array(wait) * 1000)}')
    if not t0:
        print('no t0: pass --t0 or stamp with --stamp-t0 to get the one-way numbers'); return
    sent = {r['seq']: t0 + r['stamp_ms'] / 1000 for r in ok}
    vid = np.array([r['t_recv'] - sent[r['seq']] for r in ok]) * 1000
    proc = np.array([r['t_proc'] - sent[r['seq']] for r in ok]) * 1000
    print(f'video one-way (frame sent at the bridge -> received by the agent): {_q(vid)}, min {vid.min():.0f} ms (n={len(vid)}; '
          f'two hosts: includes their clock skew, the min bounds it)')
    print(f'frame sent -> processed (received + sampling wait): {_q(proc)}')
    if not (agent_log and on):
        return
    cues = [c for c in (json.loads(l) for l in open(agent_log) if l.strip()) if c.get('status') == 'played']
    cues.sort(key=lambda c: c['t_push']); used = set(); rows = []
    for o in on['onsets']:                                     # same matching rule as facetime/latency_report.py
        cand = [i for i, c in enumerate(cues) if i not in used and c['t_push'] <= o['t'] + 0.05]
        if cand:
            used.add(cand[-1]); rows.append((o, cues[cand[-1]]))
    print(f"{'cue':28} {'intent':10} {'sent->recv':>10} {'recv->dec':>9} {'dec->onset':>10} {'sent->onset':>11}")
    tot = []
    for o, c in rows:
        ms = c.get('stamp_ms')
        if ms is None:
            print(f"{c['text'][:28]:28} {c['intent']:10} (frame had no stamp)"); continue
        t_sent = t0 + ms / 1000
        a, b, d, e = c['t_recv'] - t_sent, c['t_decision'] - c['t_recv'], o['t'] - c['t_decision'], o['t'] - t_sent
        tot.append(e * 1000)
        print(f"{c['text'][:28]:28} {c['intent']:10} {1000 * a:8.0f}ms {1000 * b:7.0f}ms {1000 * d:8.0f}ms {1000 * e:9.0f}ms")
    if tot:
        print(f'frame sent -> audio onset at the bridge (skew-free, both on the bridge clock): {_q(tot)} (n={len(tot)})')

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--stamps', required=True); ap.add_argument('--agent-log'); ap.add_argument('--onsets'); ap.add_argument('--t0', type=float)
    a = ap.parse_args()
    report(a.stamps, a.agent_log, a.onsets, a.t0)

if __name__ == '__main__':
    main()
