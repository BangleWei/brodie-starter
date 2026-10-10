"""Join the agent's cue log (coach_agent --log) with the bridge's audio onsets (fake_bridge --onsets) and
print the event-to-sound latency per cue: onset minus decision time, and onset minus first audio push.
Both clocks are time.time() on the same machine. Each onset is matched to the latest cue whose t_push is
at most 0.05 s after it and not already matched.

Run:  .venv/bin/python -m facetime.latency_report --agent-log data/tmp/agent.jsonl --onsets data/tmp/onsets.json [--room X]
"""
import argparse, json
import numpy as np

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--agent-log', required=True); ap.add_argument('--onsets', required=True); ap.add_argument('--room', default=None)
    a = ap.parse_args()
    on = json.load(open(a.onsets)); room = a.room or on.get('room')
    cues = [json.loads(l) for l in open(a.agent_log) if l.strip()]
    cues = [c for c in cues if c.get('status') == 'played' and (room is None or c.get('room') == room)]
    cues.sort(key=lambda c: c['t_push']); used = set(); rows = []
    for o in on['onsets']:
        cand = [i for i, c in enumerate(cues) if i not in used and c['t_push'] <= o['t'] + 0.05]
        if not cand:
            rows.append((o, None)); continue
        i = cand[-1]; used.add(i); rows.append((o, cues[i]))
    print(f"room {room}: {len(cues)} cues played by the agent, {len(on['onsets'])} audio onsets heard by the bridge")
    print(f"{'onset wall':>14} {'cue':28} {'intent':10} {'dec->push':>9} {'push->onset':>11} {'dec->onset':>10}  {'cap->onset':>10}")
    d2o = []; p2o = []; c2o = []
    for o, c in rows:
        if c is None:
            print(f"{o['t']:14.3f} {'(no cue matched)':28}"); continue
        dp = c['t_push'] - c['t_decision']; po = o['t'] - c['t_push']; do = o['t'] - c['t_decision']
        co = (o['t'] - c['t_capture']) if c.get('t_capture') else float('nan')
        d2o.append(do); p2o.append(po); c2o.append(co)
        print(f"{o['t']:14.3f} {c['text'][:28]:28} {c['intent']:10} {1000 * dp:8.0f}ms {1000 * po:10.0f}ms {1000 * do:9.0f}ms  {1000 * co:9.0f}ms")
    missed = [c for i, c in enumerate(cues) if i not in used]
    for c in missed:
        print(f"{'':14} {c['text'][:28]:28} {c['intent']:10}  (played by the agent, no separate onset heard; likely merged with the previous cue)")
    if d2o:
        q = lambda v: np.percentile(v, [50, 90]) * 1000
        print(f"decision->onset p50/p90 {q(d2o)[0]:.0f}/{q(d2o)[1]:.0f} ms (n={len(d2o)}); push->onset p50/p90 {q(p2o)[0]:.0f}/{q(p2o)[1]:.0f} ms; "
              f"decision->push p50/p90 {q(np.array(d2o) - np.array(p2o))[0]:.0f}/{q(np.array(d2o) - np.array(p2o))[1]:.0f} ms")
        if not any(np.isnan(c2o)): print(f"frame capture(recv)->onset p50/p90 {q(c2o)[0]:.0f}/{q(c2o)[1]:.0f} ms")

if __name__ == '__main__':
    main()
