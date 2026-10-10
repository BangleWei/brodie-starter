"""Tier (b) minus the network: the fake bridge and the coach agent cross-wired in one process through the
LiveKit media objects (VideoSource -> LocalVideoTrack -> VideoStream, AudioSource -> LocalAudioTrack ->
AudioStream), no room, no SFU. Same code paths as the live agent (frame conversion, 10 fps sampling,
pose, Coach, cue player, onset detection); what it cannot measure is the WebRTC transport legs.
Use it when the box cannot reach LiveKit media (this cloud sandbox blocks UDP and intercepts TLS on 443).

Run:  .venv/bin/python -m facetime.loopback_inproc --video data/tmp/0013_squats.mp4 --exercise squats \
        --log data/tmp/inproc_agent.jsonl --onsets data/tmp/inproc_onsets.json --out data/tmp/inproc_rx.wav
Then: .venv/bin/python -m facetime.latency_report --agent-log data/tmp/inproc_agent.jsonl --onsets data/tmp/inproc_onsets.json
"""
import argparse, asyncio, os, time
from livekit import rtc
from facetime import DATA, VIDEO_TRACK_NAME, AUDIO_TRACK_NAME, BRIDGE_IDENTITY, AGENT_IDENTITY
from facetime.coach_agent import Agent, build_parser as agent_parser
from facetime.fake_bridge import Bridge, build_parser as bridge_parser
from rtcoach.video import read_frames

class Stub:
    def __init__(self, **kw): self.__dict__.update(kw)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], parents=[agent_parser(False), bridge_parser(False)], conflict_handler='resolve')
    a = ap.parse_args(); a.room = 'inproc'; a.wait_agent = 0
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    loop.run_until_complete(run(a, loop))

async def run(a, loop):
    fr = read_frames(a.video); n, H, W = fr.shape[0], fr.shape[1], fr.shape[2]
    print(f'[inproc] {a.video}: {n} frames {W}x{H} at {a.fps} fps', flush=True)
    agent = Agent(a, loop); bridge = Bridge(a, loop)
    source = rtc.VideoSource(W, H); vtrack = rtc.LocalVideoTrack.create_video_track(VIDEO_TRACK_NAME, source)
    agent.on_track_subscribed(vtrack, Stub(name=VIDEO_TRACK_NAME, width=W, height=H, simulcasted=False), Stub(identity=BRIDGE_IDENTITY))
    bridge.on_track_subscribed(agent.player.track, Stub(name=AUDIO_TRACK_NAME), Stub(identity=AGENT_IDENTITY))
    await agent.start()
    t0, t_end, k = await bridge.send(source, fr)
    await asyncio.sleep(a.tail)
    await agent.shutdown()
    if bridge.audio_task: bridge.audio_task.cancel()
    agent.summary(); bridge.write_outputs(t0, t_end)

if __name__ == '__main__':
    main()
