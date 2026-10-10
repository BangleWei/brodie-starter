# Starter events: the perception contract for the hackathon agent (schema v1, 2026-10-09)

The realtime coach's eyes as a stream of events. One process, one participant, one audio track: `facetime/worker.py` is a
LiveKit Agents worker that runs `facetime/perception.py` (FramePipeline + Session exactly as the demo coach runs them, with
`Session.say` replaced by an event sink and no audio of its own). By default the worker also attaches the demo coach's cue
player (`facetime/audio_out.py`, unchanged) so counts and corrections are spoken from the pre-rendered bank 0.2 s after the
decision; a conversational layer consumes the events and speaks only when the session opens a slot. The demo path
(`coach_agent.py`, `pipeline.py`, `session.py`, `audio_out.py`, `modal/webhook.py`) is untouched; replays of the three demo
clips are byte-identical with and without the new code.

Files: `facetime/perception.py` (service, schema, JSONL writer), `facetime/worker.py` (LiveKit Agents worker),
`facetime/replay_local.py --events PATH` (the same stream from a recording), `scripts/test_perception.py` (contract test).

## 1. Schema

One JSON object per event. Every event has:

| key | meaning |
|---|---|
| `v` | schema version, `"1"` (`perception.EVENT_SCHEMA_VERSION`) |
| `t` | seconds since the service started, on the producer's monotonic clock; for frame events the frame's receive time. A replay passes `t0=0` and the clip time, so `t` is the clip time there. Not comparable across processes: use `t_wall` for that. |
| `t_wall` | epoch seconds when the event was emitted |
| `seq` | 0, 1, 2 ... per service |
| `type` | one of `rep`, `exercise_seen`, `fault`, `cue`, `visibility`, `gesture`, `probe`, `session`, `status`; with `--voice gemini` also `voice` {text, role, producer} (section 2c), on the same queue and log, with its own `seq` and no `frame` |
| `frame` | index of the processed frame the event came from (all types except `status`) |

The worker's JSONL log adds `room`, `call_id` (from a Chert bridge identity `chert-bridge-<call_id>`, null until that bridge is
bound and null for the whole of a phone-page call, whose participant is plain `chert-bridge`: key those logs on `room` and
`t_wall`) and `job` to every line. Lines below are from a sample recording, `data/tmp/mac_handoff/crops/run3.mp4`
(portrait 288 px, 10 fps), plan `squats:10,pushups:8,jumping jacks:20,lunges:10`, through `replay_local --events`; with the current policy the 7.5 s fault below is no longer spoken as a `cue` (section 5), the other lines are unchanged.

**rep** {exercise, n, confidence, turning_t, lag_s}: the counter confirmed rep `n` of the current set. `confidence` is from
RepCounter2: `agree` = share of the vote's slots (3 for the masked squat and pushup votes, 5 otherwise; unfired trackers vote 0)
whose count reached `n`, `amp` = median reference rep amplitude of the voting trackers (torso lengths or normalised joint
angle), `trackers` = voting trackers; null with the E1 counter. `turning_t` is the rep's turning point (the bottom) on the event
clock, `lag_s = t - turning_t` is how long after it the count landed (0.1 to 0.3 s offline; the audio adds 0.2 s on the fast path).
```
{"v": "1", "t": 11.9, "t_wall": 1791526458.6, "seq": 25, "type": "rep", "frame": 119, "exercise": "squats", "n": 1, "confidence": {"agree": 1.0, "amp": 1.816, "trackers": 3}, "turning_t": 11.7, "lag_s": 0.2}
```
**exercise_seen** {name, p}: the classifier's exercise head, top class and probability, at most once per second while the top
class is unchanged, whenever the head ran (every armed frame with the trigger on; not during waits and rests).
```
{"v": "1", "t": 4.5, "t_wall": 1791526456.235, "seq": 6, "type": "exercise_seen", "frame": 45, "name": "floor touches", "p": 0.942}
```
**fault** {label, cue_text, p}: the Coach decided a correction on this frame (the trigger fired and a label won), before the
session's hold/drop scheduling. `p` is the Coach's EMA probability of the label (null for a side-rule pseudo-label). The
spoken version, if it was spoken, is a later `cue` with `intent: correction`.
```
{"v": "1", "t": 7.5, "t_wall": 1791526457.219, "seq": 12, "type": "fault", "frame": 75, "label": "squats - shallow", "cue_text": "Sink lower", "p": 0.545}
```
**cue** {text, intent, why, state, exercise, count, t_session, held_s?, t_decided?}: what the demo coach says, in the order its
player queues it: `intent` is `count`, `correction`, `positive` (praise, "Rest for a moment"), `announce` ("Let's do X",
"Moving on to X", "Looks like X", "All done, great session"), `milestone` ("That's nine, nice work", "Okay, moving on"),
`visibility` ("Step back so I can see your whole body", "Great, I can see you", "I will wait here ..."). A correction or praise
held for the count carries `held_s` and `t_decided`. In events-only mode nothing speaks these; they are the script.
```
{"v": "1", "t": 11.9, "t_wall": 1791526458.601, "seq": 26, "type": "cue", "frame": 119, "text": "One", "intent": "count", "why": "", "state": "active", "exercise": "squats", "count": 1, "t_session": 11.9}
{"v": "1", "t": 33.6, "t_wall": 1791526465.613, "seq": 73, "type": "cue", "frame": 336, "text": "Nice work", "intent": "positive", "why": "idle", "state": "active", "exercise": "squats", "count": 9, "t_session": 33.6, "held_s": 0.9, "t_decided": 32.7}
```
**visibility** {state, paused, no_pose_s}: `state` is `full` (nose and both ankles in frame), `partial` (a pose, not the whole
body) or `none` (no pose), reported after it has held 0.5 s; `paused` is the pipeline's hysteretic pause (3 s of partial or no
pose: counting, corrections and the session clock stop; "Step back ..." is the matching cue) and is reported at once.
```
{"v": "1", "t": 0.4, "t_wall": 1791526455.096, "seq": 2, "type": "visibility", "frame": 4, "state": "partial", "paused": false, "no_pose_s": 0.0}
```
**gesture** {name}: a T-pose held 1.5 s (`facetime/probe.py TPose`), at most once per 5 s. Emitted only when the pipeline runs with
`--gesture-next` (without the flag no T-pose detector runs); the session then acts on it ("Okay, moving on"). On a device it has fired
only after "All done", with nothing to move on from. No T-pose in this recording; the shape is:
```
{"v": "1", "t": 61.2, "t_wall": 1791526473.1, "seq": 118, "type": "gesture", "frame": 612, "name": "tpose"}
```
**probe** {arms_up, on_box_ms?, span_ms?, t_first_up?, beep?}: both wrists above the head after 1 s below (the latency probe). Live,
`on_box_ms` is the time from the firing frame's receipt to this event (the receive-to-decision leg, as coach_agent measures it, no
audio leg); `span_ms` counts from the first up frame's receipt (the probe confirms over two frames, so this includes a frame interval:
the 106 to 165 ms the Mac saw). The worker adds `beep`: true only when it runs with `--latency-probe` (env `RTCOACH_LATENCY_PROBE=1`;
off by default because a jumping jack is an arm raise and beeps mid-set), the cue player is attached, no set is active and
`--no-probe-beep` is not given; the cue log then shows first-up-frame receipt to beep push. The event is emitted and logged either
way (`[event] probe ... (probe off ...)` or `(during a set: no beep, no floor window)`). A replay has no timing.
```
{"v": "1", "t": 52.0, "t_wall": 1791526471.628, "seq": 104, "type": "probe", "frame": 520, "arms_up": true}
```
**session** {state, exercise, target, count, item, done, mode}: on every state change of the session script. Plan mode:
`init` -> `wait` (announced, counter only, up to 3 s or first movement) -> `active` (counting, correcting) -> `rest` (5 s)
-> `wait` of the next item ... -> `done`. Single mode: `init` -> `active`; auto mode: `init` -> `detect` -> `active`.
```
{"v": "1", "t": 78.0, "t_wall": 1791526480.102, "seq": 149, "type": "session", "frame": 780, "state": "rest", "exercise": null, "target": 10, "count": 9, "item": 0, "done": false, "mode": "plan"}
```
With `--perception off` (section 2d) there is no session script: the worker emits ONE synthetic session event once the model
session is up (the caller is bound, the model listens and, with `--vision`, watches), so a handler can greet there. `exercise`,
`target` and `item` are null, `count` 0, `done` false, `mode` is `"voice"`, `frame` is the number of video frames received so
far. No `rep`, `exercise_seen`, `fault`, `cue`, `visibility`, `gesture` or `probe` event is ever emitted in that mode.
```
{"v": "1", "t": 3.2, "t_wall": 1791654028.1, "seq": 0, "type": "session", "frame": 31, "state": "open", "exercise": null, "target": null, "count": 0, "item": null, "done": false, "mode": "voice"}
```
**status** {fps_recv, fps_processed, pose_ms, clf_ms, pose_model, recv_to_decision_p50_ms?, dropped?, underruns?,
cues_played?, cues_dropped?, note?, state, exercise, count}: every 5 s. `fps_recv`, `dropped` and the delay are set by the
transport (null in a replay); `underruns` and the cue counts only when the player is attached; `note` when the pose model
dropped to lite.
```
{"v": "1", "t": 15.0, "t_wall": 1791526459.617, "seq": 34, "type": "status", "fps_recv": null, "fps_processed": 10.0, "pose_ms": 26.6, "clf_ms": 4.9, "pose_model": "full", "state": "active", "exercise": "squats", "count": 2}
```

## 2. The run modes of the worker

```
.venv/bin/python -m facetime.worker --help
.venv/bin/python -m facetime.worker [--plan "squats:10,pushups:8"] dev                    # (a) default: eyes + fast mouth
EVENTS_ONLY=1 .venv/bin/python -m facetime.worker dev                                     # (b) events only, no audio at all
.venv/bin/python -m facetime.worker --events-only --publish-data start                   # production, events also on the data channel
GEMINI_API_KEY=... .venv/bin/python -m facetime.worker --voice gemini dev                 # (c) eyes + fast mouth + Gemini Live between sets
GEMINI_API_KEY=... .venv/bin/python -m facetime.worker --perception off --vision --voice gemini dev   # (d) no perception: voice always on, video to the model
```
At the hackathon `GEMINI_API_KEY` is the team token and `GEMINI_BASE_URL=https://james-72765--rtcoach-gemini.modal.run` must be
exported with it (Brodie's relay, docs/STARTER_README.md section 4); a Google key of your own works with `GEMINI_BASE_URL` unset.
(a) **Default (fast mouth attached).** The worker publishes ONE audio track (`coach-audio`, 24 kHz mono) right after joining
and feeds it with the demo coach's CuePlayer and CueBank, unchanged: counts, closed-vocabulary corrections, praise, announces,
milestones and visibility lines, through the same priority slots, 145 to 225 ms frame-to-heard. The conversational layer gets
the events and must not speak over the fast path during a set (section 4).
(b) **Events only** (`--events-only` or `EVENTS_ONLY=1`). No CuePlayer, no AudioSource, no track: the audio track belongs to
whoever owns the conversation. `cue` events are still emitted as the script of what the fast path would have said.
(c) **Voice** (`--voice gemini`, needs `GEMINI_API_KEY`; `facetime/voice.py`): (a) plus Gemini Live (voice Puck, model
`GEMINI_LIVE_MODEL`, default `gemini-3.8-live`, language `en-US` via `GEMINI_LANGUAGE`) in the same participant. Its audio goes
through the cue player's one AudioSource (`CuePlayer.attach_mixer()`: pushed between clips, ducked about 10 dB under a count, delaying
a count by at most 60 ms: one 20 ms frame beyond the 40 ms lead) and it may only speak while the floor is open: session state `wait`,
`rest` or `done`, four seconds after a `gesture` or an armed `probe` (never during an active set: a gesture or probe there is logged,
not acted on), or when the caller talks during a rest (the caller's audio is not forwarded during a set). The cut when a set starts
is hard (otherwise the voice is heard under the counts): the reply is interrupted, the sink's buffer and the mixer queue are
cleared, and any frame that still arrives while the floor is closed is dropped. At the `rest` and `done` openings the model gets a
text summary of the set (exercise, reps, corrections spoken, the next item), never per rep and nothing at `wait` (a line there was
answered under the first counts). `ask_voice` prompts queued while a set runs are dropped when they are older than 15 s at the
opening or when the next set starts first. The model session starts when a caller is bound, never on a warm job. Model turns are
logged as `voice` events {text, t, role coach}; the caller's transcribed turns are NOT emitted or logged unless the worker runs with
`--log-caller` (then role caller), so by default no caller words leave the model session.
Logging lines to read a call by: `[voice] speaking (floor open|closed, state X, N ms after the ask)` when the first audio frame of a
reply reaches the sink (the transcript line `[voice] coach: ...` can arrive tens of seconds later, so it is not the time the voice
was heard), `[voice] cut: N frames dropped (floor closed), M ms of buffered audio cleared; S s of the reply was heard`,
`[voice] dropped queued ask (why): ...`, `[voice] context: ...` at each opening, `[voice] caller spoke (N words; not logged)`.
Extension points in `starter/`: `persona.md` (the prompt), `plan.txt` (the plan when `--plan`/`PLAN` are not given), `handlers.py`
with `on_event(event, say_now, ask_voice)`: `say_now(text)` is a bank string through the fast path (about 0.2 s, never over a
count), `ask_voice(prompt)` asks the model (about 1 s to first audio; queued for the next opening while a set runs). `handlers.py`
is loaded in every mode whenever the file exists (`--handlers PATH` for another file, `--handlers none` for nothing); without the
voice `ask_voice` returns None. The default handler asks for one short sentence at each `rest` and at `done`, nothing else.

Events go, in both modes, to `job.events` (an `asyncio.Queue` in the job process, for a conversational layer: register
`@facetime.worker.consumer async def fn(job)`), to the JSONL log `--event-log` / `RTCOACH_EVENT_LOG` (default
`data/tmp/worker_events.jsonl`, appended, one line per event plus room, call_id, job), and with `--publish-data` to the room data
channel (topic `brodie.events`, reliable, JSON per event; off by default because on a Chert line the room holds exactly two
participants and nobody else is there to read it; in a room of your own LiveKit project a third participant can read it). Cues the player finished or dropped go to `--cue-log` /
`RTCOACH_CUE_LOG` (the coach_agent `--log` record shape). The plan comes from `--plan`, else env `PLAN`, else `starter/plan.txt`
(`--plan-file`), else the demo plan `squats:10,pushups:8,jumping jacks:20,lunges:10`; `--exercise X` for one exercise, `--auto` for
auto-detect. `--latency-probe` (env `RTCOACH_LATENCY_PROBE=1`) arms the arms-up probe (beep and a 4 s voice window, never during a
set; off by default); `--gesture-next` turns the T-pose detector on; `--log-caller` keeps the caller's transcripts in the voice events.

(d) **Voice plus vision, perception off** (`--perception off [--vision] --voice gemini`, docs/STARTER_README.md section 13):
```
.venv/bin/python -m facetime.worker --perception off --vision --voice gemini --persona starter/examples/meal_prep/persona.md --plan-file starter/examples/meal_prep/recipe.md --handlers starter/examples/meal_prep/handlers.py dev
```
No Perception at all: prewarm loads no pose model, no session script runs, no `rep`, `exercise_seen`, `fault`, `cue`,
`visibility`, `gesture` or `probe` event exists. The worker joins, publishes the cue track, binds to the first remote participant
that publishes video (or audio when there is no video), starts the model session on that bind (never on a warm job), emits the
synthetic `session` `{state: "open", mode: "voice"}` event (section 1) once the session is up, then a `status` event every 5 s
`{fps_recv, state, exercise: null, count: 0, mode: "voice", voice: on|off, vision_frames?, vision_fps?, vision_replaced?, underruns?,
cues_played?, cues_dropped?}` (no pose fields), and ends on the same paths as (a) to (c): the caller's leave after `--leave-grace-s`,
the room's disconnect, `--max-s` after the first frame, a connect failure. Flags:
- `--perception {on,off}` (env `PERCEPTION`): `off` is this mode. With `off` the plan file (`--plan-file`, `--plan`, `PLAN`) is
  appended to the persona as free text ("The plan you are coaching, step by step: ..."), not parsed as exercises.
- `--floor {gated,open}` (env `FLOOR`): `open` is the default with `--perception off` (`gated` with perception on). Open means the
  Floor never closes: the caller's audio is always forwarded, `ask_voice` is spoken at once (never queued, never dropped),
  nothing is ever interrupted or cut; a `session`, `gesture` or `probe` event changes nothing. `--floor open` with perception
  on is allowed and warned about (the voice would talk over the counts).
- `--vision` (env `VISION=1`), `--vision-fps N` (env `VISION_FPS`, default 1, at most 2): the caller's video track reaches the
  model session. The worker already opens one `rtc.VideoStream` on the bound track (that is how a caller is detected); with
  `--vision` that read loop also offers each frame to `facetime/voice.py VideoTee`, a `livekit.agents` `VideoInput` set as
  `AgentSession.input.video` before `start()` (the session is built with `video_sampler=None`, so the tee's one frame per 1/N s
  is the only sampling; the newest frame replaces an unread one). The framework's own forward task pulls from the tee and
  hands every frame to the realtime session's `push_video`, which JPEG-encodes it (quality 75, scaled to fit 1024x1024: a
  1280x720 frame goes out as 1024x576). The worker's stream is RGB24 (Perception's format) and the plugin's encoder cannot
  convert that (`rgb24 to Rgba is not supported` in the rtc FFI), so the tee hands each
  sampled frame over as RGBA (`voice.rgba_frame`, under 10 ms). RoomIO's own video input stays off: it selects tracks by source (camera, screenshare)
  and the bridge's track source is not relied on. The tee is enabled only after the model session started: frames offered
  before that are counted (`vision_early`) and never sent; a warm job has no video and offers nothing. Logged once:
  `[vision] forwarding video to the model at 1 fps (1280x720)`; the `[status]` line and event carry `vision_frames` (sent).
  `--vision` without `--voice gemini` forwards nothing (said at start).
`say_now` still plays bank strings; `ask_voice` returns `'spoken'` at once. The probe and the T-pose detector do not run.
Tested offline only (scripts/test_worker_shutdown.py E and F, scripts/test_voice.py 9 to 11): a fake room with a fake
10 fps video track and a no-room model session that accepts one pushed frame. Not yet on a FaceTime call.

Models (pose, classifier, trigger, thresholds) and the cue bank load in the worker's prewarm hook (about 5 s here, 0.7 s of it
after imports); one preloaded job process is kept idle (`--idle-processes`, env `RTCOACH_IDLE_PROCS`), so a job start costs no
model load.

## 3. Dispatch: your own LiveKit rooms (automatic) first, the Chert CLI (named) second

- The worker registers with LiveKit under `AGENT_NAME` (env or `--agent-name`, default `brodie-coach` when the variable is
  missing). An EMPTY name (`AGENT_NAME=` in `.env.local`, or `--agent-name ""`) is automatic dispatch: LiveKit hands the worker a
  job for every new room of the project. That is the phone-page path (docs/STARTER_README.md section 3) and `fake_bridge.py`.
  Automatic dispatch fires when a room is CREATED, so start the worker first and mint a fresh room name per attempt; a room
  created while the worker was down never gets a job. The phone page joins as `chert-bridge` (the `--identity` default of
  `mint_token.py`) and publishes one video track `facetime-remote-video` (720x1280 from a portrait phone); the worker binds to it,
  logs `call id n/a`, and `call_id` is null on every line of that call. The job ends 2 s after the bound participant leaves
  (`--leave-grace-s -1` keeps it).
- A named worker receives only explicit dispatches and never joins a phone room. `chert dev --agent <name>` (the three Chert
  teams) creates a room in the team's LiveKit project with a RoomAgentDispatch for that name; LiveKit dispatches a job to the
  worker; the call is accepted once a participant of kind AGENT is in the room.
- Chert rooms: `chert-warm-<random>` (prewarm, `chert doctor --agent`) and `chert-<operation_id>` (a call). The bridge joins as
  `chert-bridge-<call_id>` and publishes the caller's camera (1280x720 at 15 fps); the worker binds to the first non-agent
  participant that publishes video, whoever it is, and logs the call id. Chert rooms hold two participants; the bridge binds to
  the one participant with an unmuted audio track, so the audio track is published at join time, before any video arrives. In
  a room of your own there is no two-seat limit; keep one audio publisher.
- **Warm jobs** (job metadata `{"chert_warm": true}`, or any job with no caller): the worker joins, publishes its (silent) audio
  track in mode (a), loads nothing more, says nothing, starts no session script, and stays joined until a participant publishes
  video or the room closes. Metadata keys `chert_call` and `media` are logged, never acted on.
- The worker ends the job itself (livekit-agents runs shutdown callbacks only after the entrypoint returns): 2 s after the bound
  participant leaves unless somebody rebinds (`--leave-grace-s`), on the room's disconnect (Chert deletes the room when the call
  ends), when connect() fails (a doctor or warm room torn down early), on `--max-s N` (N s after the first frame, tests) and on a
  perception error; cleanup is bounded (about 250 ms on test calls) and the process exits clean. A video track that goes away
  and comes back within the grace (camera switch, rejoin) is bound again and the session continues.
- Env: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` (the team's LiveKit project; the worker needs nothing else but the
  bundle on disk), `AGENT_NAME`, `PLAN`, `EVENTS_ONLY`, `RTCOACH_EVENT_LOG`, `RTCOACH_CUE_LOG`, `CUE_AUDIO_DIR` (voice bank);
  with `--voice gemini` also `GEMINI_API_KEY` and, at the hackathon, `GEMINI_BASE_URL`. Requires `livekit-agents==1.8.5`
  (requirements.txt).

## 4. What a conversational layer should do with the events

- **Speak only between sets or when the session opens a slot.** Slots: `session.state` `rest` (5 s after a set), `done`, the
  `wait` state right after an `announce` cue (the caller is getting into position; the floor is open for a question, but do not
  volunteer a line: one can be heard under the first counts), a `gesture` (`tpose`: the caller is asking for the next thing;
  4 s), an armed `probe` outside a set (4 s), and a caller's question. A `visibility.paused` is NOT a slot: the fast path has
  said "Step back ..." and the floor does not open on a pause. Never during `active`: the fast path owns the set, the first
  count cuts a voice reply, and an LLM voice lands 1 to 2 s after a rep bottom, one to two reps late at jacks cadence (the
  Gemini-in-the-loop baseline matched 0 of 16 counts).
- **Gesture and probe rules for the caller** (`facetime/probe.py`): the arm raise fires only after both
  wrists have been below the nose for a full second, so drop the arms and wait a second between raises; hold the T-pose for a
  good two seconds (1.5 s of detection, one missed frame allowed) and drop the arms once the coach speaks, since it will not fire
  again for 5 s; neither is acted on during a set (a jumping jack is an arm raise), and after "All done" a T-pose has nothing
  to move on to.
- **Do not repeat the fast path.** `cue` events are what was just said; use them as the transcript of the coach's half. In
  events-only mode the layer owns the only voice; if it speaks the counts itself it must design around 1 s, not 0.1 s.
- **Shape the next rest from the set.** `fault` events (label, p) say which corrections were decided and `cue`/`correction` which
  were spoken; `rep.confidence.agree` below 0.67 marks a rep only two of three trackers agreed on; `rep.lag_s` is the detection
  delay. Summarise at `rest`: count reached, the dominant fault, the tempo ("Slow down" / "Pick up the pace" cues).
- **Trust the plan, not `exercise_seen`.** The head is about 0.75 accurate on real sessions (above: "floor touches" at 0.94 while
  the caller stands); in plan mode it is informational. Only auto mode acts on it (3 s above 0.5 to declare).
- **Latency classes.** Fast path about 0.8 s rep bottom to count heard on the phone (0.81 s p50, 0.72 to 0.88; about 0.3 s of it is the counter's reversal detection); LLM voice 1 to
  2 s. Perception adds about 40 ms receive-to-decision at 10 fps on 4 CPUs.
- **Iterate without a call.** `replay_local --events PATH --fast --video my.mp4 --plan ...` writes the same stream for any phone recording that is already 10 fps (the replay does not resample: convert with the
  ffmpeg line in docs/STARTER_README.md section 6; the pipeline resizes to short side 288 itself); a handler can be
  developed against the file with the harness in docs/BUILD_ON_BRODIE.md section 3 (`replay_local` itself does not run handlers). The phone
  page costs nothing per attempt (a fresh room name each time); a Chert line is shared by time slot and Offline for a cron cycle
  after each call; on both, this is the inner loop.

## 5. Known limits

- Counter (current defaults validated offline on the sample recordings): pushups from a propped front camera
  can read +2 at a bad floor angle; jacks +1 on hands to the head after a set; a person who walks into place and squats within
  a second loses that first rep (walk veto); the first count needs 0.8 s of standing still. Above, run3's squat set reached 9
  of 10 and closed on the 75 s set timeout, and the first pushup landed at 137.8 s.
- A correction decided before a set's first count waits for "One" and is dropped if no rep comes, and no praise is spoken
  before the first rep (the `fault` at 7.5 s above is still emitted, its cue is not); cues decided
  before a visibility pause are dropped, never replayed.
- Visibility: whole-body rule with a 0.9 margin, pauses after 3 s; laptop framing and jumps at the top of the frame produce
  `partial` flicker (0.5 s hold) and can pause sets; no hysteresis on the per-frame state.
- `gesture` (T-pose) has fired on a device only after the plan was done, so "Okay, moving on" is unexercised mid-set on a
  call; `probe.on_box_ms` is on-box only (106 to 165 ms first-up-frame to beep on a FaceTime call).
- `exercise_seen` accuracy about 0.75; `status.fps_recv`, `dropped` and the delay are null in replays.
- `t` is per process and monotonic; `t_wall` is the only cross-process clock. `seq` restarts per job.
- Both paths are verified end to end. Phone page: the worker under automatic dispatch binds to the page's `facetime-remote-video`
  track at 720x1280, prints `first frame 720x1280` and "Let's do squats" is heard (the demo coach ran the full plan twice on the
  same page, every set on target). Chert: a FaceTime Video call through `chert dev --agent brodie-coach` ran the whole plan on
  a Mac (2 min 47 s, squats, jacks and lunges on target) and `--voice gemini` ran on a call too. `--publish-data` cannot be used in a two-participant Chert room (it can in your own).
- Voice: the hard cut above keeps the model off the counts, the persona and `language=en-US` keep it in English, and the
  default handler asks for a line at each rest. The voice holds until a clip ends instead of playing under it, the caller's
  audio reaches the model only when no Brodie audio has played for 0.3 s, the rest and wait clock pauses while either side
  talks (up to 25 s), and only one Brodie line plays at a time (an ask waits while any Brodie audio plays); prompts are never
  read aloud. These pass 61 offline and 63 live checks and have not been verified on a device yet.
- The voice seam: the bank is Gemini Puck; a conversational layer brings its own voice. Use the plain Puck voice for both the
  bank and the conversation, not the Australian bank; mixing voices sounds like two speakers.
