# Brodie real-time coach: start here (hackathon, Sat 2026-10-10, Tangen Hall)

You are a team with a LiveKit Cloud project of your own (cloud.livekit.io, the free tier is enough), a laptop running
macOS or Linux (no Windows), Python 3.12 with uv for the coach, and a phone (or a Mac with Safari) to stand in front of.
The caller joins a room of your project through our static phone page and your worker joins the same room (section 3):
that is the main path for every team. Three teams also have one of the three Chert FaceTime lines; for them the same
worker answers a real FaceTime call (section 5, with the Chert CLI facts from docs.trychert.com/facetime-cli). This page
is the whole read-me; the event schema is in docs/STARTER_EVENTS.md and what you can build is in docs/BUILD_ON_BRODIE.md.

## 1. What you are getting, and what you change

A working real-time video coach in one Python process, one LiveKit participant, one audio track. It runs the same
whether the caller arrives through the phone page in a room of your own LiveKit project or through a Chert line (a
Chert room has exactly two seats, the caller's bridge and your agent, so there everything you add runs inside this
process; in your own room there is no such limit, but keep one audio publisher). Its eyes are
`facetime/perception.py`: MediaPipe pose on every sampled frame at 10 fps, a rep counter and a fault detector, turned
into a versioned stream of events (`rep`, `exercise_seen`, `fault`, `cue`, `visibility`, `gesture`, `probe`, `session`,
`status`, and `voice` for the model's turns with `--voice gemini`). Its fast mouth is the pre-rendered Brodie cue bank (`data/cues_audio_gemini`, 379 clips, Gemini TTS, voice Puck;
an Australian-direction rendering of the same strings, `data/cues_audio_gemini_au`, is optional via `CUE_AUDIO_DIR`):
counts, corrections, praise and announcements are heard about 0.2 s
after the decision (145 to 225 ms measured over the LiveKit SFU). Its conversation is Gemini Live in the same Puck
voice, gated to the gaps between sets (`--voice gemini`, `facetime/voice.py`, section 2c of docs/STARTER_EVENTS.md);
the raw event stream is also there for your own layer (`@facetime.worker.consumer`, section 2 of that page).

You change three files in `starter/` (you write coaching, not perception):
- `starter/persona.md`: the system prompt of the conversational voice (who Brodie is, how it talks between sets).
- `starter/plan.txt`: the session script, in `--plan` syntax: `squats:10,pushups:8,jumping jacks:20,lunges:10` (27 names
  are accepted; these four count and are validated on a device, nine more count untested, the rest run as timed correction
  blocks: docs/BUILD_ON_BRODIE.md section 1).
- `starter/handlers.py` (loaded by default whenever the file exists, with or without `--voice gemini`; `--handlers none`
  turns it off): `def on_event(event, say_now, ask_voice)`, called once per event with the event dict
  (every event carries `v`, `t`, `t_wall`, `seq`, `type` and, except `status`, `frame`). Two verbs with two latencies:
  `say_now(text)` plays a bank line through the fast path's priority slots, heard about 0.2 s later, only strings in
  the bank (`data/cues_audio_gemini/manifest.json`, keyed on the exact string); `ask_voice(prompt)` hands the LLM a
  prompt and is heard 1 to 2 s later; the shipped handler asks for one sentence at each `rest` and a warm close at `done`.
  Call `ask_voice` only when the session opens a slot: `session.state` is `rest` or `done`, a `gesture` `tpose`, or a
  caller's question (a visibility pause never opens the floor; a line asked during the `wait` before a set is cut by its
  first count). Never during `active`: the fast path owns the set, the first count of a set cuts any reply still
  playing, an ask queued during a set is dropped at the next set start or after 15 s, and `cue` events are the
  transcript of what the fast path already said, so do not repeat them.
  The same events are read raw with `@facetime.worker.consumer async def fn(job): ev = await job.events.get()`.

## 2. Setup

Ten minutes of typing, 25 to 45 minutes of waiting.
```
git clone https://github.com/seraphimlabs-dev/brodie-starter.git brodie-rt-coach && cd brodie-rt-coach
git checkout -b team-<your-team>   # work on a branch of your own
uv venv --python 3.12 .venv && uv pip install -r requirements.txt
mkdir -p models && for m in lite full heavy; do [ -f models/pose_landmarker_$m.task ] || curl -sSfL -o models/pose_landmarker_$m.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_$m/float16/latest/pose_landmarker_$m.task; done
```
`uv pip install` is the long step: torch, mediapipe, transformers, modal and livekit (on Linux torch pulls about 3 GB of
CUDA wheels; run `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` first to skip them),
then the tarballs and the pose models (about 120 MB); the three Chert teams also install Node and the Chert CLI and run
the `chert login` browser flow (section 5). No `uv` yet: `pip install uv` or `brew install uv`.
You do not need Modal or any cloud of ours: the worker runs on your laptop (a Mac processed 10.7 fps with pose at 10 ms), the `modal` package in requirements.txt is a dependency of our demo path and stays unused, and nothing here
calls our deployment. The two shared conveniences are the phone page (section 3), whose link `mint_token --url` prints is
served from our host but is a static page with the url and token of YOUR LiveKit project in it (opening
`facetime/phone_test.html` as a file in Safari on a Mac works the same), and the Gemini relay (section 4), which stands in
for a Google account.
The data folder is never in git. The starter bundle is three tarballs (`rtcoach_starter_part1_models_recordings`,
`part2_bank_au`, `part3_bank_puck`, dated 2026-10-09); unpack part 1 and part 3 at the repo root (part 2 is optional) and
everything lands under `data/`: `data/models/clf_pose.pt` and `trigger_lr.json`, the voice (`data/cues_audio_gemini/`,
plain Puck, 379 clips, part 3, the bank the demo uses), the optional Australian-direction bank (`data/cues_audio_gemini_au/`,
378 clips, part 2; select it with `CUE_AUDIO_DIR=data/cues_audio_gemini_au`), the sample recordings (section 6) and
`data/STARTER_BUNDLE_README.txt`. The player prefers the plain Puck bank; use it for both the bank and the conversation, since mixing voices sounds like two speakers. The thresholds, `cues.json` and `cue_prior.json` are in `src/rtcoach/` in git. Check:
```
ls data/models/clf_pose.pt data/models/trigger_lr.json data/cues_audio_gemini/manifest.json models/pose_landmarker_full.task   # plus data/cues_audio_gemini_au/manifest.json if you unpacked part 2
```
Before the first call, prove the install on a bundle recording (about 60 s; expect squats 9 of 10 on the 75 s timeout and the
first pushup at 137.8 s), then the contract test (PASS in about 30 s) and the offline voice tests (83 checks, no key needed):
```
python -m facetime.replay_local --video data/tmp/mac_handoff/crops/run3.mp4 --plan "squats:10,pushups:8,jumping jacks:20,lunges:10" --fast --events data/tmp/run3.events.jsonl
python scripts/test_perception.py --video data/tmp/mac_handoff/segments/run3_squats.mp4 --plan squats:10
python scripts/test_voice.py
```
Then read docs/BUILD_ON_BRODIE.md: what you can build on this and how.
Create `.env.local` in the repo folder (never commit it). The three LiveKit values come from one LiveKit Cloud project of
your own: create it at cloud.livekit.io (the free tier is enough), then Settings -> Keys gives the URL
(`wss://<project>.livekit.cloud`), the key and the secret:
```
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=<key>
LIVEKIT_API_SECRET=<secret>
AGENT_NAME=
GEMINI_API_KEY=<the team token on your card>
GEMINI_BASE_URL=https://james-72765--rtcoach-gemini.modal.run
```
`AGENT_NAME=` is set EMPTY on purpose: an empty name is automatic dispatch, the worker joins every new room of your
project, which is what the phone page needs (section 3). A named worker (the default `brodie-coach` when the variable is
missing) waits for a named dispatch and never joins a phone room; only the three Chert teams set
`AGENT_NAME=<the dispatch name on the card>`, and only for the line (section 5). The worker itself needs nothing but these
three LiveKit values and the bundle on disk; `GEMINI_API_KEY` and `GEMINI_BASE_URL` are for `--voice gemini` and for
re-rendering the bank, through our relay (section 4). You re-render the bank with `python -m facetime.cues_tts --backend
gemini` only if you add strings to `cues.json` (plain Puck into `data/cues_audio_gemini`, only the strings still missing;
about 4 minutes and $0.16 for the whole bank). For the optional Australian bank add `--style "(Australian accent, warm,
upbeat trainer) " --out-dir data/cues_audio_gemini_au` (the bracketed direction is not read aloud, a spoken prefix would be).

## 3. First call on the phone page (LiveKit)

The phone (or a Mac in Safari) joins a room of your LiveKit project through `facetime/phone_test.html`, publishes its camera
and plays the coach; your worker joins the same room by automatic dispatch. Nothing loads `.env.local` for you (the LiveKit
Agents CLI does not read it, and `mint_token` needs the key and secret exported), so export it in both terminals.

Terminal A, the coach:
```
cd brodie-rt-coach && source .venv/bin/activate && set -a && source .env.local && set +a
python -m facetime.worker dev                                   # the plan in starter/plan.txt (ships as squats:10,pushups:8,jumping jacks:20,lunges:10)
python -m facetime.worker --plan "squats:10,pushups:8" dev      # your plan
python -m facetime.worker --voice gemini dev                    # plus Gemini Live (Puck): talks only between sets, needs GEMINI_API_KEY and GEMINI_BASE_URL (section 4)
```
`AGENT_NAME=` empty in `.env.local` (or `--agent-name ""` before `dev`) is what makes the worker join your room; with a
dispatch name set it registers and then waits forever.

Terminal B, the link (same folder, same exports; a fresh room name every attempt):
```
cd brodie-rt-coach && source .venv/bin/activate && set -a && source .env.local && set +a
python -m facetime.mint_token --room team1-try1 --url
```
It prints THREE lines: the token, the query string, and the https link on our host,
`https://james-72765--rtcoach-webhook.modal.run/phone?url=...&token=...&room=...`. The THIRD line is what the phone opens:
AirDrop it, send it in Messages, or make a QR code of it. Add `&facing=user` for the front camera (the default is the
back camera) and `&autojoin=1` to join without a tap; `&agent=<identity>` changes which participant the page calls the
coach (below). The page is static: our host only serves the HTML, and the `url` and `token` in the link are your own
LiveKit project (the `room` in the link is informational, the token carries the room). On a Mac, open
`facetime/phone_test.html` as a file in Safari with the same query string: it works the same, because `file://` is a
secure context for the camera, which is also why the phone needs the https link and not http from your laptop. The page
loads `livekit-client` from jsdelivr, publishes one camera track named `facetime-remote-video` asking for 1280x720 at
30 fps (a phone held in portrait delivers 720x1280; the worker samples at 10 fps), plays every audio track it subscribes
to, and shows "Tap to enable audio" when the browser blocks autoplay: tap it.

Order matters: start Terminal A before the phone joins, and use a new room name per attempt. Automatic dispatch fires when
a room is CREATED (the phone's join creates it), so a room the phone opened while the worker was down, or a room reused from
an earlier attempt, never gets a job. The job ends 2 s after the phone leaves (`--leave-grace-s -1` keeps it; a page reload
within 2 s binds again). Prop the phone (section 7) and step in.

What it prints when it works. Terminal A: `[worker] agent ''; mode plan (...); audio cue player; data channel off; events ->
data/tmp/worker_events.jsonl`, `[worker] prewarm: loaded in 5.x s (pose full, counter v2, trigger
data/models/trigger_lr.json, ...)`, the LiveKit line `registered worker` (the notices `dev mode is deprecated ... use lk agent
dev` and `in-process auto-reload has been removed` are livekit-agents 1.8.5 talking; ignore them); when the phone joins
`[worker] job ...: joined room 'team1-try1'`, `[worker] published audio track coach-audio`, `[worker] participant joined:
chert-bridge`, `[worker] bound to 'chert-bridge' (call id n/a): video 'facetime-remote-video' 720x1280`, `[worker] first frame
720x1280`, `[cue] HH:MM:SS [announce] "Let's do squats"`, then `[event] rep squats 1 (agree 1.0, ...)` and `[cue] ... [count]
'One' decision->push N ms` per rep, and a `[status]` line every 5 s (expect processed about 10 fps, pose about 20 ms,
underruns 0). On the phone: "Let's do squats" from the speaker, then the counts.

The page's "Coach present" indicator is green ONLY for a participant whose identity equals `agent` exactly (default
`brodie-agent`). The worker under `dev` joins as `agent-<job id>`, so the indicator stays "no" on a working call; the real
signal is the page's "Coach audio: playing from agent-..." line, and the audio itself. To make it green, join one room
directly under that identity instead of waiting for a dispatch: `python -m facetime.worker connect --room team1-try1
--participant-identity brodie-agent` (`connect` is a LiveKit Agents CLI subcommand).

On this path the caller's participant is plain `chert-bridge` and there is no call id: `call_id` is null in your logs, and
your hand-in is keyed on the room name and the time of the call (section 10). In your own room there is no two-seat limit (a
second phone or a dashboard participant can join; the worker binds to the first non-agent participant that publishes video
and ignores other video), but keep one audio publisher.

`--voice gemini` on a call: on a FaceTime call over the Chert CLI the model session started 0.9 s after the worker bound to
the caller, the caller's words were transcribed but not logged, and the worker ended 252 ms after the hang-up. The first count
of a set cuts the voice, and the persona keeps it in English. It waits for a clip to end instead of playing under it, the caller's
audio reaches the model only when no Brodie audio has played for 0.3 s, the rest clock pauses while either side is talking
(up to 25 s), and only one Brodie line plays at a time (an ask waits while any Brodie audio plays); prompts are never read
aloud. These rules pass 61 offline and 63 live checks but have not been verified on a device yet, so listen for overlapping
voices on your first call.

## 4. The Gemini relay

Your card carries a team token, not a Google key. With `GEMINI_API_KEY=<team token>` and
`GEMINI_BASE_URL=https://james-72765--rtcoach-gemini.modal.run` exported, every Gemini byte from your laptop goes to our relay
(`modal/gemini_relay.py`), which swaps the token for our key and forwards to Google: the Live websocket (`/ws/...`, the
worker's `--voice gemini`) and plain HTTPS (`generateContent`, `models.list`, TTS for `cues_tts`, streaming `alt=sse`). The
token travels in the `x-goog-api-key` header; the google-genai SDK does that itself once the two variables are set, so the
worker and `cues_tts` need no change. An unknown token gets HTTP 401 (JSON `{"error": "unknown team token"}`) or a websocket
close 1008. Each team has 240 Live minutes; past the cap the relay answers 429 or close 1008 "team cap reached, ask the
organiser". `GET /health` is public and shows token counts and totals, never a key or token. A team's own Google key works
too: put it in `GEMINI_API_KEY` and leave `GEMINI_BASE_URL` unset. Smoke test (replace the token, never paste it anywhere):
```
curl -s -X POST "https://james-72765--rtcoach-gemini.modal.run/v1beta/models/gemini-2.5-flash:generateContent" -H "x-goog-api-key: <team token>" -H "content-type: application/json" -d '{"contents":[{"parts":[{"text":"hi"}]}]}'
```

## 5. If your team has one of the three Chert lines

Everything above still applies; the line adds a Chert project (one FaceTime line), the Chert CLI, a dispatch name, and a
real FaceTime call in place of the phone page. The CLI facts are from docs.trychert.com/facetime-cli.

The Chert CLI (Node 22.23.2 or 24.x, the same Node in both terminals):
```
node --version                       # 22.23.2 or 24.x; same Node in both terminals
npm install -g @trychert/facetime@0.1.0-beta.4 && chert --version    # pin the version: the unpinned install can fail
chert login                          # browser sign-in for your project; `chert login --device` on a remote shell
chert whoami
chert doctor                         # run from the repo folder: it reads LIVEKIT_* from .env.local; read-only
```
In `.env.local` set `AGENT_NAME=<the dispatch name on your assignment card>` while you work the line: the CLI dispatches
to that name, and a worker with an empty name is not what it dispatches to. Put it back to empty for the phone page.

Terminal A, the coach (export `.env.local` as in section 3):
```
cd brodie-rt-coach && source .venv/bin/activate && set -a && source .env.local && set +a
python -m facetime.worker dev                                   # or --plan "...", or --voice gemini, as in section 3
```
Terminal B, the line (same folder, same Node):
```
cd brodie-rt-coach && set -a && source .env.local && set +a
chert doctor --agent "$AGENT_NAME"   # dispatches a real job into a temporary room; Terminal A must already be running
chert dev --agent "$AGENT_NAME"
```
Then place a FaceTime VIDEO call from your own phone to the address on your card, prop the phone (section 7) and step in.

What it prints when it works. Terminal A, after the start-up lines of section 3: for Chert's waiting room `[worker] job ...:
joined room 'chert-warm-...'`, `[worker] published audio track coach-audio`, `[worker] warm job (chert_warm): no caller;
staying joined, nothing is said until a participant publishes video`; on your call `[worker] participant joined:
chert-bridge-<call_id>`, `[worker] bound to 'chert-bridge-<call_id>' ... 1280x720`, `[worker] first frame 1280x720`,
`[cue] HH:MM:SS [announce] "Let's do squats"` about 7 s after the accept, then the rep, count and `[status]` lines as in
section 3. Terminal B: `Connected to Chert. Call readiness is reported separately.`, then
`"<name>" joined a waiting room. Media readiness is not verified.`, and after you hang up `Call ended: room removed.`

Caveat: Chert's documented starter is FaceTime Audio only and the docs say video "require[s] separate development and
validation". In testing, a FaceTime Video call through `chert dev` on our dev line reached exactly this worker at
1280x720, 15 fps, and ran the whole plan, so the CLI path passes video on a validated line; whether your workshop line is validated is the organiser's question to Chert. If your call connects
but Terminal A never prints `bound to`, the line is not passing video: tell the organiser and iterate on recordings
(section 6) and the phone page (section 3) meanwhile.

The rules of the line:

- One line = one connected CLI = one active call at a time; more agent names add no capacity. Start Terminal A before
  B: the call is only accepted once an agent participant is in the room, and the worker needs its models preloaded.
- `chert dev` prepares and refills waiting rooms with no caller, and `chert doctor --agent` dispatches a real job. The
  worker treats both as warm jobs: it joins, publishes its silent track, says nothing and starts no script until video
  arrives. Keep it that way if you touch `worker.py`: never greet or start the plan on job start.
- The line is Offline for a few minutes after each call: on our dev line Chert's worker showed "Offline, no
  recent successful command poll" after every call, and calls in that window ring out; the line refreshes on Chert's
  next cron cycle (`chert doctor` can also report the machine behind the line offline for a while). The
  CLI docs expect a second call right after cleanup, so try it, but do not plan on it: recordings are your inner loop
  (section 6) and the phone page costs nothing per attempt (section 3); the line is for checking.
- After changing code, restart both terminals so no prepared room runs old code: end the call, Ctrl+C in Terminal B
  and wait for `Stopped. The line no longer routes to this agent.`, then stop Terminal A, then start A, then B.
- Handoff at the end of your slot: the caller hangs up, you wait for `Call ended: room removed.`, Ctrl+C `chert dev`
  and wait for the release, then stop the agent. The next team runs `chert doctor` before starting. If cleanup or
  release stays pending, hold the handoff, keep `~/.chert` intact, do not delete state, and contact the organiser.
  `--takeover` replaces another team's session and is not yours to run.
- When `chert doctor` fails: `project_not_signed_in`, run `chert login` for the right project (and `chert setup --clear`
  if a stale saved setup is reported); `project_role_insufficient`, ask the organiser for owner/admin; no line or more
  than one line, ask the organiser for exactly one; busy line or offline worker, wait or ask, never force it; missing
  LiveKit keys, run from the folder with `.env.local` and check for stale shell `LIVEKIT_*` overrides. Line provisioning
  and media troubleshooting belong to the organiser and Chert, not to you.

## 6. Iterate without calling

Any video of a person goes through the same perception and session code and writes the same event stream:
```
python -m facetime.replay_local --video my_session.mp4 --plan "squats:10,pushups:8" --fast --events events.jsonl
```
It prints every cue with its clip time, and `events.jsonl` is schema v1 (section 1 of docs/STARTER_EVENTS.md) with `t`
equal to the clip time; `--fast` skips wall-clock pacing; `--exercise squats` for one exercise, `--gesture-next` to
try the T-pose. Record with your phone in portrait, propped as in section 7, whole body, 2 to 3 m away; walk in,
stand still for a second, then do the set. Convert the clip to 10 fps first: the replay treats every file as 10 fps and loads every frame into memory (a raw
30 fps phone clip runs as 3x slow motion and two minutes of 1080p is 20 GB of frames):
`ffmpeg -i IMG_1234.mov -vf "fps=10,scale='if(gt(iw,ih),-2,288)':'if(gt(iw,ih),288,-2)'" -an -c:v libx264 -preset fast -crf 23 data/tmp/mine.mp4`
(the pipeline resizes to short side 288 itself, the scale only keeps the file small). `replay_local` does not run
`handlers.py`: feed the events file to your `on_event` with the ten-line harness in docs/BUILD_ON_BRODIE.md section 3,
then check once on the phone page (section 3) or, for the three Chert teams, on the line (section 5).

Three sample recordings ship in the bundle, already cropped to the camera view
(portrait 288 px, 10 fps) at `data/tmp/mac_handoff/crops/`: `run3.mp4` (a clean full plan on the LiveKit phone page),
`run2.mp4` (end of squats, pushups with phone handling, 20 jacks, 10 lunges) and `live.mp4` (the full plan on a real
FaceTime call), plus per-exercise cuts in `data/tmp/mac_handoff/segments/`. Their event logs sit beside them as `<name>.events.jsonl`, written by an earlier policy version: the current code
writes fewer `cue` events on the same crop (run3: 18 instead of 27; the nine that went are a standing "Sink lower" before
the first count and idle praise during the uncounted pushup set; reps and session states are identical). Regenerate them
with the command above (`--video data/tmp/mac_handoff/crops/run3.mp4`, about 60 s) rather than diffing against the
shipped files.

Latency from a recording: film the demo phone's screen and speaker with a second phone (a FaceTime screen recording has
no audio), then `python scripts/e2e_from_recording.py film.mp4 --crop W:H:X:Y --t0 11 --t1 36 --exercise squats`
prints rep bottom to count heard per rep and the p50 for one set.

## 7. Framing and physics

- Phone propped at knee to waist height, 2 to 3 m from the floor spot, whole body in frame with the feet above the
  bottom edge and room above the head, a plain wall behind if you can. A laptop camera cannot frame a standing adult
  top to bottom at that distance; a phone in portrait can.
- On the phone page the camera reaches the coach as the phone holds it, 720x1280 portrait at up to 30 fps, sampled at
  10 fps. On a Chert line FaceTime video arrives as 1280x720 landscape at 15 fps whatever the phone's orientation (Fill
  framing can crop a portrait phone to 16:9 and cut the feet or head). The pose works on both; keep the body central.
- Step into place before the start line and stand still for a second; do not reach for the phone mid-set.
- Pushups need the phone low and side-on to the floor spot, about 2 m away. Lunges count one per leg (ten = five pairs).
- If the head or feet leave the frame the coach says "Step back so I can see your whole body" (at most every 20 s),
  pauses counting and the set clock, and resumes with "Great, I can see you".
- Arm raise (the latency probe, `--latency-probe`) and T-pose (`--gesture-next`): both are off by default. The probe
  never acts during a set; between sets, keep both hands below the nose for a full second, then raise both arms (one
  beep per raise). The T-pose closes the current set or skips a rest: hold both arms straight out at shoulder height
  for two seconds, facing the camera, and drop the arms once it says "Okay, moving on"; it cannot fire again for 5 s,
  and after "All done" there is nothing left to move on to. Alternating raises and Ts with no pause between them fires
  nothing.

## 8. Latency you can count on

Two classes. Fast path (bank): decision to heard about 0.2 s (145 to 225 ms over the SFU); rep bottom to count heard
about 0.8 s (0.81 s p50 measured on a phone-page recording, 0.72 to 0.88; about 0.3 s of it is the counter's
reversal detection; the two FaceTime legs add a little more by ear). Anything through the LLM: 0.8 to 1.2 s to first audio for every 2026
realtime model, so a count spoken by the LLM lands 1.4 to 1.9 s after the bottom, one to two reps late at jacks cadence
(the Gemini-in-the-loop baseline matched 0 of 16 counts). Put counts and form cues on the fast path, conversation on
the LLM. Perception itself costs about 40 ms receive-to-decision at 10 fps on 4 CPU cores.

## 9. What the coach is good and bad at

Good: counting squats, pushups, lunges and jumping jacks from a propped phone with the whole body in frame; a plan that
announces each exercise, counts to a target and rests; corrections for the common faults (depth, back, knees, pace);
side cues for one-sided stretches; the step-back line; a count never stuck behind a correction, and no correction after
the set has closed.
Bad: shallow reps can be undercounted (counter exact 0.87 on MM-Fit, 0.59 against coaches' spoken numbers on
home sessions); corrections agree with the human coach about 44 percent of the time when both spoke, are generic ("Sink
lower") rather than specific, and about 5 per minute are not matched by a coach's; auto exercise detection
(`exercise_seen`, `--auto`) is about 0.75 accurate and needs 3 s, so trust your plan and keep plan mode; the false reps
that remain: pushups from a propped front camera can read +2 at a bad floor angle, jacks +1 on hands to the head after
a set, a person who walks in and squats within a second loses that first rep, and the first count needs 0.8 s of
standing still; a correction decided before a set's first count waits for "One" and is dropped if no rep comes, and no praise is spoken before the first rep; jumps at the top of the frame
can flicker `partial` and pause a set; the T-pose ("Okay, moving on", `--gesture-next`) has fired on a device only after
"All done", with no set left to move on from, so mid-set use is untested on a call; it needs the 1.5 s hold and 5 s between fires (section 7)
and is ignored by design while the whole body is not visible.

## 10. Telemetry and privacy

By default the worker writes two local files and nothing else: `data/tmp/worker_events.jsonl` (every event plus `room`,
`call_id`, `job`; `--event-log` / `RTCOACH_EVENT_LOG`) and `data/tmp/worker_cues.jsonl` (cues played or dropped;
`--cue-log` / `RTCOACH_CUE_LOG`). No frames, no thumbnails, no caller audio, no transcripts, no FaceTime handles, no
tokens: the only caller identifier is the Chert bridge's `call_id`, and on the phone page there is none (`call_id` is
null, the participant is plain `chert-bridge`). Opt out: `--event-log /dev/null --cue-log /dev/null`.
Brodie asks you to leave the logs on and hand them in after the event (the same files, keyed on the room name and the
time of the call; on a Chert line also on `call_id`). The Gemini
Live layer (`--voice gemini`) logs its own turns as `voice` events and keeps the caller's words out of every sink unless
you opt in with `--log-caller`; caller audio is never written anywhere by this code. Tell your callers one
line before the set: "This call is coached by software: it reads your body pose from the video and keeps a log of rep
counts and cues, not the video or the audio."

## 11. Licence and attribution

- `clf_pose.pt`, `trigger_lr.json`, `thresholds_e3c.json` and `cue_prior.json` are results of a 30-day QEVD evaluation
  licence and ship with the starter under its terms: keep them private (no public GitHub, no LFS, no public URL), never
  identify a participant, no face recognition. The cue wording in `src/rtcoach/cues.json` and the audio bank are ours.
- What ships from the QEVD taxonomy is its fault label names, used verbatim as keys in `src/rtcoach/cues.json`, the
  thresholds files and `cue_prior.json` (for example `squats - knees over toes`), and nothing else. No QEVD clips, frames,
  annotations or coach phrases are included, and none may be requested; the sample recordings are our own, used with consent. The
  counter benchmark data (MM-Fit, CC BY 4.0) is not included either.
- MediaPipe (pose models and SDK): Apache License 2.0. LiveKit Agents SDK and client: Apache License 2.0. Your LiveKit
  Cloud project: your own account under LiveKit's terms. The Chert CLI and FaceTime line (three teams): your organiser's
  agreement with Chert.

## 12. Getting help

Paste, in this order: the Terminal A lines from `[worker] prewarm` through the failure and the last `[status]` line; the
room name you minted and the time of the attempt; what the phone page shows in its log panel and on its "Coach audio" and
"Coach present" lines (remember that "Coach present" stays "no" under `dev`, section 3); and whether `curl` against the
relay's `/health` answers (section 4). A Chert team adds the full `chert doctor` output, `node --version`, `chert --version`
and its `AGENT_NAME`. Never paste tokens, API keys, `.env.local`, the phone link (it carries your token), or an unsanitised
log. Your support contact is on the card.

## 13. Your own coach: beyond reps

The worker has two modes.

**Reps (as shipped).** `facetime/perception.py` watches the caller at 10 fps (pose, rep counter, fault detector, the session
script from `starter/plan.txt`), the fast path speaks counts and corrections from the bank about 0.2 s after the decision, and
the Gemini voice is gated to the gaps between sets. Everything in sections 1 to 12 is about this mode.

**Voice plus vision, perception off.** `--perception off` runs no pose model, no counter and no script: the worker still joins,
publishes its audio track, binds to the first caller that publishes video (or audio, when there is no video), starts the Gemini
Live session on that bind, and from then on the model is always allowed to talk (`--floor open`, the default here): the caller's
audio is always forwarded, `ask_voice` is spoken at once, nothing is ever cut. `--vision` hands the caller's video to the model at
about 1 fps (`--vision-fps`, at most 2): each frame goes out as a JPEG scaled by the plugin to fit 1024x1024 (a phone-page
720x1280 frame as 576x1024, a FaceTime 1280x720 frame as 1024x576), which is what
the Live API wants for a scene that changes slowly, like a kitchen. The plan file is appended to the persona as free text (a
recipe, a checklist), not parsed as exercises. Your `handlers.py` sees ONE synthetic `session` event, `{state: "open",
exercise: null, count: 0, mode: "voice"}`, when the model session is up (greet there), a `status` event every 5 s (frames
received, frames sent to the model, no pose fields) and the `voice` events; no `rep`, `cue`, `fault` or `visibility` events.

The meal-prep example in `starter/examples/meal_prep/` (a persona that names what it sees before it instructs, one step at a
time, safety first with knives and heat; a 15-minute garlic butter pasta with greens as numbered, timed steps; a handler that
greets on the session open event and a `later(seconds, prompt, ask_voice)` helper for timed nudges on the worker's event loop):
```
.venv/bin/python -m facetime.worker --perception off --vision --voice gemini --persona starter/examples/meal_prep/persona.md --plan-file starter/examples/meal_prep/recipe.md --handlers starter/examples/meal_prep/handlers.py dev
```
Tell the caller: prop the phone so the camera sees the counter and the stove (landscape, about an arm's length away, not pointed
at the ceiling), keep your hands in view when you cut or stir, and say "ready" or hold the board up to the camera when you want
the next step. The coach asks you to show the pan or the board when it is unsure.

What to expect. The first audio arrives about 1 s after the model decides to speak (measured 0.9 to 1.3 s to first audio on a
text turn with `gemini-3.8-live`). Video goes at 1 fps, so the model sees a step change (the pasta going in, the garlic
colouring) within a second or two; it only sees what the phone camera sees, so framing is everything. `[vision] forwarding video
to the model at 1 fps (1280x720)` is logged once on the first forwarded frame and the `[status]` line carries the count of frames
sent. The greeting comes from your handler on the session open event, about 1 to 2 s after the caller is bound.

Limits. There is no fast path for cooking: every word goes through the model (about 1 s), the bank `say_now` still plays the
strings that are in the bank (`data/cues_audio_gemini/manifest.json`: counts, "Nice work", "Rest for a moment" and the like), and the
arms-up probe and the T-pose gesture do nothing in this mode (no pose model runs). The model cannot read a thermometer or taste;
it works from what it sees at one frame per second and what the caller says. This mode has been tested offline only (fake
room, fake frames, a no-room model session with one pushed frame); it has not been on a call, phone page or FaceTime. Expect to tune the persona
on the first call: if the model narrates too much, tell it so in `persona.md`.
