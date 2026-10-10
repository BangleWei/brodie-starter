# Build on Brodie: the skills you can make in a day, and how (2026-10-10)

Read this after section 2 of docs/STARTER_README.md (the setup). Every command below runs from the repo root with the
virtualenv activated (`source .venv/bin/activate`, so `python` is `.venv/bin/python`) and the starter bundle unpacked under
`data/`. Nothing here needs a call until it says "on a call": a call is the phone page in a room of your own LiveKit
project (README section 3; a fresh room name per attempt, nothing to wait for) or, for the three teams that have one, the
Chert FaceTime line (README section 5; Offline for a few minutes after every call). Calls are for checking, recordings are
the inner loop. The event schema is docs/STARTER_EVENTS.md.

## Brodie in five sentences

1. **Eyes.** `facetime/perception.py` runs MediaPipe pose on the caller's video at 10 fps, a rep counter (`src/rtcoach/repcount2.py`)
   and a fault classifier, and turns them into a stream of events: `rep`, `fault`, `cue`, `visibility`, `gesture`, `probe`,
   `exercise_seen`, `session`, `status` (plus `voice` for the model's turns). Your code reads events; it never touches frames.
2. **Fast mouth.** The bank: 377 pre-rendered strings (379 clips, Gemini TTS, voice Puck, `data/cues_audio_gemini/manifest.json`)
   pushed into the one audio track by the cue player about 0.2 s after a decision (145 to 225 ms measured). Counts, corrections,
   praise, announcements and your `say_now(text)` lines all come from the bank, and only strings in the bank can be said.
3. **Conversation.** Gemini Live (same Puck voice, `facetime/voice.py`) in the same process, mixed into the same track, allowed to
   speak only when the session opens a slot (the rest after a set, the wait before one, the end, a gesture, a caller's question
   during a rest). About 1 s from a prompt to the first audio, so it never counts.
4. **The session script.** `starter/plan.txt` (or `--plan`) is a list `name:count,...`: for each item the coach announces it, waits
   for movement (up to 3 s), counts to the target with corrections, says "That's ten, nice work", rests 5 s, moves on, and ends with
   "All done, great session"; a set that has not reached its target after 75 s is closed with the count reached (`--set-timeout`).
5. **One participant, one audio track.** Everything you build runs inside `python -m facetime.worker` (`facetime/worker.py`): on a
   Chert line the room has exactly two seats, the caller's bridge and your worker, so there is no second process on that call and
   no third seat; in your own LiveKit room there is no such limit, but keep one audio publisher.

## Twenty minutes before any call, before anything else

Prove the install with the bundle, then build against recordings. Times below are on the bundle files, on 4 CPU cores:

```
python -m facetime.replay_local --video data/tmp/mac_handoff/crops/run3.mp4 --plan "squats:10,pushups:8,jumping jacks:20,lunges:10" --fast --events data/tmp/run3.events.jsonl
```
About 60 s. It prints every cue with its clip time (`t=11.90s [count] 'One'`) and writes 263 events. Expect squats 9 of 10
(the set closes on the 75 s timeout), the first pushup at 137.8 s. The shipped `data/tmp/mac_handoff/crops/run3.events.jsonl`
was written by an earlier policy version: it has 27 cue events where the current code writes 18 (the nine that went are
a standing "Sink lower" before the first count and idle praise during the uncounted pushup set). Reps
and session states are identical. Regenerate the logs rather than diffing against the shipped ones.

```
python scripts/test_perception.py --video data/tmp/mac_handoff/segments/run3_squats.mp4 --plan squats:10   # 32 s, prints PASS
python scripts/test_voice.py                                                                              # 31 s, 83/83 offline, no key needed
```
(`test_perception.py` without `--video` looks for `data/tmp/squats_clip.mp4`, which is not in the bundle.)

Hear the coach on a recording, no call: `--audio-out` mixes the bank clips at their cue times into a wav.
```
python -m facetime.replay_local --video data/tmp/mac_handoff/segments/run3_squats.mp4 --plan "squats:10" --fast --audio-out data/tmp/squats.wav
```

**Your own phone video must be 10 fps first.** The replay treats every file as 10 fps and loads every frame into RAM (a raw
30 fps 1080p clip plays as 3x slow motion, counts land late, sets time out, and two minutes of it is 20 GB of frames). Convert
once; the pipeline resizes to short side 288 itself, this just keeps the file small and the clock right:
```
ffmpeg -i IMG_1234.mov -vf "fps=10,scale='if(gt(iw,ih),-2,288)':'if(gt(iw,ih),288,-2)'" -an -c:v libx264 -preset fast -crf 23 data/tmp/mine.mp4
python -m facetime.replay_local --video data/tmp/mine.mp4 --plan "squats:10" --fast --events data/tmp/mine.events.jsonl
```
(A 30 fps 1080 copy of `run3_squats` replayed as 111.9 s and timed out at seven; after the one-liner it gave the
same nine reps at the same clip times as the original.) Film in portrait, phone propped at knee height 2 to 3 m away, whole body
in frame, stand still a second before the set (README section 7).

## Decision table

| I want to ... | change this file / use this flag | latency class |
|---|---|---|
| run a different workout | `starter/plan.txt` or `--plan "name:count,..."` (names in section 1) | fast path, 0.2 s |
| change what a correction says | the `cue` strings in `src/rtcoach/cues.json`, then `python -m facetime.cues_tts --backend gemini` | fast path |
| say something new at a moment of my choosing | a new entry in `cues.json` (any key) + re-render, then `say_now(text)` in `starter/handlers.py` | fast path |
| speak at halfway, after a correction streak, at the end | `starter/handlers.py` `on_event` with `say_now` (bank) or `ask_voice` (model) | 0.2 s / 1 s |
| change how Brodie talks between sets | `starter/persona.md`, `ask_voice(prompt)` in handlers, `--voice gemini` | model, 1 s |
| let the caller skip ahead with a gesture | `--gesture-next` (T-pose 1.5 s), a `gesture` event, 4 s of open floor | perception event |
| drop the plan and let the coach guess | `--auto` (`--auto-classes all` for every class) | perception, 3 s to declare |
| develop without calling | `python -m facetime.replay_local --events ...` on a crop, then the harness in section 3 | offline |
| coach something that is not reps (a recipe, a checklist) | `--perception off --vision --voice gemini --persona ... --plan-file ...` (section 8) | model, 1 s; video 1 fps |
| a different voice or accent for the bank | `cues_tts --voice NAME` / `--style "(direction) "` into `--out-dir`, then `CUE_AUDIO_DIR=...` | fast path |
| a longer rest, shorter sets, fewer corrections | `--rest-s`, `--set-timeout`, `--tempo`, `--params '{"gap": 6}'`, `--no-announce` | no code |
| count one exercise with no target | `--exercise squats` (single mode) | fast path |
| a live dashboard of events | `--publish-data` in a LiveKit room you own (not a Chert room) + your own listener | events |
| a laptop-only demo with no phone | `python -m facetime.webcam_local --camera 0 --plan ... --show` (no voice, no handlers) | fast path |
| hand in your call logs | leave `data/tmp/worker_events.jsonl` and `worker_cues.jsonl` on (default) | - |

## The catalogue

Each entry: what it is, the extension point, the command, the latency class, a first version you can have in 30 minutes, and
the limits as measured, not as hoped.

### 1. Your own session plan

**What.** A different list of exercises and targets, in `starter/plan.txt` (read whenever `--plan` and `PLAN` are absent) or on the
command line. Syntax `name:count,name:count`; a typo exits at start with the list of accepted names.

**Names the plan accepts** (27; `facetime/pipeline.py accepted_names()`), and what the counter does with each:
- Counted, validated on a device (the four demo exercises; masks and gates in `src/rtcoach/repcount2.py`): `squats` (vote over
  hip height, knee angle, knee width, shoulder height, shoulder to hip; upright gate; settle rule; walk veto), `pushups` (vote over
  elbow, elbow width, shoulder height and width, wrist height; prone gate), `lunges` or `walking lunges` (upright gate, settle
  rule, the first count must come from a leg or torso tracker; one count per leg), `jumping jacks` (full five-signal vote, no gate).
- Counted, QEVD-trained only, never on a device: `squat jumps`, `squat kicks`, `boxing squat punches`, `lunge jumps`,
  `good mornings`, `good morning beginner`, `plank taps`, `puddle jumps`, `floor touches`, `toe touchers`. These use the full
  top-5 vote with no body-state gate, so any repetitive movement counts (`squat jumps:5` on the jumping-jacks segment
  counts five).
- Accepted but not counted (corrections only; the set runs to `--set-timeout` and closes with "Nice work"): `high knees`,
  `butt kickers`, `quick feet`, `mountain climbers`, `moving plank`, `air jump rope`, `shoulder gators`, `standing kicks`,
  `quad stretch left`, `quad stretch right`, `armcrosschest left`, `armcrosschest right` (the four stretches get the side cues).
  There is no `name:30s` syntax; a timed block is `high knees:1 --set-timeout 30` and the timeout applies to every item.

**Corrections the bank has for the demo four** (`cues.json`, 556 QEVD fault labels over 50 exercises, each mapped to a short
string): squats "Sink lower", "Squat deeper", "Hips back", "Keep back straight"; pushups "Go lower", "Hips down", "Keep going",
"Keep knees up", "Stay in plank"; jumping jacks "Use your arms", "Bring arms up", "Reach arms higher", "Jump feet wider",
"Step feet wider", "Add the jump", "Move your legs", "Sync arms and legs", "Keep movements sharp", "Pick up the pace"; lunges
"Lunge lower", "Sink deeper", "Step deeper", "Keep chest up", "Bend back knee", "Alternate your legs", "Pick up pace"; plus the
tempo cues "Slow down" and "Pick up the pace" for all four.

**Command.**
```
python -m facetime.replay_local --video data/tmp/mac_handoff/crops/run2.mp4 --plan "pushups:8,jumping jacks:20,lunges:10" --fast --events data/tmp/plan.events.jsonl
python -m facetime.worker --plan "pushups:8,jumping jacks:20,lunges:10" dev          # on a call (README section 3)
```
**Latency.** Fast path throughout (counts 0.81 s p50 after the rep bottom on the phone page; announces and milestones 0.2 s).

**30 minutes.** Write the plan, replay it on `run2.mp4` or `live.mp4`, read the `session` events to see each set open, close and
rest, and set `--set-timeout 45` if a set you expect to be short keeps running to 75 s (pushups did on a test call).

**Limits.** Shallow reps are undercounted (exact 0.87 on MM-Fit, 0.59 against coaches' spoken numbers on home sessions); pushups
from a propped front camera can read +2; jacks +1 on hands to the head; a squat within a second of walking in is lost; the
first count needs 0.8 s of standing still; lunges count one per leg; the non-demo countable names have no device numbers at all.

### 2. New cue wording and new bank strings

**What.** Change what a correction says, or add a string of your own that `say_now` can play. The bank is keyed on the exact string:
nothing is synthesised at run time.

**Extension point.** `src/rtcoach/cues.json` (`{"<exercise> - <label>": {"cue": "Sink lower", "kind": "fault"}}`): edit a `cue` value to
reword a correction. For a brand-new line add an entry under any key of your own (the Coach only looks keys up by classifier
label, so `"brodie extras - halfway": {"cue": "Halfway there", "kind": "good"}` is harmless and puts "Halfway there" in the
vocabulary). Then render only what is missing (`GEMINI_API_KEY` exported; at the hackathon that is your team token
and `GEMINI_BASE_URL` must be exported too, so the render goes through Brodie's Gemini relay like the worker's Live session does):
```
python -m facetime.cues_tts --backend gemini --dry-run     # lists the strings to render, no API call
python -m facetime.cues_tts --backend gemini               # renders them into data/cues_audio_gemini, idempotent
```
Every start line then reports coverage: `[cues] bank .../data/cues_audio_gemini: 380 clips, covers 378/378`. A string that is not
rendered is skipped with a warning (`no cue audio for 'Halfway there'`) and `say_now` returns a record with `status: 'missing'`.

**Latency.** Fast path, 0.2 s.

**Cost and time.** About 55 audio tokens per clip; the whole 369-string bank cost $0.16 and the Australian re-render
$0.17 plus re-renders; a handful of new strings is cents. A full bank takes about 4 minutes.

**30 minutes.** Reword the four squat corrections to your coach's voice, add "Halfway there", render, replay `run3_squats.mp4`
and hear it with `--audio-out`.

**Limits.** Counts ("One" to "Twenty", digits to 30), "That's N, nice work", "Let's do X", "Rest for a moment", the praise list
("Nice work", "Looking strong", "Good pace", "That's it", "Keep it up", "Great rhythm"), the visibility lines and "Okay, moving
on" are generated in code (`facetime/cues_tts.py vocabulary()`, `src/rtcoach/coach.py PRAISE`, `facetime/session.py`), not in
`cues.json`: rewording them means editing those strings in code and re-rendering. Keep the plain Puck voice for the strings
you add so the bank stays one voice (an edited `cues.json` key set still carries the QEVD label names; section 11 of the README).

### 3. Handlers: bank lines at moments the shipped coach is silent

**What.** `starter/handlers.py` `on_event(event, say_now, ask_voice)` is called for every event, in order, on the worker's event
loop (return quickly, never block; exceptions are logged and ignored). It loads by default whenever the file exists, in every
mode (`--handlers PATH` for another file, `--handlers none` for nothing). Two verbs: `say_now(text, intent='other')` plays a bank
string through the fast path at priority 'other' (never over a count; `intent='announce'` replaces a pending cue), about 0.2 s;
`ask_voice(prompt)` asks the model (about 1 s to first audio), spoken at once while the floor is open, else queued for the next
opening (dropped when a set starts first or when older than 15 s; at most 3 queued); returns `'spoken'`, `'queued'`, or `None`
without `--voice gemini`.

**Moments the shipped coach does not use** (the default handler asks for one sentence at `rest` and a close at `done`):
the halfway rep (`rep.n == target // 2`, the target from the `session` event at `wait`); a pep line at `rest` after two or more
`cue` events with `intent: correction` in the set; a cool-down ask at `done` (the floor stays open after "All done" until the
caller hangs up, so a timed `later(seconds, prompt, ask_voice)` as in `starter/examples/meal_prep/handlers.py` works there too);
a bank countdown in a timed block (`say_now('Ten')` ... from a timer, in a non-counted item nothing competes with it).

**Develop against a recording, no call.** `replay_local` does not load handlers; it writes the events, and this feeds them to
your file (paste as is; the handler is the one below, tested on `run3`):
```
python - data/tmp/run3.events.jsonl starter/handlers.py <<'EOF'
import json, sys, importlib.util
events, path = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location('h', path); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
say = lambda text, intent='other': print(f'          -> say_now {text!r}') or {'status': 'ok'}
ask = lambda prompt: print(f'          -> ask_voice {prompt[:60]!r}...') or 'queued'
for line in open(events):
    ev = json.loads(line)
    if ev['type'] == 'session': print(f"{ev['t']:7.1f}  session {ev['state']:6} {ev.get('exercise') or '-':14} target {ev.get('target')} count {ev.get('count')}")
    elif ev['type'] == 'cue': print(f"{ev['t']:7.1f}  cue     {ev['intent']:10} {ev['text']!r}")
    h.on_event(ev, say, ask)
EOF
```
A 30-minute handler (halfway marker, pep line after a correction streak, cool-down ask):
```python
STATE = {'target': 0, 'corrections': 0}

def on_event(event, say_now, ask_voice):
    t = event['type']
    if t == 'session':
        st = event.get('state')
        if st == 'wait':                                   # a new set is announced: remember its target, reset the streak
            STATE['target'] = event.get('target') or 0; STATE['corrections'] = 0
        elif st == 'rest':
            if STATE['corrections'] >= 2: say_now('Keep it up')          # bank string, about 0.2 s, never over a count
            ask_voice('The set just ended: in ONE short sentence, say how it went and name what comes next.')
        elif st == 'done':
            ask_voice('The session is over: close it warmly in one short sentence and suggest one 30 second stretch.')
    elif t == 'rep' and STATE['target'] >= 6 and event['n'] == STATE['target'] // 2:
        say_now('Looking strong')                                        # the halfway rep
    elif t == 'cue' and event.get('intent') == 'correction':
        STATE['corrections'] += 1
```
On `run3` this says "Looking strong" right after "Five" of the squat set and asks the voice at the 78.0 s rest. On a call the
default command picks the file up: `python -m facetime.worker --voice gemini dev` (without `--voice gemini` only `say_now` works).

**Latency.** `say_now` 0.2 s; `ask_voice` 1 s plus whatever the floor rules impose.

**Limits.** Only bank strings (section 2 to add one). `cue` events are the transcript of what the fast path already said, do not
repeat them. Never ask the voice during `active`: the first count of a set cuts a reply still playing and drops queued asks.
A `visibility` pause is not a slot. `say_now` during a set is legal but shares the track with the counts: keep it to one short
string, the player plays it right after the count in flight.

### 4. A between-sets conversational persona

**What.** Who Brodie is when it talks: `starter/persona.md` is the system prompt of the Gemini Live session (`--persona PATH` for
another file). At every `rest` and at `done` the model also receives a context line it did not ask for, built from the events
(`facetime/voice.py SetSummary`): "Set finished: squats, 10 of 10 reps counted. Corrections spoken: Squat deeper, Sink lower.
Faults seen: squats - shallow x3. Next up: pushups x 8 after a five second rest. You may say one or two short sentences now.";
after a gesture "The caller held a T-pose (the "next" gesture). You have a few seconds; one short line at most."; at `done` the
whole plan's counts and "Close the session warmly in one or two sentences." Nothing is sent at `wait` (a line there can be heard
under the first counts). The model replies only when a handler calls `ask_voice` or the caller talks.

**Command.**
```
python -m facetime.worker --voice gemini dev                                             # starter/persona.md, starter/handlers.py
python -m facetime.worker --voice gemini --persona my_persona.md --handlers my_handlers.py dev
python scripts/test_voice.py --live                                                      # the model session for real, needs GEMINI_API_KEY (41 checks on the Mac)
```
`--voice-name` (env `GEMINI_VOICE`, default Puck) picks another Gemini voice; `GEMINI_LANGUAGE` (default `en-US`) and the persona's
"Always speak English" line set the language; `--log-caller` keeps the caller's transcripts in the `voice` events (off by default).

**Latency.** Model: 0.9 to 1.3 s to first audio with `gemini-3.8-live`; the model session starts 0.9 to 1.0 s after the caller is
bound; a rest is 5 s plus the wait before the next first rep, so ask for one sentence.

**30 minutes.** Rewrite `persona.md` (a drill sergeant, a physio, a parent coaching a child), keep the English line and the
"never count, never correct" rule, change the two `ask_voice` prompts in `handlers.py` to match, run `test_voice.py --live` to
hear a turn, then one call.

**Limits.** The floor is enforced in code, not by the prompt: the model cannot speak during a set whatever the persona says.
Use the plain Puck voice for both the bank and the conversation; mixing voices sounds like two speakers. Another
language only changes the conversation: the bank stays English (section 9 for a partial fix). The fixes for a reply over the
counts and a reply in German pass offline and live checks but have not been verified on a device yet.

### 5. Gesture-driven flows (the T-pose)

**What.** With `--gesture-next` the pipeline runs a T-pose detector (`facetime/probe.py TPose`: both arms straight out at shoulder
height, held 1.5 s, one missed frame allowed, then 5 s before it can fire again). On a fire the session acts: in `active` the set
closes early with "Okay, moving on", then the rest and the next item; in `rest` the next exercise starts at once; in `wait` the
Coach is armed; after `done` nothing. A `gesture {name: tpose}` event is emitted and, with `--voice gemini`, the floor opens for
4 s (not during a set), so a handler can `ask_voice` on it ("the caller wants the next thing").

**Command.**
```
python -m facetime.replay_local --video data/tmp/mine.mp4 --plan "squats:5,lunges:6" --fast --gesture-next --events data/tmp/g.events.jsonl
python -m facetime.worker --gesture-next --voice gemini dev
```
**Latency.** Perception event: 1.5 s of hold plus one frame; the line itself 0.2 s.

**30 minutes.** Record yourself: walk in, three squats, hold a T for two full seconds facing the camera, drop the arms, then the
lunges. Replay with `--gesture-next` and look for the `gesture` event and "Okay, moving on". None of the shipped recordings has a
T-pose (`live.mp4` with `--gesture-next` gives 0 gesture events), so this needs your own clip.

**Limits.** "Okay, moving on" has fired on a device only after "All done" (twice, nothing left to move on from); mid-set
it is unexercised on a call. The detector is ignored while the whole body is not visible, and alternating raises and T-poses
with no pause fires nothing. The arms-up probe (`--latency-probe`) is the other gesture: a beep and
a 4 s floor window, never during a set, off by default because a jumping jack is an arm raise.

### 6. Auto exercise detection

**What.** No plan, no exercise: the classifier's exercise head, smoothed and summed over families, declares an exercise after
3 s above 0.5 ("Looks like jumping jacks"), the counter and corrections switch to it, and a challenger can take over every 10 s.
`--auto-classes demo` (default) declares only the four demo exercises and folds look-alikes in; `all` keeps every class.

**Command.**
```
python -m facetime.replay_local --video data/tmp/mac_handoff/segments/run3_jacks.mp4 --fast --events data/tmp/auto.events.jsonl   # no --plan: auto
python -m facetime.worker --auto dev
```
On `run3_jacks.mp4`: "Looks like jumping jacks" at 4.5 s, then One to Seventeen.

**Latency.** 3 s to declare, then the fast path.

**30 minutes.** A "just start moving" coach: `--auto` plus a handler that asks the voice to name the exercise on each `announce` cue
with intent `announce` and text starting "Looks like".

**Limits.** The head is about 0.75 accurate on real sessions (`exercise_seen` said "floor touches" at 0.94 while the caller stood);
there is no target in auto mode, so no "That's ten" and no rest; trust the plan for anything you demo.

### 7. Replay-driven development

**What.** Any 10 fps recording goes through the same pipeline and session as a call and writes the same events (schema v1, `t` is
the clip time). The bundle has three sample calls cropped to the camera view: `data/tmp/mac_handoff/crops/run3.mp4` (clean
full plan, phone page), `run2.mp4` (end of squats, pushups with phone handling, 20 jacks, 10 lunges), `live.mp4` (the full plan
on a real FaceTime call), and per-exercise cuts in `data/tmp/mac_handoff/segments/` (`run3_squats`, `run3_pushups`, `run3_jacks`,
`run3_lunges`, `run2_*`, `live_*`).

**Commands.** The first block of this page, plus:
```
python -m facetime.replay_local --video data/tmp/mac_handoff/segments/run3_squats.mp4 --exercise squats --fast      # single mode: counts, no target
python -m facetime.replay_local --video data/tmp/mac_handoff/crops/live.mp4 --plan "squats:10,pushups:8,jumping jacks:20,lunges:10" --fast --set-timeout 45 --rest-s 8 --events data/tmp/live.events.jsonl
python scripts/e2e_from_recording.py film.mp4 --crop W:H:X:Y --t0 11 --t1 36 --exercise squats                     # rep bottom to count heard, from a film of the phone
```
**Latency.** None: `--fast` runs at about 28 fps here; without it the replay paces at wall clock like the live agent.

**30 minutes.** Record one set of each exercise you plan to demo with the phone propped as on the call, convert, replay, and read
the counts before you ever dial.

**Limits.** A replay has no transport timing (`status.fps_recv`, `dropped` and the delay are null; `probe` has no `on_box_ms`);
handlers are not run by `replay_local` (section 3 harness); the visibility rule is the same as live, so a laptop-framed clip
will pause.

### 8. A voice-first skill with vision, perception off

**What.** `--perception off` runs no pose model, no counter, no script: the worker joins, publishes the track, binds to the first
caller, starts the model session, and the floor is always open (`--floor open` is then the default): the caller's audio is always
forwarded, `ask_voice` is spoken at once, nothing is cut. `--vision` hands the caller's video to the model at 1 fps (`--vision-fps`,
at most 2; each frame a JPEG scaled to 1024x576). The plan file is appended to the persona as free text, not parsed. The handler
sees one synthetic `session {state: "open", mode: "voice"}` event when the session is up, a `status` every 5 s, and `voice` events.
The example is a cooking coach: `starter/examples/meal_prep/` (a persona that names what it sees before it instructs, a 15-minute
garlic butter pasta as timed steps in `recipe.md`, and a handler that greets on the open event with a `later(seconds, prompt,
ask_voice)` helper for timed nudges).

**Command** (reports `mode voice (perception off, vision on, floor open)` at start; the call itself needs the keys):
```
python -m facetime.worker --perception off --vision --voice gemini --persona starter/examples/meal_prep/persona.md --plan-file starter/examples/meal_prep/recipe.md --handlers starter/examples/meal_prep/handlers.py dev
```
**Latency.** Everything through the model: about 1 s to first audio; the model sees a change in the scene within a second or two.

**30 minutes.** Copy `starter/examples/meal_prep/` to `starter/examples/<yours>/`, replace the recipe with your checklist (a
stretching routine, a desk setup check, a packing list), and change the greeting prompt. Tell the caller how to prop the phone:
the model only sees what the camera sees.

**Limits.** No fast path: every word costs about 1 s. `say_now` still plays bank strings (counts, "Nice work", "Rest for a
moment") and nothing else. No probe, no T-pose, no `rep`, `fault` or `visibility` events. The model cannot read a thermometer
or taste. This mode has been tested offline only (fake room, fake frames, a no-room model session with one pushed frame); it has
not been on a call, phone page or FaceTime. Expect to tune the persona on the first call. `--vision` with perception on is wired in code but untested anywhere,
and the floor stays gated there.

### 9. The bank in another voice, accent or language

**What.** Re-render the 377 strings with a Gemini TTS voice or a stage direction into a new directory and select it with
`CUE_AUDIO_DIR`. The Australian-direction bank already exists in the bundle (`data/cues_audio_gemini_au`, 378 clips, "(Australian
accent, warm, upbeat trainer)"); by default the player prefers the plain Puck bank, then the Australian one.

**Command.**
```
CUE_AUDIO_DIR=data/cues_audio_gemini_au python -m facetime.worker dev                                                   # the Australian bank
python -m facetime.cues_tts --backend gemini --style "(Calm, slow, physiotherapist) " --out-dir data/cues_audio_gemini_calm --dry-run
python -m facetime.cues_tts --backend gemini --style "(Calm, slow, physiotherapist) " --out-dir data/cues_audio_gemini_calm
CUE_AUDIO_DIR=data/cues_audio_gemini_calm python -m facetime.replay_local --video data/tmp/mac_handoff/segments/run3_squats.mp4 --plan "squats:10" --fast --audio-out data/tmp/calm.wav
```
`--voice NAME` picks another prebuilt Gemini voice (default Puck). The bracketed direction is not spoken and not part of the
manifest key; a spoken prefix would be. Always a new `--out-dir`: banks never mix.

**Cost and time.** $0.16 to $0.21 per full bank, about 4 minutes. The
Australian bank needed a duration gate: 21 clips came out far longer than their Puck twins (the direction read aloud or drawn
out) and were re-rendered up to six times each; check clip durations against the Puck manifest before you trust a new bank.

**30 minutes.** A dry run, a render, a replay with `--audio-out`, listen.

**Limits.** Another language is partial: translate the `cue` strings in `cues.json` and the bank follows, but the counts,
"That's N, nice work", "Let's do X" and the praise list are English in code (section 2). A bank voice other than Puck and the
Gemini Live voice will sound like two people; change both (`--voice-name`) or neither. That is why the
default is the plain Puck bank, not the Australian one.

### 10. Logging and telemetry hand-in

**What.** Two local files by default, nothing else: `data/tmp/worker_events.jsonl` (every event plus `room`, `call_id`, `job`;
`--event-log` / `RTCOACH_EVENT_LOG`) and `data/tmp/worker_cues.jsonl` (cues played or dropped; `--cue-log` / `RTCOACH_CUE_LOG`).
No frames, no audio, no transcripts, no handles; the caller's words stay out unless `--log-caller`. Opt out with
`--event-log /dev/null --cue-log /dev/null`. Brodie asks you to leave them on and hand them in after the event, keyed on the room
name and the time of the call (`call_id` is null on the phone page, where the caller is plain `chert-bridge`; a Chert call also has
its `call_id`). A replay's `--events` file is the same schema, so a call log replays through the section 3 harness.

**30 minutes.** A set summary script over `worker_events.jsonl`: reps per set from `session` events at `rest`, corrections from
`cue` events with `intent: correction`, the dominant `fault` label; print one line per room (and per call id on a Chert line).

**Limits.** `t` is per process; `t_wall` is the only cross-process clock; `seq` restarts per job.

### 11. Your own LiveKit room: the phone page (the main path), a data channel, the laptop webcam

Most teams have no Chert line, and even the three that do get a dozen attempts an hour only in a room of their own: the phone
page over your LiveKit project is the main path (README section 3 has the two terminals and what prints; this is the short form).

- **The phone page** (`facetime/phone_test.html`, `facetime/mint_token.py`): a phone or a Mac joins a room of your project,
  publishes its camera as one track named `facetime-remote-video` (asking 1280x720 at 30 fps; a portrait phone delivers
  720x1280, the worker samples at 10 fps) and plays every audio track it subscribes to. Run the worker with automatic dispatch
  (`AGENT_NAME=` empty in `.env.local`, or `--agent-name ""`) so it joins every new room, start it first, and mint a fresh room
  name per attempt (dispatch fires when the room is created, which the phone's join does):
  ```
  python -m facetime.worker --agent-name "" --plan "squats:10" dev
  python -m facetime.mint_token --room rtcoach-team-1 --url      # three lines: the token, the ?url=...&token=...&room=... query string, and the https phone link
  ```
  `mint_token` needs `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` exported (and `LIVEKIT_URL` for the link); nothing loads
  `.env.local` for you. The THIRD printed line is the phone link: `https://james-72765--rtcoach-webhook.modal.run/phone?url=...`
  served from our host, a static page whose `url` and `token` are your own project (`room` is informational, the token carries
  it). AirDrop it, Messages it or QR it; `&facing=user` for the front camera, `&autojoin=1` to skip the tap. On a Mac, open
  `facetime/phone_test.html` as a file in Safari with the same query string (file:// is a secure context for the camera; http
  from a laptop will not get a phone's camera, hence https). The worker binds to the first non-agent participant that publishes
  video and prints `bound to 'chert-bridge' (call id n/a): video 'facetime-remote-video' 720x1280`, then `first frame 720x1280`,
  then "Let's do squats" is heard. The page's "Coach present" indicator is green only for a participant whose identity equals
  `agent` exactly (default `brodie-agent`, `&agent=<identity>` to change it); a worker under `dev` joins as `agent-<job id>`, so
  it stays "no" on a working call and "Coach audio: playing from agent-..." is the real signal. `python -m facetime.worker
  connect --room <room> --participant-identity brodie-agent` joins one room directly under that identity and makes it green.
  The job ends 2 s after the phone leaves (`--leave-grace-s -1` keeps it). `call_id` is null on this path (section 10).
- **A data channel for a dashboard** (`--publish-data`, topic `--data-topic`, default `brodie.events`): every event as JSON on the
  room's reliable data channel. A third participant (a token from `mint_token.py --identity dashboard`) reads it with the
  LiveKit client's `data_received` event; no listener ships in the repo. Only in rooms you own (a Chert room has no third seat);
  untested on a call.
- **The laptop webcam** (`facetime/webcam_local.py`): the camera, the same pipeline and session, cues on the speaker through
  `sounddevice` (or a wav when there is no output device). Tested with a file as the camera:
  ```
  python -m facetime.webcam_local --camera 0 --plan "squats:10,pushups:8" --show
  python -m facetime.webcam_local --camera data/tmp/mac_handoff/segments/run3_squats.mp4 --plan "squats:10" --audio wav --audio-out data/tmp/webcam.wav --max-s 40
  ```
  No handlers, no voice, no events queue: it is for the perception and the bank only. A laptop camera cannot frame a standing
  adult top to bottom at 2 to 3 m (README section 7).
- **A fake bridge** (`facetime/fake_bridge.py`): publishes an mp4 into a room as the bridge and records the coach's audio with
  onset times, for latency numbers over your own LiveKit project; pair with `facetime/latency_report.py`. Used on the demo coach,
  not re-run with the worker.
- `python -m facetime.worker ... connect --room <room> [--participant-identity brodie-agent]` joins one room directly, no
  dispatch (a LiveKit Agents CLI subcommand; the identity is what the phone page's indicator looks for).

### 12. Tuning the shipped coach without code

All on `facetime.worker`, `facetime.replay_local` and `facetime.webcam_local` alike: `--rest-s 5` (the rest), `--set-timeout 75`
(close a set at this many seconds; 45 suits short sets), `--tempo '{"squats": 2.0}'` (seconds per rep; defaults squats 2.5,
pushups 2.0, jumping jacks 0.9, lunges 2.0; "Slow down" below 0.6x, "Pick up the pace" above 1.8x, at most once per 15 s),
`--no-announce` (no "Let's do X"), `--params '{"gap": 6, "praise_after": 20, "max_cues_per_label": 1}'` (the Coach policy, defaults
in `facetime/pipeline.py POLICY_DEFAULT`: gap 3 s between cues, cooldown 8 s per label, praise every 8 s, 3 cues per label per
set), `--trigger none` (the threshold rule instead of the learned trigger; fewer, later corrections), `--counter e1` (the older
pose-PCA counter), `--exercise squats` (single mode: one exercise for the whole call, no target, no rest), `--pose lite` (faster,
less accurate pose model). Replay first: the same flags give the same cues on a recording.

### 13. Bring your own mouth (advanced)

`--events-only` (or `EVENTS_ONLY=1`) publishes no audio at all: no cue player, no track. The events still flow to `job.events`
(register `@facetime.worker.consumer async def fn(job): ev = await job.events.get()` in a module you import before the worker
starts), to the JSONL log and, with `--publish-data`, to the room. You then own the only voice in the room and must publish
it from the same process. Honest cost: you lose the 0.2 s fast path and must design counts around 1 s (the Gemini-in-the-loop
baseline matched 0 of 16 counts at jacks cadence). A day's work with no shipped example; the shipped `--voice gemini` path is the
same idea with the fast path kept.

## What Brodie cannot do today

- **No new fault detectors.** The classifier heads are fixed: the 50 QEVD exercises and their 556 labels, trained under a licence
  whose data you do not have. A new exercise or a new fault needs training data; today you can only reword what exists, change
  when it fires (`--params`, `--trigger`) and which labels count (`--thr`).
- **No model-driven counts.** Counts come from the pose counter on the fast path; a realtime model answers in 0.8 to 1.2 s, one to
  two reps late at jacks cadence. Keep counts and form cues on the fast path, conversation on the model.
- **Vision for the model is about 1 fps** (at most 2), JPEG, and only tested offline; it is a scene reader, not a rep watcher.
- **Two seats per room on a Chert line.** The caller's bridge and your worker. No sidecar, no dashboard participant, no second audio
  publisher on a Chert call; `--publish-data` has no reader there. In your own LiveKit room there is no such limit (keep one audio
  publisher).
- **A Chert line is Offline for a few minutes after each call** (Chert's worker refreshes on its cron cycle); calls in that window ring
  out. The phone page has no such wait (a fresh room name per attempt); recordings are the inner loop on both.
- **No Windows.** macOS or Linux, Python 3.12; Node 22.23.2 or 24.x for the Chert CLI (the three Chert teams only).
- **The voice cannot speak during a set** whatever the persona says, and a reply still playing is cut by the first count.
- **Only bank strings are spoken on the fast path**; nothing is synthesised at run time. A new string is a render step (section 2).
- **Counting is validated on a device for four exercises only**; the other nine countable names have QEVD numbers at best.
- **No time-based plan items** (`name:count` only); a timed block is `name:1 --set-timeout N`, and the timeout is global.
- **The T-pose mid-set and the latest voice fixes have not been verified on a device**; the meal-prep mode has not been on a call.
- **The avatar face (`--avatar`) and the mirror (`--mirror`: the caller's own frame with the pose, the count and the last cue drawn
  on it as the coach's video) are on the demo coach (`facetime/coach_agent.py`), not on the starter worker.**
- **No hot reload**: restart Terminal A after every code change (the `dev mode is deprecated, use lk agent dev` notice from
  livekit-agents 1.8.5 is expected; ignore it); on a Chert line also restart `chert dev` so no prepared room runs old code.

## Latency classes, in one place

| path | measured | where |
|---|---|---|
| bank line, decision to heard | 145 to 225 ms over the LiveKit SFU | PROGRESS.md loopbacks |
| rep bottom to count heard | 0.81 s p50 (0.72 to 0.88) on the phone page; about 0.3 s is the counter's reversal detection | `scripts/e2e_from_recording.py`, a sample recording |
| receive to decision, perception | 27 to 40 ms p50 at 10 fps on the Mac over the CLI path | reports/MAC_TEST_2026-10-08.md sections 5 and 6 |
| model, prompt to first audio | 0.9 to 1.3 s (`gemini-3.8-live`), 1.2 to 1.3 s in the live tests | `facetime/voice.py`, `scripts/test_voice.py --live` |
| model session up after the caller is bound | 0.9 to 1.0 s | Mac section 6 |
| T-pose | 1.5 s hold, then 5 s refractory | `facetime/probe.py` |
| auto-detect | 3 s above 0.5 to declare | `facetime/session.py` |
| ring to answer on a Chert line | 8.6 to 8.7 s (Chert's side), then a 0.1 s warm-room accept, first cue about 7 s after the accept | Mac sections 5 and 6 |
