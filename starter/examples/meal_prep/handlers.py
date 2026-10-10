"""Meal-prep coach handlers (docs/STARTER_README.md "Your own coach: beyond reps"). Run with:

  .venv/bin/python -m facetime.worker --perception off --vision --voice gemini --persona starter/examples/meal_prep/persona.md \
      --plan-file starter/examples/meal_prep/recipe.md --handlers starter/examples/meal_prep/handlers.py dev

With --perception off there are no rep, cue, fault or visibility events: the worker emits ONE synthetic `session` event
{state: "open", exercise: null, count: 0, mode: "voice"} once the model session is up (the caller is bound and the model is
listening and watching), then a `status` event every 5 s and `voice` events for the model's turns. The floor is always open,
so ask_voice(prompt) is spoken at once (about 1 s to first audio) and nothing is ever cut; say_now(text) still plays a bank
string through the fast path (only strings in the bank, data/cues_audio_gemini/manifest.json).

on_event runs on the worker's event loop: return quickly, never block. later(seconds, prompt, ask_voice) schedules an
ask_voice on that same loop (asyncio.call_later) and returns the handle (handle.cancel() to drop it).
"""
import asyncio

TIMERS = []       # handles of scheduled asks, so a handler can cancel them (TIMERS[-1].cancel())


def later(seconds, prompt, ask_voice):
    """Ask the voice `prompt` after `seconds` on the worker's event loop; returns the asyncio TimerHandle (cancel() to drop it).
    Example: later(8 * 60, 'The pasta has been in for 8 minutes: ask the caller to check one strand.', ask_voice)"""
    loop = asyncio.get_running_loop()
    h = loop.call_later(seconds, ask_voice, prompt); TIMERS.append(h)
    return h


def on_event(event, say_now, ask_voice):
    """Greet once the session is open (the model can see and hear the caller): say hello and ask what is on the counter.
    Nothing else: the model holds the recipe and runs the steps from what it sees and hears."""
    if event['type'] == 'session' and event.get('state') == 'open' and event.get('mode') == 'voice':
        ask_voice('The call just started. In one or two short sentences: say hello, say you can see their kitchen, and ask what they '
                  'have on the counter so far. Do not start the recipe yet.')
        # A timed nudge, when a step has a clock on it (the pasta goes in at the caller's "it is in"):
        # later(8 * 60, 'The pasta has been in for about 8 minutes: ask the caller to fish out one strand and bite it.', ask_voice)
