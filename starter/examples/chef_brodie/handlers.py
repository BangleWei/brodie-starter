"""Chef Brodie handlers (guacamole), adapted from starter/examples/meal_prep. Meal-prep coach handlers (docs/STARTER_README.md "Your own coach: beyond reps"). Run with:

  .venv/bin/python -m facetime.worker --perception off --vision --voice gemini --persona starter/examples/chef_brodie/persona.md \
      --plan-file starter/examples/chef_brodie/recipe.md --handlers starter/examples/chef_brodie/handlers.py dev

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
    """Greet once the session is open, ask the caller to prop the phone so the board is in view, and nudge once if the caller
    goes quiet mid-recipe. The model holds the recipe and runs the steps from what it sees and hears."""
    if event['type'] == 'session' and event.get('state') == 'open' and event.get('mode') == 'voice':
        ask_voice('The call just started. In one or two short sentences: introduce yourself as Chef Brodie, say you are making '
                  'guacamole together, and ask them to prop the phone so you can see the cutting board and their hands. '
                  'Do not start the recipe yet.')
        # One check-in at 4 minutes, roughly when the knife work should be underway.
        later(4 * 60, 'About four minutes in. If the caller is cutting, comment on their grip in one short sentence; '
                      'otherwise ask in one short sentence which step they are on.', ask_voice)
