"""Learned cue selector (E6, scripts/e6_cue_selector.py): given the trigger's feature vector at the frame a correction
fires, pick WHICH of the exercise's fault cues to say. numpy only; weights in data/models/cue_selector.json.

Hook (trigger-mode branch of Coach.step, right after `fire,li=decide(...)` and before the cue_prior / thr branches):
    if fire and P.get('cue_selector'):
        from .cue_selector import select_cue
        li=select_cue(self.tstate['X'], self.ex, fallback=li)     # label index; tstate['X'] is set by predict_frame
The selector returns a label index of the chosen cue (the highest-EMA label among the labels sharing that cue), so
`cue_text(self.tstate, li)` and `why` keep working unchanged. Exercises without a selector return `fallback`."""
import json, os, numpy as np
from .config import DATA
_MODEL=None
def load_model(path=None):
    m=json.load(open(path or os.path.join(DATA,'models','cue_selector.json')))
    for s in m['selectors'].values():
        if 'W' in s: s['W']=np.asarray(s['W'],np.float32); s['b']=np.asarray(s['b'],np.float32)
        if 'idx' in s: s['idx']=np.asarray(s['idx'],np.int64)
    off={}; o=0
    for g,d in m['groups']: off[g]=o; o+=d
    m['off']=off; return m
def _model(model):
    global _MODEL
    if model is not None: return model
    if _MODEL is None: _MODEL=load_model()
    return _MODEL
def cue_probs(x, exercise, model=None):
    """probabilities over the exercise's cue list (selectors[ex]['cues']); None if no selector for the exercise."""
    m=_model(model); s=m['selectors'].get(exercise)
    if s is None: return None
    P=np.zeros(len(s['cues']),np.float64)
    if 'W' not in s:
        if s['classes']: P[s['classes'][0]]=1
        return P
    x=np.asarray(x,np.float32); z=(x[s['idx']] if s['feat']=='own' else x)@s['W'].T+s['b']; z=z-z.max(); p=np.exp(z); p/=p.sum()
    P[s['classes']]=p; return P
def select_cue(state_features, exercise, model=None, fallback=None):
    """state_features: the trigger's feature vector (rtcoach.trigger state['X'], D=2218). Returns a label index."""
    m=_model(model); s=m['selectors'].get(exercise); P=cue_probs(state_features,exercise,m)
    if P is None or not P.any(): return fallback
    cue=s['cues'][int(P.argmax())]; o=m['off']['ema05']; x=np.asarray(state_features,np.float32)
    return int(max(s['cue_labels'][cue], key=lambda j: x[o+j]))
