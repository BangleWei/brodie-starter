"""Learned correction trigger for the live coach (E3b, scripts/e3b_trigger.py). numpy only: the weights are exported
to data/models/trigger_lr.json by the training script, no sklearn/torch at runtime.

Per evaluated frame (the classifier's label/exercise probability vectors) it returns p = P(a correction is due within
the next 1.5 s). It decides only WHEN to speak: the cue text comes from the coach's own rule (highest EMA fault label of
the current exercise, else the global top fault label), exposed here as state['choice'].

    m=load_model(); st=init_state(m, exercise)              # once per episode / exercise (resets the temporal context)
    for each frame i (episode-relative, 10 fps) with classifier output lab, ex:
        p=predict_frame(st, lab, ex)                           # updates the EMA / max / time state
        fire,label=decide(st, p, i)                            # tau, gap and cooldown in frames; label = labs index
        ... if the coach speaks anything else (praise, count) call note_said(st, i) so the gap applies."""
import json, os, numpy as np
from .config import DATA
CUES=json.load(open(os.path.join(os.path.dirname(__file__),'cues.json')))
def load_model(path=None):
    m=json.load(open(path or os.path.join(DATA,'models','trigger_lr.json')))
    f=m['features']; f['lab_ex']=np.asarray(f['lab_ex']); f['fault']=np.array([CUES.get(l,{}).get('kind')=='fault' and bool(CUES.get(l,{}).get('cue')) for l in f['labs']])
    w=m['model']
    if w['kind']=='lr': w['w']=np.asarray(w['w'],np.float32); w['b']=float(w['b'])
    else: w['W1']=np.asarray(w['W1'],np.float32); w['b1']=np.asarray(w['b1'],np.float32); w['W2']=np.asarray(w['W2'],np.float32); w['b2']=float(w['b2'])
    return m
def init_state(model, exercise=None):
    """exercise: classifier exercise name (rtcoach.exmap.LR2CLF value) or None if unknown."""
    f=model['features']; ei=f['exs'].index(exercise) if exercise in f['exs'] else None
    m=f['fault'] if ei is None else (f['fault']&(f['lab_ex']==ei))
    if not m.any(): m=f['fault']
    exo=np.zeros(len(f['exs']),np.float32)
    if ei is not None: exo[ei]=1
    return dict(model=model, ei=ei, mask=m, sel=np.where(m)[0], exo=exo, ema=None, ring=[], k=0, last_say=-10**9, last_corr=-10**9, choice=None, p=None)
def predict_frame(state, lab, ex):
    """lab (n_lab,), ex (n_ex,) probabilities for this evaluated frame -> p in [0,1]. Call once per evaluated frame, in
    order, at the stride the model was trained with (features.every, 1 = every frame)."""
    f=state['model']['features']; lab=np.asarray(lab,np.float32); ex=np.asarray(ex,np.float32)
    a0,a1=f['ema_rates']
    if state['ema'] is None: state['ema']=[lab.astype(np.float64).copy(), lab.astype(np.float64).copy()]
    else:
        state['ema'][0]=a0*state['ema'][0]+(1-a0)*lab; state['ema'][1]=a1*state['ema'][1]+(1-a1)*lab
    ring=state['ring']; ring.append(lab)
    if len(ring)>f['maxwin']: del ring[0]
    e05=state['ema'][0].astype(np.float32); e85=state['ema'][1].astype(np.float32); mx=np.max(ring,0)
    t=min(state['k']*f['every']/f['fps'], f['tcap']); tf=np.array([t/f['tcap'], np.log1p(t)/np.log1p(f['tcap'])],np.float32)
    m=state['mask']; cur=np.array([lab[m].max(),e05[m].max(),e85[m].max(),mx[m].max()],np.float32)
    X=np.concatenate([lab,ex,e05,e85,mx,state['exo'],cur,tf]).astype(np.float32)
    w=state['model']['model']
    if w['kind']=='lr': z=float(X@w['w']+w['b'])
    else: z=float(np.maximum(X@w['W1']+w['b1'],0)@w['W2']+w['b2'])
    p=float(1/(1+np.exp(-z))); state['k']+=1; state['p']=p; state['X']=X   # X: the feature vector, for rtcoach.cue_selector
    state['choice']=int(state['sel'][e05[state['sel']].argmax()])
    return p
def decide(state, p, i):
    """i: episode frame index (10 fps). Returns (fire, label_index). Applies tau, gap (since anything said) and
    cooldown (since the last correction), all in integer frames like the training-time decode."""
    d=state['model']['decode']; fps=state['model']['features']['fps']
    G=int(round(d['gap']*fps)); CD=int(round(d['cooldown']*fps))
    if p>d['tau'] and i-state['last_say']>=G and i-state['last_corr']>=CD:
        state['last_say']=i; state['last_corr']=i; return True, state['choice']
    return False, None
def note_said(state, i):
    """the coach spoke something else (praise, count, start line) at frame i."""
    state['last_say']=i
def cue_text(state, label_index):
    return CUES[state['model']['features']['labs'][label_index]]['cue']
